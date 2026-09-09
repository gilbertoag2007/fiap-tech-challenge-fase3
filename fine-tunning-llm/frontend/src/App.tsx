import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Edge,
  Handle,
  Node,
  Position,
  ReactFlow,
} from "@xyflow/react";
import {
  Activity,
  Bot,
  Check,
  ChevronDown,
  ChevronUp,
  Clock3,
  Cpu,
  Database,
  GitBranch,
  HardDrive,
  LoaderCircle,
  Pause,
  Send,
  ShieldCheck,
  Sparkles,
  UserRoundCheck,
  X,
} from "lucide-react";

type Modelo = {
  chave: "llama" | "qwen10" | "qwen80";
  rotulo: string;
  modelo_base: string;
  disponivel_cpu: boolean;
  exige_cuda: boolean;
  carregado: boolean;
  adapter_local: boolean;
};

type Evento = {
  sequencia: number;
  no: string;
  status: string;
  duracao_ms?: number;
  erro?: string;
  revisao?: Revisao;
  resposta?: Resposta;
};

type Revisao = {
  id_execucao: string;
  id_registro: string;
  rascunho: string;
  fontes: string[];
  alertas: string[];
  aviso: string;
};

type Resposta = {
  situacao: "aprovada" | "rejeitada";
  resposta: string | null;
  alertas: string[];
  aviso: string;
};

/** Resumo de um prontuário, usado no seletor de registros. */
type RegistroResumo = {
  id_registro: string;
  especialidade_medica?: string;
  hipotese_clinica?: string;
  diagnostico_confirmado?: string;
  tipo_pergunta?: string;
  contexto_solicitacao?: string;
  pergunta_sugerida?: string;
};

/** Metadados sanitizados de um checkpoint persistido pelo InMemorySaver. */
type Checkpoint = {
  checkpoint_id: string;
  passo: number | null;
  origem: string;
  criado_em: string | null;
  proximos_nos: string[];
  tarefas_pendentes: string[];
  chaves_estado: string[];
  chaves_gravadas: string[];
  interrompido: boolean;
};

const API = import.meta.env.VITE_API_URL ?? "";

const definicaoNos = [
  ["receber_pergunta", "Pergunta", Bot, 30, 30],
  ["carregar_modelo", "Modelo", Cpu, 250, 30],
  ["validar_entrada", "Validar", ShieldCheck, 470, 30],
  ["consultar_registro", "Prontuário", Database, 690, 30],
  ["gerar_rascunho", "Rascunho", Sparkles, 690, 180],
  ["validar_seguranca", "Segurança", ShieldCheck, 470, 180],
  ["solicitar_revisao_humana", "Revisão HITL", UserRoundCheck, 250, 180],
  ["finalizar_aprovacao", "Aprovado", Check, 30, 135],
  ["finalizar_rejeicao", "Rejeitado", X, 30, 235],
] as const;

const arestas: Edge[] = [
  ["receber_pergunta", "carregar_modelo"],
  ["carregar_modelo", "validar_entrada"],
  ["validar_entrada", "consultar_registro"],
  ["consultar_registro", "gerar_rascunho"],
  ["gerar_rascunho", "validar_seguranca"],
  ["validar_seguranca", "solicitar_revisao_humana"],
  ["solicitar_revisao_humana", "finalizar_aprovacao"],
  ["solicitar_revisao_humana", "finalizar_rejeicao"],
].map(([source, target]) => ({
  id: `${source}-${target}`,
  source,
  target,
  animated: true,
  style: { stroke: "#52525b" },
}));

function NoFluxo({
  data,
}: {
  data: {
    label: string;
    status: string;
    duracao?: number;
    Icone: typeof Bot;
  };
}) {
  const { label, status, duracao, Icone } = data;
  return (
    <div className={`flow-node flow-node--${status}`}>
      <Handle type="target" position={Position.Left} />
      <Icone size={16} />
      <div>
        <strong>{label}</strong>
        <span>
          {status === "iniciado" ? "processando" : status.replaceAll("_", " ")}
          {duracao !== undefined && ` · ${duracao.toFixed(0)} ms`}
        </span>
      </div>
      {status === "iniciado" && <LoaderCircle className="spin" size={14} />}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { fluxo: NoFluxo };

function horaCurta(iso: string | null): string {
  if (!iso) return "--:--:--";
  const data = new Date(iso);
  return Number.isNaN(data.getTime())
    ? "--:--:--"
    : data.toLocaleTimeString("pt-BR", { hour12: false });
}

/** Log dos checkpoints do InMemorySaver, fixo no canto inferior direito. */
function PainelCheckpoints({
  checkpoints,
  threadId,
  aberto,
  alternar,
}: {
  checkpoints: Checkpoint[];
  threadId: string;
  aberto: boolean;
  alternar: () => void;
}) {
  const lista = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const elemento = lista.current;
    if (elemento) elemento.scrollTop = elemento.scrollHeight;
  }, [checkpoints.length, aberto]);

  return (
    <section className={`saver ${aberto ? "saver--aberto" : ""}`}>
      <button className="saver-header" onClick={alternar} type="button">
        <HardDrive size={14} />
        <div>
          <strong>InMemorySaver</strong>
          <small>
            {threadId
              ? `thread ${threadId.slice(0, 8)} · ${checkpoints.length} checkpoints`
              : "checkpointer do LangGraph"}
          </small>
        </div>
        {aberto ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
      </button>

      {aberto && (
        <div className="saver-body" ref={lista}>
          {checkpoints.length === 0 && (
            <p className="saver-vazio">
              Nenhum checkpoint gravado. Envie uma pergunta para o grafo
              persistir o estado em memória.
            </p>
          )}
          {checkpoints.map((checkpoint) => (
            <article className="saver-item" key={checkpoint.checkpoint_id}>
              <header>
                <span className={`saver-passo saver-passo--${checkpoint.origem}`}>
                  passo {checkpoint.passo ?? "?"}
                </span>
                <span className="saver-origem">{checkpoint.origem}</span>
                {checkpoint.interrompido && (
                  <span className="saver-interrompido">
                    <Pause size={10} /> interrupt
                  </span>
                )}
                <time>{horaCurta(checkpoint.criado_em)}</time>
              </header>
              <code>{checkpoint.checkpoint_id.slice(0, 18)}</code>
              <dl>
                <div>
                  <dt>next</dt>
                  <dd>
                    {checkpoint.proximos_nos.length > 0
                      ? checkpoint.proximos_nos.join(", ")
                      : "END"}
                  </dd>
                </div>
                <div>
                  <dt>writes</dt>
                  <dd>
                    {checkpoint.chaves_gravadas.length > 0
                      ? checkpoint.chaves_gravadas.join(", ")
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>state</dt>
                  <dd>{checkpoint.chaves_estado.length} chaves</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [modelos, setModelos] = useState<Modelo[]>([]);
  const [modelo, setModelo] = useState<Modelo["chave"]>("qwen10");
  const [cudaDisponivel, setCudaDisponivel] = useState(false);
  const [baseIndisponivel, setBaseIndisponivel] = useState("");
  const [idRegistro, setIdRegistro] = useState("");
  const [sugestoes, setSugestoes] = useState<RegistroResumo[]>([]);
  const [listaAberta, setListaAberta] = useState(false);
  const [registroEscolhido, setRegistroEscolhido] = useState<RegistroResumo | null>(null);
  const [pergunta, setPergunta] = useState(
    "Quais informações clínicas e orientações gerais estão disponíveis neste registro?",
  );
  const [eventos, setEventos] = useState<Evento[]>([]);
  const [status, setStatus] = useState("pronto");
  const [revisao, setRevisao] = useState<Revisao | null>(null);
  const [textoRevisado, setTextoRevisado] = useState("");
  const [resposta, setResposta] = useState<Resposta | null>(null);
  const [erro, setErro] = useState("");
  const [idSessao, setIdSessao] = useState("");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [logsAberto, setLogsAberto] = useState(true);
  const idExecucao = useRef("");
  const eventSource = useRef<EventSource | null>(null);
  const atrasoVisual = useRef(0);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/modelos`).then((r) => r.json()),
      fetch(`${API}/api/health`).then((r) => r.json()),
    ])
      .then(([listaModelos, saude]) => {
        setModelos(listaModelos);
        setCudaDisponivel(Boolean(saude.cuda));
        if (saude.base_prontuarios && !saude.base_prontuarios.disponivel) {
          setBaseIndisponivel(String(saude.base_prontuarios.caminho));
        }
      })
      .catch(() => setErro("Backend indisponível."));
    return () => eventSource.current?.close();
  }, []);

  // Busca registros conforme o usuário digita. O termo vale tanto como ID
  // (prefixo numérico) quanto como texto livre de especialidade/diagnóstico.
  useEffect(() => {
    if (!listaAberta) return;
    let ativo = true;
    const temporizador = window.setTimeout(async () => {
      try {
        const retorno = await fetch(
          `${API}/api/registros?limite=20&busca=${encodeURIComponent(idRegistro)}`,
        );
        if (!retorno.ok) return;
        const dados = await retorno.json();
        if (ativo) setSugestoes(dados);
      } catch {
        // O seletor é auxiliar: digitar o ID direto continua funcionando.
      }
    }, 250);
    return () => {
      ativo = false;
      window.clearTimeout(temporizador);
    };
  }, [idRegistro, listaAberta]);

  // Acompanha o InMemorySaver: recarrega a cada mudança de status e, enquanto o
  // grafo estiver ativo, também por polling curto.
  useEffect(() => {
    if (!idSessao) return;
    let ativo = true;

    async function buscarCheckpoints() {
      try {
        const retorno = await fetch(
          `${API}/api/assistente/sessoes/${idSessao}/checkpoints`,
        );
        if (!retorno.ok) return;
        const dados = await retorno.json();
        if (ativo) setCheckpoints(dados.checkpoints ?? []);
      } catch {
        // O painel de logs é auxiliar: falhas não interrompem a consulta.
      }
    }

    buscarCheckpoints();
    if (["aprovada", "rejeitada", "falha"].includes(status)) {
      return () => {
        ativo = false;
      };
    }
    const temporizador = window.setInterval(buscarCheckpoints, 1500);
    return () => {
      ativo = false;
      window.clearInterval(temporizador);
    };
  }, [idSessao, status]);

  const estadosNos = useMemo(() => {
    const mapa: Record<string, Evento> = {};
    for (const evento of eventos) mapa[evento.no] = evento;
    return mapa;
  }, [eventos]);

  const nodes: Node[] = definicaoNos.map(
    ([id, label, Icone, x, y]): Node => {
      const evento = estadosNos[id];
      return {
        id,
        type: "fluxo",
        position: { x, y },
        data: {
          label,
          Icone,
          status: evento?.status ?? "pendente",
          duracao: evento?.duracao_ms,
        },
      };
    },
  );

  function escolherRegistro(registro: RegistroResumo) {
    setIdRegistro(registro.id_registro);
    setRegistroEscolhido(registro);
    setListaAberta(false);
    // A pergunta original acompanha o prontuário no corpus: pré-preenchê-la
    // mantém pergunta e contexto coerentes entre si. O texto segue editável.
    if (registro.pergunta_sugerida) setPergunta(registro.pergunta_sugerida);
  }

  function abrirEventos(id: string) {
    eventSource.current?.close();
    const stream = new EventSource(
      `${API}/api/assistente/sessoes/${id}/eventos`,
    );
    eventSource.current = stream;
    stream.addEventListener("fluxo", (mensagem) => {
      const evento = JSON.parse((mensagem as MessageEvent).data) as Evento;
      atrasoVisual.current += 280;
      window.setTimeout(() => {
        setEventos((atuais) => [...atuais, evento]);
        setStatus(evento.status);
        if (evento.erro) setErro(evento.erro);
        if (evento.revisao) {
          setRevisao(evento.revisao);
          setTextoRevisado(evento.revisao.rascunho);
        }
        if (evento.resposta) {
          setResposta(evento.resposta);
          stream.close();
        }
      }, atrasoVisual.current);
    });
    stream.onerror = () => {
      if (!revisao && !resposta) setErro("Conexão de eventos interrompida.");
    };
  }

  async function enviar(evento: FormEvent) {
    evento.preventDefault();
    setEventos([]);
    setRevisao(null);
    setResposta(null);
    setErro("");
    setCheckpoints([]);
    setIdSessao("");
    setStatus("enviando");
    atrasoVisual.current = 0;
    const retorno = await fetch(`${API}/api/assistente/sessoes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id_registro: idRegistro,
        pergunta_clinica: pergunta,
        modelo,
      }),
    });
    if (!retorno.ok) {
      setErro("Não foi possível iniciar a consulta.");
      return;
    }
    const sessao = await retorno.json();
    idExecucao.current = sessao.id_execucao;
    setIdSessao(sessao.id_execucao);
    abrirEventos(sessao.id_execucao);
  }

  async function decidir(acao: "aprovar" | "rejeitar" | "editar") {
    setErro("");
    const retorno = await fetch(
      `${API}/api/assistente/sessoes/${idExecucao.current}/decisao`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          acao,
          texto_revisado: acao === "editar" ? textoRevisado : null,
        }),
      },
    );
    if (!retorno.ok) {
      const detalhe = await retorno.json();
      setErro(detalhe.detail ?? "Não foi possível concluir a revisão.");
    }
  }

  const processando = ![
    "pronto",
    "aguardando_revisao",
    "aprovada",
    "rejeitada",
    "falha",
  ].includes(status);

  return (
    <main>
      <header>
        <div className="brand">
          <div className="brand-icon"><Activity /></div>
          <div>
            <h1>Assistente Clínico</h1>
            <p>LangChain + LangGraph · revisão humana obrigatória</p>
          </div>
        </div>
        <div className="online"><span /> Backend conectado</div>
      </header>

      <div className="workspace">
        <section className="chat-panel">
          <div className="welcome">
            <GitBranch size={32} />
            <h2>Consulte um prontuário anonimizado</h2>
            <p>Acompanhe cada etapa da resposta e revise antes de liberar.</p>
          </div>

          {resposta && (
            <article className={`answer answer--${resposta.situacao}`}>
              <span>{resposta.situacao}</span>
              <p>{resposta.resposta ?? "O rascunho não foi liberado."}</p>
            </article>
          )}

          {baseIndisponivel && (
            <div className="error error--aviso">
              <strong>Base de prontuários não encontrada.</strong>
              <p>
                O backend não localizou <code>{baseIndisponivel}</code>. Em
                Docker esse caminho é um volume: confirme que
                <code> fine-tunning-llm/app/data/processado/dados_medicos_auditoria.xlsx </code>
                existe no host e que o <code>docker compose</code> foi iniciado a
                partir da raiz do repositório. Se houver um stack antigo no ar,
                pare-o antes (<code>docker compose ls -a</code>).
              </p>
            </div>
          )}

          {erro && <div className="error">{erro}</div>}

          <form className="composer" onSubmit={enviar}>
            <div className="controls">
              <label>
                <span>Modelo</span>
                <select
                  value={modelo}
                  onChange={(e) => setModelo(e.target.value as Modelo["chave"])}
                >
                  {modelos.map((item) => (
                    <option
                      key={item.chave}
                      value={item.chave}
                      disabled={item.exige_cuda && !cudaDisponivel}
                    >
                      {item.rotulo}
                      {item.exige_cuda
                        ? cudaDisponivel
                          ? " · GPU"
                          : " · indisponível sem CUDA"
                        : " · CPU/GPU"}
                    </option>
                  ))}
                </select>
              </label>
              <label
                className="campo-registro"
                title="Digite o ID ou busque por especialidade/diagnóstico"
              >
                <span>Prontuário</span>
                <input
                  value={idRegistro}
                  onChange={(e) => {
                    setIdRegistro(e.target.value);
                    setRegistroEscolhido(null);
                    setListaAberta(true);
                  }}
                  onFocus={() => setListaAberta(true)}
                  onBlur={() => window.setTimeout(() => setListaAberta(false), 150)}
                  placeholder="ID (ex.: 7) ou busca"
                  autoComplete="off"
                />
                {listaAberta && sugestoes.length > 0 && (
                  <ul className="sugestoes">
                    {sugestoes.map((registro) => (
                      <li key={registro.id_registro}>
                        <button
                          type="button"
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={() => escolherRegistro(registro)}
                        >
                          <strong>#{registro.id_registro}</strong>
                          <span>
                            {[
                              registro.especialidade_medica,
                              registro.hipotese_clinica,
                              registro.tipo_pergunta,
                            ]
                              .filter(Boolean)
                              .join(" · ") || "registro sem resumo"}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </label>
            </div>
            <textarea
              value={pergunta}
              onChange={(e) => setPergunta(e.target.value)}
              placeholder="Digite sua pergunta clínica..."
              rows={4}
            />
            <div className="composer-footer">
              <small>
                {registroEscolhido
                  ? `Registro #${registroEscolhido.id_registro} · ${
                      registroEscolhido.especialidade_medica ?? "sem especialidade"
                    } · pergunta original carregada`
                  : "Dados anonimizados · conteúdo sujeito à revisão clínica"}
              </small>
              <button disabled={processando || !pergunta.trim() || !idRegistro.trim()}>
                {processando ? <LoaderCircle className="spin" /> : <Send />}
                Enviar
              </button>
            </div>
          </form>
        </section>

        <aside className="process-panel">
          <div className="panel-title">
            <div>
              <span>FLUXO EM TEMPO REAL</span>
              <h2>Processamento LangGraph</h2>
            </div>
            <span className={`status status--${status}`}>{status.replaceAll("_", " ")}</span>
          </div>
          <div className="flow">
            <ReactFlow
              nodes={nodes}
              edges={arestas}
              nodeTypes={nodeTypes}
              fitView
              minZoom={0.4}
              maxZoom={1.3}
              nodesDraggable={false}
              nodesConnectable={false}
              panOnDrag={false}
              zoomOnScroll={false}
            >
              <Background color="#27272a" gap={20} />
            </ReactFlow>
          </div>

          <div className="timeline">
            <h3><Clock3 size={16} /> Linha do tempo</h3>
            {eventos.length === 0 && <p>Aguardando uma pergunta...</p>}
            {eventos.slice(-6).map((evento, indice) => (
              <div className="timeline-item" key={`${evento.sequencia}-${indice}`}>
                <span />
                <div>
                  <strong>{evento.no.replaceAll("_", " ")}</strong>
                  <small>
                    {evento.status.replaceAll("_", " ")}
                    {evento.duracao_ms !== undefined &&
                      ` · ${evento.duracao_ms.toFixed(0)} ms`}
                  </small>
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>

      <PainelCheckpoints
        checkpoints={checkpoints}
        threadId={idSessao}
        aberto={logsAberto}
        alternar={() => setLogsAberto((atual) => !atual)}
      />

      {revisao && !resposta && (
        <div className="modal-backdrop">
          <section className="review-modal">
            <div className="review-title">
              <UserRoundCheck />
              <div>
                <span>HUMAN-IN-THE-LOOP · ID CONFIRMADO: {revisao.id_registro}</span>
                <h2>Revise o rascunho</h2>
              </div>
            </div>
            <textarea
              value={textoRevisado}
              onChange={(e) => setTextoRevisado(e.target.value)}
              rows={14}
            />
            <p className="notice">{revisao.aviso}</p>
            <div className="review-actions">
              <button className="danger" onClick={() => decidir("rejeitar")}>
                <X /> Rejeitar
              </button>
              <button className="secondary" onClick={() => decidir("editar")}>
                <Sparkles /> Salvar edição
              </button>
              <button onClick={() => decidir("aprovar")}>
                <Check /> Aprovar original
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
