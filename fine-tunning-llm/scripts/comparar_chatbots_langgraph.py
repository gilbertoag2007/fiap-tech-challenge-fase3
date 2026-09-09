"""Segundo teste de 20 perguntas, agora com o pipeline LangChain/LangGraph.

O primeiro teste (`chatbot-modelo-fine-tunning/comparar_qwen_llama.py`) chamou
`model.generate` diretamente, sem LangChain, sem LangGraph e sem os controles
anti-repetição adotados depois. Este script repete a medição em três braços
sobre **os mesmos 20 casos**, para separar o que vem do modelo do que vem da
orquestração:

1. `direto_sem_penalidade` — replica o decoding do primeiro teste
   (`do_sample=False`, sem `repetition_penalty`, sem `no_repeat_ngram_size`);
2. `direto_com_penalidade` — caminho público atual do serviço
   (`repetition_penalty=1.15`, `no_repeat_ngram_size=4`);
3. `pipeline_langgraph` — fluxo completo: tool `buscar_prontuario`,
   `AssistenteChain`, nó `validar_seguranca` e HITL aprovado.

Os 20 casos vêm do split de **teste** do dataset de fine-tuning, ou seja, não
participaram do treino de nenhum adaptador — diferente do primeiro teste, cujas
perguntas eram redigidas à mão e, quando reais, podiam estar no treino.

Execute a partir de `fine-tunning-llm/`:

    .venv/bin/python scripts/comparar_chatbots_langgraph.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from app.assistente.auditoria import ServicoAuditoriaAssistente  # noqa: E402
from app.assistente.chain import AssistenteChain  # noqa: E402
from app.assistente.fluxo import FluxoAssistenteMedico  # noqa: E402
from app.assistente.modelo_chat import ModeloChatLocal  # noqa: E402
from app.assistente.modelos import (  # noqa: E402
    DecisaoHumana,
    SolicitacaoAssistente,
)
from app.assistente.repositorio import RepositorioProntuariosExcel  # noqa: E402
from app.config import CAMINHO_PRONTUARIOS, DIRETORIO_RELATORIOS  # noqa: E402
from app.services.arquivo_service import ArquivoService  # noqa: E402
from app.services.fine_tuning_service import FineTuningService  # noqa: E402


QUANTIDADE_CASOS = 20
MAX_NOVOS_TOKENS = 384
MODELOS_AVALIADOS = ("qwen80", "llama")
BRACOS = (
    "direto_sem_penalidade",
    "direto_com_penalidade",
    "pipeline_langgraph",
)
SECOES_OBRIGATORIAS = (
    "Resposta",
    "Considerações clínicas",
    "Conduta/Orientação",
    "Limitações",
)
CAMINHO_AUDITORIA_EXPERIMENTO = (
    DIRETORIO_RELATORIOS / "auditoria_experimento_langgraph.jsonl"
)
CAMINHO_JSON = DIRETORIO_RELATORIOS / "comparacao_chatbots_langgraph_20q.json"
CAMINHO_MD = DIRETORIO_RELATORIOS / "comparacao_chatbots_langgraph_20q.md"


# ---------------------------------------------------------------- seleção ----
def selecionar_casos(quantidade: int = QUANTIDADE_CASOS) -> list[dict[str, str]]:
    """Escolhe casos do split de teste, um por especialidade, de forma estável.

    Regra determinística: entre os registros do split de teste com todos os
    campos necessários preenchidos, ordenam-se as especialidades por frequência
    (empate pelo nome) e toma-se o registro de menor `id` de cada uma das
    `quantidade` primeiras. Sem amostragem aleatória, para que a seleção seja
    reproduzível sem depender de semente.
    """
    servico_arquivo = ArquivoService()
    auditoria = servico_arquivo.gerar_dataframe(
        FineTuningService.CAMINHO_ARQUIVO_AUDITORIA
    )
    fine_tuning = servico_arquivo.gerar_dataframe(
        FineTuningService.CAMINHO_ARQUIVO_FINE_TUNING
    )

    ids_teste = set(
        fine_tuning.loc[fine_tuning["split"] == "teste", "id_exemplo"]
    )
    dados = auditoria[auditoria["id"].isin(ids_teste)].copy()
    for coluna in (
        "prontuario_contexto_anonimizado",
        "pergunta_original_anonimizado",
        "especialidade_medica",
        "papel_solicitante",
        "contexto_solicitacao",
    ):
        dados = dados[
            dados[coluna].notna()
            & dados[coluna].astype(str).str.strip().ne("")
        ]

    frequencia = dados["especialidade_medica"].value_counts()
    especialidades = sorted(
        frequencia.index,
        key=lambda nome: (-int(frequencia[nome]), str(nome)),
    )[:quantidade]
    posicao = {nome: indice for indice, nome in enumerate(especialidades)}

    escolhidos = (
        dados[dados["especialidade_medica"].isin(especialidades)]
        .sort_values("id")
        .groupby("especialidade_medica", sort=False)
        .head(1)
    )
    escolhidos = escolhidos.assign(
        _ordem=escolhidos["especialidade_medica"].map(posicao)
    ).sort_values("_ordem")

    casos: list[dict[str, str]] = []
    for numero, (_, linha) in enumerate(escolhidos.iterrows(), start=1):
        casos.append(
            {
                "codigo": f"C{numero:02d}",
                "id_registro": str(linha["id"]).strip(),
                "especialidade": str(linha["especialidade_medica"]).strip(),
                "tipo_pergunta": str(linha["tipo_pergunta"]).strip(),
                "papel_solicitante": str(linha["papel_solicitante"]).strip(),
                "contexto_solicitacao": str(
                    linha["contexto_solicitacao"]
                ).strip(),
                "prontuario": str(
                    linha["prontuario_contexto_anonimizado"]
                ).strip(),
                "pergunta": str(linha["pergunta_original_anonimizado"]).strip(),
            }
        )
    return casos


def montar_mensagem_usuario(caso: dict[str, str]) -> str:
    """Reproduz o formato de solicitação usado no treino e no primeiro teste."""
    return (
        f"Papel do solicitante: {caso['papel_solicitante']}\n"
        f"Contexto da solicitacao: {caso['contexto_solicitacao']}\n"
        f"Prontuario: {caso['prontuario']}\n"
        f"Pergunta: {caso['pergunta']}"
    )


# ----------------------------------------------------------------- métricas --
def avaliar_estrutura_heuristica(texto: str) -> int:
    """Mesma contagem frouxa do primeiro teste, mantida para comparabilidade."""
    minusculo = texto.lower()
    return sum(
        1
        for chave in ("resposta", "considera", "conduta", "limita")
        if chave in minusculo
    )


def avaliar_estrutura_estrita(texto: str) -> int:
    """Conta seções no formato exigido pelo nó `validar_seguranca`."""
    return sum(
        1
        for secao in SECOES_OBRIGATORIAS
        if re.search(
            rf"^[ \t]*{re.escape(secao)}[ \t]*:",
            texto,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )


def detectar_loop(texto: str) -> bool:
    """Mesmo detector de repetição de trigramas usado no primeiro teste."""
    palavras = texto.lower().split()
    if len(palavras) < 40:
        return False
    trigramas = [" ".join(palavras[i : i + 3]) for i in range(len(palavras) - 2)]
    if not trigramas:
        return False
    mais_comum = max(trigramas.count(item) for item in set(trigramas))
    return mais_comum >= 8


def medir(texto: str) -> dict[str, object]:
    return {
        "chars": len(texto),
        "secoes_heuristica": avaliar_estrutura_heuristica(texto),
        "secoes_estritas": avaliar_estrutura_estrita(texto),
        "loop": detectar_loop(texto),
    }


# ---------------------------------------------------------------- inferência --
def gerar_replicando_primeiro_teste(
    servico: FineTuningService,
    mensagem_system: str,
    mensagem_usuario: str,
    max_novos_tokens: int = MAX_NOVOS_TOKENS,
) -> str:
    """Gera sem os controles anti-repetição, como no primeiro teste.

    Reimplementado aqui de propósito: o serviço da aplicação sempre aplica
    `repetition_penalty` e `no_repeat_ngram_size`, e o objetivo deste braço é
    justamente medir o comportamento anterior a essa decisão. Reaproveita o
    modelo já carregado pelo serviço, sem carregar pesos duas vezes.
    """
    if servico.tokenizer is None or servico.modelo is None:
        raise RuntimeError("O modelo precisa estar carregado antes do braço direto.")

    mensagens = [
        {"role": "system", "content": mensagem_system.strip()},
        {"role": "user", "content": mensagem_usuario.strip()},
    ]
    kwargs_template: dict[str, object] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if servico.chave_modelo.startswith("qwen"):
        kwargs_template["enable_thinking"] = False
    prompt = servico.tokenizer.apply_chat_template(mensagens, **kwargs_template)
    entradas = servico.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=servico.max_tokens_entrada,
    )
    entradas = entradas.to(next(servico.modelo.parameters()).device)

    with torch.inference_mode():
        gerado = servico.modelo.generate(
            **entradas,
            max_new_tokens=max_novos_tokens,
            do_sample=False,
            pad_token_id=servico.tokenizer.eos_token_id,
            eos_token_id=servico.tokenizer.eos_token_id,
        )
    inicio = entradas["input_ids"].shape[1]
    return servico.tokenizer.decode(
        gerado[0][inicio:],
        skip_special_tokens=True,
    ).strip()


class ServicoComGravacao:
    """Proxy que registra a saída bruta do modelo dentro do pipeline.

    O `AssistenteChain` completa seções ausentes e anexa fontes; sem capturar o
    texto antes disso, não é possível saber se a aderência ao formato vem do
    modelo ou do pós-processamento determinístico.
    """

    def __init__(self, servico: FineTuningService) -> None:
        self._servico = servico
        self.ultima_saida_bruta = ""

    @property
    def NOME_MODELO_BASE(self) -> str:  # noqa: N802 - contrato do protocolo
        return self._servico.NOME_MODELO_BASE

    @property
    def chave_modelo(self) -> str:
        return self._servico.chave_modelo

    def gerar_resposta_modelo_ajustado(
        self,
        mensagem_system: str,
        mensagem_usuario: str,
        max_novos_tokens: int = MAX_NOVOS_TOKENS,
    ) -> str:
        saida = self._servico.gerar_resposta_modelo_ajustado(
            mensagem_system=mensagem_system,
            mensagem_usuario=mensagem_usuario,
            max_novos_tokens=max_novos_tokens,
        )
        self.ultima_saida_bruta = saida
        return saida


# --------------------------------------------------------------- experimento --
def executar(
    casos: list[dict[str, str]],
    modelos: tuple[str, ...] = MODELOS_AVALIADOS,
) -> dict[str, object]:
    servico_arquivo = ArquivoService()
    servico = FineTuningService(servico_arquivo=servico_arquivo)
    gravador = ServicoComGravacao(servico)

    repositorio = RepositorioProntuariosExcel(
        servico_arquivo=servico_arquivo,
        caminho_arquivo=CAMINHO_PRONTUARIOS,
    )
    eventos_no: dict[str, list[dict[str, object]]] = defaultdict(list)

    def observar(evento: dict[str, object]) -> None:
        eventos_no[str(evento.get("id_execucao", ""))].append(dict(evento))

    fluxo = FluxoAssistenteMedico(
        repositorio=repositorio,
        chain_assistente=AssistenteChain(ModeloChatLocal(
            servico_fine_tuning=gravador,
            max_novos_tokens=MAX_NOVOS_TOKENS,
        )),
        auditoria=ServicoAuditoriaAssistente(
            caminho_arquivo=CAMINHO_AUDITORIA_EXPERIMENTO
        ),
        observador_eventos=observar,
    )

    resultados: dict[str, object] = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "quantidade_casos": len(casos),
        "max_novos_tokens": MAX_NOVOS_TOKENS,
        "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
        "split_origem": "teste",
        "bracos": list(BRACOS),
        "modelos_avaliados": list(modelos),
        "casos": casos,
        "modelos": {},
    }

    for chave in modelos:
        configuracao = FineTuningService.MODELOS_DISPONIVEIS[chave]
        print(f"\n=== {configuracao['rotulo']} ===", flush=True)
        servico.configurar_modelo(chave)
        # Aquecimento curto: carrega os pesos pela via pública, para que o
        # braço sem penalidade reaproveite o modelo já residente.
        servico.gerar_resposta_modelo_ajustado(
            mensagem_system=FineTuningService.MENSAGEM_SYSTEM,
            mensagem_usuario="Aquecimento do modelo.",
            max_novos_tokens=8,
        )

        execucoes: list[dict[str, object]] = []
        for caso in casos:
            print(
                f"  > {caso['codigo']} id={caso['id_registro']} "
                f"({caso['especialidade']})",
                flush=True,
            )
            registro: dict[str, object] = {
                "codigo": caso["codigo"],
                "id_registro": caso["id_registro"],
                "especialidade": caso["especialidade"],
                "bracos": {},
            }
            mensagem_usuario = montar_mensagem_usuario(caso)

            # 1) caminho público atual do serviço (com anti-repetição).
            inicio = perf_counter()
            texto = servico.gerar_resposta_modelo_ajustado(
                mensagem_system=FineTuningService.MENSAGEM_SYSTEM,
                mensagem_usuario=mensagem_usuario,
                max_novos_tokens=MAX_NOVOS_TOKENS,
            )
            registro["bracos"]["direto_com_penalidade"] = {
                "texto": texto,
                "duracao_s": round(perf_counter() - inicio, 2),
                **medir(texto),
            }

            # 2) decoding do primeiro teste, sem anti-repetição.
            inicio = perf_counter()
            texto = gerar_replicando_primeiro_teste(
                servico,
                FineTuningService.MENSAGEM_SYSTEM,
                mensagem_usuario,
            )
            registro["bracos"]["direto_sem_penalidade"] = {
                "texto": texto,
                "duracao_s": round(perf_counter() - inicio, 2),
                **medir(texto),
            }

            # 3) pipeline completo com LangGraph e HITL aprovado.
            inicio = perf_counter()
            gravador.ultima_saida_bruta = ""
            solicitacao = SolicitacaoAssistente(
                id_registro=caso["id_registro"],
                pergunta_clinica=caso["pergunta"],
            )
            revisao = fluxo.iniciar(solicitacao)
            resposta = fluxo.retomar(
                revisao.id_execucao,
                DecisaoHumana(acao="aprovar", observacao="teste automatizado"),
            )
            duracao = round(perf_counter() - inicio, 2)
            bruto = gravador.ultima_saida_bruta
            registro["bracos"]["pipeline_langgraph"] = {
                "texto": resposta.resposta or "",
                "texto_bruto_modelo": bruto,
                "duracao_s": duracao,
                "alertas": list(resposta.alertas),
                "fontes": list(resposta.fontes),
                "situacao": resposta.situacao,
                "nos_executados": [
                    evento["no"]
                    for evento in eventos_no.get(revisao.id_execucao, [])
                    if evento.get("status") == "concluido"
                ],
                "bruto": medir(bruto),
                **medir(resposta.resposta or ""),
            }
            execucoes.append(registro)

            for nome in BRACOS:
                dados = registro["bracos"][nome]
                print(
                    f"      {nome:<22} secoes={dados['secoes_estritas']}/4 "
                    f"(heur={dados['secoes_heuristica']}) "
                    f"loop={dados['loop']} chars={dados['chars']} "
                    f"{dados['duracao_s']}s",
                    flush=True,
                )

        resultados["modelos"][chave] = {
            "rotulo": configuracao["rotulo"],
            "base": configuracao["nome_base"],
            "adapter": str(configuracao["caminho_adapter"]),
            "execucoes": execucoes,
        }
        servico.descarregar_modelo()

    return resultados


# ------------------------------------------------------------------ relatório --
def resumir(resultados: dict[str, object]) -> None:
    """Calcula as médias por modelo e braço e grava no próprio resultado."""
    total = int(resultados["quantidade_casos"])
    for bloco in resultados["modelos"].values():
        metricas: dict[str, dict[str, object]] = {}
        for nome in BRACOS:
            dados = [
                execucao["bracos"][nome] for execucao in bloco["execucoes"]
            ]
            metricas[nome] = {
                "media_secoes_estritas": round(
                    sum(item["secoes_estritas"] for item in dados) / total, 2
                ),
                "media_secoes_heuristica": round(
                    sum(item["secoes_heuristica"] for item in dados) / total, 2
                ),
                "quatro_secoes": sum(
                    1 for item in dados if item["secoes_estritas"] == 4
                ),
                "loops": sum(1 for item in dados if item["loop"]),
                "media_chars": round(
                    sum(item["chars"] for item in dados) / total, 1
                ),
                "media_duracao_s": round(
                    sum(item["duracao_s"] for item in dados) / total, 2
                ),
            }
            if nome == "pipeline_langgraph":
                brutos = [item["bruto"] for item in dados]
                metricas[nome]["bruto"] = {
                    "media_secoes_estritas": round(
                        sum(item["secoes_estritas"] for item in brutos) / total,
                        2,
                    ),
                    "quatro_secoes": sum(
                        1 for item in brutos if item["secoes_estritas"] == 4
                    ),
                    "loops": sum(1 for item in brutos if item["loop"]),
                    "media_chars": round(
                        sum(item["chars"] for item in brutos) / total, 1
                    ),
                }
                contagem_alertas: dict[str, int] = defaultdict(int)
                for item in dados:
                    for alerta in item["alertas"]:
                        contagem_alertas[alerta] += 1
                metricas[nome]["alertas"] = dict(
                    sorted(contagem_alertas.items())
                )
        bloco["metricas"] = metricas


def placar(resultados: dict[str, object], braco: str) -> dict[str, int] | None:
    """Placar par a par entre os dois modelos, no mesmo critério do 1º teste."""
    if not {"qwen80", "llama"} <= set(resultados["modelos"]):
        return None
    qwen = resultados["modelos"]["qwen80"]["execucoes"]
    llama = resultados["modelos"]["llama"]["execucoes"]
    vitorias = {"qwen": 0, "llama": 0, "empate": 0}
    for execucao_qwen, execucao_llama in zip(qwen, llama):
        dados_qwen = execucao_qwen["bracos"][braco]
        dados_llama = execucao_llama["bracos"][braco]
        nota_qwen = dados_qwen["secoes_estritas"] - (
            2 if dados_qwen["loop"] else 0
        )
        nota_llama = dados_llama["secoes_estritas"] - (
            2 if dados_llama["loop"] else 0
        )
        if nota_llama > nota_qwen:
            vitorias["llama"] += 1
        elif nota_qwen > nota_llama:
            vitorias["qwen"] += 1
        else:
            vitorias["empate"] += 1
    return vitorias


def escrever_markdown(resultados: dict[str, object]) -> None:
    total = int(resultados["quantidade_casos"])
    linhas = [
        "# Comparacao chatbot 20Q (2a rodada, com LangGraph integrado)",
        "",
        f"Gerado em: {resultados['gerado_em']}",
        f"Casos: {total} (split de teste, nao vistos no treino)",
        f"Dispositivo: {resultados['dispositivo']} | "
        f"max_new_tokens: {resultados['max_novos_tokens']}",
        "",
        "Bracos comparados:",
        "",
        "1. `direto_sem_penalidade` — decoding do 1o teste "
        "(sem repetition_penalty / no_repeat_ngram_size);",
        "2. `direto_com_penalidade` — via publica atual do servico "
        "(repetition_penalty=1.15, no_repeat_ngram_size=4);",
        "3. `pipeline_langgraph` — fluxo completo (tool, chain, "
        "validar_seguranca, HITL aprovado).",
        "",
        "Secoes contadas pelo criterio estrito do no `validar_seguranca` "
        "(`^Secao:`). A coluna `heur` repete a contagem frouxa do 1o teste.",
        "",
        "## Resumo automatico",
        "",
        "| Modelo | Braco | Media secoes | heur | 4/4 secoes | Loops | "
        "Media chars | Media s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bloco in resultados["modelos"].values():
        for nome in BRACOS:
            metrica = bloco["metricas"][nome]
            linhas.append(
                f"| {bloco['rotulo']} | `{nome}` | "
                f"{metrica['media_secoes_estritas']:.2f} | "
                f"{metrica['media_secoes_heuristica']:.2f} | "
                f"{metrica['quatro_secoes']}/{total} | "
                f"{metrica['loops']}/{total} | "
                f"{metrica['media_chars']:.0f} | "
                f"{metrica['media_duracao_s']:.1f} |"
            )

    linhas.extend(
        [
            "",
            "## Texto bruto do modelo dentro do pipeline",
            "",
            "Medido antes de o `AssistenteChain` completar secoes ausentes e "
            "anexar fontes — separa o merito do modelo do pos-processamento "
            "deterministico.",
            "",
            "| Modelo | Media secoes (bruto) | 4/4 (bruto) | Loops (bruto) | "
            "Media chars (bruto) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bloco in resultados["modelos"].values():
        bruto = bloco["metricas"]["pipeline_langgraph"]["bruto"]
        linhas.append(
            f"| {bloco['rotulo']} | {bruto['media_secoes_estritas']:.2f} | "
            f"{bruto['quatro_secoes']}/{total} | {bruto['loops']}/{total} | "
            f"{bruto['media_chars']:.0f} |"
        )

    linhas.extend(["", "## Alertas emitidos por `validar_seguranca`", ""])
    for bloco in resultados["modelos"].values():
        alertas = bloco["metricas"]["pipeline_langgraph"]["alertas"]
        if alertas:
            detalhe = ", ".join(
                f"`{chave}` {valor}/{total}" for chave, valor in alertas.items()
            )
        else:
            detalhe = "nenhum"
        linhas.append(f"- {bloco['rotulo']}: {detalhe}")

    linhas.extend(["", "## Placar estrutural par a par", ""])
    for nome in BRACOS:
        vitorias = placar(resultados, nome)
        if vitorias is None:
            linhas.append(f"- `{nome}`: indisponivel (execucao com um modelo)")
            continue
        linhas.append(
            f"- `{nome}`: Llama **{vitorias['llama']}/{total}** | "
            f"Qwen **{vitorias['qwen']}/{total}** | "
            f"empates **{vitorias['empate']}/{total}**"
        )
    resultados["placar_estrutura"] = {
        nome: placar(resultados, nome) for nome in BRACOS
    }

    linhas.extend(["", "## Casos avaliados", "", "| Codigo | id | Especialidade | Tipo |", "|---|---|---|---|"])
    for caso in resultados["casos"]:
        linhas.append(
            f"| {caso['codigo']} | {caso['id_registro']} | "
            f"{caso['especialidade']} | {caso['tipo_pergunta']} |"
        )

    for caso in resultados["casos"]:
        linhas.extend(
            [
                "",
                f"## {caso['codigo']} — id {caso['id_registro']} — "
                f"{caso['especialidade']}",
                "",
                "### Solicitacao",
                "",
                "```",
                montar_mensagem_usuario(caso),
                "```",
            ]
        )
        for bloco in resultados["modelos"].values():
            execucao = next(
                item
                for item in bloco["execucoes"]
                if item["codigo"] == caso["codigo"]
            )
            for nome in BRACOS:
                dados = execucao["bracos"][nome]
                cabecalho = (
                    f"_secoes={dados['secoes_estritas']}/4 | "
                    f"heur={dados['secoes_heuristica']}/4 | "
                    f"chars={dados['chars']} | loop={dados['loop']}_"
                )
                if nome == "pipeline_langgraph":
                    cabecalho += (
                        f"\n\n_alertas: "
                        f"{', '.join(dados['alertas']) or 'nenhum'}_"
                        f"\n\n_bruto do modelo: "
                        f"secoes={dados['bruto']['secoes_estritas']}/4, "
                        f"loop={dados['bruto']['loop']}, "
                        f"chars={dados['bruto']['chars']}_"
                    )
                linhas.extend(
                    [
                        "",
                        f"### {bloco['rotulo']} — `{nome}`",
                        "",
                        cabecalho,
                        "",
                        dados["texto"] or "_(vazio)_",
                    ]
                )

    CAMINHO_MD.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--casos",
        type=int,
        default=QUANTIDADE_CASOS,
        help="Quantidade de casos do split de teste (padrao: 20).",
    )
    parser.add_argument(
        "--modelos",
        nargs="+",
        default=list(MODELOS_AVALIADOS),
        choices=sorted(FineTuningService.MODELOS_DISPONIVEIS),
        help="Modelos avaliados (padrao: qwen80 llama).",
    )
    argumentos = parser.parse_args()

    DIRETORIO_RELATORIOS.mkdir(parents=True, exist_ok=True)
    casos = selecionar_casos(argumentos.casos)
    print(f"Casos selecionados: {[caso['id_registro'] for caso in casos]}")

    resultados = executar(casos, tuple(argumentos.modelos))
    resumir(resultados)
    escrever_markdown(resultados)
    CAMINHO_JSON.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSalvo em:\n- {CAMINHO_JSON}\n- {CAMINHO_MD}")
    for bloco in resultados["modelos"].values():
        print(f"\n{bloco['rotulo']}")
        for nome in BRACOS:
            metrica = bloco["metricas"][nome]
            print(
                f"  {nome:<22} secoes={metrica['media_secoes_estritas']:.2f} "
                f"4/4={metrica['quatro_secoes']}/{len(casos)} "
                f"loops={metrica['loops']}/{len(casos)} "
                f"chars={metrica['media_chars']:.0f}"
            )


if __name__ == "__main__":
    main()
