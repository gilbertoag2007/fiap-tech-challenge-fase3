"""Orquestração LangGraph do assistente médico com revisão humana obrigatória."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any, Callable, Literal, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.assistente.auditoria import ServicoAuditoriaAssistente
from app.assistente.chain import AVISO_REVISAO_HUMANA, AssistenteChain
from app.assistente.ferramentas import (
    criar_ferramentas_assistente,
    invocar_buscar_prontuario,
    mapa_ferramentas,
)
from app.assistente.modelos import (
    DecisaoHumana,
    EstadoAssistente,
    RespostaAssistente,
    RevisaoPendente,
    SolicitacaoAssistente,
)
from app.assistente.repositorio import RepositorioProntuarios


class FluxoAssistenteMedico:
    """Coordena geração segura e liberação exclusiva após decisão humana."""

    def __init__(
        self,
        repositorio: RepositorioProntuarios,
        chain_assistente: AssistenteChain,
        auditoria: ServicoAuditoriaAssistente,
        observador_eventos: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.repositorio = repositorio
        self.chain_assistente = chain_assistente
        self.auditoria = auditoria
        self.observador_eventos = observador_eventos
        self.ferramentas = criar_ferramentas_assistente(repositorio)
        self.ferramentas_por_nome = mapa_ferramentas(self.ferramentas)
        self.grafo = self._construir_grafo()

    def iniciar(self, solicitacao: SolicitacaoAssistente) -> RevisaoPendente:
        """Executa o grafo até a interrupção obrigatória para revisão humana."""
        try:
            resultado = self.grafo.invoke(
                solicitacao.model_dump(),
                config=self._configuracao(solicitacao.id_execucao),
            )
            payload = self._extrair_payload_interrupcao(resultado)
            return RevisaoPendente.model_validate(payload)
        except Exception as erro:
            try:
                self._registrar_falha(solicitacao.id_execucao, "iniciar", erro)
            except Exception:
                pass
            raise

    def retomar(
        self,
        id_execucao: str,
        decisao: DecisaoHumana,
    ) -> RespostaAssistente:
        """Retoma uma execução interrompida e expõe somente a resposta pública."""
        try:
            estado = self.grafo.invoke(
                Command(resume=decisao.model_dump()),
                config=self._configuracao(id_execucao),
            )
            return RespostaAssistente(
                id_execucao=estado["id_execucao"],
                id_registro=estado["id_registro"],
                situacao=cast(Literal["aprovada", "rejeitada"], estado["situacao"]),
                resposta=estado.get("resposta"),
                fontes=estado.get("fontes", []),
                alertas=estado.get("alertas", []),
                aviso=estado["aviso"],
            )
        except Exception as erro:
            try:
                self._registrar_falha(id_execucao, "retomar", erro)
            except Exception:
                pass
            raise

    def historico_checkpoints(
        self,
        id_execucao: str,
        limite: int = 60,
    ) -> list[dict[str, object]]:
        """Lê o InMemorySaver e devolve metadados sanitizados dos checkpoints.

        Expõe apenas telemetria de persistência do grafo (passo, nós pendentes,
        chaves gravadas no estado). Nenhum valor clínico é retornado, mantendo o
        painel de logs alinhado à política de auditoria sem dados sensíveis.
        """
        instantaneos = list(
            self.grafo.get_state_history(self._configuracao(id_execucao))
        )
        instantaneos.reverse()  # ordem cronológica: do passo -1 até o último.

        registros: list[dict[str, object]] = []
        chaves_anteriores: set[str] = set()
        for instantaneo in instantaneos:
            chaves_atuais = set(instantaneo.values or {})
            tarefas = getattr(instantaneo, "tasks", ()) or ()
            metadados = dict(instantaneo.metadata or {})
            registros.append(
                {
                    "checkpoint_id": str(
                        instantaneo.config.get("configurable", {}).get(
                            "checkpoint_id",
                            "",
                        )
                    ),
                    "passo": metadados.get("step"),
                    "origem": metadados.get("source", ""),
                    "criado_em": instantaneo.created_at,
                    "proximos_nos": list(instantaneo.next or ()),
                    "tarefas_pendentes": [tarefa.name for tarefa in tarefas],
                    "chaves_estado": sorted(chaves_atuais),
                    "chaves_gravadas": sorted(chaves_atuais - chaves_anteriores),
                    "interrompido": any(
                        getattr(tarefa, "interrupts", ()) for tarefa in tarefas
                    ),
                }
            )
            chaves_anteriores = chaves_atuais
        return registros[-limite:]

    def _construir_grafo(self):
        fluxo = StateGraph(EstadoAssistente)
        fluxo.add_node(
            "validar_entrada",
            self._envolver_no("validar_entrada", self._validar_entrada),
        )
        fluxo.add_node(
            "consultar_registro",
            self._envolver_no("consultar_registro", self._consultar_registro),
        )
        fluxo.add_node(
            "gerar_rascunho",
            self._envolver_no("gerar_rascunho", self._gerar_rascunho),
        )
        fluxo.add_node(
            "validar_seguranca",
            self._envolver_no("validar_seguranca", self._validar_seguranca),
        )
        fluxo.add_node(
            "solicitar_revisao_humana",
            self._envolver_no(
                "solicitar_revisao_humana",
                self._solicitar_revisao_humana,
            ),
        )
        fluxo.add_node(
            "finalizar_aprovacao",
            self._envolver_no("finalizar_aprovacao", self._finalizar_aprovacao),
        )
        fluxo.add_node(
            "finalizar_rejeicao",
            self._envolver_no("finalizar_rejeicao", self._finalizar_rejeicao),
        )

        fluxo.add_edge(START, "validar_entrada")
        fluxo.add_edge("validar_entrada", "consultar_registro")
        fluxo.add_edge("consultar_registro", "gerar_rascunho")
        fluxo.add_edge("gerar_rascunho", "validar_seguranca")
        fluxo.add_edge("validar_seguranca", "solicitar_revisao_humana")
        fluxo.add_conditional_edges(
            "solicitar_revisao_humana",
            self._rota_finalizacao,
            {
                "aprovar": "finalizar_aprovacao",
                "rejeitar": "finalizar_rejeicao",
            },
        )
        fluxo.add_edge("finalizar_aprovacao", END)
        fluxo.add_edge("finalizar_rejeicao", END)
        return fluxo.compile(checkpointer=InMemorySaver())

    def _envolver_no(
        self,
        nome: str,
        funcao: Callable[[EstadoAssistente], EstadoAssistente],
    ) -> Callable[[EstadoAssistente], EstadoAssistente]:
        """Emite telemetria transitória de cada nó sem conteúdo clínico."""

        def executar(estado: EstadoAssistente) -> EstadoAssistente:
            inicio = perf_counter()
            self._emitir_evento(estado, nome, "iniciado")
            try:
                resultado = funcao(estado)
            except Exception:
                self._emitir_evento(
                    estado,
                    nome,
                    "falha",
                    duracao_ms=(perf_counter() - inicio) * 1000,
                )
                raise
            self._emitir_evento(
                estado,
                nome,
                "concluido",
                duracao_ms=(perf_counter() - inicio) * 1000,
            )
            return resultado

        return executar

    def _validar_entrada(self, estado: EstadoAssistente) -> EstadoAssistente:
        id_registro = estado["id_registro"].strip()
        pergunta_clinica = estado["pergunta_clinica"].strip()
        if not id_registro:
            raise ValueError("O identificador do registro é obrigatório.")
        if not pergunta_clinica:
            raise ValueError("A pergunta clínica é obrigatória.")
        atualizacao: EstadoAssistente = {
            "id_registro": id_registro,
            "pergunta_clinica": pergunta_clinica,
        }
        self._auditar(estado, "validar_entrada", "concluida")
        return atualizacao

    def _consultar_registro(self, estado: EstadoAssistente) -> EstadoAssistente:
        # Tool calling adaptado: o grafo invoca a tool de forma determinística
        # (seguro para modelos locais sem function-calling nativo confiável).
        payload = invocar_buscar_prontuario(
            self.ferramentas_por_nome,
            estado["id_registro"],
        )
        id_retornado = str(payload.get("id_registro", "")).strip()
        if id_retornado != estado["id_registro"].strip():
            raise RuntimeError(
                "O ID retornado pela ferramenta não corresponde ao ID solicitado."
            )
        campos = dict(payload.get("campos", {}))
        fontes = list(payload.get("fontes", []))
        atualizacao: EstadoAssistente = {
            "contexto_clinico": campos,
            "campos": campos,
            "fontes": fontes,
        }
        self._auditar(atualizacao | estado, "consultar_registro", "concluida")
        return atualizacao

    def _gerar_rascunho(self, estado: EstadoAssistente) -> EstadoAssistente:
        from app.assistente.modelos import RegistroClinico

        registro = RegistroClinico(
            id_registro=estado["id_registro"],
            campos=estado["campos"],
            fontes=estado["fontes"],
        )
        rascunho = self.chain_assistente.gerar_rascunho(
            estado["pergunta_clinica"],
            registro,
        )
        atualizacao: EstadoAssistente = {"rascunho": rascunho}
        self._auditar(estado, "gerar_rascunho", "concluida")
        return atualizacao

    def _validar_seguranca(self, estado: EstadoAssistente) -> EstadoAssistente:
        rascunho = estado.get("rascunho", "")
        alertas = self._alertas_seguranca(rascunho, estado.get("fontes", []))
        atualizacao: EstadoAssistente = {
            "alertas": alertas,
            "aviso": AVISO_REVISAO_HUMANA,
        }
        self._auditar(atualizacao | estado, "validar_seguranca", "concluida")
        return atualizacao

    def _solicitar_revisao_humana(self, estado: EstadoAssistente) -> EstadoAssistente:
        decisao = DecisaoHumana.model_validate(
            interrupt(
                {
                    "id_execucao": estado["id_execucao"],
                    "id_registro": estado["id_registro"],
                    "rascunho": estado["rascunho"],
                    "fontes": estado.get("fontes", []),
                    "alertas": estado.get("alertas", []),
                    "aviso": estado["aviso"],
                }
            )
        )
        return {
            "decisao_humana": decisao.acao_efetiva != "rejeitar",
            "acao_humana": decisao.acao_efetiva,
            "observacao_humana": decisao.observacao,
            "texto_revisado": (decisao.texto_revisado or "").strip(),
        }

    def _finalizar_aprovacao(self, estado: EstadoAssistente) -> EstadoAssistente:
        resposta = estado["rascunho"]
        if estado.get("acao_humana") == "editar":
            resposta = estado.get("texto_revisado", "").strip()
            if AVISO_REVISAO_HUMANA not in resposta:
                resposta = f"{resposta}\n\n{AVISO_REVISAO_HUMANA}"
        alertas = self._alertas_seguranca(
            resposta,
            estado.get("fontes", []),
        )
        atualizacao: EstadoAssistente = {
            "situacao": "aprovada",
            "resposta": resposta,
            "alertas": alertas,
        }
        self._auditar(atualizacao | estado, "finalizar_aprovacao", "concluida")
        return atualizacao

    def _finalizar_rejeicao(self, estado: EstadoAssistente) -> EstadoAssistente:
        atualizacao: EstadoAssistente = {
            "situacao": "rejeitada",
            "resposta": None,
        }
        self._auditar(atualizacao | estado, "finalizar_rejeicao", "concluida")
        return atualizacao

    @staticmethod
    def _rota_finalizacao(estado: EstadoAssistente) -> str:
        return "aprovar" if estado["decisao_humana"] else "rejeitar"

    @staticmethod
    def _alertas_seguranca(rascunho: str, fontes: list[str]) -> list[str]:
        alertas: list[str] = []
        for secao, codigo in (
            ("Resposta", "SECAO_RESPOSTA_AUSENTE"),
            ("Considerações clínicas", "SECAO_CONSIDERACOES_CLINICAS_AUSENTE"),
            ("Conduta/Orientação", "SECAO_CONDUTA_ORIENTACAO_AUSENTE"),
            ("Limitações", "SECAO_LIMITACOES_AUSENTE"),
        ):
            if not re.search(
                rf"^[ \\t]*{re.escape(secao)}[ \\t]*:",
                rascunho,
                flags=re.IGNORECASE | re.MULTILINE,
            ):
                alertas.append(codigo)
        if not fontes:
            alertas.append("FONTES_AUSENTES")
        if AVISO_REVISAO_HUMANA not in rascunho:
            alertas.append("AVISO_REVISAO_HUMANA_AUSENTE")
        return alertas

    @staticmethod
    def _configuracao(id_execucao: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": id_execucao}}

    @staticmethod
    def _extrair_payload_interrupcao(resultado: dict[str, object]) -> dict[str, object]:
        interrupcoes = resultado.get("__interrupt__")
        if not interrupcoes:
            raise RuntimeError("O fluxo não solicitou revisão humana.")
        primeira_interrupcao = interrupcoes[0]
        payload = getattr(primeira_interrupcao, "value", primeira_interrupcao)
        if not isinstance(payload, dict):
            raise RuntimeError("A interrupção de revisão humana é inválida.")
        return payload

    def _auditar(self, estado: EstadoAssistente, etapa: str, situacao: str) -> None:
        self.auditoria.registrar(
            id_execucao=estado["id_execucao"],
            etapa=etapa,
            situacao=situacao,
            fontes=estado.get("fontes", []),
            alertas=estado.get("alertas", []),
            decisao_humana=estado.get("decisao_humana"),
            acao_humana=estado.get("acao_humana"),
        )

    def _registrar_falha(self, id_execucao: str, etapa: str, erro: Exception) -> None:
        self.auditoria.registrar(
            id_execucao=id_execucao,
            etapa=etapa,
            situacao="falha",
            tipo_erro=type(erro).__name__,
        )

    def _emitir_evento(
        self,
        estado: EstadoAssistente,
        no: str,
        status: str,
        duracao_ms: float | None = None,
    ) -> None:
        if self.observador_eventos is None:
            return
        evento: dict[str, object] = {
            "id_execucao": estado.get("id_execucao", ""),
            "no": no,
            "status": status,
        }
        if duracao_ms is not None:
            evento["duracao_ms"] = round(duracao_ms, 2)
        self.observador_eventos(evento)
