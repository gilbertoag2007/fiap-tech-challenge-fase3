"""Orquestração LangGraph do assistente médico com revisão humana obrigatória."""

from __future__ import annotations

import re
from typing import Literal, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.assistente.auditoria import ServicoAuditoriaAssistente
from app.assistente.chain import AVISO_REVISAO_HUMANA, AssistenteChain
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
    ) -> None:
        self.repositorio = repositorio
        self.chain_assistente = chain_assistente
        self.auditoria = auditoria
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

    def _construir_grafo(self):
        fluxo = StateGraph(EstadoAssistente)
        fluxo.add_node("validar_entrada", self._validar_entrada)
        fluxo.add_node("consultar_registro", self._consultar_registro)
        fluxo.add_node("gerar_rascunho", self._gerar_rascunho)
        fluxo.add_node("validar_seguranca", self._validar_seguranca)
        fluxo.add_node("solicitar_revisao_humana", self._solicitar_revisao_humana)
        fluxo.add_node("finalizar_aprovacao", self._finalizar_aprovacao)
        fluxo.add_node("finalizar_rejeicao", self._finalizar_rejeicao)

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
        registro = self.repositorio.buscar_por_id(estado["id_registro"])
        atualizacao: EstadoAssistente = {
            "contexto_clinico": dict(registro.campos),
            "campos": dict(registro.campos),
            "fontes": list(registro.fontes),
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
            "decisao_humana": decisao.aprovado,
            "observacao_humana": decisao.observacao,
        }

    def _finalizar_aprovacao(self, estado: EstadoAssistente) -> EstadoAssistente:
        atualizacao: EstadoAssistente = {
            "situacao": "aprovada",
            "resposta": estado["rascunho"],
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
        )

    def _registrar_falha(self, id_execucao: str, etapa: str, erro: Exception) -> None:
        self.auditoria.registrar(
            id_execucao=id_execucao,
            etapa=etapa,
            situacao="falha",
            tipo_erro=type(erro).__name__,
        )
