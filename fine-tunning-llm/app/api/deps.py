"""Dependências singleton da API."""

from __future__ import annotations

import asyncio

from app.api.estado import RegistroSessoes
from app.assistente.auditoria import ServicoAuditoriaAssistente
from app.assistente.chain import AssistenteChain
from app.assistente.fluxo import FluxoAssistenteMedico
from app.assistente.modelo_chat import ModeloChatLocal
from app.assistente.repositorio import RepositorioProntuariosExcel
from app.config import CAMINHO_AUDITORIA, CAMINHO_PRONTUARIOS
from app.services.arquivo_service import ArquivoService
from app.services.fine_tuning_service import FineTuningService


registro_sessoes = RegistroSessoes()
trava_inferencia = asyncio.Lock()

servico_arquivos = ArquivoService()
servico_fine_tuning = FineTuningService(servico_arquivo=servico_arquivos)
repositorio = RepositorioProntuariosExcel(
    servico_arquivo=servico_arquivos,
    caminho_arquivo=CAMINHO_PRONTUARIOS,
)
modelo_chat = ModeloChatLocal(servico_fine_tuning=servico_fine_tuning)
chain_assistente = AssistenteChain(modelo_chat)
auditoria = ServicoAuditoriaAssistente(caminho_arquivo=CAMINHO_AUDITORIA)


def observar_evento(evento: dict[str, object]) -> None:
    id_execucao = str(evento.get("id_execucao", ""))
    if not id_execucao:
        return
    try:
        registro_sessoes.adicionar_evento(id_execucao, evento)
    except KeyError:
        pass


fluxo_assistente = FluxoAssistenteMedico(
    repositorio=repositorio,
    chain_assistente=chain_assistente,
    auditoria=auditoria,
    observador_eventos=observar_evento,
)
