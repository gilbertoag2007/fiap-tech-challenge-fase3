"""FastAPI do chatbot clínico com eventos LangGraph por SSE."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.api import deps
from app.api.schemas import (
    CriarSessaoRequest,
    CriarSessaoResponse,
    DecisaoRequest,
    RegistroResumo,
    StatusModelo,
    novo_id_execucao,
)
from app.assistente.modelos import DecisaoHumana, SolicitacaoAssistente
from app.config import CAMINHO_PRONTUARIOS, CORS_ORIGINS


app = FastAPI(
    title="Assistente Clínico FIAP",
    version="1.0.0",
    description="LangChain + LangGraph + HITL com modelos LoRA locais.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    # A base é um volume montado: sinalizar sua ausência aqui evita que o erro
    # apareça só na primeira consulta, como um caminho de container sem contexto.
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "modelo_carregado": (
            deps.servico_fine_tuning.chave_modelo
            if deps.servico_fine_tuning.modelo_ajustado_carregado
            else None
        ),
        "base_prontuarios": {
            "disponivel": CAMINHO_PRONTUARIOS.is_file(),
            "caminho": str(CAMINHO_PRONTUARIOS),
        },
    }


@app.get("/api/modelos", response_model=list[StatusModelo])
def listar_modelos() -> list[StatusModelo]:
    atual = deps.servico_fine_tuning.chave_modelo
    carregado = deps.servico_fine_tuning.modelo_ajustado_carregado
    resultado: list[StatusModelo] = []
    for chave, configuracao in deps.servico_fine_tuning.MODELOS_DISPONIVEIS.items():
        caminho = Path(configuracao["caminho_adapter"])
        resultado.append(
            StatusModelo(
                chave=chave,
                rotulo=str(configuracao["rotulo"]),
                modelo_base=str(configuracao["nome_base"]),
                disponivel_cpu=chave.startswith("qwen"),
                exige_cuda=chave == "llama",
                carregado=carregado and atual == chave,
                adapter_local=(caminho / "adapter_config.json").exists(),
                repo_adapter=(
                    str(configuracao["repo_adapter"])
                    if configuracao.get("repo_adapter")
                    else None
                ),
            )
        )
    return resultado


@app.post("/api/modelos/{chave}")
async def selecionar_modelo(chave: str) -> dict[str, str]:
    if chave not in deps.servico_fine_tuning.MODELOS_DISPONIVEIS:
        raise HTTPException(status_code=404, detail="Modelo não encontrado.")
    async with deps.trava_inferencia:
        await asyncio.to_thread(
            deps.servico_fine_tuning.configurar_modelo,
            chave,
        )
    return {"modelo": chave, "status": "selecionado"}


@app.get("/api/registros", response_model=list[RegistroResumo])
async def listar_registros(busca: str = "", limite: int = 20) -> list[RegistroResumo]:
    """Alimenta o seletor de prontuários da interface.

    Devolve apenas os campos de identificação do caso e a pergunta original que
    acompanha o registro; o contexto clínico completo continua acessível somente
    pela tool do fluxo, sob a allowlist do repositório.
    """
    try:
        registros = await asyncio.to_thread(
            deps.repositorio.listar,
            busca,
            limite,
        )
    except FileNotFoundError as erro:
        raise HTTPException(status_code=503, detail=str(erro)) from erro
    return [RegistroResumo.model_validate(registro) for registro in registros]


@app.get("/api/assistente/grafo")
def obter_grafo() -> dict[str, object]:
    return {
        "nos": [
            "receber_pergunta",
            "carregar_modelo",
            "validar_entrada",
            "consultar_registro",
            "gerar_rascunho",
            "validar_seguranca",
            "solicitar_revisao_humana",
            "finalizar_aprovacao",
            "finalizar_rejeicao",
        ],
        "arestas": [
            ["receber_pergunta", "carregar_modelo"],
            ["carregar_modelo", "validar_entrada"],
            ["validar_entrada", "consultar_registro"],
            ["consultar_registro", "gerar_rascunho"],
            ["gerar_rascunho", "validar_seguranca"],
            ["validar_seguranca", "solicitar_revisao_humana"],
            ["solicitar_revisao_humana", "finalizar_aprovacao"],
            ["solicitar_revisao_humana", "finalizar_rejeicao"],
        ],
    }


@app.post(
    "/api/assistente/sessoes",
    response_model=CriarSessaoResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def criar_sessao(payload: CriarSessaoRequest) -> CriarSessaoResponse:
    id_execucao = novo_id_execucao()
    deps.registro_sessoes.criar(id_execucao, payload.modelo)
    asyncio.create_task(_executar_sessao(id_execucao, payload))
    return CriarSessaoResponse(id_execucao=id_execucao)


async def _executar_sessao(
    id_execucao: str,
    payload: CriarSessaoRequest,
) -> None:
    try:
        async with deps.trava_inferencia:
            deps.registro_sessoes.adicionar_evento(
                id_execucao,
                {"no": "carregar_modelo", "status": "iniciado"},
            )
            await asyncio.to_thread(
                deps.servico_fine_tuning.configurar_modelo,
                payload.modelo,
            )
            deps.registro_sessoes.adicionar_evento(
                id_execucao,
                {"no": "carregar_modelo", "status": "concluido"},
            )
            revisao = await asyncio.to_thread(
                deps.fluxo_assistente.iniciar,
                SolicitacaoAssistente(
                    id_registro=payload.id_registro,
                    pergunta_clinica=payload.pergunta_clinica,
                    id_execucao=id_execucao,
                ),
            )
        deps.registro_sessoes.atualizar(
            id_execucao,
            status="aguardando_revisao",
            revisao=revisao.model_dump(),
        )
        deps.registro_sessoes.adicionar_evento(
            id_execucao,
            {
                "no": "solicitar_revisao_humana",
                "status": "aguardando_revisao",
                "revisao": revisao.model_dump(),
            },
        )
    except Exception as erro:
        deps.registro_sessoes.atualizar(
            id_execucao,
            status="falha",
            erro=f"{type(erro).__name__}: {erro}",
        )
        deps.registro_sessoes.adicionar_evento(
            id_execucao,
            {
                "no": "processamento",
                "status": "falha",
                "erro": str(erro),
            },
        )


@app.get("/api/assistente/sessoes/{id_execucao}")
def obter_sessao(id_execucao: str) -> dict[str, object]:
    try:
        return deps.registro_sessoes.obter(id_execucao)
    except KeyError as erro:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.") from erro


@app.get("/api/assistente/sessoes/{id_execucao}/checkpoints")
def checkpoints_sessao(id_execucao: str) -> dict[str, object]:
    """Expõe os checkpoints do InMemorySaver (LangGraph) já sanitizados."""
    try:
        deps.registro_sessoes.obter(id_execucao)
    except KeyError as erro:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.") from erro
    registros = deps.fluxo_assistente.historico_checkpoints(id_execucao)
    return {
        "id_execucao": id_execucao,
        "checkpointer": "InMemorySaver",
        "thread_id": id_execucao,
        "total": len(registros),
        "checkpoints": registros,
    }


@app.get("/api/assistente/sessoes/{id_execucao}/eventos")
async def eventos_sessao(
    id_execucao: str,
    request: Request,
) -> StreamingResponse:
    try:
        deps.registro_sessoes.obter(id_execucao)
    except KeyError as erro:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.") from erro

    async def transmitir() -> AsyncIterator[str]:
        sequencia = 0
        ciclos_sem_evento = 0
        while not await request.is_disconnected():
            eventos = deps.registro_sessoes.eventos_desde(
                id_execucao,
                sequencia,
            )
            for evento in eventos:
                sequencia += 1
                yield f"event: fluxo\ndata: {json.dumps(evento, ensure_ascii=False)}\n\n"
            if eventos:
                ciclos_sem_evento = 0
            else:
                ciclos_sem_evento += 1
            if ciclos_sem_evento >= 50:
                yield "event: heartbeat\ndata: {}\n\n"
                ciclos_sem_evento = 0
            await asyncio.sleep(0.2)

    return StreamingResponse(
        transmitir(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/assistente/sessoes/{id_execucao}/decisao")
async def decidir(
    id_execucao: str,
    payload: DecisaoRequest,
) -> dict[str, object]:
    try:
        sessao = deps.registro_sessoes.obter(id_execucao)
    except KeyError as erro:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.") from erro
    if sessao["status"] != "aguardando_revisao":
        raise HTTPException(
            status_code=409,
            detail="A sessão não está aguardando revisão.",
        )
    try:
        resposta = await asyncio.to_thread(
            deps.fluxo_assistente.retomar,
            id_execucao,
            DecisaoHumana(
                acao=payload.acao,
                texto_revisado=payload.texto_revisado,
                observacao=payload.observacao,
            ),
        )
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
    deps.registro_sessoes.atualizar(
        id_execucao,
        status=resposta.situacao,
        resposta=resposta.model_dump(),
    )
    deps.registro_sessoes.adicionar_evento(
        id_execucao,
        {
            "no": (
                "finalizar_rejeicao"
                if resposta.situacao == "rejeitada"
                else "finalizar_aprovacao"
            ),
            "status": resposta.situacao,
            "resposta": resposta.model_dump(),
        },
    )
    return resposta.model_dump()
