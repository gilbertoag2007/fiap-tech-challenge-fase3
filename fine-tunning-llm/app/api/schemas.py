"""Contratos HTTP da aplicação."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class CriarSessaoRequest(BaseModel):
    id_registro: str = Field(min_length=1)
    pergunta_clinica: str = Field(min_length=1)
    modelo: Literal["llama", "qwen10", "qwen80"] = "llama"


class CriarSessaoResponse(BaseModel):
    id_execucao: str
    status: str = "iniciando"


class DecisaoRequest(BaseModel):
    acao: Literal["aprovar", "rejeitar", "editar"]
    texto_revisado: str | None = None
    observacao: str = ""


class StatusModelo(BaseModel):
    chave: str
    rotulo: str
    modelo_base: str
    disponivel_cpu: bool
    exige_cuda: bool
    carregado: bool
    adapter_local: bool
    repo_adapter: str | None = None


class RegistroResumo(BaseModel):
    """Resumo de um registro, usado apenas para popular o seletor da interface."""

    id_registro: str
    especialidade_medica: str | None = None
    hipotese_clinica: str | None = None
    diagnostico_confirmado: str | None = None
    tipo_pergunta: str | None = None
    contexto_solicitacao: str | None = None
    pergunta_sugerida: str | None = None


def novo_id_execucao() -> str:
    return str(uuid4())
