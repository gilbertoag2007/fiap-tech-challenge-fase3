"""Contratos tipados do assistente médico."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


TextoObrigatorio = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ModeloImutavel(BaseModel):
    """Base estrita e imutável para os contratos públicos."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class SolicitacaoAssistente(ModeloImutavel):
    """Entrada solicitada para uma execução do assistente."""

    id_registro: TextoObrigatorio
    pergunta_clinica: TextoObrigatorio
    id_execucao: TextoObrigatorio = Field(default_factory=lambda: str(uuid4()))


class RegistroClinico(ModeloImutavel):
    """Registro clínico anonimizado recuperado pela fonte de dados."""

    id_registro: TextoObrigatorio
    campos: dict[str, str]
    fontes: list[str]


class DecisaoHumana(ModeloImutavel):
    """Decisão obtida na etapa obrigatória de revisão humana."""

    aprovado: bool
    observacao: str = ""


class RevisaoPendente(ModeloImutavel):
    """Rascunho aguardando decisão humana antes de ser liberado."""

    id_execucao: TextoObrigatorio
    id_registro: TextoObrigatorio
    rascunho: TextoObrigatorio
    fontes: list[str]
    alertas: list[str]
    aviso: TextoObrigatorio


class RespostaAssistente(ModeloImutavel):
    """Resposta final, aprovada ou rejeitada pela revisão humana."""

    id_execucao: TextoObrigatorio
    id_registro: TextoObrigatorio
    situacao: Literal["aprovada", "rejeitada"]
    resposta: TextoObrigatorio | None
    fontes: list[str]
    alertas: list[str]
    aviso: TextoObrigatorio


class EstadoAssistente(TypedDict, total=False):
    """Estado serializável compartilhado pelo fluxo LangGraph."""

    id_registro: str
    pergunta_clinica: str
    id_execucao: str
    contexto_clinico: dict[str, str]
    campos: dict[str, str]
    fontes: list[str]
    rascunho: str
    alertas: list[str]
    aviso: str
    decisao_humana: bool
    observacao_humana: str
    situacao: Literal["aprovada", "rejeitada"]
    resposta: str | None
