"""Ferramentas LangChain usadas pelo fluxo do assistente médico."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, StructuredTool

from app.assistente.repositorio import RepositorioProntuarios


def criar_ferramenta_buscar_prontuario(
    repositorio: RepositorioProntuarios,
) -> BaseTool:
    """Cria a tool de consulta de prontuário ancorada no repositório Excel."""

    def _buscar_prontuario(id_registro: str) -> str:
        """Consulta o prontuário clínico anonimizado pelo identificador do Excel.

        Use esta ferramenta quando precisar recuperar o contexto clínico
        estruturado de um registro existente em dados_medicos_auditoria.xlsx.
        """
        registro = repositorio.buscar_por_id(id_registro)
        return json.dumps(
            {
                "id_registro": registro.id_registro,
                "campos": registro.campos,
                "fontes": registro.fontes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    return StructuredTool.from_function(
        func=_buscar_prontuario,
        name="buscar_prontuario",
        description=(
            "Busca um prontuário clínico anonimizado pelo ID do Excel de "
            "auditoria. Retorna campos permitidos e fontes consultadas."
        ),
    )


def criar_ferramentas_assistente(
    repositorio: RepositorioProntuarios,
) -> list[BaseTool]:
    """Retorna o conjunto de tools disponíveis ao fluxo LangGraph."""
    return [criar_ferramenta_buscar_prontuario(repositorio)]


def mapa_ferramentas(ferramentas: list[BaseTool]) -> dict[str, BaseTool]:
    """Indexa tools pelo nome para invocação determinística no grafo."""
    return {ferramenta.name: ferramenta for ferramenta in ferramentas}


def invocar_buscar_prontuario(
    ferramentas: dict[str, BaseTool],
    id_registro: str,
) -> dict[str, object]:
    """Invoca a tool buscar_prontuario e devolve o payload tipado."""
    ferramenta = ferramentas.get("buscar_prontuario")
    if ferramenta is None:
        raise RuntimeError("A ferramenta buscar_prontuario não está disponível.")
    bruto = ferramenta.invoke({"id_registro": id_registro})
    if isinstance(bruto, str):
        payload = json.loads(bruto)
    elif isinstance(bruto, dict):
        payload = bruto
    else:
        raise RuntimeError("Resposta inválida da ferramenta buscar_prontuario.")
    return payload
