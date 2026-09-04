"""Acesso controlado aos prontuários anonimizados."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from app.assistente.modelos import RegistroClinico


class RegistroNaoEncontradoError(LookupError):
    """Indica que não há um registro para o identificador informado."""


class RegistroDuplicadoError(LookupError):
    """Indica que há mais de um registro para o identificador informado."""


class RepositorioProntuarios(Protocol):
    """Fronteira de consulta de prontuários anonimizados."""

    def buscar_por_id(self, id_registro: str) -> RegistroClinico:
        pass


class RepositorioProntuariosExcel:
    """Consulta um arquivo Excel anonimizado por identificador de registro."""

    CAMPOS_PERMITIDOS = (
        "prontuario_contexto_anonimizado",
        "hipotese_clinica",
        "diagnostico_confirmado",
        "exames_relevantes",
        "medicamentos_utilizados",
        "alergias",
        "diagnosticos_anteriores",
        "especialidade_medica",
    )
    COLUNAS_OBRIGATORIAS = ("id", "prontuario_contexto_anonimizado")

    def __init__(self, servico_arquivo: object, caminho_arquivo: Path | str) -> None:
        self.servico_arquivo = servico_arquivo
        self.caminho_arquivo = Path(caminho_arquivo)

    def buscar_por_id(self, id_registro: str) -> RegistroClinico:
        """Retorna exatamente um registro, com campos explicitamente permitidos."""
        dataframe = self.servico_arquivo.gerar_dataframe(self.caminho_arquivo)
        self._validar_colunas_obrigatorias(dataframe)

        identificador_normalizado = self._normalizar_identificador(id_registro)
        correspondencias = dataframe.loc[
            dataframe["id"].map(self._normalizar_identificador)
            == identificador_normalizado
        ]

        if len(correspondencias) == 0:
            raise RegistroNaoEncontradoError("Registro clínico não encontrado.")
        if len(correspondencias) > 1:
            raise RegistroDuplicadoError("Foram encontrados registros clínicos duplicados.")

        linha = correspondencias.iloc[0]
        campos = self._extrair_campos_permitidos(linha)
        if "prontuario_contexto_anonimizado" not in campos:
            raise ValueError(
                "O campo obrigatório prontuario_contexto_anonimizado não pode estar vazio."
            )
        return RegistroClinico(
            id_registro=identificador_normalizado,
            campos=campos,
            fontes=list(campos),
        )

    @classmethod
    def _validar_colunas_obrigatorias(cls, dataframe: pd.DataFrame) -> None:
        colunas_ausentes = [
            coluna
            for coluna in cls.COLUNAS_OBRIGATORIAS
            if coluna not in dataframe.columns
        ]
        if colunas_ausentes:
            raise ValueError(
                "Colunas obrigatórias ausentes: " + ", ".join(colunas_ausentes)
            )

    @classmethod
    def _extrair_campos_permitidos(cls, linha: pd.Series) -> dict[str, str]:
        campos: dict[str, str] = {}
        for campo in cls.CAMPOS_PERMITIDOS:
            if campo not in linha.index:
                continue
            valor = linha[campo]
            if cls._valor_vazio(valor):
                continue
            campos[campo] = str(valor)
        return campos

    @staticmethod
    def _normalizar_identificador(valor: object) -> str:
        if RepositorioProntuariosExcel._valor_vazio(valor):
            return ""
        return str(valor).strip()

    @staticmethod
    def _valor_vazio(valor: object) -> bool:
        if valor is None:
            return True
        if isinstance(valor, str):
            return not valor.strip()
        return bool(pd.isna(valor))
