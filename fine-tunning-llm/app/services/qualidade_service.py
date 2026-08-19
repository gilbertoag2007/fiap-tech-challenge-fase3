from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.services.excel_service import ExcelService

# Colunas monitoradas para identificar ausências nos registros do dataframe.
COLUNAS_MONITORADAS: tuple[str, ...] = (
    "papel_solicitante",
    "contexto_solicitacao",
    "pergunta_original",
    "prontuario_contexto",
    "resposta_estruturada",
    "hipotese_clinica",
    "especialidade_medica",
    "tipo_pergunta",
    "diagnostico_confirmado",
)
CAMINHO_ARQUIVO_TRATADO = Path("app/data/processed/dados_medicos_base_V3_tratado.xlsx")

@dataclass(frozen=True)
class AnaliseInconsistenciaResult:
    caminho_relatorio: Path
    mensagem_resumo: str


@dataclass(frozen=True)
class TratamentoDuplicadosResult:
    dataframe_tratado: pd.DataFrame
    linhas_repetidas_removidas: int
    caminho_arquivo_tratado: Path


@dataclass(frozen=True)
class TratamentoAusentesResult:
    dataframe_tratado: pd.DataFrame
    linhas_com_ausencias_removidas: int
    caminho_arquivo_tratado: Path


class QualidadeService:
    def __init__(
        self,
        colunas: tuple[str, ...] = COLUNAS_MONITORADAS,
        servico_excel: ExcelService | None = None,
    ) -> None:
        self.colunas_monitoradas = colunas
        self.servico_excel = servico_excel or ExcelService()

    def analisar_registros_repetidos(
        self,
        dataframe: pd.DataFrame,
        caminho_relatorio: Path,
    ) -> AnaliseInconsistenciaResult:
        linhas_repetidas = dataframe.index[dataframe.duplicated(keep=False)]
        linhas_repetidas_formatadas = [str(indice + 2) for indice in linhas_repetidas]

        caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)
        with caminho_relatorio.open("w", encoding="utf-8") as arquivo_relatorio:
            if not linhas_repetidas_formatadas:
                arquivo_relatorio.write("Nenhum registro repetido foi identificado.\n")
            else:
                arquivo_relatorio.write("Linhas repetidas identificadas:\n")
                for linha in linhas_repetidas_formatadas:
                    arquivo_relatorio.write(f"Linha {linha}\n")

        if not linhas_repetidas_formatadas:
            mensagem_resumo = "Nenhuma inconsistência encontrada nos registros repetidos."
        else:
            mensagem_resumo = (
                f"Foram identificadas {len(linhas_repetidas_formatadas)} linhas repetidas."
            )

        return AnaliseInconsistenciaResult(
            caminho_relatorio=caminho_relatorio,
            mensagem_resumo=mensagem_resumo,
        )

    def analisar_registros_com_colunas_ausentes(
        self,
        dataframe: pd.DataFrame,
        caminho_relatorio: Path,
    ) -> AnaliseInconsistenciaResult:
        linhas_ausentes = self._coletar_valores_ausentes(dataframe)

        caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)
        with caminho_relatorio.open("w", encoding="utf-8") as arquivo_relatorio:
            if not linhas_ausentes:
                arquivo_relatorio.write("Nenhuma coluna ausente foi identificada.\n")
            else:
                arquivo_relatorio.write("Registros com colunas ausentes:\n")
                for linha in linhas_ausentes:
                    arquivo_relatorio.write(f"{linha}\n")

        if not linhas_ausentes:
            mensagem_resumo = "Nenhuma inconsistência encontrada nas colunas ausentes."
        else:
            mensagem_resumo = (
                f"Foram identificados {len(linhas_ausentes)} registros com colunas ausentes."
            )

        return AnaliseInconsistenciaResult(
            caminho_relatorio=caminho_relatorio,
            mensagem_resumo=mensagem_resumo,
        )

    def remover_registros_repetidos(
        self,
        dataframe: pd.DataFrame,
        caminho_arquivo_tratado: Path = CAMINHO_ARQUIVO_TRATADO,
    ) -> TratamentoDuplicadosResult:
        dataframe_tratado = dataframe.drop_duplicates().reset_index(drop=True)
        total_linhas_removidas = len(dataframe) - len(dataframe_tratado)
        caminho_arquivo_tratado = self.servico_excel.criar_excel(
            dataframe_tratado,
            caminho_arquivo_tratado,
        )

        return TratamentoDuplicadosResult(
            dataframe_tratado=dataframe_tratado,
            linhas_repetidas_removidas=total_linhas_removidas,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
        )

    def remover_registros_com_colunas_ausentes(
        self,
        dataframe: pd.DataFrame,
        caminho_arquivo_tratado: Path = CAMINHO_ARQUIVO_TRATADO,
    ) -> TratamentoAusentesResult:
        indices_ausentes = self._coletar_indices_ausentes(dataframe)
        mascara_ausentes = dataframe.index.isin(indices_ausentes)

        dataframe_tratado = dataframe.loc[~mascara_ausentes].reset_index(drop=True)
        total_linhas_removidas = len(dataframe) - len(dataframe_tratado)
        caminho_arquivo_tratado = self.servico_excel.criar_excel(
            dataframe_tratado,
            caminho_arquivo_tratado,
        )

        return TratamentoAusentesResult(
            dataframe_tratado=dataframe_tratado,
            linhas_com_ausencias_removidas=total_linhas_removidas,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
        )

    def _coletar_indices_ausentes(self, dataframe: pd.DataFrame) -> set[int]:
        indices_ausentes: set[int] = set()

        for nome_coluna in self.colunas_monitoradas:
            if nome_coluna not in dataframe.columns:
                indices_ausentes.update(dataframe.index.tolist())
                continue

            mascara_ausente = (
                dataframe[nome_coluna].isna() | dataframe[nome_coluna].astype(str).str.strip().eq("")
            )
            indices_ausentes.update(dataframe.index[mascara_ausente].tolist())

        return indices_ausentes

    def _coletar_valores_ausentes(self, dataframe: pd.DataFrame) -> list[str]:
        linhas_ausentes: list[str] = []

        for nome_coluna in self.colunas_monitoradas:
            if nome_coluna not in dataframe.columns:
                linhas_ausentes.append(f"COLUNA AUSENTE NO DATASET: {nome_coluna}")
                continue

            mascara_ausente = (
                dataframe[nome_coluna].isna() | dataframe[nome_coluna].astype(str).str.strip().eq("")
            )
            indices_ausentes = dataframe.index[mascara_ausente]

            for indice in indices_ausentes:
                numero_linha = indice + 2
                linhas_ausentes.append(
                    f"Linha {numero_linha} | Coluna {nome_coluna} | Valor ausente"
                )

        return linhas_ausentes
