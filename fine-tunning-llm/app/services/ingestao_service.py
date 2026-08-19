from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# Resume o resultado da ingestão do Excel, incluindo origem, destino e metadados básicos.
@dataclass(frozen=True)
class IngestaoResultado:
    dataframe: pd.DataFrame
    qtd_linhas: int
    colunas_df: list[str]


class ExcelIngestaoService:
    def __init__(self, caminho_origem: Path) -> None:
        self.caminho_arquivo_origem = caminho_origem

    # Gera o dataframe a partir do arquivo Excel de origem e retorna metadados.
    def gerar_dataframe(self) -> IngestaoResultado:
        if not self.caminho_arquivo_origem.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.caminho_arquivo_origem}")

        dataframe_gerado = pd.read_excel(self.caminho_arquivo_origem)
        self.dataframe = dataframe_gerado

        return IngestaoResultado(
            dataframe=self.dataframe,
            qtd_linhas=len(self.dataframe),
            colunas_df=list(self.dataframe.columns),
        )
