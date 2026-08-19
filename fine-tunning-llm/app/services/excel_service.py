from __future__ import annotations

from pathlib import Path

import pandas as pd


class ExcelService:
    """Centraliza a leitura e a gravação de arquivos Excel do pipeline."""

    def ler_excel(self, caminho_arquivo: Path) -> pd.DataFrame:
        """Lê um arquivo Excel e retorna seu conteúdo em um dataframe."""
        if not caminho_arquivo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

        return pd.read_excel(caminho_arquivo)

    def criar_excel(self, dataframe: pd.DataFrame, caminho_arquivo: Path) -> Path:
        """Cria um arquivo Excel a partir de um dataframe e retorna seu caminho."""
        if caminho_arquivo.suffix.lower() != ".xlsx":
            raise ValueError("O arquivo de destino deve possuir a extensão .xlsx.")

        caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_excel(caminho_arquivo, index=False)
        return caminho_arquivo
