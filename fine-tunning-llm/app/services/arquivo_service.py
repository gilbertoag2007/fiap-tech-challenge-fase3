from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


class ArquivoService:
    """Centraliza a leitura e a gravação de arquivos Excel do pipeline."""

    def gerar_dataframe(self, caminho_arquivo: Path) -> pd.DataFrame:
        """Lê um arquivo Excel e retorna seu conteúdo em um dataframe."""
        if not caminho_arquivo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

        return pd.read_excel(caminho_arquivo)

    def criar_excel(self, dataframe: pd.DataFrame, caminho_arquivo: Path) -> Path:
        """Cria um arquivo Excel a partir de um dataframe e retorna seu caminho."""
        self._validar_extensao_xlsx(caminho_arquivo)

        caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
        if caminho_arquivo.exists():
            caminho_arquivo.unlink()

        dataframe.to_excel(caminho_arquivo, index=False)
        return caminho_arquivo

    def criar_arquivo_txt(
        self,
        titulo: str,
        conteudo: str,
        caminho_arquivo: Path,
    ) -> Path | None:
        """Cria um arquivo TXT com título e conteúdo e retorna seu caminho."""
        try:
            # Garante que o relatório anterior seja substituído no mesmo caminho.
            caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
            if caminho_arquivo.exists():
                caminho_arquivo.unlink()

            caminho_arquivo.write_text(
                f"{titulo}\n{conteudo}",
                encoding="utf-8",
            )
            return caminho_arquivo
        except OSError:
            return None

    def atualizar_excel(
        self,
        dataframe: pd.DataFrame,
        caminho_arquivo: Path,
    ) -> Path:
        """Atualiza um arquivo Excel com as colunas e linhas do dataframe."""
        self._validar_extensao_xlsx(caminho_arquivo)

        caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
        if not caminho_arquivo.exists():
            dataframe.to_excel(caminho_arquivo, index=False)
            return caminho_arquivo

        nome_aba = self._obter_nome_aba_principal(caminho_arquivo)

        with pd.ExcelWriter(
            caminho_arquivo,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as escritor:
            dataframe.to_excel(escritor, sheet_name=nome_aba, index=False)

        return caminho_arquivo

    @staticmethod
    def _validar_extensao_xlsx(caminho_arquivo: Path) -> None:
        if caminho_arquivo.suffix.lower() != ".xlsx":
            raise ValueError("O arquivo de destino deve possuir a extensao .xlsx.")

    @staticmethod
    def _obter_nome_aba_principal(caminho_arquivo: Path) -> str:
        """Recupera a aba principal para manter o mesmo nome no arquivo."""
        workbook = load_workbook(caminho_arquivo)
        try:
            return workbook.active.title
        finally:
            workbook.close()
