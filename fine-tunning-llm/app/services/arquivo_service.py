from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


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
        with NamedTemporaryFile(
            dir=caminho_arquivo.parent,
            prefix=f".{caminho_arquivo.stem}_",
            suffix=caminho_arquivo.suffix,
            delete=False,
        ) as arquivo_temporario:
            caminho_temporario = Path(arquivo_temporario.name)

        try:
            # Grava um XLSX completo antes de substituir o destino existente.
            dataframe.to_excel(caminho_temporario, index=False)
            caminho_temporario.replace(caminho_arquivo)
        finally:
            # Remove o temporario caso a gravacao ou a substituicao falhe.
            if caminho_temporario.exists():
                caminho_temporario.unlink()

        return caminho_arquivo

    @staticmethod
    def _validar_extensao_xlsx(caminho_arquivo: Path) -> None:
        if caminho_arquivo.suffix.lower() != ".xlsx":
            raise ValueError("O arquivo de destino deve possuir a extensao .xlsx.")
