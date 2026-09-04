from __future__ import annotations

from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


class ArquivoService:
    """Centraliza a leitura e a gravação de arquivos Excel do pipeline."""

    def gerar_dataframe(
        self,
        caminho_arquivo: Path,
        percentual_registros: float = 100.0,
        quantidade_minima: int = 0,
    ) -> pd.DataFrame:
        """Lê um Excel e retorna o percentual solicitado dos registros."""
        if not caminho_arquivo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

        dataframe = pd.read_excel(caminho_arquivo)
        return self.selecionar_percentual_registros(
            dataframe,
            percentual_registros,
            quantidade_minima,
        )

    @staticmethod
    def selecionar_percentual_registros(
        dataframe: pd.DataFrame,
        percentual_registros: float,
        quantidade_minima: int = 0,
    ) -> pd.DataFrame:
        """Seleciona o percentual inicial do dataframe, preservando sua ordem."""
        if (
            isinstance(quantidade_minima, bool)
            or not isinstance(quantidade_minima, int)
            or quantidade_minima < 0
        ):
            raise ValueError(
                "A quantidade mínima de registros deve ser um inteiro "
                "maior ou igual a zero."
            )

        percentual_validado = ArquivoService.validar_percentual_registros(
            percentual_registros
        )
        quantidade_registros = ceil(
            len(dataframe) * percentual_validado / 100
        )
        quantidade_registros = min(
            len(dataframe),
            max(quantidade_registros, quantidade_minima),
        )
        return dataframe.iloc[:quantidade_registros].copy()

    @staticmethod
    def validar_percentual_registros(percentual_registros: float) -> float:
        """Valida e normaliza um percentual entre zero e cem."""
        if isinstance(percentual_registros, bool):
            raise ValueError("O percentual de registros deve estar entre 0 e 100.")

        try:
            percentual_validado = float(percentual_registros)
        except (TypeError, ValueError) as erro:
            raise ValueError(
                "O percentual de registros deve ser um número entre 0 e 100."
            ) from erro

        if not 0 < percentual_validado <= 100:
            raise ValueError("O percentual de registros deve estar entre 0 e 100.")

        return percentual_validado

    def criar_excel(self, dataframe: pd.DataFrame, caminho_arquivo: Path) -> Path:
        """Cria um arquivo Excel a partir de um dataframe e retorna seu caminho."""
        return self.atualizar_excel(dataframe, caminho_arquivo)

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
        return self.atualizar_excel_com_abas(
            {"Sheet1": dataframe},
            caminho_arquivo,
        )

    def atualizar_excel_com_abas(
        self,
        dataframes: dict[str, pd.DataFrame],
        caminho_arquivo: Path,
    ) -> Path:
        """Atualiza atomicamente um Excel composto por uma ou mais abas."""
        self._validar_extensao_xlsx(caminho_arquivo)
        if not dataframes:
            raise ValueError("Informe ao menos uma aba para gerar o arquivo Excel.")

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
            with pd.ExcelWriter(caminho_temporario, engine="openpyxl") as escritor:
                for nome_aba, dataframe in dataframes.items():
                    dataframe.to_excel(
                        escritor,
                        sheet_name=nome_aba,
                        index=False,
                    )
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
