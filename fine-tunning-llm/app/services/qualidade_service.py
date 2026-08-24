from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.services.arquivo_service import ArquivoService

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



@dataclass(frozen=True)
class TratamentoResult:
    dataframe_tratado: pd.DataFrame
    linhas_tratadas: int
    caminho_arquivo_tratado: Path


class QualidadeService:
    def __init__(
        self,
        colunas: tuple[str, ...] = COLUNAS_MONITORADAS,
        servico_arquivo: ArquivoService | None = None,
    ) -> None:
        self.colunas_monitoradas = colunas
        self.servico_arquivo = servico_arquivo or ArquivoService()

    def analisar_registros_repetidos(
        self,
        dataframe: pd.DataFrame,
        caminho_relatorio: Path,
    ) -> Path | None:
        # Gera o relatório apenas quando houver pelo menos um registro repetido.
        linhas_repetidas = dataframe.index[dataframe.duplicated(keep=False)]
        conteudo = (
            "\n".join(f"Linha {indice + 2}" for indice in linhas_repetidas)
            if len(linhas_repetidas) > 0
            else "Registros repetidos não encontrados."
        )
        return self.servico_arquivo.criar_arquivo_txt(
            "Linhas repetidas identificadas:",
            conteudo,
            caminho_relatorio,
        )

    def analisar_registros_com_colunas_ausentes(
        self,
        dataframe: pd.DataFrame,
        caminho_relatorio: Path,
    ) -> Path | None:
        # Gera o relatório apenas quando houver registros com colunas ausentes.
        linhas_ausentes = self._coletar_valores_ausentes(dataframe)
        conteudo = (
            "\n".join(linhas_ausentes)
            if linhas_ausentes
            else "Registros com colunas ausentes não encontrados."
        )
        return self.servico_arquivo.criar_arquivo_txt(
            "Registros com colunas ausentes:",
            conteudo,
            caminho_relatorio,
        )

    def remover_registros_repetidos(
        self,
        dataframe: pd.DataFrame,
        caminho_arquivo_tratado: Path
    ) -> TratamentoResult:
        """Remove registros repetidos, salva o resultado e retorna o resumo."""
        # Remove as duplicidades e reorganiza os índices do dataframe tratado.
        dataframe_tratado = dataframe.drop_duplicates().reset_index(drop=True)
        total_linhas_removidas = len(dataframe) - len(dataframe_tratado)
        # Salva o dataframe tratado no caminho informado.
        caminho_arquivo_tratado = self.servico_arquivo.atualizar_excel(
            dataframe_tratado,
            caminho_arquivo_tratado,
        )

        return TratamentoResult(
            dataframe_tratado=dataframe_tratado,
            linhas_tratadas=total_linhas_removidas,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
        )

    def remover_registros_com_colunas_ausentes(
        self,
        dataframe: pd.DataFrame,
        caminho_arquivo_tratado: Path 
    ) -> TratamentoResult:
        """Remove registros com valores ausentes, salva o resultado e retorna o resumo."""
        # Identifica as linhas que possuem alguma coluna monitorada ausente.
        indices_ausentes = self._coletar_indices_ausentes(dataframe)
        mascara_ausentes = dataframe.index.isin(indices_ausentes)

        # Mantém somente os registros completos e reorganiza seus índices.
        dataframe_tratado = dataframe.loc[~mascara_ausentes].reset_index(drop=True)
        total_linhas_removidas = len(dataframe) - len(dataframe_tratado)
        # Salva o dataframe tratado no caminho informado.
        caminho_arquivo_tratado = self.servico_arquivo.atualizar_excel(
            dataframe_tratado,
            caminho_arquivo_tratado,
        )

        return TratamentoResult(
            dataframe_tratado=dataframe_tratado,
            linhas_tratadas=total_linhas_removidas,
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
