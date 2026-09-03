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

COLUNAS_RESUMO: tuple[str, ...] = (
    "tipo_inconsistencia",
    "quantidade_antes",
    "registros_removidos",
    "quantidade_depois",
    "resultado",
)

COLUNAS_OCORRENCIAS: tuple[str, ...] = (
    "momento",
    "tipo_inconsistencia",
    "linha_excel",
    "id_registro",
    "coluna",
    "descricao",
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

    def gerar_relatorio_qualidade(
        self,
        dataframe_antes: pd.DataFrame,
        caminho_relatorio: Path,
        dataframe_depois: pd.DataFrame | None = None,
        registros_repetidos_removidos: int = 0,
        registros_ausentes_removidos: int = 0,
    ) -> Path:
        """Consolida o resumo e as ocorrências de qualidade em um Excel."""
        ocorrencias_antes = self._coletar_ocorrencias(
            dataframe_antes,
            momento="Antes",
        )
        ocorrencias_depois = (
            self._coletar_ocorrencias(dataframe_depois, momento="Depois")
            if dataframe_depois is not None
            else []
        )
        ocorrencias = pd.DataFrame(
            [*ocorrencias_antes, *ocorrencias_depois],
            columns=COLUNAS_OCORRENCIAS,
        )

        def contar_ocorrencias(
            registros: list[dict[str, object]],
            repetidos: bool,
        ) -> int:
            if repetidos:
                return sum(
                    registro["tipo_inconsistencia"] == "Registro repetido"
                    for registro in registros
                )
            return sum(
                registro["tipo_inconsistencia"] != "Registro repetido"
                for registro in registros
            )

        configuracoes = (
            (
                "Registros repetidos",
                contar_ocorrencias(ocorrencias_antes, repetidos=True),
                contar_ocorrencias(ocorrencias_depois, repetidos=True),
                registros_repetidos_removidos,
            ),
            (
                "Valores ou colunas ausentes",
                contar_ocorrencias(ocorrencias_antes, repetidos=False),
                contar_ocorrencias(ocorrencias_depois, repetidos=False),
                registros_ausentes_removidos,
            ),
        )

        resumo = []
        tratamento_executado = dataframe_depois is not None
        for tipo, quantidade_antes, quantidade_depois, removidos in configuracoes:
            if not tratamento_executado:
                resultado = (
                    "Aguardando tratamento"
                    if quantidade_antes
                    else "Sem ocorrências"
                )
            elif quantidade_depois:
                resultado = "Pendente"
            else:
                resultado = "Tratado" if quantidade_antes else "Sem ocorrências"

            resumo.append(
                {
                    "tipo_inconsistencia": tipo,
                    "quantidade_antes": quantidade_antes,
                    "registros_removidos": (
                        removidos if tratamento_executado else None
                    ),
                    "quantidade_depois": (
                        quantidade_depois if tratamento_executado else None
                    ),
                    "resultado": resultado,
                }
            )

        dataframe_resumo = pd.DataFrame(resumo, columns=COLUNAS_RESUMO)
        return self.servico_arquivo.atualizar_excel_com_abas(
            {
                "resumo": dataframe_resumo,
                "ocorrencias": ocorrencias,
            },
            caminho_relatorio,
        )

    def remover_registros_repetidos(
        self,
        dataframe: pd.DataFrame,
        caminho_arquivo_tratado: Path,
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
        caminho_arquivo_tratado: Path,
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
                dataframe[nome_coluna].isna()
                | dataframe[nome_coluna].astype(str).str.strip().eq("")
            )
            indices_ausentes.update(dataframe.index[mascara_ausente].tolist())

        return indices_ausentes

    def _coletar_ocorrencias(
        self,
        dataframe: pd.DataFrame,
        momento: str,
    ) -> list[dict[str, object]]:
        """Estrutura duplicidades e ausências para o relatório consolidado."""
        ocorrencias: list[dict[str, object]] = []

        for indice in dataframe.index[dataframe.duplicated(keep=False)]:
            identificador = (
                dataframe.at[indice, "id"]
                if "id" in dataframe.columns
                and not pd.isna(dataframe.at[indice, "id"])
                else ""
            )
            ocorrencias.append(
                {
                    "momento": momento,
                    "tipo_inconsistencia": "Registro repetido",
                    "linha_excel": indice + 2,
                    "id_registro": identificador,
                    "coluna": "",
                    "descricao": "Registro duplicado considerando todas as colunas",
                }
            )

        for nome_coluna in self.colunas_monitoradas:
            if nome_coluna not in dataframe.columns:
                ocorrencias.append(
                    {
                        "momento": momento,
                        "tipo_inconsistencia": "Coluna ausente no dataset",
                        "linha_excel": "",
                        "id_registro": "",
                        "coluna": nome_coluna,
                        "descricao": "Coluna monitorada ausente no dataset",
                    }
                )
                continue

            mascara_ausente = (
                dataframe[nome_coluna].isna()
                | dataframe[nome_coluna].astype(str).str.strip().eq("")
            )
            for indice in dataframe.index[mascara_ausente]:
                identificador = (
                    dataframe.at[indice, "id"]
                    if "id" in dataframe.columns
                    and not pd.isna(dataframe.at[indice, "id"])
                    else ""
                )
                ocorrencias.append(
                    {
                        "momento": momento,
                        "tipo_inconsistencia": "Valor ausente",
                        "linha_excel": indice + 2,
                        "id_registro": identificador,
                        "coluna": nome_coluna,
                        "descricao": "Valor nulo ou vazio",
                    }
                )

        return ocorrencias
