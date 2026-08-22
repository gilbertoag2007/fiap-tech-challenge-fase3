from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import RecognizerResult

from app.services.excel_service import ExcelService


CAMINHO_ARQUIVO_TRATADO = Path("app/data/processed/dados_medicos_base_V3_tratado.xlsx")
COLUNA_PERGUNTA_ANONIMIZADA = "pergunta anonimizada"

@dataclass(frozen=True)
class EvidenciaIdentificacao:
    coluna: str
    linha: int
    valor_encontrado: str
    tipo: str


@dataclass(frozen=True)
class ResultadoIdentificacao:
    dataframe_tratado: pd.DataFrame
    caminho_arquivo_tratado: Path
    caminho_relatorio: Path | None
    total_evidencias: int
    evidencias: list[EvidenciaIdentificacao]


@dataclass(frozen=True)
class ResultadoAnonimizacao:
    dataframe_tratado: pd.DataFrame
    caminho_arquivo_tratado: Path
    total_registros_anonimizados: int


class PiiService:
    def __init__(self, servico_excel: ExcelService | None = None) -> None:
        """Inicializa as regras de PII e o servico responsavel pela gravacao."""
        self.servico_excel = servico_excel or ExcelService()
        self.anonymizer = AnonymizerEngine()
        self.padroes_pii = {
            # CPF sem pontuacao precisa estar identificado; isso evita confundir telefone com CPF.
            "cpf": re.compile(
                r"(?i)(?:\bcpf\b\s*[:\-]?\s*\d{11}|\b\d{3}\.\d{3}\.\d{3}-\d{2}\b)"
            ),
            "data_nascimento": re.compile(
                r"(?<!\d)(?:0[1-9]|[12]\d|3[01])[-/](?:0[1-9]|1[0-2])[-/]\d{4}(?!\d)"
            ),
        }
    def identificar_pii(
        self,
        dataframe: pd.DataFrame,
        colunas_analisadas: Sequence[str],
        caminho_relatorio: Path | None = None,
        caminho_arquivo_tratado: Path = CAMINHO_ARQUIVO_TRATADO,
    ) -> ResultadoIdentificacao:
        """Identifica CPF e datas nas colunas informadas e grava o dataframe tratado."""
        colunas_analisadas = self._validar_colunas_analisadas(
            dataframe,
            colunas_analisadas,
        )
        dataframe["possui_pii"] = "Não"
        evidencias = self._coletar_evidencias(dataframe, colunas_analisadas)
        caminho_arquivo_tratado = self.servico_excel.criar_excel(
            dataframe,
            caminho_arquivo_tratado,
        )
        self._salvar_relatorio(caminho_relatorio, "Evidências de PII", evidencias)
        return ResultadoIdentificacao(
            dataframe_tratado=dataframe,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
            caminho_relatorio=caminho_relatorio,
            total_evidencias=len(evidencias),
            evidencias=evidencias,
        )

    def anonimizar_pii_deterministico(
        self,
        dataframe: pd.DataFrame,
        caminho_arquivo_tratado: Path = CAMINHO_ARQUIVO_TRATADO,
    ) -> ResultadoAnonimizacao:
        """Copia e anonimiza a pergunta apenas nos registros marcados com PII."""
        self._validar_colunas_anonimizacao(dataframe)
        dataframe[COLUNA_PERGUNTA_ANONIMIZADA] = dataframe["pergunta_original"]
        total_registros_anonimizados = 0

        for indice, possui_pii in dataframe["possui_pii"].items():
            if possui_pii != "Sim":
                continue

            valor_original = dataframe.at[indice, "pergunta_original"]
            if pd.isna(valor_original):
                continue

            texto_original = str(valor_original)
            resultados = self._coletar_resultados_anonymizer(texto_original)
            if not resultados:
                continue

            dataframe.at[indice, COLUNA_PERGUNTA_ANONIMIZADA] = (
                self.anonymizer.anonymize(
                    texto_original,
                    resultados,
                    operators={
                        "cpf": OperatorConfig("replace", {"new_value": "[CPF]"}),
                        "data_nascimento": OperatorConfig(
                            "replace", {"new_value": "[DATA_NASCIMENTO]"}
                        ),
                    },
                ).text
            )
            total_registros_anonimizados += 1

        caminho_arquivo_tratado = self.servico_excel.criar_excel(
            dataframe,
            caminho_arquivo_tratado,
        )
        return ResultadoAnonimizacao(
            dataframe_tratado=dataframe,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
            total_registros_anonimizados=total_registros_anonimizados,
        )

    def _coletar_evidencias(
        self,
        dataframe: pd.DataFrame,
        colunas_analisadas: Sequence[str],
    ) -> list[EvidenciaIdentificacao]:
        """Percorre as colunas selecionadas e registra as linhas com padroes de PII."""
        evidencias: list[EvidenciaIdentificacao] = []

        for nome_coluna in colunas_analisadas:
            for indice, valor in dataframe[nome_coluna].items():
                if pd.isna(valor):
                    continue

                valor_texto = str(valor).strip()
                if not valor_texto:
                    continue

                if self._valor_tem_padrao_pii(valor_texto):
                    dataframe.at[indice, "possui_pii"] = "Sim"
                    evidencias.append(
                        EvidenciaIdentificacao(
                            coluna=nome_coluna,
                            linha=indice + 2,
                            valor_encontrado=valor_texto,
                            tipo="PII",
                        )
                    )

        return evidencias

    def _validar_colunas_analisadas(
        self,
        dataframe: pd.DataFrame,
        colunas_analisadas: Sequence[str],
    ) -> tuple[str, ...]:
        """Valida, remove duplicidades e retorna as colunas usadas na analise."""
        if isinstance(colunas_analisadas, str):
            raise ValueError("Informe as colunas em uma lista ou tupla.")

        colunas_unicas = tuple(dict.fromkeys(colunas_analisadas))
        if not colunas_unicas:
            raise ValueError("Informe ao menos uma coluna para a análise de PII.")

        colunas_ausentes = [
            nome_coluna
            for nome_coluna in colunas_unicas
            if nome_coluna not in dataframe.columns
        ]
        if colunas_ausentes:
            raise ValueError(
                "Colunas não encontradas no dataframe: "
                + ", ".join(colunas_ausentes)
            )

        return colunas_unicas

    def _valor_tem_padrao_pii(self, valor: str) -> bool:
        """Informa se o texto possui CPF ou data compativel com as regexes configuradas."""
        return any(padrao.search(valor) for padrao in self.padroes_pii.values())

    def _coletar_resultados_anonymizer(
        self,
        valor: str,
    ) -> list[RecognizerResult]:
        """Converte os matches das regexes em resultados aceitos pelo Presidio."""
        resultados: list[RecognizerResult] = []
        for tipo, padrao in self.padroes_pii.items():
            for correspondencia in padrao.finditer(valor):
                resultados.append(
                    RecognizerResult(
                        entity_type=tipo,
                        start=correspondencia.start(),
                        end=correspondencia.end(),
                        score=1.0,
                    )
                )
        return resultados

    def _validar_colunas_anonimizacao(self, dataframe: pd.DataFrame) -> None:
        """Confirma que a identificacao foi executada antes da anonimização."""
        colunas_necessarias = {"pergunta_original", "possui_pii"}
        colunas_ausentes = colunas_necessarias.difference(dataframe.columns)
        if colunas_ausentes:
            raise ValueError(
                "Execute a identificacao de PII antes da anonimização. "
                "Colunas ausentes: "
                + ", ".join(sorted(colunas_ausentes))
            )

    def _salvar_relatorio(
        self,
        caminho_relatorio: Path | None,
        titulo: str,
        evidencias: list[EvidenciaIdentificacao],
    ) -> None:
        """Salva as evidencias encontradas em um relatorio texto, quando solicitado."""
        if caminho_relatorio is None:
            return

        caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)
        with caminho_relatorio.open("w", encoding="utf-8") as arquivo_relatorio:
            arquivo_relatorio.write(f"{titulo}\n")
            if not evidencias:
                arquivo_relatorio.write("Nenhuma evidência identificada.\n")
                return

            for evidencia in evidencias:
                arquivo_relatorio.write(
                    f"Linha {evidencia.linha} | Coluna {evidencia.coluna} | Tipo {evidencia.tipo} | Valor {evidencia.valor_encontrado}\n"
                )
