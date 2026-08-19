from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from app.services.excel_service import ExcelService


CAMINHO_ARQUIVO_TRATADO = Path("app/data/processed/dados_medicos_base_V3_tratado.xlsx")

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


class PiiService:
    def __init__(self, servico_excel: ExcelService | None = None) -> None:
        self.servico_excel = servico_excel or ExcelService()
        self.padroes_pii = {
            "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
            "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
            "telefone": re.compile(r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?(?:9?\d{4})-?\d{4}\b"),
            "data_nascimento": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
        }
        self.palavras_chave_pii = (
            "nome",
            "cpf",
            "rg",
            "email",
            "telefone",
            "celular",
            "endereco",
            "bairro",
            "cidade",
            "cep",
            "nascimento",
        )
    def identificar_pii(
        self,
        dataframe: pd.DataFrame,
        caminho_relatorio: Path | None = None,
        caminho_arquivo_tratado: Path = CAMINHO_ARQUIVO_TRATADO,
    ) -> ResultadoIdentificacao:
        dataframe["possui_pii"] = "Não"
        evidencias = self._coletar_evidencias(dataframe)
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

    def _coletar_evidencias(
        self,
        dataframe: pd.DataFrame,
    ) -> list[EvidenciaIdentificacao]:
        evidencias: list[EvidenciaIdentificacao] = []

        for nome_coluna in dataframe.columns:
            nome_coluna_normalizado = nome_coluna.lower()
            corresponde_coluna = any(
                palavra_chave in nome_coluna_normalizado for palavra_chave in self.palavras_chave_pii
            )

            for indice, valor in dataframe[nome_coluna].items():
                if pd.isna(valor):
                    continue

                valor_texto = str(valor).strip()
                if not valor_texto:
                    continue

                if corresponde_coluna or self._valor_tem_padrao_pii(valor_texto):
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

    def _valor_tem_padrao_pii(self, valor: str) -> bool:
        return any(padrao.search(valor) for padrao in self.padroes_pii.values())

    def _salvar_relatorio(
        self,
        caminho_relatorio: Path | None,
        titulo: str,
        evidencias: list[EvidenciaIdentificacao],
    ) -> None:
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
