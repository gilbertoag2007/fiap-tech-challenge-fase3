from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine

from app.services.arquivo_service import ArquivoService


@dataclass
class ResultadoIdentificacaoPii:
    """Armazena o dataframe analisado e o caminho do arquivo atualizado."""

    dataframe_resultado: pd.DataFrame
    caminho_arquivo_tratado: Path | None


class PiiService:
    """Identifica PII nas colunas textuais de um dataframe."""

    ENTIDADES_PII = (
        "PERSON",
        "PHONE_NUMBER",
        "DATE_TIME",
        "CPF",
    )

    def __init__(self, servico_arquivo: ArquivoService | None = None) -> None:
        """Inicializa o Presidio com o modelo local de portugues."""
        self.servico_arquivo = servico_arquivo or ArquivoService()
        motor_nlp = SpacyNlpEngine(
            models=[{"lang_code": "pt", "model_name": "pt_core_news_sm"}]
        )
        self.analyzer = AnalyzerEngine(
            nlp_engine=motor_nlp,
            supported_languages=["pt"],
        )
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="CPF",
                patterns=[
                    Pattern(
                        name="cpf_brasileiro",
                        regex=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
                        score=0.85,
                    )
                ],
                supported_language="pt",
            )
        )

    def identificar_e_tratar_pii(
        self,
        dataframe: pd.DataFrame,
        colunas_analisar: list[str],
        caminho_arquivo_tratado: Path | None = None,
        percentual_dataframe: float = 100.0,
    ) -> ResultadoIdentificacaoPii:
        """Coordena a identificacao e o futuro tratamento das PII."""
        # Executa a identificacao e inclui as informacoes de PII no dataframe.
        resultado = self.identificar_pii(
            dataframe=dataframe,
            colunas_analisar=colunas_analisar,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
            percentual_dataframe=percentual_dataframe,
        )

        # A chamada da futura anonimizacao sera adicionada neste ponto do fluxo.

        # Retorna o dataframe atualizado e o caminho do arquivo Excel tratado.
        return resultado

    def identificar_pii(
        self,
        dataframe: pd.DataFrame,
        colunas_analisar: list[str],
        caminho_arquivo_tratado: Path | None = None,
        percentual_dataframe: float = 100.0,
    ) -> ResultadoIdentificacaoPii:
        """Identifica PII nas colunas informadas e atualiza o arquivo Excel."""
        # Garante que o percentual represente uma parcela valida do dataframe.
        if not 0 < percentual_dataframe <= 100:
            raise ValueError("O percentual do dataframe deve estar entre 0 e 100.")

        # Inclui as colunas de resultado no dataframe recebido.
        dataframe_resultado = dataframe
        dataframe_resultado["entidades identificadas"] = ""
        dataframe_resultado["possui_pii"] = ""

        # Arredonda para cima para analisar ao menos um registro de dataframes nao vazios.
        quantidade_registros = ceil(
            len(dataframe_resultado) * percentual_dataframe / 100
        )
        registros_analisar = dataframe_resultado.iloc[:quantidade_registros]

        for indice, registro in registros_analisar.iterrows():
            entidades_identificadas: list[str] = []
            dataframe_resultado.at[indice, "possui_pii"] = "Não"

            for coluna in colunas_analisar:
                valor = registro[coluna]

                # Ignora celulas vazias antes de enviar o conteudo ao Presidio.
                if pd.isna(valor) or not str(valor).strip():
                    continue

                resultados = self.analyzer.analyze(
                    text=str(valor),
                    language="pt",
                    entities=list(self.ENTIDADES_PII),
                )
                entidades_identificadas.extend(
                    resultado.entity_type for resultado in resultados
                )

            if entidades_identificadas:
                # Remove repeticoes e separa por virgula as entidades do registro.
                entidades_identificadas = list(dict.fromkeys(entidades_identificadas))
                dataframe_resultado.at[indice, "entidades identificadas"] = ", ".join(
                    entidades_identificadas
                )
                dataframe_resultado.at[indice, "possui_pii"] = "Sim"

        # Atualiza o Excel somente quando um caminho de destino for informado.
        if caminho_arquivo_tratado is not None:
            caminho_arquivo_tratado = self.servico_arquivo.atualizar_excel(
                dataframe_resultado,
                caminho_arquivo_tratado,
            )

        return ResultadoIdentificacaoPii(
            dataframe_resultado=dataframe_resultado,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
        )
