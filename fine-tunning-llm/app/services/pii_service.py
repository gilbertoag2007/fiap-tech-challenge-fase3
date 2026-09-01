from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.services.arquivo_service import ArquivoService


@dataclass
class ResultadoIdentificacaoPii:
    """Armazena o dataframe analisado e o caminho do arquivo atualizado."""

    dataframe_resultado: pd.DataFrame
    caminho_arquivo_tratado: Path | None


class PiiService:
    """Identifica e anonimiza PII nas colunas textuais de um dataframe."""

    CAMINHO_ARQUIVO_AUDITORIA = Path(
        "app/data/processado/dados_medicos_auditoria.xlsx"
    )

    ENTIDADES_PII = (
        "PERSON",
        "PHONE_NUMBER",
        "DATE_TIME",
        "CPF",
    )

    TOKENS_ANONIMIZACAO = {
        "PERSON": "[nome do paciente]",
        "PHONE_NUMBER": "[Telefone do paciente]",
        "DATE_TIME": "[Data de nascimento]",
        "CPF": "[cpf]",
    }

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
        self.anonymizer = AnonymizerEngine()
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
        """Coordena a identificacao e a anonimizacao das PII."""
        # Executa a identificacao e inclui as informacoes de PII no dataframe.
        resultado = self.identificar_pii(
            dataframe=dataframe,
            colunas_analisar=colunas_analisar,
            caminho_arquivo_tratado=caminho_arquivo_tratado,
            percentual_dataframe=percentual_dataframe,
        )

        dataframe_anonimizado = self.anonimizar_pii(
            dataframe=resultado.dataframe_resultado,
            colunas_analisar=["pergunta_original"],
        )

        if dataframe_anonimizado is not None:
            resultado.dataframe_resultado = dataframe_anonimizado
            resultado.caminho_arquivo_tratado = self.CAMINHO_ARQUIVO_AUDITORIA

        prontuario_anonimizado = self.anonimizar_prontuario_contexto(
            dataframe=resultado.dataframe_resultado,
        )

        if prontuario_anonimizado is not None:
            resultado.dataframe_resultado = prontuario_anonimizado
            resultado.caminho_arquivo_tratado = self.CAMINHO_ARQUIVO_AUDITORIA

        # Retorna o dataframe atualizado e o caminho do arquivo Excel tratado.
        return resultado

    def anonimizar_prontuario_contexto(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Remove nome e CPF do prontuario e reorganiza os dados do paciente."""
        coluna_original = "prontuario_contexto"
        coluna_anonimizada = "prontuario_contexto_anonimizado"

        if coluna_original not in dataframe.columns:
            return None

        houve_anonimizacao = False
        linhas_com_pii = dataframe["possui_pii"] == "Sim"
        entidades_por_campo = {"nome": "PERSON", "cpf": "CPF"}

        for indice, valor in dataframe.loc[linhas_com_pii, coluna_original].items():
            if pd.isna(valor) or not str(valor).strip():
                continue

            texto_original = str(valor)
            linha_sexo = ""
            demais_linhas: list[str] = []

            for linha in texto_original.splitlines():
                linha = linha.strip()
                campo, separador, conteudo = linha.partition(":")
                campo_normalizado = campo.casefold()

                if separador and campo_normalizado in entidades_por_campo:
                    entidade = entidades_por_campo[campo_normalizado]
                    resultados = self.analyzer.analyze(
                        text=conteudo,
                        language="pt",
                        entities=[entidade],
                    )
                    # O Presidio remove o valor sensivel antes de excluir o campo.
                    self.anonymizer.anonymize(
                        text=conteudo,
                        analyzer_results=resultados,
                        operators={entidade: OperatorConfig("redact", {})},
                    )
                    continue

                if separador and campo_normalizado == "sexo":
                    sexo = conteudo.strip().removesuffix(".")
                    linha_sexo = f"Paciente do Sexo {sexo}."
                    continue

                if linha:
                    demais_linhas.append(linha)

            linhas_anonimizadas = [linha_sexo, *demais_linhas]
            texto_anonimizado = "\n".join(
                linha for linha in linhas_anonimizadas if linha
            )

            if texto_anonimizado == texto_original:
                continue

            if coluna_anonimizada not in dataframe.columns:
                # Preserva o conteudo das linhas que nao precisam de anonimizacao.
                dataframe[coluna_anonimizada] = dataframe[coluna_original].astype(
                    "object"
                )

            dataframe.at[indice, coluna_anonimizada] = texto_anonimizado
            houve_anonimizacao = True

        if not houve_anonimizacao:
            return None

        self.servico_arquivo.atualizar_excel(
            dataframe,
            self.CAMINHO_ARQUIVO_AUDITORIA,
        )
        return dataframe

    def anonimizar_pii(
        self,
        dataframe: pd.DataFrame,
        colunas_analisar: list[str],
    ) -> pd.DataFrame | None:
        """Anonimiza as PII identificadas e atualiza o arquivo de auditoria."""
        dataframe_resultado = dataframe
        linhas_com_pii = dataframe_resultado["possui_pii"] == "Sim"
        houve_atualizacao = False

        for indice, registro in dataframe_resultado.loc[linhas_com_pii].iterrows():
            entidades = {
                entidade.strip()
                for entidade in str(registro["entidades identificadas"]).split(",")
                if entidade.strip() in self.TOKENS_ANONIMIZACAO
            }

            if not entidades:
                continue

            operadores = {
                entidade: OperatorConfig(
                    "replace",
                    {"new_value": self.TOKENS_ANONIMIZACAO[entidade]},
                )
                for entidade in entidades
            }

            for coluna in colunas_analisar:
                valor = registro[coluna]

                if pd.isna(valor) or not str(valor).strip():
                    continue

                texto_original = str(valor)
                resultados = self.analyzer.analyze(
                    text=texto_original,
                    language="pt",
                    entities=list(entidades),
                )

                if not resultados:
                    continue

                texto_anonimizado = self.anonymizer.anonymize(
                    text=texto_original,
                    analyzer_results=resultados,
                    operators=operadores,
                ).text

                if texto_anonimizado == texto_original:
                    continue

                coluna_anonimizada = f"{coluna}_anonimizado"
                if coluna_anonimizada not in dataframe_resultado.columns:
                    # Mantem os valores originais nas linhas que nao possuem PII.
                    dataframe_resultado[coluna_anonimizada] = dataframe_resultado[
                        coluna
                    ].astype("object")

                dataframe_resultado.at[indice, coluna_anonimizada] = texto_anonimizado
                houve_atualizacao = True

        if not houve_atualizacao:
            return None

        self.servico_arquivo.atualizar_excel(
            dataframe_resultado,
            self.CAMINHO_ARQUIVO_AUDITORIA,
        )
        return dataframe_resultado

    def identificar_pii(
        self,
        dataframe: pd.DataFrame,
        colunas_analisar: list[str],
        caminho_arquivo_tratado: Path | None = None,
        percentual_dataframe: float = 100.0,
    ) -> ResultadoIdentificacaoPii:
        """Identifica PII nas colunas informadas e atualiza o arquivo Excel."""
        # Inclui as colunas de resultado no dataframe recebido.
        dataframe_resultado = dataframe
        dataframe_resultado["entidades identificadas"] = ""
        dataframe_resultado["possui_pii"] = ""

        registros_analisar = self.servico_arquivo.selecionar_percentual_registros(
            dataframe_resultado,
            percentual_dataframe,
        )

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
