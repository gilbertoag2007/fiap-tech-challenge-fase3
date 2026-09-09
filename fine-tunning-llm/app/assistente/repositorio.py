"""Acesso controlado aos prontuários anonimizados."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Protocol

import pandas as pd

from app.assistente.modelos import RegistroClinico


class RegistroNaoEncontradoError(LookupError):
    """Indica que não há um registro para o identificador informado."""


class RegistroDuplicadoError(LookupError):
    """Indica que há mais de um registro para o identificador informado."""


class RepositorioProntuarios(Protocol):
    """Fronteira de consulta de prontuários anonimizados."""

    def buscar_por_id(self, id_registro: str) -> RegistroClinico:
        pass

    def listar(self, busca: str = "", limite: int = 20) -> list[dict[str, str]]:
        pass


class RepositorioProntuariosExcel:
    """Consulta um arquivo Excel anonimizado por identificador de registro."""

    CAMPOS_PERMITIDOS = (
        "prontuario_contexto_anonimizado",
        "hipotese_clinica",
        "diagnostico_confirmado",
        "exames_relevantes",
        "medicamentos_utilizados",
        "alergias",
        "diagnosticos_anteriores",
        "especialidade_medica",
    )
    COLUNAS_OBRIGATORIAS = ("id", "prontuario_contexto_anonimizado")

    # Campos exibidos apenas no seletor da interface. Deliberadamente separados
    # de CAMPOS_PERMITIDOS: o que entra no prompt do modelo — e, portanto, na
    # lista de fontes — continua sendo exclusivamente a allowlist acima.
    CAMPOS_LISTAGEM = (
        "especialidade_medica",
        "hipotese_clinica",
        "diagnostico_confirmado",
        "tipo_pergunta",
        "contexto_solicitacao",
    )
    CAMPO_PERGUNTA_SUGERIDA = "pergunta_original_anonimizado"

    def __init__(self, servico_arquivo: object, caminho_arquivo: Path | str) -> None:
        self.servico_arquivo = servico_arquivo
        self.caminho_arquivo = Path(caminho_arquivo)
        self._cache: tuple[float, int, pd.DataFrame] | None = None
        self._trava_cache = Lock()

    def _carregar_dataframe(self) -> pd.DataFrame:
        """Lê a planilha uma vez e reaproveita enquanto o arquivo não mudar.

        A busca do seletor dispara a cada digitação; reler 14 mil linhas de
        Excel a cada tecla tornaria a interface inutilizável.
        """
        try:
            estado = self.caminho_arquivo.stat()
            assinatura = (estado.st_mtime, estado.st_size)
        except OSError:
            assinatura = None

        with self._trava_cache:
            if (
                assinatura is not None
                and self._cache is not None
                and self._cache[:2] == assinatura
            ):
                return self._cache[2]
            dataframe = self.servico_arquivo.gerar_dataframe(self.caminho_arquivo)
            if assinatura is not None:
                self._cache = (assinatura[0], assinatura[1], dataframe)
            return dataframe

    def listar(self, busca: str = "", limite: int = 20) -> list[dict[str, str]]:
        """Lista registros para o seletor, opcionalmente filtrados por texto.

        Um termo puramente numérico casa pelo início do ID, para que digitar o
        identificador continue sendo o caminho mais curto.
        """
        dataframe = self._carregar_dataframe()
        self._validar_colunas_obrigatorias(dataframe)

        termo = (busca or "").strip().lower()
        if termo:
            identificadores = dataframe["id"].map(self._normalizar_identificador)
            if termo.isdigit():
                selecao = identificadores.str.startswith(termo)
            else:
                selecao = pd.Series(False, index=dataframe.index)
                for coluna in self.CAMPOS_LISTAGEM:
                    if coluna in dataframe.columns:
                        selecao |= (
                            dataframe[coluna]
                            .astype(str)
                            .str.lower()
                            .str.contains(termo, regex=False, na=False)
                        )
            dataframe = dataframe.loc[selecao]

        limite = max(1, min(int(limite), 50))
        resultados: list[dict[str, str]] = []
        for _, linha in dataframe.head(limite).iterrows():
            item: dict[str, str] = {
                "id_registro": self._normalizar_identificador(linha["id"])
            }
            for coluna in self.CAMPOS_LISTAGEM:
                if coluna in linha.index and not self._valor_vazio(linha[coluna]):
                    item[coluna] = str(linha[coluna]).strip()
            pergunta = linha.get(self.CAMPO_PERGUNTA_SUGERIDA)
            if not self._valor_vazio(pergunta):
                item["pergunta_sugerida"] = str(pergunta).strip()
            resultados.append(item)
        return resultados

    def buscar_por_id(self, id_registro: str) -> RegistroClinico:
        """Retorna exatamente um registro, com campos explicitamente permitidos."""
        dataframe = self._carregar_dataframe()
        self._validar_colunas_obrigatorias(dataframe)

        identificador_normalizado = self._normalizar_identificador(id_registro)
        correspondencias = dataframe.loc[
            dataframe["id"].map(self._normalizar_identificador)
            == identificador_normalizado
        ]

        if len(correspondencias) == 0:
            raise RegistroNaoEncontradoError("Registro clínico não encontrado.")
        if len(correspondencias) > 1:
            raise RegistroDuplicadoError("Foram encontrados registros clínicos duplicados.")

        linha = correspondencias.iloc[0]
        campos = self._extrair_campos_permitidos(linha)
        if "prontuario_contexto_anonimizado" not in campos:
            raise ValueError(
                "O campo obrigatório prontuario_contexto_anonimizado não pode estar vazio."
            )
        return RegistroClinico(
            id_registro=identificador_normalizado,
            campos=campos,
            fontes=list(campos),
        )

    @classmethod
    def _validar_colunas_obrigatorias(cls, dataframe: pd.DataFrame) -> None:
        colunas_ausentes = [
            coluna
            for coluna in cls.COLUNAS_OBRIGATORIAS
            if coluna not in dataframe.columns
        ]
        if colunas_ausentes:
            raise ValueError(
                "Colunas obrigatórias ausentes: " + ", ".join(colunas_ausentes)
            )

    @classmethod
    def _extrair_campos_permitidos(cls, linha: pd.Series) -> dict[str, str]:
        campos: dict[str, str] = {}
        for campo in cls.CAMPOS_PERMITIDOS:
            if campo not in linha.index:
                continue
            valor = linha[campo]
            if cls._valor_vazio(valor):
                continue
            campos[campo] = str(valor)
        return campos

    @staticmethod
    def _normalizar_identificador(valor: object) -> str:
        if RepositorioProntuariosExcel._valor_vazio(valor):
            return ""
        return str(valor).strip()

    @staticmethod
    def _valor_vazio(valor: object) -> bool:
        if valor is None:
            return True
        if isinstance(valor, str):
            return not valor.strip()
        return bool(pd.isna(valor))
