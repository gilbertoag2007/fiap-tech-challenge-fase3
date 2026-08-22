from pathlib import Path

import pandas as pd
import pytest

from app.services.pii_service import PiiService


class ServicoExcelFalso:
    def criar_excel(self, dataframe: pd.DataFrame, caminho_arquivo: Path) -> Path:
        return caminho_arquivo


def test_identificar_pii_analisa_somente_as_colunas_informadas(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "texto_analisado": ["CPF: 123.456.789-00", "Sem dados pessoais"],
            "texto_ignorado": ["CPF 123.456.789-00", "Telefone 11999999999"],
        }
    )
    servico = PiiService(servico_excel=ServicoExcelFalso())

    resultado = servico.identificar_pii(
        dataframe=dataframe,
        colunas_analisadas=("texto_analisado",),
        caminho_arquivo_tratado=tmp_path / "tratado.xlsx",
    )

    assert resultado.total_evidencias == 1
    assert resultado.evidencias[0].coluna == "texto_analisado"
    assert dataframe.loc[0, "possui_pii"] == "Sim"
    assert dataframe.loc[1, "possui_pii"] != "Sim"


def test_identificar_pii_rejeita_coluna_inexistente(tmp_path: Path) -> None:
    dataframe = pd.DataFrame({"texto": ["Sem dados pessoais"]})
    servico = PiiService(servico_excel=ServicoExcelFalso())

    with pytest.raises(ValueError, match="coluna_inexistente"):
        servico.identificar_pii(
            dataframe=dataframe,
            colunas_analisadas=("coluna_inexistente",),
            caminho_arquivo_tratado=tmp_path / "tratado.xlsx",
        )

    assert "possui_pii" not in dataframe.columns


def test_identificar_pii_nao_classifica_numeros_e_datas_comuns(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "texto": [
                "Exame realizado em 15/08/2024. Codigo do procedimento: 12345678.",
                "Sem data ou identificador pessoal.",
            ]
        }
    )
    servico = PiiService(servico_excel=ServicoExcelFalso())

    servico.identificar_pii(
        dataframe=dataframe,
        colunas_analisadas=("texto",),
        caminho_arquivo_tratado=tmp_path / "tratado.xlsx",
    )

    assert dataframe.loc[0, "possui_pii"] == "Sim"
    assert dataframe.loc[1, "possui_pii"] != "Sim"


def test_identificar_pii_nao_confunde_original_com_rg(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "pergunta_original": [
                "Quem teve hepatite A pode tomar vacina contra febre amarela?"
            ],
            "resposta_estruturada": ["Nao ha contraindicacao."],
        }
    )
    servico = PiiService(servico_excel=ServicoExcelFalso())

    servico.identificar_pii(
        dataframe=dataframe,
        colunas_analisadas=("pergunta_original", "resposta_estruturada"),
        caminho_arquivo_tratado=tmp_path / "tratado.xlsx",
    )

    assert dataframe.loc[0, "possui_pii"] != "Sim"


def test_anonimizar_pii_cria_pergunta_anonimizada_apenas_para_sim(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "pergunta_original": [
                "CPF: 123.456.789-00",
                "Pergunta sem identificador pessoal.",
            ],
            "possui_pii": ["Sim", "NÃ£o"],
        }
    )
    servico = PiiService(servico_excel=ServicoExcelFalso())

    resultado = servico.anonimizar_pii_deterministico(
        dataframe=dataframe,
        caminho_arquivo_tratado=tmp_path / "tratado.xlsx",
    )

    assert resultado.total_registros_anonimizados == 1
    assert dataframe.loc[0, "pergunta anonimizada"] == "CPF: [CPF]"
    assert dataframe.loc[1, "pergunta anonimizada"] == dataframe.loc[1, "pergunta_original"]
