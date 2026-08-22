from pathlib import Path

import pandas as pd

from app.services.excel_service import ExcelService


def test_criar_excel_substitui_arquivo_existente(tmp_path: Path) -> None:
    caminho_arquivo = tmp_path / "tratado.xlsx"
    servico = ExcelService()

    servico.criar_excel(
        pd.DataFrame({"coluna_antiga": ["valor antigo"]}),
        caminho_arquivo,
    )
    servico.criar_excel(
        pd.DataFrame({"coluna_nova": ["valor novo"]}),
        caminho_arquivo,
    )

    resultado = pd.read_excel(caminho_arquivo)

    assert list(resultado.columns) == ["coluna_nova"]
    assert resultado.loc[0, "coluna_nova"] == "valor novo"
