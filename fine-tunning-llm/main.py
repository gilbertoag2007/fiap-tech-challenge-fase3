from pathlib import Path

from app.services.ingestao_service import ExcelIngestaoService
from app.services.pii_service import PiiService
from app.services.qualidade_service import QualidadeService


def main() -> None:
    caminho_arquivo_origem = Path("app/data/raw/dados_medicos_base_V3.xlsx")
    caminho_arquivo_tratado = Path("app/data/processed/dados_medicos_base_V3_tratado.xlsx")
    caminho_relatorio_repetidos_antes = Path("app/data/processed/registros_repetidos_antes.txt")
    caminho_relatorio_ausentes_antes = Path("app/data/processed/registros_com_colunas_ausentes_antes.txt")
    caminho_relatorio_repetidos_depois = Path("app/data/processed/registros_repetidos_depois.txt")
    caminho_relatorio_ausentes_depois = Path("app/data/processed/registros_com_colunas_ausentes_depois.txt")
    caminho_relatorio_pii = Path("app/data/processed/identificacao_pii.txt")

    servico_ingestao = ExcelIngestaoService(caminho_arquivo_origem)
    servico_qualidade = QualidadeService()
    servico_pii = PiiService()
    dataframe_original = None
    dataframe_tratado = None

    opcoes_menu = {
        "1": "Ler arquivo excel e gerar dataframe",
        "2": "Verificar qualidade do arquivo e tratar erros",
        "3": "Tratar inconsistências encontradas",
        "4": "Identificar PII",
        "5": "Executar Fine Tunning",
        "0": "Sair",
    }

    while True:
        print("\nMenu principal")
        for numero_opcao, descricao_opcao in opcoes_menu.items():
            print(f"{numero_opcao} - {descricao_opcao}")

        # Mantém o menu ativo até que o usuário informe uma opção válida.
        opcao_escolhida = input("Informe a opcao desejada: ").strip()

        if opcao_escolhida == "0":
            print("Encerrando o programa.")
            break

        if opcao_escolhida not in opcoes_menu:
            print("Opcao invalida. Escolha uma das opcoes do menu.")
            continue

        print(f"Opcao selecionada: {opcoes_menu[opcao_escolhida]}")

        if opcao_escolhida == "1":
            resultado_ingestao = servico_ingestao.gerar_dataframe()
            dataframe_original = resultado_ingestao.dataframe
            dataframe_tratado = resultado_ingestao.dataframe

            print(
                "Dataframe gerado com sucesso: "
                f"{resultado_ingestao.qtd_linhas} linhas e "
                f"{resultado_ingestao.colunas_df} colunas."
            )
            continue

        if opcao_escolhida == "2":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 1 antes de executar a qualidade.")
                continue

            resultado_repetidos = servico_qualidade.analisar_registros_repetidos(
                dataframe_original,
                caminho_relatorio_repetidos_antes,
            )
            resultado_ausentes = servico_qualidade.analisar_registros_com_colunas_ausentes(
                dataframe_original,
                caminho_relatorio_ausentes_antes,
            )

            print(resultado_repetidos.mensagem_resumo)
            print(f"Relatório gerado em: {resultado_repetidos.caminho_relatorio}")
            print(resultado_ausentes.mensagem_resumo)
            print(f"Relatório gerado em: {resultado_ausentes.caminho_relatorio}")
            continue

        if opcao_escolhida == "3":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 1 antes de tratar inconsistências.")
                continue

            resultado_tratamento = servico_qualidade.remover_registros_repetidos(
                dataframe_original,
                caminho_arquivo_tratado,
            )
            dataframe_tratado = resultado_tratamento.dataframe_tratado

            resultado_repetidos = servico_qualidade.analisar_registros_repetidos(
                dataframe_tratado,
                caminho_relatorio_repetidos_depois,
            )

            resultado_tratamento_ausentes = servico_qualidade.remover_registros_com_colunas_ausentes(
                dataframe_tratado,
                caminho_arquivo_tratado,
            )
            dataframe_tratado = resultado_tratamento_ausentes.dataframe_tratado
            resultado_ausentes = servico_qualidade.analisar_registros_com_colunas_ausentes(
                dataframe_tratado,
                caminho_relatorio_ausentes_depois,
            )
            print("Inconsistências tratadas com sucesso.")
            print(
                "Registros repetidos removidos: "
                f"{resultado_tratamento.linhas_repetidas_removidas}"
            )

            print(
                "Registros com colunas ausentes tratados: "
                f"{resultado_tratamento_ausentes.linhas_com_ausencias_removidas}"
            )
            print(
                "Arquivo Excel tratado gerado em: "
                f"{resultado_tratamento_ausentes.caminho_arquivo_tratado}"
            )
            continue

        if opcao_escolhida == "4":
            if dataframe_tratado is None:
                print("Carregue o dataframe na opção 1 antes de identificar PII.")
                continue

            resultado_pii = servico_pii.identificar_pii(
                dataframe_tratado,
                caminho_relatorio_pii,
                caminho_arquivo_tratado,
            )
            dataframe_tratado = resultado_pii.dataframe_tratado

            print(f"Evidências de PII identificadas: {resultado_pii.total_evidencias}")
            print(f"Relatório de PII gerado em: {resultado_pii.caminho_relatorio}")
            print(f"Arquivo Excel atualizado em: {resultado_pii.caminho_arquivo_tratado}")
            continue

        if opcao_escolhida == "5":
            print("A etapa de fine tuning ainda não foi implementada.")
            continue


if __name__ == "__main__":
    main()
