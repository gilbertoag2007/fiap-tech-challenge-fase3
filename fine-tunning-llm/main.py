from pathlib import Path

from app.services.ingestao_service import ExcelIngestaoService
from app.services.pii_service import PiiService
from app.services.qualidade_service import QualidadeService


COLUNAS_ANALISADAS_PII: tuple[str, ...] = (
    "pergunta_original",
    "resposta_estruturada",
)


def exibir_menu(opcoes_menu: dict[str, str]) -> None:
    print("\nMenu principal")
    for numero_opcao, descricao_opcao in opcoes_menu.items():
        print(f"{numero_opcao} - {descricao_opcao}")


def executar_etapa_1(servico_ingestao: ExcelIngestaoService):
    resultado_ingestao = servico_ingestao.gerar_dataframe()
    print(
        "Dataframe gerado com sucesso: "
        f"{resultado_ingestao.qtd_linhas} linhas e "
        f"{resultado_ingestao.colunas_df} colunas."
    )
    return resultado_ingestao.dataframe


def executar_etapa_2(
    servico_qualidade: QualidadeService,
    dataframe_original,
    caminho_relatorio_repetidos_antes: Path,
    caminho_relatorio_ausentes_antes: Path,
) -> None:
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


def executar_etapa_3(
    servico_qualidade: QualidadeService,
    dataframe_original,
    caminho_arquivo_tratado: Path,
    caminho_relatorio_repetidos_depois: Path,
    caminho_relatorio_ausentes_depois: Path,
):
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
    print(f"Validação pós-tratamento de repetidos: {resultado_repetidos.mensagem_resumo}")
    print(
        "Registros com colunas ausentes tratados: "
        f"{resultado_tratamento_ausentes.linhas_com_ausencias_removidas}"
    )
    print(f"Validação pós-tratamento de ausentes: {resultado_ausentes.mensagem_resumo}")
    print(
        "Arquivo Excel tratado gerado em: "
        f"{resultado_tratamento_ausentes.caminho_arquivo_tratado}"
    )
    return dataframe_tratado


def executar_etapa_4(
    servico_pii: PiiService,
    dataframe_tratado,
    caminho_relatorio_pii: Path,
    caminho_arquivo_tratado: Path,
):
    resultado_pii = servico_pii.identificar_pii(
        dataframe=dataframe_tratado,
        colunas_analisadas=COLUNAS_ANALISADAS_PII,
        caminho_relatorio=caminho_relatorio_pii,
        caminho_arquivo_tratado=caminho_arquivo_tratado,
    )

    print(f"Evidências de PII identificadas: {resultado_pii.total_evidencias}")
    print(f"Relatório de PII gerado em: {resultado_pii.caminho_relatorio}")
    print(f"Arquivo Excel atualizado em: {resultado_pii.caminho_arquivo_tratado}")
    return resultado_pii.dataframe_tratado


def executar_etapa_4_2(
    servico_pii: PiiService,
    dataframe_tratado,
    caminho_arquivo_tratado: Path,
):
    resultado_anonimizacao = servico_pii.anonimizar_pii_deterministico(
        dataframe=dataframe_tratado,
        caminho_arquivo_tratado=caminho_arquivo_tratado,
    )

    print(
        "Registros anonimizados: "
        f"{resultado_anonimizacao.total_registros_anonimizados}"
    )
    print(
        "Arquivo Excel anonimizado em: "
        f"{resultado_anonimizacao.caminho_arquivo_tratado}"
    )
    return resultado_anonimizacao.dataframe_tratado


def executar_todas_as_etapas(
    servico_ingestao: ExcelIngestaoService,
    servico_qualidade: QualidadeService,
    servico_pii: PiiService,
    caminho_arquivo_tratado: Path,
    caminho_relatorio_repetidos_antes: Path,
    caminho_relatorio_ausentes_antes: Path,
    caminho_relatorio_repetidos_depois: Path,
    caminho_relatorio_ausentes_depois: Path,
    caminho_relatorio_pii: Path,
) -> None:
    dataframe_original = executar_etapa_1(servico_ingestao)
    executar_etapa_2(
        servico_qualidade,
        dataframe_original,
        caminho_relatorio_repetidos_antes,
        caminho_relatorio_ausentes_antes,
    )
    dataframe_tratado = executar_etapa_3(
        servico_qualidade,
        dataframe_original,
        caminho_arquivo_tratado,
        caminho_relatorio_repetidos_depois,
        caminho_relatorio_ausentes_depois,
    )
    executar_etapa_4(
        servico_pii,
        dataframe_tratado,
        caminho_relatorio_pii,
        caminho_arquivo_tratado,
    )


def main() -> None:
    caminho_arquivo_origem = Path("app/data/raw/dados_medicos_base_V3.xlsx")
    caminho_arquivo_tratado = Path("app/data/processed/dados_medicos_base_V3_tratado.xlsx")
    caminho_relatorio_repetidos_antes = Path("app/data/relatorios/registros_repetidos_antes.txt")
    caminho_relatorio_ausentes_antes = Path("app/data/relatorios/registros_com_colunas_ausentes_antes.txt")
    caminho_relatorio_repetidos_depois = Path("app/data/relatorios/registros_repetidos_depois.txt")
    caminho_relatorio_ausentes_depois = Path("app/data/relatorios/registros_com_colunas_ausentes_depois.txt")
    caminho_relatorio_pii = Path("app/data/relatorios/identificacao_pii.txt")

    servico_ingestao = ExcelIngestaoService(caminho_arquivo_origem)
    servico_qualidade = QualidadeService()
    servico_pii = PiiService()
    dataframe_original = None
    dataframe_tratado = None

    opcoes_menu = {
        "0": "Executar todas as etapas",
        "1": "Ler arquivo excel e gerar dataframe",
        "2": "Verificar qualidade do arquivo e tratar erros",
        "3": "Tratar inconsistências encontradas",
        "4": "Identificar PII",
        "4.2": "Anonimizar PII Deterministico",
        "5": "Executar Fine Tunning",
        "6": "Sair",
    }

    while True:
        exibir_menu(opcoes_menu)

        # Mantém o menu ativo até que o usuário informe uma opção válida.
        opcao_escolhida = input("Informe a opcao desejada: ").strip()

        if opcao_escolhida == "6":
            print("Encerrando o programa.")
            break

        if opcao_escolhida not in opcoes_menu:
            print("Opcao invalida. Escolha uma das opcoes do menu.")
            continue

        print(f"Opcao selecionada: {opcoes_menu[opcao_escolhida]}")

        if opcao_escolhida == "0":
            dataframe_original = executar_etapa_1(servico_ingestao)
            executar_etapa_2(
                servico_qualidade,
                dataframe_original,
                caminho_relatorio_repetidos_antes,
                caminho_relatorio_ausentes_antes,
            )
            dataframe_tratado = executar_etapa_3(
                servico_qualidade,
                dataframe_original,
                caminho_arquivo_tratado,
                caminho_relatorio_repetidos_depois,
                caminho_relatorio_ausentes_depois,
            )
            dataframe_tratado = executar_etapa_4(
                servico_pii,
                dataframe_tratado,
                caminho_relatorio_pii,
                caminho_arquivo_tratado,
            )
            dataframe_tratado = executar_etapa_4_2(
                servico_pii,
                dataframe_tratado,
                caminho_arquivo_tratado,
            )
            continue

        if opcao_escolhida == "1":
            dataframe_original = executar_etapa_1(servico_ingestao)
            dataframe_tratado = dataframe_original
            continue

        if opcao_escolhida == "2":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 1 antes de executar a qualidade.")
                continue

            executar_etapa_2(
                servico_qualidade,
                dataframe_original,
                caminho_relatorio_repetidos_antes,
                caminho_relatorio_ausentes_antes,
            )
            continue

        if opcao_escolhida == "3":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 1 antes de tratar inconsistências.")
                continue

            dataframe_tratado = executar_etapa_3(
                servico_qualidade,
                dataframe_original,
                caminho_arquivo_tratado,
                caminho_relatorio_repetidos_depois,
                caminho_relatorio_ausentes_depois,
            )
            continue

        if opcao_escolhida == "4":
            if dataframe_tratado is None:
                print("Carregue o dataframe na opção 1 antes de identificar PII.")
                continue

            dataframe_tratado = executar_etapa_4(
                servico_pii,
                dataframe_tratado,
                caminho_relatorio_pii,
                caminho_arquivo_tratado,
            )
            continue

        if opcao_escolhida == "4.2":
            if dataframe_tratado is None or "possui_pii" not in dataframe_tratado.columns:
                print("Execute a opcao 4 antes de anonimizar PII.")
                continue

            dataframe_tratado = executar_etapa_4_2(
                servico_pii,
                dataframe_tratado,
                caminho_arquivo_tratado,
            )
            continue

        if opcao_escolhida == "5":
            print("A etapa de fine tuning ainda não foi implementada.")
            continue


if __name__ == "__main__":
    main()
