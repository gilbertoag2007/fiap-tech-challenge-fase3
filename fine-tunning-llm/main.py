from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from app.services.arquivo_service import ArquivoService

from app.services.pii_service import PiiService
from app.services.qualidade_service import QualidadeService


""" Colunas do arquivo a serem analisadas para verificar existencia de PII """
COLUNAS_ANALISADAS_PII: tuple[str, ...] = (
    "papel_solicitante",
    "contexto_solicitacao",
    "pergunta_original",
    "prontuario_contexto",
    "resposta_estruturada",
    "hipotese_clinica",
    "especialidade_medica",
    "tipo_pergunta",
    "diagnostico_confirmado",
    "exames_relevantes",
    "medicamentos_utilizados",
    "alergias",
    "diagnosticos_anteriores",
    


)


def exibir_menu(opcoes_menu: dict[str, str]) -> None:
    print("\nMenu principal")
    for numero_opcao, descricao_opcao in opcoes_menu.items():
        print(f"{numero_opcao} - {descricao_opcao}")


def executar_etapa_1(servico_arquivos: ArquivoService, caminho_arquivo: Path) -> pd.DataFrame:

    print (f"INICIANDO A ETAPA 1 - LEITURA DO ARQUIVO EXCEL: {caminho_arquivo}")

    dataframe_original = servico_arquivos.gerar_dataframe(caminho_arquivo)
    print(
        "Dataframe gerado com sucesso: "
        f"{dataframe_original.shape[0]} linhas e "
        f"{dataframe_original.shape[1]} colunas."
    )
    print ("ETAPA 1 CONCLUÍDA")
    print ("*" * 50)
    return dataframe_original


def executar_etapa_2(
    servico_qualidade: QualidadeService,
    dataframe_original   
) -> None:

    print(f"INICIANDO A ETAPA 2 - VERIFICAÇÃO REGISTROS REPETIDOS E COLUNAS AUSENTES")
            
    caminho_rel_repetidos = servico_qualidade.analisar_registros_repetidos(
        dataframe_original,
        Path("app/data/relatorios/registros_repetidos_antes.txt"),
    )
    caminho_rel_ausentes = servico_qualidade.analisar_registros_com_colunas_ausentes(
        dataframe_original,
        Path("app/data/relatorios/registros_ausentes_antes.txt"),
    )

    print(f"Relatório de registros repetidos: {caminho_rel_repetidos}")
    print(f"Relatório de registros com colunas ausentes: {caminho_rel_ausentes}")
    print(f"ETAPA 2 CONCLUÍDA")
    print ("*" * 50)

def executar_etapa_3(
    servico_qualidade: QualidadeService,
    dataframe_original,
    caminho_arquivo_tratado: Path,
    caminho_relatorio_repetidos_depois: Path,
    caminho_relatorio_ausentes_depois: Path,
):
    print(f"INICIANDO A ETAPA 3 - REMOVER REGISTROS REPETIDOS E COM COLUNAS AUSENTES")    

    resultado_tratamento_repetidos = servico_qualidade.remover_registros_repetidos(
        dataframe_original,
        caminho_arquivo_tratado,
    )
    dataframe_auditoria = resultado_tratamento_repetidos.dataframe_tratado

    caminho_rel_repetidos_pos_tratamento = servico_qualidade.analisar_registros_repetidos(
        dataframe_auditoria,
        caminho_relatorio_repetidos_depois,
    )

    resultado_tratamento_ausentes = servico_qualidade.remover_registros_com_colunas_ausentes(
        dataframe_auditoria,
        caminho_arquivo_tratado,
    )
    dataframe_auditoria = resultado_tratamento_ausentes.dataframe_tratado
    caminho_rel_ausentes_pos_tratamento = servico_qualidade.analisar_registros_com_colunas_ausentes(
        dataframe_auditoria,
        caminho_relatorio_ausentes_depois,
    )

    print("Inconsistências tratadas com sucesso.")
    print(
        "Registros repetidos removidos: "
        f"{resultado_tratamento_repetidos.linhas_tratadas}"
    )
    print (f"Relatório de registros repetidos gerado em: {caminho_rel_repetidos_pos_tratamento}")
    
    print(
        "Registros com colunas ausentes tratados: "
        f"{resultado_tratamento_ausentes.linhas_tratadas}"
    )
    print(f"Relatório de registros com colunas ausentes gerado em: {caminho_rel_ausentes_pos_tratamento}")

    print(
        "Arquivo Excel tratado gerado em: "
        f"{resultado_tratamento_ausentes.caminho_arquivo_tratado}"
    )

    print ("ETAPA 3 CONCLUÍDA")
    print ("*" * 50)

    return dataframe_auditoria


def executar_etapa_4(
    servico_pii: PiiService,
    dataframe_tratado,
    caminho_arquivo_tratado: Path,
):
    # Registra o momento inicial e inicia a medição da duração da etapa.
    data_hora_inicio = datetime.now()
    inicio_execucao = perf_counter()

    print("INICIANDO A ETAPA 4 - IDENTIFICAÇÃO E TRATAMENTO DE PII")
    print(f"Data e hora de início: {data_hora_inicio:%d/%m/%Y %H:%M:%S}")

    # Executa o fluxo de identificação e futuro tratamento das PII.
    resultado_pii = servico_pii.identificar_e_tratar_pii(
        dataframe=dataframe_tratado,
        colunas_analisar=list(COLUNAS_ANALISADAS_PII),
        caminho_arquivo_tratado=caminho_arquivo_tratado,
        percentual_dataframe=1,
    )

    print(
        "Arquivo Excel atualizado em: "
        f"{resultado_pii.caminho_arquivo_tratado}"
    )
    # Calcula e apresenta o tempo total depois que o processamento termina.
    duracao_execucao = perf_counter() - inicio_execucao
    minutos, segundos = divmod(duracao_execucao, 60)
    print(
        f"Tempo de execução: {int(minutos)} minutos e "
        f"{segundos:.2f} segundos"
    )
    print("ETAPA 4 CONCLUÍDA")
    print("*" * 50)

    return resultado_pii.dataframe_resultado

def main() -> None:
    caminho_arquivo_auditoria = Path("app/data/processado/dados_medicos_auditoria.xlsx")

    servico_arquivos = ArquivoService()
    servico_qualidade = QualidadeService()
    servico_pii = PiiService()
    dataframe_original = None
    dataframe_auditoria = None

    opcoes_menu = {
        "0": "Executar todas as etapas",
        "1": "Ler arquivo excel e gerar dataframe",
        "2": "Identificar registros repetidos e colunas ausentes",
        "3": "Tratar inconsistências encontradas",
        "4": "Identificar e tratar PII",
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

            # Etapa 1 - Leitura do arquivo Excel e geração do dataframe
            dataframe_original = executar_etapa_1(servico_arquivos,Path("app/data/original/dados_medicos_base.xlsx")) 

            # Etapa 2 - Identificação de registros repetidos e colunas ausentes
            executar_etapa_2( servico_qualidade, dataframe_original)
          
            # Etapa 3 - Tratamento de registros repetidos e colunas ausentes
            dataframe_auditoria = executar_etapa_3(
                servico_qualidade,
                dataframe_original,
                caminho_arquivo_auditoria,
                Path("app/data/relatorios/registros_repetidos_depois.txt"),
                Path("app/data/relatorios/registros_com_colunas_ausentes_depois.txt")
   
            )

            # Etapa 4 - Identificação e tratamento de PII
            dataframe_auditoria = executar_etapa_4(
                servico_pii,
                dataframe_auditoria,
                caminho_arquivo_auditoria,
            )

        if opcao_escolhida == "1":
            dataframe_original = executar_etapa_1(servico_arquivos)
           
            continue

        if opcao_escolhida == "2":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 1 antes de executar a qualidade.")
                continue

            executar_etapa_2( servico_qualidade, dataframe_original)
            continue

        if opcao_escolhida == "3":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 1 antes de tratar inconsistências.")
                continue

            dataframe_auditoria = executar_etapa_3(
                            servico_qualidade,
                            dataframe_original,
                            caminho_arquivo_auditoria,
                            Path("app/data/relatorios/registros_repetidos_depois.txt"),
                            Path("app/data/relatorios/registros_com_colunas_ausentes_depois.txt")
            )                
            continue

        if opcao_escolhida == "4":
            if dataframe_auditoria is None:
                print("Execute a opção 3 antes de identificar e tratar PII.")
                continue

            dataframe_auditoria = executar_etapa_4(
                servico_pii,
                dataframe_auditoria,
                caminho_arquivo_auditoria,
            )
            continue

      
        if opcao_escolhida == "5":
            print("A etapa de fine tuning ainda não foi implementada.")
            continue


if __name__ == "__main__":
    main()
