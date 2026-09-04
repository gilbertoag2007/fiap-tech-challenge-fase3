from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.assistente.auditoria import ServicoAuditoriaAssistente
from app.assistente.chain import AssistenteChain
from app.assistente.fluxo import FluxoAssistenteMedico
from app.assistente.modelo_chat import ModeloChatQwenLocal
from app.assistente.modelos import DecisaoHumana, SolicitacaoAssistente
from app.assistente.repositorio import (
    RegistroDuplicadoError,
    RegistroNaoEncontradoError,
    RepositorioProntuariosExcel,
)

if TYPE_CHECKING:
    from app.services.arquivo_service import ArquivoService
    from app.services.fine_tuning_service import FineTuningService
    from app.services.pii_service import PiiService
    from app.services.qualidade_service import QualidadeService
else:
    class ResultadoTratamentoQualidade(Protocol):
        """Resultado produzido pelas operações de tratamento de qualidade."""

        dataframe_tratado: pd.DataFrame
        linhas_tratadas: int
        caminho_arquivo_tratado: Path

    class ResultadoIdentificacaoPii(Protocol):
        """Resultado produzido pela identificação e anonimização de PII."""

        dataframe_resultado: pd.DataFrame
        caminho_arquivo_tratado: Path | None

    class ArquivoService(Protocol):
        """Contrato leve do serviço de arquivos usado pelo menu."""

        def gerar_dataframe(
            self,
            caminho_arquivo: Path,
            percentual_registros: float = 100.0,
            quantidade_minima: int = 0,
        ) -> pd.DataFrame:
            """Lê um dataframe a partir do arquivo indicado."""

        @staticmethod
        def validar_percentual_registros(percentual_registros: float) -> float:
            """Valida o percentual de registros solicitado."""

    class FineTuningService(Protocol):
        """Contrato leve do serviço de fine-tuning usado pelo menu."""

        NOME_MODELO_BASE: str
        CAMINHO_ARQUIVO_FINE_TUNING: Path
        CAMINHO_RELATORIO_METRICAS: Path
        CAMINHO_RELATORIO_TECNICO: Path
        limite_registros_fine_tuning: int | None
        quantidade_epocas_fine_tuning: int
        max_tokens_entrada: int
        rank_lora: int

        def gerar_dataframe_fine_tuning(self) -> pd.DataFrame:
            """Prepara o dataframe usado no fine-tuning."""

        def realizar_inferencia_base(self) -> Path:
            """Executa a inferência do modelo-base."""

        def realizar_fine_tuning(self) -> Path:
            """Executa o treinamento supervisionado."""

        def realizar_inferencia_fine_tuning(self) -> Path:
            """Executa a inferência do modelo ajustado."""

        def comparar_inferencias(self) -> Path:
            """Gera a comparação entre inferências."""

        def gerar_resposta_modelo_ajustado(
            self,
            mensagem_system: str,
            mensagem_usuario: str,
            max_novos_tokens: int = 384,
        ) -> str:
            """Gera uma resposta com o adaptador ajustado."""

    class PiiService(Protocol):
        """Contrato leve do serviço de PII usado pelo menu."""

        def identificar_e_tratar_pii(
            self,
            dataframe: pd.DataFrame,
            colunas_analisar: list[str],
            caminho_arquivo_tratado: Path | None = None,
            percentual_dataframe: float = 100.0,
        ) -> ResultadoIdentificacaoPii:
            """Identifica e anonimiza PII no dataframe informado."""

    class QualidadeService(Protocol):
        """Contrato leve do serviço de qualidade usado pelo menu."""

        def analisar_registros_repetidos(
            self,
            dataframe: pd.DataFrame,
            caminho_relatorio: Path,
        ) -> Path | None:
            """Gera o relatório de registros repetidos."""

        def analisar_registros_com_colunas_ausentes(
            self,
            dataframe: pd.DataFrame,
            caminho_relatorio: Path,
        ) -> Path | None:
            """Gera o relatório de colunas ausentes."""

        def remover_registros_repetidos(
            self,
            dataframe: pd.DataFrame,
            caminho_arquivo_tratado: Path,
        ) -> ResultadoTratamentoQualidade:
            """Remove registros repetidos e retorna seu resumo."""

        def remover_registros_com_colunas_ausentes(
            self,
            dataframe: pd.DataFrame,
            caminho_arquivo_tratado: Path,
        ) -> ResultadoTratamentoQualidade:
            """Remove registros com colunas ausentes e retorna seu resumo."""


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

# Configuracao centralizada da etapa 8 para facilitar novos experimentos.
LIMITE_REGISTROS_FINE_TUNING = None
QUANTIDADE_EPOCAS_FINE_TUNING = 3
RANK_LORA_FINE_TUNING = 16
MAX_TOKENS_ENTRADA_FINE_TUNING = 512

CONSOLE = Console()


def obter_icone(emoji: str, alternativa: str) -> str:
    """Usa emoji somente quando a codificação do terminal oferece suporte."""
    try:
        emoji.encode(CONSOLE.encoding)
    except (LookupError, UnicodeEncodeError):
        return alternativa
    return emoji


GRUPOS_MENU: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (f"{obter_icone('⚡', '>>')} FLUXOS COMPLETOS", "bright_cyan", ("0", "1")),
    (
        f"{obter_icone('🧹', '--')} PREPARAÇÃO DOS DADOS",
        "bright_blue",
        ("2", "3", "4", "5", "6"),
    ),
    (
        f"{obter_icone('🧠', '**')} MODELO E AVALIAÇÃO",
        "bright_magenta",
        ("7", "8", "9", "10"),
    ),
    (f"{obter_icone('🩺', '+')} ASSISTENTE MÉDICO", "bright_green", ("11",)),
    (f"{obter_icone('⚙', '--')} SISTEMA", "bright_black", ("12",)),
)

ICONES_MENU: dict[str, str] = {
    "0": obter_icone("🚀", ">>"),
    "1": obter_icone("🔬", "::"),
    "2": obter_icone("📂", "+"),
    "3": obter_icone("🔎", "?"),
    "4": obter_icone("🧽", "~"),
    "5": obter_icone("🔐", "#"),
    "6": obter_icone("🧩", "+"),
    "7": obter_icone("💬", ">"),
    "8": obter_icone("🛠", "*"),
    "9": obter_icone("✨", "*"),
    "10": obter_icone("📊", "%"),
    "11": obter_icone("🩺", "+"),
    "12": obter_icone("👋", "<"),
}


def exibir_menu(
    opcoes_menu: dict[str, str],
    percentual_registros: float,
    dataframe_original_carregado: bool,
    dataframe_auditoria_carregado: bool,
    dataframe_fine_tuning_carregado: bool,
) -> None:
    """Exibe as opções do pipeline agrupadas e o estado da sessão atual."""
    titulo = Text(
        f"{obter_icone('🩺', '+')}  PIPELINE DE FINE-TUNING DE LLM",
        style="bold bright_white",
    )
    subtitulo = Text(
        "Dados médicos  •  Qwen3-0.6B  •  LoRA",
        style="cyan",
    )
    cabecalho = Text.assemble(titulo, "\n", subtitulo)
    CONSOLE.print()
    CONSOLE.print(
        Panel(
            cabecalho,
            border_style="bright_cyan",
            padding=(1, 3),
        )
    )

    tabela = Table(
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white on dark_cyan",
        show_lines=False,
        expand=True,
    )
    tabela.add_column("Opção", justify="center", width=7, no_wrap=True)
    tabela.add_column("Ação", ratio=4)
    tabela.add_column("Estado", ratio=2, no_wrap=True)

    estados = {
        "0": "[cyan]fluxo completo[/cyan]",
        "1": "[magenta]fluxo completo[/magenta]",
        "2": "[green]disponível[/green]",
        "3": (
            "[green]disponível[/green]"
            if dataframe_original_carregado
            else "[yellow]requer etapa 2[/yellow]"
        ),
        "4": (
            "[green]disponível[/green]"
            if dataframe_original_carregado
            else "[yellow]requer etapa 2[/yellow]"
        ),
        "5": (
            "[green]disponível[/green]"
            if dataframe_auditoria_carregado
            else "[yellow]requer etapa 4[/yellow]"
        ),
        "6": (
            "[green]concluída[/green]"
            if dataframe_fine_tuning_carregado
            else (
                "[green]disponível[/green]"
                if dataframe_auditoria_carregado
                else "[yellow]requer etapa 4[/yellow]"
            )
        ),
        "7": "[blue]sob demanda[/blue]",
        "8": "[blue]sob demanda[/blue]",
        "9": "[blue]sob demanda[/blue]",
        "10": "[blue]sob demanda[/blue]",
        "11": "[green]revisão humana[/green]",
        "12": "[bright_black]encerrar[/bright_black]",
    }

    for indice_grupo, (nome_grupo, cor_grupo, opcoes_grupo) in enumerate(GRUPOS_MENU):
        if indice_grupo:
            tabela.add_section()
        tabela.add_row("", f"[bold {cor_grupo}]{nome_grupo}[/]", "")
        for numero_opcao in opcoes_grupo:
            tabela.add_row(
                f"[bold {cor_grupo}][ {numero_opcao} ][/]",
                f"{ICONES_MENU[numero_opcao]}  {opcoes_menu[numero_opcao]}",
                estados[numero_opcao],
            )

    CONSOLE.print(tabela)
    CONSOLE.print(
        f"[bright_black]Amostra configurada:[/bright_black] "
        f"[bold cyan]{percentual_registros:.2f}%[/bold cyan] dos registros\n"
    )


def executar_etapa_2(
    servico_arquivos: ArquivoService,
    caminho_arquivo: Path,
    percentual_registros: float,
) -> pd.DataFrame:

    print (f"INICIANDO A ETAPA 2 - LEITURA DO ARQUIVO EXCEL: {caminho_arquivo}")
    print(f"Percentual de registros utilizado: {percentual_registros:.2f}%")

    dataframe_original = servico_arquivos.gerar_dataframe(
        caminho_arquivo,
        percentual_registros=percentual_registros,
        quantidade_minima=3,
    )
    print(
        "Dataframe gerado com sucesso: "
        f"{dataframe_original.shape[0]} linhas e "
        f"{dataframe_original.shape[1]} colunas."
    )
    print ("ETAPA 2 CONCLUÍDA")
    print ("*" * 50)
    return dataframe_original


def executar_etapa_3(
    servico_qualidade: QualidadeService,
    dataframe_original   
) -> None:

    print(f"INICIANDO A ETAPA 3 - VERIFICAÇÃO REGISTROS REPETIDOS E COLUNAS AUSENTES")
            
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
    print(f"ETAPA 3 CONCLUÍDA")
    print ("*" * 50)

def executar_etapa_4(
    servico_qualidade: QualidadeService,
    dataframe_original,
    caminho_arquivo_tratado: Path,
    caminho_relatorio_repetidos_depois: Path,
    caminho_relatorio_ausentes_depois: Path,
):
    data_hora_inicio = datetime.now()
    inicio_execucao = perf_counter()

    print(f"INICIANDO A ETAPA 4 - REMOVER REGISTROS REPETIDOS E COM COLUNAS AUSENTES")
    print(f"Data e hora de início: {data_hora_inicio:%d/%m/%Y %H:%M}")

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

    data_hora_termino = datetime.now()
    duracao_minutos = (perf_counter() - inicio_execucao) / 60
    print(f"Data e hora de término: {data_hora_termino:%d/%m/%Y %H:%M}")
    print(f"Duração da execução: {duracao_minutos:.2f} minutos")
    print ("ETAPA 4 CONCLUÍDA")
    print ("*" * 50)

    return dataframe_auditoria


def executar_etapa_5(
    servico_pii: PiiService,
    dataframe_tratado,
    caminho_arquivo_tratado: Path,
):
    # Registra o momento inicial e inicia a medição da duração da etapa.
    data_hora_inicio = datetime.now()
    inicio_execucao = perf_counter()

    print("INICIANDO A ETAPA 5 - IDENTIFICAÇÃO E TRATAMENTO DE PII")
    print(f"Data e hora de início: {data_hora_inicio:%d/%m/%Y %H:%M}")

    # Executa o fluxo de identificação e futuro tratamento das PII.
    resultado_pii = servico_pii.identificar_e_tratar_pii(
        dataframe=dataframe_tratado,
        colunas_analisar=list(COLUNAS_ANALISADAS_PII),
        caminho_arquivo_tratado=caminho_arquivo_tratado,
        # Processa integralmente o subconjunto escolhido na leitura inicial.
        percentual_dataframe=100.0,
    )

    print(
        "Arquivo Excel atualizado em: "
        f"{resultado_pii.caminho_arquivo_tratado}"
    )
    # Calcula e apresenta o tempo total depois que o processamento termina.
    data_hora_termino = datetime.now()
    duracao_minutos = (perf_counter() - inicio_execucao) / 60
    print(f"Data e hora de término: {data_hora_termino:%d/%m/%Y %H:%M}")
    print(f"Duração da execução: {duracao_minutos:.2f} minutos")
    print("ETAPA 5 CONCLUÍDA")
    print("*" * 80)

    return resultado_pii.dataframe_resultado


def executar_etapa_6(
    servico_fine_tuning: FineTuningService,
) -> pd.DataFrame:
    print("INICIANDO A ETAPA 6 - PREPARACAO DO DATAFRAME PARA FINE-TUNING")

    dataframe_fine_tuning = servico_fine_tuning.gerar_dataframe_fine_tuning()

    print(
        "Dataframe de fine-tuning gerado com sucesso: "
        f"{dataframe_fine_tuning.shape[0]} linhas e "
        f"{dataframe_fine_tuning.shape[1]} colunas."
    )
    print(
        "Arquivo Excel gerado em: "
        f"{servico_fine_tuning.CAMINHO_ARQUIVO_FINE_TUNING}"
    )
    print("ETAPA 6 CONCLUIDA")
    print("*" * 80)

    return dataframe_fine_tuning


def executar_etapa_7(servico_fine_tuning: FineTuningService) -> Path:
    """Executa a inferencia-base nos registros reservados para teste."""
    data_hora_inicio = datetime.now()
    inicio_execucao = perf_counter()

    print("INICIANDO A ETAPA 7 - INFERENCIA BASE")
    print(f"Data e hora de início: {data_hora_inicio:%d/%m/%Y %H:%M}")
    caminho_relatorio = servico_fine_tuning.realizar_inferencia_base()

    print(f"Relatorio de inferencia base gerado em: {caminho_relatorio}")
    data_hora_termino = datetime.now()
    duracao_minutos = (perf_counter() - inicio_execucao) / 60
    print(f"Data e hora de término: {data_hora_termino:%d/%m/%Y %H:%M}")
    print(f"Duração da execução: {duracao_minutos:.2f} minutos")
    print("ETAPA 7 CONCLUIDA")
    print("*" * 80)
    return caminho_relatorio


def executar_etapa_8(servico_fine_tuning: FineTuningService) -> Path:
    """Executa o fine-tuning supervisionado com adaptador LoRA."""
    data_hora_inicio = datetime.now()
    inicio_execucao = perf_counter()

    print("INICIANDO A ETAPA 8 - FINE-TUNING COM LORA")
    print(f"Data e hora de início: {data_hora_inicio:%d/%m/%Y %H:%M}")
    print("O treinamento sera executado em CPU e pode ser demorado.")
    print(
        "Configuracao: "
        f"registros={servico_fine_tuning.limite_registros_fine_tuning}, "
        f"epocas={servico_fine_tuning.quantidade_epocas_fine_tuning}, "
        f"max_tokens={servico_fine_tuning.max_tokens_entrada}, "
        f"lora_r={servico_fine_tuning.rank_lora}."
    )

    caminho_modelo = servico_fine_tuning.realizar_fine_tuning()

    print(f"Adaptador LoRA salvo em: {caminho_modelo}")
    print(
        "Metricas do fine-tuning salvas em: "
        f"{servico_fine_tuning.CAMINHO_RELATORIO_METRICAS}"
    )
    print(
        "Relatorio tecnico do fine-tuning salvo em: "
        f"{servico_fine_tuning.CAMINHO_RELATORIO_TECNICO}"
    )
    data_hora_termino = datetime.now()
    duracao_minutos = (perf_counter() - inicio_execucao) / 60
    print(f"Data e hora de término: {data_hora_termino:%d/%m/%Y %H:%M}")
    print(f"Duração da execução: {duracao_minutos:.2f} minutos")
    print("ETAPA 8 CONCLUIDA")
    print("*" * 80)
    return caminho_modelo


def executar_etapa_9(servico_fine_tuning: FineTuningService) -> Path:
    """Executa a inferencia com o adaptador LoRA treinado."""
    data_hora_inicio = datetime.now()
    inicio_execucao = perf_counter()

    print("INICIANDO A ETAPA 9 - INFERENCIA APOS FINE-TUNING")
    print(f"Data e hora de início: {data_hora_inicio:%d/%m/%Y %H:%M}")

    caminho_relatorio = servico_fine_tuning.realizar_inferencia_fine_tuning()

    print(f"Relatorio da inferencia ajustada gerado em: {caminho_relatorio}")
    data_hora_termino = datetime.now()
    duracao_minutos = (perf_counter() - inicio_execucao) / 60
    print(f"Data e hora de término: {data_hora_termino:%d/%m/%Y %H:%M}")
    print(f"Duração da execução: {duracao_minutos:.2f} minutos")
    print("ETAPA 9 CONCLUIDA")
    print("*" * 80)
    return caminho_relatorio


def executar_etapa_10(servico_fine_tuning: FineTuningService) -> Path:
    """Gera o relatorio comparativo das inferencias do split de teste."""
    print("INICIANDO A ETAPA 10 - COMPARACAO DAS INFERENCIAS")

    caminho_relatorio = servico_fine_tuning.comparar_inferencias()

    print(f"Relatorio comparativo gerado em: {caminho_relatorio}")
    print("ETAPA 10 CONCLUIDA")
    print("*" * 80)
    return caminho_relatorio


def executar_etapa_11(fluxo_assistente: FluxoAssistenteMedico) -> None:
    """Gera um rascunho e exige revisão humana antes da liberação."""
    id_registro = CONSOLE.input("Informe o identificador do registro: ").strip()
    pergunta_clinica = CONSOLE.input("Informe a pergunta clínica: ").strip()
    solicitacao = SolicitacaoAssistente(
        id_registro=id_registro,
        pergunta_clinica=pergunta_clinica,
    )
    revisao = fluxo_assistente.iniciar(solicitacao)
    fontes = ", ".join(revisao.fontes) or "Nenhuma fonte informada."
    alertas = ", ".join(revisao.alertas) or "Nenhum alerta."
    CONSOLE.print(
        Panel(
            f"Rascunho:\n{revisao.rascunho}\n\n"
            f"Fontes: {fontes}\n"
            f"Alertas: {alertas}\n"
            f"Aviso: {revisao.aviso}",
            title="Revisão humana obrigatória",
            border_style="yellow",
        )
    )

    while True:
        decisao_informada = CONSOLE.input(
            "Aprovar o rascunho? (s/n): "
        ).strip().lower()
        if decisao_informada in {"s", "n"}:
            break
        CONSOLE.print("Decisão inválida. Informe apenas 's' ou 'n'.")

    observacao = CONSOLE.input("Observação da revisão (opcional): ").strip()
    resposta = fluxo_assistente.retomar(
        revisao.id_execucao,
        DecisaoHumana(
            aprovado=decisao_informada == "s",
            observacao=observacao,
        ),
    )
    if resposta.situacao == "aprovada":
        fontes = ", ".join(resposta.fontes) or "Nenhuma fonte informada."
        CONSOLE.print(
            Panel(
                f"Resposta:\n{resposta.resposta}\n\n"
                f"Fontes: {fontes}\n"
                f"Aviso: {resposta.aviso}",
                title="Resposta aprovada",
                border_style="green",
            )
        )
        return

    CONSOLE.print("Rascunho rejeitado. Nenhum conteúdo foi liberado.")


def solicitar_percentual_registros() -> float:
    """Solicita uma única vez o percentual usado em toda a execução."""
    from app.services.arquivo_service import ArquivoService

    while True:
        valor_informado = CONSOLE.input(
            "[bold cyan]Informe o percentual de registros que será utilizado "
            "(0 a 100): [/bold cyan]"
        ).strip()
        try:
            return ArquivoService.validar_percentual_registros(
                valor_informado.replace(",", ".")
            )
        except ValueError as erro:
            CONSOLE.print(
                f"[bold red]{obter_icone('✖', 'X')} Percentual inválido:[/bold red] "
                f"{erro}"
            )


def main(percentual_registros: float | None = None) -> None:
    from app.services.arquivo_service import ArquivoService
    from app.services.fine_tuning_service import FineTuningService
    from app.services.pii_service import PiiService
    from app.services.qualidade_service import QualidadeService

    if percentual_registros is None:
        percentual_registros = solicitar_percentual_registros()
    else:
        percentual_registros = ArquivoService.validar_percentual_registros(
            percentual_registros
        )

    caminho_arquivo_auditoria = Path("app/data/processado/dados_medicos_auditoria.xlsx")

    servico_arquivos = ArquivoService()
    servico_qualidade = QualidadeService(servico_arquivo=servico_arquivos)
    servico_pii = PiiService(servico_arquivo=servico_arquivos)
    servico_fine_tuning = FineTuningService(
        servico_arquivo=servico_arquivos,
        limite_registros_fine_tuning=LIMITE_REGISTROS_FINE_TUNING,
        quantidade_epocas_fine_tuning=QUANTIDADE_EPOCAS_FINE_TUNING,
        rank_lora=RANK_LORA_FINE_TUNING,
        max_tokens_entrada=MAX_TOKENS_ENTRADA_FINE_TUNING,
    )
    repositorio_prontuarios = RepositorioProntuariosExcel(
        servico_arquivo=servico_arquivos,
        caminho_arquivo=caminho_arquivo_auditoria,
    )
    modelo_assistente = ModeloChatQwenLocal(
        servico_fine_tuning=servico_fine_tuning,
    )
    chain_assistente = AssistenteChain(modelo_assistente)
    auditoria_assistente = ServicoAuditoriaAssistente(
        caminho_arquivo=Path("app/data/relatorios/auditoria_assistente.jsonl")
    )
    fluxo_assistente = FluxoAssistenteMedico(
        repositorio=repositorio_prontuarios,
        chain_assistente=chain_assistente,
        auditoria=auditoria_assistente,
    )
    dataframe_original = None
    dataframe_auditoria = None
    dataframe_fine_tuning = None

    opcoes_menu = {
        "0": "Preparar dados — executar etapas 2 a 6",
        "1": "Treinar e avaliar — executar etapas 7 a 10",
        "2": "Ler arquivo Excel e gerar dataframe",
        "3": "Identificar registros repetidos e colunas ausentes",
        "4": "Tratar inconsistências encontradas",
        "5": "Identificar e tratar PII",
        "6": "Preparar dataframe para fine-tuning",
        "7": "Executar inferência-base",
        "8": "Executar fine-tuning com LoRA",
        "9": "Executar inferência após fine-tuning",
        "10": "Comparar inferências",
        "11": "Consultar assistente médico com revisão humana",
        "12": "Sair",
    }

    while True:
        exibir_menu(
            opcoes_menu=opcoes_menu,
            percentual_registros=percentual_registros,
            dataframe_original_carregado=dataframe_original is not None,
            dataframe_auditoria_carregado=dataframe_auditoria is not None,
            dataframe_fine_tuning_carregado=dataframe_fine_tuning is not None,
        )

        # Mantém o menu ativo até que o usuário informe uma opção válida.
        opcao_escolhida = CONSOLE.input(
            f"[bold bright_cyan]{obter_icone('❯', '>')} Informe a opção desejada: "
            "[/bold bright_cyan]"
        ).strip()

        if opcao_escolhida == "12":
            CONSOLE.print(
                f"\n[bold green]{obter_icone('👋', '<')} Programa encerrado. "
                "Até a próxima![/bold green]"
            )
            break

        if opcao_escolhida not in opcoes_menu:
            CONSOLE.print(
                f"[bold red]{obter_icone('✖', 'X')} Opção inválida.[/bold red] "
                "Escolha um dos números exibidos no menu."
            )
            continue

        CONSOLE.print(
            f"\n[bold green]{obter_icone('▶', '>')} Opção selecionada:[/bold green] "
            f"{ICONES_MENU[opcao_escolhida]} {opcoes_menu[opcao_escolhida]}\n"
        )

        if opcao_escolhida == "0":

            # Etapa 2 - Leitura do arquivo Excel e geração do dataframe
            dataframe_original = executar_etapa_2(
                servico_arquivos,
                Path("app/data/original/dados_medicos_base.xlsx"),
                percentual_registros,
            )

            # Etapa 3 - Identificação de registros repetidos e colunas ausentes
            executar_etapa_3(servico_qualidade, dataframe_original)
          
            # Etapa 4 - Tratamento de registros repetidos e colunas ausentes
            dataframe_auditoria = executar_etapa_4(
                servico_qualidade,
                dataframe_original,
                caminho_arquivo_auditoria,
                Path("app/data/relatorios/registros_repetidos_depois.txt"),
                Path("app/data/relatorios/registros_com_colunas_ausentes_depois.txt"),
            )

            # Etapa 5 - Identificação e tratamento de PII
            dataframe_auditoria = executar_etapa_5(
                servico_pii,
                dataframe_auditoria,
                caminho_arquivo_auditoria,
            )

            # Etapa 6 - Preparacao do dataframe para fine-tuning
            try:
                dataframe_fine_tuning = executar_etapa_6(servico_fine_tuning)
            except ValueError as erro:
                dataframe_fine_tuning = None
                print(f"Não foi possível executar a etapa 6: {erro}")

            continue

        if opcao_escolhida == "1":
            # Etapa 7 - Inferencia com o modelo antes do fine-tuning
            executar_etapa_7(servico_fine_tuning)

            # Etapa 8 - Fine-tuning com LoRA
            executar_etapa_8(servico_fine_tuning)

            # Etapa 9 - Inferencia com o modelo ajustado
            executar_etapa_9(servico_fine_tuning)

            # Etapa 10 - Comparacao das inferencias
            executar_etapa_10(servico_fine_tuning)

            continue

        if opcao_escolhida == "2":
            dataframe_original = executar_etapa_2(
                servico_arquivos,
                Path("app/data/original/dados_medicos_base.xlsx"),
                percentual_registros,
            )
            continue

        if opcao_escolhida == "3":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 2 antes de executar a qualidade.")
                continue

            executar_etapa_3(servico_qualidade, dataframe_original)
            continue

        if opcao_escolhida == "4":
            if dataframe_original is None:
                print("Carregue o dataframe na opção 2 antes de tratar inconsistências.")
                continue

            dataframe_auditoria = executar_etapa_4(
                servico_qualidade,
                dataframe_original,
                caminho_arquivo_auditoria,
                Path("app/data/relatorios/registros_repetidos_depois.txt"),
                Path("app/data/relatorios/registros_com_colunas_ausentes_depois.txt"),
            )
            continue

        if opcao_escolhida == "5":
            if dataframe_auditoria is None:
                print("Execute a opção 4 antes de identificar e tratar PII.")
                continue

            dataframe_auditoria = executar_etapa_5(
                servico_pii,
                dataframe_auditoria,
                caminho_arquivo_auditoria,
            )
            continue

        if opcao_escolhida == "6":
            if dataframe_auditoria is None:
                print(
                    "Execute as opções 2 a 5 nesta execução antes de preparar "
                    "o dataframe de fine-tuning."
                )
                continue

            try:
                dataframe_fine_tuning = executar_etapa_6(servico_fine_tuning)
            except ValueError as erro:
                dataframe_fine_tuning = None
                print(f"Não foi possível executar a etapa 6: {erro}")
            continue

        if opcao_escolhida == "7":
            executar_etapa_7(servico_fine_tuning)
            continue

        if opcao_escolhida == "8":
            executar_etapa_8(servico_fine_tuning)
            continue

        if opcao_escolhida == "9":
            executar_etapa_9(servico_fine_tuning)
            continue

        if opcao_escolhida == "10":
            executar_etapa_10(servico_fine_tuning)
            continue

        if opcao_escolhida == "11":
            try:
                executar_etapa_11(fluxo_assistente)
            except (
                ValueError,
                RegistroNaoEncontradoError,
                RegistroDuplicadoError,
                FileNotFoundError,
                RuntimeError,
            ) as erro:
                CONSOLE.print(f"Não foi possível executar o assistente: {erro}")
            continue


if __name__ == "__main__":
    main()
