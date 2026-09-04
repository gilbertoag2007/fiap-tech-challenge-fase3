from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

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
from app.services.arquivo_service import ArquivoService
from app.services.qualidade_service import QualidadeService

if TYPE_CHECKING:
    from app.services.fine_tuning_service import FineTuningService
    from app.services.pii_service import PiiService


# Colunas do arquivo analisadas na identificação de PII.
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

OPCOES_MENU: dict[str, str] = {
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

ETAPAS_POR_ATALHO: dict[str, tuple[str, ...]] = {
    "0": ("2", "3", "4", "5", "6"),
    "1": ("7", "8", "9", "10"),
}


def exibir_menu(
    percentual_registros: float,
    dataframe_original_carregado: bool,
    dataframe_auditoria_carregado: bool,
    fine_tuning_preparado: bool,
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
            if fine_tuning_preparado
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
                f"{ICONES_MENU[numero_opcao]}  {OPCOES_MENU[numero_opcao]}",
                estados[numero_opcao],
            )

    CONSOLE.print(tabela)
    CONSOLE.print(
        f"[bright_black]Amostra configurada:[/bright_black] "
        f"[bold cyan]{percentual_registros:.2f}%[/bold cyan] dos registros\n"
    )


def iniciar_etapa(numero_etapa: int, descricao: str) -> float:
    """Exibe o cabeçalho da etapa e inicia a medição de tempo."""
    inicio_execucao = perf_counter()
    titulo = Text.assemble(
        (f" {ICONES_MENU[str(numero_etapa)]} ETAPA {numero_etapa}", "bold cyan"),
        (f" · {descricao} ", "bold bright_white"),
    )
    CONSOLE.print()
    CONSOLE.rule(titulo, style="bright_cyan")
    exibir_detalhe("Início", f"{datetime.now():%d/%m/%Y %H:%M:%S}", "bright_black")
    return inicio_execucao


def exibir_detalhe(rotulo: str, valor: object, estilo: str = "bright_white") -> None:
    """Exibe uma informação da etapa em uma linha curta e alinhada."""
    linha = Text("  ")
    linha.append(f"{obter_icone('•', '-')} ", style="bright_cyan")
    linha.append(f"{rotulo}: ", style="bright_black")
    linha.append(str(valor), style=estilo)
    CONSOLE.print(linha)


def exibir_conclusao_etapa(numero_etapa: int, inicio_execucao: float) -> None:
    """Exibe horário de término e duração da etapa."""
    duracao_segundos = perf_counter() - inicio_execucao
    if duracao_segundos < 1:
        duracao_formatada = f"{duracao_segundos * 1000:.0f} ms"
    elif duracao_segundos < 60:
        duracao_formatada = f"{duracao_segundos:.1f} s"
    elif duracao_segundos < 3600:
        duracao_formatada = f"{duracao_segundos / 60:.2f} min"
    else:
        duracao_formatada = f"{duracao_segundos / 3600:.2f} h"

    conclusao = Text("  ")
    conclusao.append(
        f"{obter_icone('✓', 'OK')} Etapa {numero_etapa} concluída",
        style="bold green",
    )
    conclusao.append(f"  •  {datetime.now():%H:%M:%S}", style="bright_black")
    conclusao.append(f"  •  {duracao_formatada}", style="cyan")
    CONSOLE.print(conclusao)
    CONSOLE.print()


def executar_etapa_2(
    servico_arquivos: ArquivoService,
    caminho_arquivo: Path,
    percentual_registros: float,
) -> pd.DataFrame:
    inicio_execucao = iniciar_etapa(2, "LEITURA DO ARQUIVO EXCEL")
    exibir_detalhe("Arquivo", caminho_arquivo, "cyan")
    exibir_detalhe(
        "Amostra",
        f"{percentual_registros:.2f}% — mínimo de 3 registros para fine-tuning",
    )

    dataframe_original = servico_arquivos.gerar_dataframe(
        caminho_arquivo,
        percentual_registros=percentual_registros,
        quantidade_minima=3,
    )
    exibir_detalhe(
        "Resultado",
        f"{dataframe_original.shape[0]} linhas × "
        f"{dataframe_original.shape[1]} colunas",
        "green",
    )
    exibir_conclusao_etapa(2, inicio_execucao)
    return dataframe_original


def executar_etapa_3(
    servico_qualidade: QualidadeService,
    dataframe_original: pd.DataFrame,
    caminho_relatorio_qualidade: Path,
) -> Path:
    inicio_execucao = iniciar_etapa(3, "VERIFICAÇÃO DE QUALIDADE")

    caminho_relatorio = servico_qualidade.gerar_relatorio_qualidade(
        dataframe_original,
        caminho_relatorio_qualidade,
    )

    exibir_detalhe("Relatório de qualidade", caminho_relatorio, "cyan")
    exibir_conclusao_etapa(3, inicio_execucao)
    return caminho_relatorio


def executar_etapa_4(
    servico_qualidade: QualidadeService,
    dataframe_original: pd.DataFrame,
    caminho_arquivo_tratado: Path,
    caminho_relatorio_qualidade: Path,
) -> pd.DataFrame:
    inicio_execucao = iniciar_etapa(4, "TRATAMENTO DE INCONSISTÊNCIAS")

    resultado_tratamento_repetidos = servico_qualidade.remover_registros_repetidos(
        dataframe_original,
        caminho_arquivo_tratado,
    )
    dataframe_auditoria = resultado_tratamento_repetidos.dataframe_tratado

    resultado_tratamento_ausentes = (
        servico_qualidade.remover_registros_com_colunas_ausentes(
            dataframe_auditoria,
            caminho_arquivo_tratado,
        )
    )
    dataframe_auditoria = resultado_tratamento_ausentes.dataframe_tratado
    caminho_relatorio = servico_qualidade.gerar_relatorio_qualidade(
        dataframe_antes=dataframe_original,
        dataframe_depois=dataframe_auditoria,
        registros_repetidos_removidos=(
            resultado_tratamento_repetidos.linhas_tratadas
        ),
        registros_ausentes_removidos=(
            resultado_tratamento_ausentes.linhas_tratadas
        ),
        caminho_relatorio=caminho_relatorio_qualidade,
    )

    exibir_detalhe(
        "Repetidos removidos",
        resultado_tratamento_repetidos.linhas_tratadas,
        "green",
    )
    exibir_detalhe(
        "Incompletos removidos",
        resultado_tratamento_ausentes.linhas_tratadas,
        "green",
    )
    exibir_detalhe(
        "Relatório de qualidade",
        caminho_relatorio,
        "cyan",
    )
    exibir_detalhe(
        "Arquivo tratado",
        resultado_tratamento_ausentes.caminho_arquivo_tratado,
        "cyan",
    )

    exibir_conclusao_etapa(4, inicio_execucao)

    return dataframe_auditoria


def executar_etapa_5(
    servico_pii: PiiService,
    dataframe_tratado: pd.DataFrame,
    caminho_arquivo_tratado: Path,
) -> pd.DataFrame:
    inicio_execucao = iniciar_etapa(5, "IDENTIFICAÇÃO E TRATAMENTO DE PII")

    resultado_pii = servico_pii.identificar_e_tratar_pii(
        dataframe=dataframe_tratado,
        colunas_analisar=list(COLUNAS_ANALISADAS_PII),
        caminho_arquivo_tratado=caminho_arquivo_tratado,
    )

    exibir_detalhe("Arquivo atualizado", resultado_pii.caminho_arquivo_tratado, "cyan")
    exibir_conclusao_etapa(5, inicio_execucao)

    return resultado_pii.dataframe_resultado


def executar_etapa_6(
    servico_fine_tuning: FineTuningService,
) -> pd.DataFrame:
    inicio_execucao = iniciar_etapa(6, "PREPARAÇÃO PARA FINE-TUNING")

    dataframe_fine_tuning = servico_fine_tuning.gerar_dataframe_fine_tuning()

    exibir_detalhe(
        "Resultado",
        f"{dataframe_fine_tuning.shape[0]} linhas × "
        f"{dataframe_fine_tuning.shape[1]} colunas",
        "green",
    )
    exibir_detalhe(
        "Arquivo gerado",
        servico_fine_tuning.CAMINHO_ARQUIVO_FINE_TUNING,
        "cyan",
    )
    exibir_conclusao_etapa(6, inicio_execucao)

    return dataframe_fine_tuning


def executar_etapa_7(servico_fine_tuning: FineTuningService) -> Path:
    """Executa a inferência-base nos registros reservados para teste."""
    inicio_execucao = iniciar_etapa(7, "INFERÊNCIA-BASE")
    caminho_relatorio = servico_fine_tuning.realizar_inferencia_base()

    exibir_detalhe("Avaliação iniciada", caminho_relatorio, "cyan")
    exibir_conclusao_etapa(7, inicio_execucao)
    return caminho_relatorio


def executar_etapa_8(servico_fine_tuning: FineTuningService) -> Path:
    """Executa o fine-tuning supervisionado com adaptador LoRA."""
    inicio_execucao = iniciar_etapa(8, "FINE-TUNING COM LoRA")
    exibir_detalhe(
        "Aviso",
        "Treinamento em CPU; a execução pode ser demorada",
        "yellow",
    )
    exibir_detalhe(
        "Configuração",
        f"registros={servico_fine_tuning.limite_registros_fine_tuning}, "
        f"épocas={servico_fine_tuning.quantidade_epocas_fine_tuning}, "
        f"max_tokens={servico_fine_tuning.max_tokens_entrada}, "
        f"lora_r={servico_fine_tuning.rank_lora}",
    )

    caminho_modelo = servico_fine_tuning.realizar_fine_tuning()

    exibir_detalhe("Adaptador LoRA", caminho_modelo, "cyan")
    exibir_detalhe(
        "Métricas",
        servico_fine_tuning.CAMINHO_RELATORIO_METRICAS,
        "cyan",
    )
    exibir_detalhe(
        "Relatório técnico",
        servico_fine_tuning.CAMINHO_RELATORIO_TECNICO,
        "cyan",
    )
    exibir_conclusao_etapa(8, inicio_execucao)
    return caminho_modelo


def executar_etapa_9(servico_fine_tuning: FineTuningService) -> Path:
    """Executa a inferência com o adaptador LoRA treinado."""
    inicio_execucao = iniciar_etapa(9, "INFERÊNCIA APÓS FINE-TUNING")

    caminho_relatorio = servico_fine_tuning.realizar_inferencia_fine_tuning()

    exibir_detalhe("Avaliação atualizada", caminho_relatorio, "cyan")
    exibir_conclusao_etapa(9, inicio_execucao)
    return caminho_relatorio


def executar_etapa_10(servico_fine_tuning: FineTuningService) -> Path:
    """Gera o relatório comparativo das inferências do split de teste."""
    inicio_execucao = iniciar_etapa(10, "COMPARAÇÃO DAS INFERÊNCIAS")

    caminho_relatorio = servico_fine_tuning.comparar_inferencias()

    exibir_detalhe("Avaliação validada", caminho_relatorio, "cyan")
    exibir_conclusao_etapa(10, inicio_execucao)
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
            Text.assemble(
                "Rascunho:\n",
                Text(revisao.rascunho),
                "\n\nFontes: ",
                fontes,
                "\nAlertas: ",
                alertas,
                "\nAviso: ",
                revisao.aviso,
            ),
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
                Text.assemble(
                    "Resposta:\n",
                    Text(resposta.resposta or ""),
                    "\n\nFontes: ",
                    fontes,
                    "\nAviso: ",
                    resposta.aviso,
                ),
                title="Resposta aprovada",
                border_style="green",
            )
        )
        return

    CONSOLE.print("Rascunho rejeitado. Nenhum conteúdo foi liberado.")


def solicitar_percentual_registros() -> float:
    """Solicita uma única vez o percentual usado em toda a execução."""
    while True:
        valor_informado = CONSOLE.input(
            "[bold cyan]Informe o percentual de registros que será utilizado "
            "(0 a 100; mínimo de 3 registros para fine-tuning): [/bold cyan]"
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
    if percentual_registros is None:
        percentual_registros = solicitar_percentual_registros()
    else:
        percentual_registros = ArquivoService.validar_percentual_registros(
            percentual_registros
        )

    caminho_arquivo_original = Path("app/data/original/dados_medicos_base.xlsx")
    caminho_arquivo_auditoria = Path("app/data/processado/dados_medicos_auditoria.xlsx")
    caminho_relatorio_qualidade = Path(
        "app/data/relatorios/relatorio_qualidade.xlsx"
    )

    servico_arquivos = ArquivoService()
    servico_qualidade = QualidadeService(servico_arquivo=servico_arquivos)
    servico_pii: PiiService | None = None
    servico_fine_tuning: FineTuningService | None = None
    fluxo_assistente: FluxoAssistenteMedico | None = None
    dataframe_original: pd.DataFrame | None = None
    dataframe_auditoria: pd.DataFrame | None = None
    fine_tuning_preparado = False

    while True:
        exibir_menu(
            percentual_registros=percentual_registros,
            dataframe_original_carregado=dataframe_original is not None,
            dataframe_auditoria_carregado=dataframe_auditoria is not None,
            fine_tuning_preparado=fine_tuning_preparado,
        )

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

        if opcao_escolhida not in OPCOES_MENU:
            CONSOLE.print(
                f"[bold red]{obter_icone('✖', 'X')} Opção inválida.[/bold red] "
                "Escolha um dos números exibidos no menu."
            )
            continue

        CONSOLE.print(
            f"\n[bold green]{obter_icone('▶', '>')} Opção selecionada:[/bold green] "
            f"{ICONES_MENU[opcao_escolhida]} {OPCOES_MENU[opcao_escolhida]}\n"
        )

        for etapa in ETAPAS_POR_ATALHO.get(opcao_escolhida, (opcao_escolhida,)):
            if etapa == "2":
                dataframe_original = executar_etapa_2(
                    servico_arquivos,
                    caminho_arquivo_original,
                    percentual_registros,
                )
                dataframe_auditoria = None
                fine_tuning_preparado = False

            elif etapa == "3":
                if dataframe_original is None:
                    CONSOLE.print(
                        "[yellow]Carregue o dataframe na opção 2 antes de "
                        "executar a qualidade.[/yellow]"
                    )
                    break
                executar_etapa_3(
                    servico_qualidade,
                    dataframe_original,
                    caminho_relatorio_qualidade,
                )

            elif etapa == "4":
                if dataframe_original is None:
                    CONSOLE.print(
                        "[yellow]Carregue o dataframe na opção 2 antes de "
                        "tratar inconsistências.[/yellow]"
                    )
                    break
                dataframe_auditoria = executar_etapa_4(
                    servico_qualidade,
                    dataframe_original,
                    caminho_arquivo_auditoria,
                    caminho_relatorio_qualidade,
                )
                fine_tuning_preparado = False

            elif etapa == "5":
                if dataframe_auditoria is None:
                    CONSOLE.print(
                        "[yellow]Execute a opção 4 antes de identificar e "
                        "tratar PII.[/yellow]"
                    )
                    break
                if servico_pii is None:
                    from app.services.pii_service import PiiService

                    servico_pii = PiiService(servico_arquivo=servico_arquivos)
                dataframe_auditoria = executar_etapa_5(
                    servico_pii,
                    dataframe_auditoria,
                    caminho_arquivo_auditoria,
                )
                fine_tuning_preparado = False

            elif etapa in {"6", "7", "8", "9", "10"}:
                if etapa == "6" and dataframe_auditoria is None:
                    CONSOLE.print(
                        "[yellow]Execute as opções 2 a 5 nesta execução antes "
                        "de preparar o dataframe de fine-tuning.[/yellow]"
                    )
                    break
                if servico_fine_tuning is None:
                    from app.services.fine_tuning_service import FineTuningService

                    servico_fine_tuning = FineTuningService(
                        servico_arquivo=servico_arquivos
                    )

                if etapa == "6":
                    try:
                        executar_etapa_6(servico_fine_tuning)
                        fine_tuning_preparado = True
                    except ValueError as erro:
                        fine_tuning_preparado = False
                        CONSOLE.print(
                            "[bold red]Não foi possível executar a etapa 6:"
                            f"[/bold red] {erro}"
                        )
                elif etapa == "7":
                    executar_etapa_7(servico_fine_tuning)
                elif etapa == "8":
                    executar_etapa_8(servico_fine_tuning)
                elif etapa == "9":
                    executar_etapa_9(servico_fine_tuning)
                else:
                    executar_etapa_10(servico_fine_tuning)

        if opcao_escolhida == "11":
            try:
                if servico_fine_tuning is None:
                    from app.services.fine_tuning_service import FineTuningService

                    servico_fine_tuning = FineTuningService(
                        servico_arquivo=servico_arquivos
                    )
                if fluxo_assistente is None:
                    repositorio_prontuarios = RepositorioProntuariosExcel(
                        servico_arquivo=servico_arquivos,
                        caminho_arquivo=caminho_arquivo_auditoria,
                    )
                    modelo_assistente = ModeloChatQwenLocal(
                        servico_fine_tuning=servico_fine_tuning,
                    )
                    chain_assistente = AssistenteChain(modelo_assistente)
                    auditoria_assistente = ServicoAuditoriaAssistente(
                        caminho_arquivo=Path(
                            "app/data/relatorios/auditoria_assistente.jsonl"
                        )
                    )
                    fluxo_assistente = FluxoAssistenteMedico(
                        repositorio=repositorio_prontuarios,
                        chain_assistente=chain_assistente,
                        auditoria=auditoria_assistente,
                    )
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
