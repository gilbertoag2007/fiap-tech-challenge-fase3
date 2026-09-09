from __future__ import annotations

import gc
import threading
from datetime import datetime
from math import ceil, exp, isfinite
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer

from app.config import (
    DIRETORIO_MODELOS,
    HF_REPO_LLAMA,
    HF_REPO_QWEN80,
    MODELO_PADRAO,
)
from app.services.arquivo_service import ArquivoService


class FineTuningService:
    """Prepara, treina e avalia o modelo usado no fine-tuning."""

    NOME_MODELO_BASE = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TOKENS_ENTRADA = 512
    LIMITE_REGISTROS_FINE_TUNING = None
    QUANTIDADE_EPOCAS_FINE_TUNING = 3
    RANK_LORA = 16
    BATCH_TREINO = 2
    ACUMULACAO_GRADIENTE = 4
    SEMENTE_ALEATORIA = 42
    LIMITE_REGISTROS_INFERENCIA_TESTE = 50
    MODULOS_LORA = ("q_proj", "k_proj", "v_proj", "o_proj")
    MODELOS_DISPONIVEIS = {
        "llama": {
            "rotulo": "Llama 3.1 8B Instruct QLoRA (treino em GPU)",
            "nome_base": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "caminho_adapter": DIRETORIO_MODELOS / "llama31_8b_instruct_lora_gpu",
            "repo_adapter": HF_REPO_LLAMA,
            "loader": "unsloth",
            "comando_download": (
                "hf download unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
            ),
        },
        "qwen10": {
            "rotulo": "Qwen3-0.6B 10% (treino em CPU)",
            "nome_base": "Qwen/Qwen3-0.6B",
            "caminho_adapter": DIRETORIO_MODELOS / "qwen3_06b_lora_cpu",
            "repo_adapter": None,
            "loader": "transformers",
            "comando_download": "hf download Qwen/Qwen3-0.6B",
        },
        "qwen80": {
            "rotulo": "Qwen3-0.6B 80% q/k/v/o (treino em GPU)",
            "nome_base": "Qwen/Qwen3-0.6B",
            "caminho_adapter": (
                DIRETORIO_MODELOS
                / "qwen_06b_lora_gpu"
                / "qwen3_06b_lora_80pct_more_projections"
            ),
            "repo_adapter": HF_REPO_QWEN80,
            "loader": "transformers",
            "comando_download": "hf download Qwen/Qwen3-0.6B",
        },
    }
    CAMINHO_ARQUIVO_AUDITORIA = Path(
        "app/data/processado/dados_medicos_auditoria.xlsx"
    )
    CAMINHO_ARQUIVO_FINE_TUNING = Path(
        "app/data/processado/dados_medicos_fine_tuning.xlsx"
    )
    CAMINHO_ARQUIVO_INFERENCIA_BASE = Path(
        "app/data/relatorios/inferencia_base.xlsx"
    )
    CAMINHO_ARQUIVO_INFERENCIA_FINE_TUNING = Path(
        "app/data/relatorios/inferencia_fine_tuning.xlsx"
    )
    CAMINHO_ARQUIVO_COMPARACAO = Path(
        "app/data/relatorios/comparacao_inferencias.xlsx"
    )
    CAMINHO_RELATORIO_METRICAS = Path(
        "app/data/relatorios/metricas_fine_tuning.txt"
    )
    CAMINHO_RELATORIO_TECNICO = Path(
        "app/data/relatorios/relatorio_tecnico_fine_tuning.xlsx"
    )
    CAMINHO_MODELO_FINE_TUNING = DIRETORIO_MODELOS / "llama31_8b_instruct_lora_gpu"
    CAMINHO_CHECKPOINTS = CAMINHO_MODELO_FINE_TUNING / "checkpoints"
    COLUNA_TOTAL_TOKENS = "total_okens_fine_tunning"
    COLUNAS_NECESSARIAS = (
        "id",
        "papel_solicitante",
        "contexto_solicitacao",
        "pergunta_original_anonimizado",
        "prontuario_contexto_anonimizado",
        "resposta_estruturada",
        "especialidade_medica",
        "tipo_pergunta",
    )
    COLUNAS_DATASET_FINE_TUNING = (
        "id_exemplo",
        "system",
        "user",
        "assistant",
        COLUNA_TOTAL_TOKENS,
        "split",
    )
    SPLITS_ESPERADOS = ("treino", "validacao", "teste")
    MENSAGEM_SYSTEM = (
        "Voce e um assistente especializado em apoio clinico. "
        "Responda somente a solicitacao, sem repetir o papel, o prontuario "
        "ou a pergunta. Use exatamente as secoes: Resposta, Consideracoes "
        "clinicas, Conduta/Orientacao e Limitacoes."
    )

    def __init__(
        self,
        servico_arquivo: ArquivoService | None = None,
        limite_registros_fine_tuning: int | None = (
            LIMITE_REGISTROS_FINE_TUNING
        ),
        quantidade_epocas_fine_tuning: int = QUANTIDADE_EPOCAS_FINE_TUNING,
        rank_lora: int = RANK_LORA,
        max_tokens_entrada: int = MAX_TOKENS_ENTRADA,
    ) -> None:
        if limite_registros_fine_tuning is not None and (
            isinstance(limite_registros_fine_tuning, bool)
            or not isinstance(limite_registros_fine_tuning, int)
            or limite_registros_fine_tuning < 3
        ):
            raise ValueError(
                "O limite de registros do fine-tuning deve ser um inteiro "
                "maior ou igual a 3, ou None para usar todos os registros."
            )

        parametros_positivos = {
            "quantidade de epocas": quantidade_epocas_fine_tuning,
            "rank LoRA": rank_lora,
            "limite de tokens": max_tokens_entrada,
        }
        for nome_parametro, valor in parametros_positivos.items():
            if (
                isinstance(valor, bool)
                or not isinstance(valor, int)
                or valor <= 0
            ):
                raise ValueError(
                    f"A {nome_parametro} deve ser um numero inteiro positivo."
                )

        self.servico_arquivo = servico_arquivo or ArquivoService()
        self.limite_registros_fine_tuning = limite_registros_fine_tuning
        self.quantidade_epocas_fine_tuning = quantidade_epocas_fine_tuning
        self.rank_lora = rank_lora
        self.max_tokens_entrada = max_tokens_entrada
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.modelo: PreTrainedModel | PeftModel | None = None
        self.modelo_ajustado_carregado = False
        self.quantidade_registros_descartados_tokens = 0
        self._fast_language_model = None
        self._trava_modelo = threading.RLock()
        self.chave_modelo = MODELO_PADRAO
        self.loader_modelo = "unsloth"
        self.configurar_modelo(MODELO_PADRAO)

    def configurar_modelo(self, chave_modelo: str) -> None:
        """Seleciona Llama ou Qwen para inferencia/assistente e limpa cache."""
        chave = chave_modelo.strip().lower()
        if chave not in self.MODELOS_DISPONIVEIS:
            disponiveis = ", ".join(sorted(self.MODELOS_DISPONIVEIS))
            raise ValueError(
                f"Modelo '{chave_modelo}' invalido. Use um de: {disponiveis}."
            )
        with self._trava_modelo:
            if chave == self.chave_modelo and self.modelo_ajustado_carregado:
                return
            self.descarregar_modelo()
            configuracao = self.MODELOS_DISPONIVEIS[chave]
            self.chave_modelo = chave
            self.loader_modelo = configuracao["loader"]
            self.NOME_MODELO_BASE = configuracao["nome_base"]
            self.CAMINHO_MODELO_FINE_TUNING = Path(
                configuracao["caminho_adapter"]
            )
            self.CAMINHO_CHECKPOINTS = (
                self.CAMINHO_MODELO_FINE_TUNING / "checkpoints"
            )

    def descarregar_modelo(self) -> None:
        """Libera o modelo atual e devolve VRAM/RAM antes de uma troca."""
        self.tokenizer = None
        self.modelo = None
        self.modelo_ajustado_carregado = False
        self._fast_language_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def gerar_resposta_modelo_ajustado(
        self,
        mensagem_system: str,
        mensagem_usuario: str,
        max_novos_tokens: int = 384,
    ) -> str:
        """Gera uma resposta usando o adaptador LoRA configurado (Llama/Qwen)."""
        with self._trava_modelo:
            self._carregar_modelo_ajustado()
            return self._gerar_resposta(
                mensagem_system=mensagem_system,
                mensagem_usuario=mensagem_usuario,
                max_novos_tokens=max_novos_tokens,
            )

    @staticmethod
    def _importar_fast_language_model():
        """Importa Unsloth apenas quando o Llama for realmente solicitado."""
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel

        return FastLanguageModel

    def _obter_caminho_modelo_base(self) -> Path:
        """Resolve o caminho local do modelo-base no cache do Hugging Face."""
        try:
            return Path(
                snapshot_download(repo_id=self.NOME_MODELO_BASE)
            )
        except OSError as erro:
            raise FileNotFoundError(
                "Modelo base nao encontrado no cache local. Execute: "
                f"hf download {self.NOME_MODELO_BASE}"
            ) from erro

    def _obter_implementacao_atencao(self) -> str:
        """Prefere Flash Attention 2; caso contrario usa SDPA."""
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            return "sdpa"
        return "flash_attention_2"

    def _carregar_modelo_unsloth(self, para_treino: bool = False) -> None:
        """Carrega tokenizer e modelo 4-bit via Unsloth FastLanguageModel."""
        if self.DISPOSITIVO != "cuda":
            raise RuntimeError(
                "O modelo Llama 8B Instruct em 4-bit exige GPU CUDA. "
                "Verifique se o PyTorch reconhece a placa de video."
            )

        FastLanguageModel = self._importar_fast_language_model()
        self._fast_language_model = FastLanguageModel
        caminho_modelo = self._obter_caminho_modelo_base()
        implementacao_atencao = self._obter_implementacao_atencao()

        kwargs_carregamento: dict[str, Any] = {
            "model_name": str(caminho_modelo),
            "max_seq_length": self.max_tokens_entrada,
            "dtype": None,
            "load_in_4bit": True,
        }
        # Unsloth escolhe o backend mais rapido; quando Flash Attn 2 existe,
        # reforcamos a preferencia pelo backend explicito.
        if implementacao_atencao == "flash_attention_2":
            kwargs_carregamento["attn_implementation"] = "flash_attention_2"

        try:
            self.modelo, self.tokenizer = FastLanguageModel.from_pretrained(
                **kwargs_carregamento
            )
        except TypeError:
            kwargs_carregamento.pop("attn_implementation", None)
            self.modelo, self.tokenizer = FastLanguageModel.from_pretrained(
                **kwargs_carregamento
            )

        self._configurar_tokenizer()
        if para_treino:
            self.modelo.train()
        else:
            FastLanguageModel.for_inference(self.modelo)
            self.modelo.eval()
        print(
            "Modelo carregado com Unsloth Fast "
            f"(atencao={implementacao_atencao}, max_seq={self.max_tokens_entrada})."
        )

    def _configurar_tokenizer(self) -> None:
        """Padroniza pad/padding para treino e inferencia com Llama."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer nao carregado.")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        self.tokenizer.clean_up_tokenization_spaces = False

    def carregar_modelo_base(self) -> None:
        """Carrega tokenizer e modelo-base com Unsloth Fast (QLoRA 4-bit)."""
        if (
            self.tokenizer is not None
            and self.modelo is not None
            and not self.modelo_ajustado_carregado
        ):
            return

        self.tokenizer = None
        self.modelo = None
        self.modelo_ajustado_carregado = False
        self._carregar_modelo_unsloth(para_treino=False)

    def gerar_dataframe_fine_tuning(self) -> pd.DataFrame:
        """Le a auditoria, prepara os exemplos e salva o novo arquivo Excel."""
        dataframe_auditoria = self.servico_arquivo.gerar_dataframe(
            self.CAMINHO_ARQUIVO_AUDITORIA
        )
        self._validar_colunas_necessarias(dataframe_auditoria)

        # Cada linha representa uma conversa completa para o treinamento.
        dataframe_fine_tuning = pd.DataFrame(
            {
                "id_exemplo": dataframe_auditoria["id"],
                "system": self.MENSAGEM_SYSTEM,
                "user": self._montar_mensagem_usuario(dataframe_auditoria),
                "assistant": dataframe_auditoria["resposta_estruturada"],
                "especialidade_medica": dataframe_auditoria[
                    "especialidade_medica"
                ],
                "tipo_pergunta": dataframe_auditoria["tipo_pergunta"],
            }
        )
        dataframe_fine_tuning[self.COLUNA_TOTAL_TOKENS] = (
            self._calcular_total_tokens_conversa(dataframe_fine_tuning)
        )
        dataframe_fine_tuning["split"] = self._gerar_splits(
            dataframe_fine_tuning
        )

        self.servico_arquivo.criar_excel(
            dataframe_fine_tuning,
            self.CAMINHO_ARQUIVO_FINE_TUNING,
        )
        return dataframe_fine_tuning

    def _carregar_tokenizer_modelo_base(self) -> Path:
        """Carrega do cache o mesmo tokenizer usado pelo fine-tuning."""
        caminho_modelo = self._obter_caminho_modelo_base()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                caminho_modelo,
                local_files_only=True,
            )
            self._configurar_tokenizer()
        except OSError as erro:
            raise FileNotFoundError(
                "Tokenizer do modelo base nao encontrado no cache local. "
                f"Execute: hf download {self.NOME_MODELO_BASE}"
            ) from erro

        return caminho_modelo

    def _calcular_total_tokens_conversa(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """Conta os tokens da conversa com o mesmo template usado no treino."""
        if self.tokenizer is None:
            self._carregar_tokenizer_modelo_base()

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer nao carregado para contar os tokens.")

        totais = []
        for registro in dataframe.itertuples(index=False):
            mensagens = [
                {"role": "system", "content": str(registro.system).strip()},
                {"role": "user", "content": str(registro.user).strip()},
                {
                    "role": "assistant",
                    "content": str(registro.assistant).strip(),
                },
            ]
            tokens = self.tokenizer.apply_chat_template(
                mensagens,
                tokenize=True,
                add_generation_prompt=False,
            )
            ids_tokens = (
                tokens["input_ids"]
                if hasattr(tokens, "keys") and "input_ids" in tokens
                else tokens
            )
            totais.append(len(ids_tokens))

        return pd.Series(totais, index=dataframe.index, dtype="int64")

    def gerar_datasets_treinamento(self) -> DatasetDict:
        """Converte o arquivo preparado em datasets conversacionais por split."""
        dataframe = self._carregar_dataframe_fine_tuning_validado()
        datasets_por_split: dict[str, Dataset] = {}

        limites_por_split: dict[str, int] | None = None
        if (
            self.limite_registros_fine_tuning is not None
            and self.limite_registros_fine_tuning < len(dataframe)
        ):
            limites_por_split = self._calcular_quantidades_splits(
                self.limite_registros_fine_tuning
            )

        for nome_split in self.SPLITS_ESPERADOS:
            registros_split = dataframe.loc[dataframe["split"] == nome_split]
            if limites_por_split is not None:
                quantidade_split = limites_por_split[nome_split]
                if len(registros_split) < quantidade_split:
                    raise ValueError(
                        f"O split {nome_split} possui {len(registros_split)} "
                        f"registros, mas {quantidade_split} sao necessarios."
                    )
                registros_split = registros_split.sample(
                    n=quantidade_split,
                    random_state=self.SEMENTE_ALEATORIA,
                )

            exemplos = [
                {
                    "id_exemplo": registro["id_exemplo"],
                    "prompt": [
                        {
                            "role": "system",
                            "content": registro["system"],
                        },
                        {
                            "role": "user",
                            "content": registro["user"],
                        },
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": registro["assistant"],
                        }
                    ],
                }
                for _, registro in registros_split.iterrows()
            ]
            datasets_por_split[nome_split] = Dataset.from_list(exemplos)

        return DatasetDict(datasets_por_split)

    def realizar_inferencia_base(
        self,
        max_novos_tokens: int = 384,
        limite_registros: int | None = LIMITE_REGISTROS_INFERENCIA_TESTE,
    ) -> Path:
        """Executa a inferencia-base em uma amostra do split de teste."""
        self.carregar_modelo_base()
        return self._realizar_inferencia_split_teste(
            nome_coluna_resposta="resposta_inferencia_base",
            caminho_arquivo=self.CAMINHO_ARQUIVO_INFERENCIA_BASE,
            max_novos_tokens=max_novos_tokens,
            limite_registros=limite_registros,
        )

    def realizar_fine_tuning(self, max_passos: int | None = None) -> Path:
        """Executa o SFT com Unsloth Fast + QLoRA e salva o adaptador."""
        datasets = self.gerar_datasets_treinamento()
        if self.tokenizer is None:
            self._carregar_tokenizer_modelo_base()

        estatisticas_tokens = self._calcular_estatisticas_tokens(datasets)
        exemplos_acima_limite = {
            nome_split: estatisticas["exemplos_acima_limite"]
            for nome_split, estatisticas in estatisticas_tokens.items()
            if estatisticas["exemplos_acima_limite"] > 0
        }
        if exemplos_acima_limite:
            resumo = ", ".join(
                f"{nome_split}={quantidade}"
                for nome_split, quantidade in exemplos_acima_limite.items()
            )
            print(
                f"Aviso: exemplos acima do limite de "
                f"{self.max_tokens_entrada} tokens serao truncados: {resumo}."
            )

        configuracao_lora = self._criar_configuracao_lora()
        configuracao_treinamento = self._criar_configuracao_treinamento(
            max_passos=max_passos
        )

        FastLanguageModel = self._importar_fast_language_model()
        self._fast_language_model = FastLanguageModel

        # Carrega Unsloth Fast em modo treino e aplica LoRA otimizado.
        self.tokenizer = None
        self.modelo = None
        self.modelo_ajustado_carregado = False
        self._carregar_modelo_unsloth(para_treino=True)
        self.modelo = FastLanguageModel.get_peft_model(
            self.modelo,
            r=configuracao_lora.r,
            target_modules=list(configuracao_lora.target_modules),
            lora_alpha=configuracao_lora.lora_alpha,
            lora_dropout=configuracao_lora.lora_dropout,
            bias=configuracao_lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=self.SEMENTE_ALEATORIA,
            max_seq_length=self.max_tokens_entrada,
        )
        if hasattr(self.modelo, "config"):
            self.modelo.config.use_cache = False
        self.modelo.train()

        treinador = SFTTrainer(
            model=self.modelo,
            args=configuracao_treinamento,
            train_dataset=datasets["treino"],
            eval_dataset=datasets["validacao"],
            processing_class=self.tokenizer,
        )
        parametros_totais = sum(
            parametro.numel() for parametro in treinador.model.parameters()
        )
        parametros_treinaveis = sum(
            parametro.numel()
            for parametro in treinador.model.parameters()
            if parametro.requires_grad
        )
        percentual_treinavel = (
            parametros_treinaveis / parametros_totais * 100
            if parametros_totais
            else 0.0
        )
        print(
            "Parametros treinaveis: "
            f"{parametros_treinaveis} de {parametros_totais} "
            f"({percentual_treinavel:.4f}%)."
        )

        # Mede o modelo com o adaptador ainda nao treinado no mesmo split usado
        # depois do treinamento, permitindo uma comparacao direta e justa.
        metricas_validacao_base = dict(treinador.evaluate())
        resultado_treinamento = treinador.train()
        metricas_treinamento = dict(resultado_treinamento.metrics)
        metricas_treinamento["parametros_totais"] = parametros_totais
        metricas_treinamento["parametros_treinaveis"] = parametros_treinaveis
        metricas_treinamento["percentual_parametros_treinaveis"] = (
            percentual_treinavel
        )
        metricas_treinamento["passos_treinamento"] = treinador.state.global_step
        metricas_treinamento["melhor_perda_validacao"] = (
            treinador.state.best_metric
        )
        metricas_validacao = dict(treinador.evaluate())
        historico_treinamento = [
            dict(registro) for registro in treinador.state.log_history
        ]

        self.CAMINHO_MODELO_FINE_TUNING.mkdir(parents=True, exist_ok=True)
        treinador.save_model(str(self.CAMINHO_MODELO_FINE_TUNING))
        self.tokenizer.save_pretrained(str(self.CAMINHO_MODELO_FINE_TUNING))
        self._salvar_metricas_fine_tuning(
            metricas_treinamento=metricas_treinamento,
            metricas_validacao_base=metricas_validacao_base,
            metricas_validacao=metricas_validacao,
            estatisticas_tokens=estatisticas_tokens,
        )
        self._salvar_relatorio_tecnico_fine_tuning(
            metricas_treinamento=metricas_treinamento,
            metricas_validacao_base=metricas_validacao_base,
            metricas_validacao=metricas_validacao,
            estatisticas_tokens=estatisticas_tokens,
            historico_treinamento=historico_treinamento,
        )

        self.modelo = treinador.model
        self.modelo.eval()
        self.modelo_ajustado_carregado = True
        return self.CAMINHO_MODELO_FINE_TUNING

    def realizar_inferencia_fine_tuning(
        self,
        max_novos_tokens: int = 384,
        limite_registros: int | None = LIMITE_REGISTROS_INFERENCIA_TESTE,
    ) -> Path:
        """Executa a inferencia ajustada em uma amostra do split de teste."""
        self._carregar_modelo_ajustado()
        return self._realizar_inferencia_split_teste(
            nome_coluna_resposta="resposta_inferencia_fine_tuning",
            caminho_arquivo=self.CAMINHO_ARQUIVO_INFERENCIA_FINE_TUNING,
            max_novos_tokens=max_novos_tokens,
            limite_registros=limite_registros,
        )

    def comparar_inferencias(self) -> Path:
        """Compara as respostas base e ajustada com a resposta esperada."""
        dataframe_referencia = self._carregar_dataframe_fine_tuning_validado()
        dataframe_referencia = dataframe_referencia.loc[
            dataframe_referencia["split"] == "teste",
            ["id_exemplo", "system", "user", "assistant"],
        ].rename(columns={"assistant": "resposta_esperada"})

        dataframe_base = self.servico_arquivo.gerar_dataframe(
            self.CAMINHO_ARQUIVO_INFERENCIA_BASE
        )
        dataframe_ajustado = self.servico_arquivo.gerar_dataframe(
            self.CAMINHO_ARQUIVO_INFERENCIA_FINE_TUNING
        )
        dataframe_base = self._validar_relatorio_inferencia(
            dataframe_base,
            "resposta_inferencia_base",
            "inferencia-base",
        )
        dataframe_ajustado = self._validar_relatorio_inferencia(
            dataframe_ajustado,
            "resposta_inferencia_fine_tuning",
            "inferencia fine-tuning",
        )

        ids_referencia = set(dataframe_referencia["id_exemplo"])
        ids_base = set(dataframe_base["id_exemplo"])
        ids_ajustado = set(dataframe_ajustado["id_exemplo"])
        ids_comuns = ids_base & ids_ajustado
        if not ids_comuns:
            raise ValueError(
                "Os relatorios de inferencia base e ajustada nao possuem "
                "registros em comum."
            )
        if ids_base != ids_ajustado:
            print(
                "Aviso: relatorios base/ajustada com IDs distintos; "
                f"usando intersecao de {len(ids_comuns)} registros."
            )
            dataframe_base = dataframe_base.loc[
                dataframe_base["id_exemplo"].isin(ids_comuns)
            ]
            dataframe_ajustado = dataframe_ajustado.loc[
                dataframe_ajustado["id_exemplo"].isin(ids_comuns)
            ]
            ids_base = ids_comuns
        ids_desconhecidos = ids_base - ids_referencia
        if ids_desconhecidos:
            raise ValueError(
                "Os relatorios de inferencia possuem registros que nao "
                "pertencem ao split de teste: "
                + ", ".join(sorted(str(item) for item in ids_desconhecidos))
            )

        dataframe_referencia = dataframe_referencia.loc[
            dataframe_referencia["id_exemplo"].isin(ids_base)
        ]

        comparacao = dataframe_referencia.merge(
            dataframe_base[
                [
                    "id_exemplo",
                    "system",
                    "user",
                    "resposta_inferencia_base",
                ]
            ],
            on="id_exemplo",
            how="left",
            validate="one_to_one",
            suffixes=("", "_base"),
        )
        comparacao = comparacao.merge(
            dataframe_ajustado[
                [
                    "id_exemplo",
                    "system",
                    "user",
                    "resposta_inferencia_fine_tuning",
                ]
            ],
            on="id_exemplo",
            how="left",
            validate="one_to_one",
            suffixes=("", "_fine_tuning"),
        )

        mensagens_divergentes = (
            comparacao["system"] != comparacao["system_base"]
        ) | (comparacao["user"] != comparacao["user_base"])
        mensagens_divergentes |= (
            comparacao["system"] != comparacao["system_fine_tuning"]
        ) | (comparacao["user"] != comparacao["user_fine_tuning"])
        if mensagens_divergentes.any():
            raise ValueError(
                "As mensagens usadas nas inferencias divergem do split de teste."
            )

        comparacao = comparacao[
            [
                "id_exemplo",
                "system",
                "user",
                "resposta_esperada",
                "resposta_inferencia_base",
                "resposta_inferencia_fine_tuning",
            ]
        ].copy()
        comparacao["avaliacao_estrutura"] = ""
        comparacao["avaliacao_relevancia_clinica"] = ""
        comparacao["avaliacao_alucinacao"] = ""
        comparacao["avaliacao_exposicao_pii"] = ""
        comparacao["observacoes"] = ""

        return self.servico_arquivo.criar_excel(
            comparacao,
            self.CAMINHO_ARQUIVO_COMPARACAO,
        )

    def _carregar_modelo_ajustado(self) -> None:
        """Carrega o modelo-base e o adaptador LoRA conforme a chave configurada."""
        self._garantir_adapter_local()
        caminho_configuracao = (
            self.CAMINHO_MODELO_FINE_TUNING / "adapter_config.json"
        )
        if not caminho_configuracao.exists():
            comando = self.MODELOS_DISPONIVEIS[self.chave_modelo][
                "comando_download"
            ]
            raise FileNotFoundError(
                "Adaptador LoRA nao encontrado. Baixe o modelo-base "
                f"({comando}) e garanta o adaptador em: "
                f"{self.CAMINHO_MODELO_FINE_TUNING}"
            )

        if (
            self.modelo_ajustado_carregado
            and self.tokenizer is not None
            and self.modelo is not None
        ):
            return

        try:
            if self.loader_modelo == "unsloth":
                self._carregar_modelo_unsloth(para_treino=False)
                FastLanguageModel = (
                    self._fast_language_model
                    or self._importar_fast_language_model()
                )
                self.modelo = PeftModel.from_pretrained(
                    self.modelo,
                    self.CAMINHO_MODELO_FINE_TUNING,
                    is_trainable=False,
                )
                FastLanguageModel.for_inference(self.modelo)
            else:
                self._carregar_modelo_transformers_peft()
        except (OSError, ValueError, ImportError) as erro:
            raise RuntimeError(
                "Nao foi possivel carregar o modelo ajustado localmente. "
                f"Modelo={self.chave_modelo}."
            ) from erro

        self.modelo.eval()
        self.modelo_ajustado_carregado = True

    def _garantir_adapter_local(self) -> None:
        """Baixa do Hub o adapter publicado quando ele não existe localmente."""
        if (self.CAMINHO_MODELO_FINE_TUNING / "adapter_config.json").exists():
            return
        configuracao = self.MODELOS_DISPONIVEIS[self.chave_modelo]
        repo_adapter = configuracao.get("repo_adapter")
        if not repo_adapter:
            raise FileNotFoundError(
                "O adapter Qwen10 distribuído com o projeto não foi encontrado em "
                f"{self.CAMINHO_MODELO_FINE_TUNING}."
            )
        self.CAMINHO_MODELO_FINE_TUNING.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=str(repo_adapter),
                local_dir=self.CAMINHO_MODELO_FINE_TUNING,
            )
        except OSError as erro:
            raise FileNotFoundError(
                f"Não foi possível baixar o adapter {repo_adapter}. "
                "Verifique internet, autenticação e acesso ao repositório."
            ) from erro

    def _carregar_modelo_transformers_peft(self) -> None:
        """Carrega Qwen (ou similar) via transformers + Peft em GPU/CPU."""
        caminho_modelo = self._obter_caminho_modelo_base()
        self.tokenizer = AutoTokenizer.from_pretrained(
            caminho_modelo,
            local_files_only=True,
        )
        self._configurar_tokenizer()
        usando_cuda = self.DISPOSITIVO == "cuda"
        tipo = torch.float32
        if usando_cuda:
            tipo = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        kwargs_modelo: dict[str, Any] = {
            "dtype": tipo,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if usando_cuda:
            kwargs_modelo["device_map"] = "auto"
        modelo_base = AutoModelForCausalLM.from_pretrained(
            caminho_modelo,
            **kwargs_modelo,
        )
        if not usando_cuda:
            modelo_base.to("cpu")
        self.modelo = PeftModel.from_pretrained(
            modelo_base,
            self.CAMINHO_MODELO_FINE_TUNING,
            is_trainable=False,
        )

    def _realizar_inferencia_split_teste(
        self,
        nome_coluna_resposta: str,
        caminho_arquivo: Path,
        max_novos_tokens: int,
        limite_registros: int | None,
    ) -> Path:
        """Executa a inferencia atual nos registros reservados para teste."""
        if max_novos_tokens <= 0:
            raise ValueError("A quantidade de novos tokens deve ser positiva.")
        if limite_registros is not None and (
            isinstance(limite_registros, bool)
            or not isinstance(limite_registros, int)
            or limite_registros <= 0
        ):
            raise ValueError(
                "O limite de registros deve ser um numero inteiro positivo."
            )

        dataframe = self._carregar_dataframe_fine_tuning_validado()
        registros_teste = dataframe.loc[
            dataframe["split"] == "teste"
        ].sort_values("id_exemplo", kind="stable")
        if limite_registros is not None:
            registros_teste = registros_teste.head(limite_registros)
        resultados_inferencia = []

        for _, registro in registros_teste.iterrows():
            resposta = self._gerar_resposta(
                mensagem_system=registro["system"],
                mensagem_usuario=registro["user"],
                max_novos_tokens=max_novos_tokens,
            )
            resultados_inferencia.append(
                {
                    "id_exemplo": registro["id_exemplo"],
                    "split": "teste",
                    "system": registro["system"],
                    "user": registro["user"],
                    nome_coluna_resposta: resposta,
                }
            )

        dataframe_inferencia = pd.DataFrame(
            resultados_inferencia,
            columns=(
                "id_exemplo",
                "split",
                "system",
                "user",
                nome_coluna_resposta,
            ),
        )
        return self.servico_arquivo.criar_excel(
            dataframe_inferencia,
            caminho_arquivo,
        )

    def _gerar_resposta(
        self,
        mensagem_system: str,
        mensagem_usuario: str,
        max_novos_tokens: int,
    ) -> str:
        """Gera uma resposta individual com o modelo atualmente carregado."""
        if not mensagem_system.strip():
            raise ValueError("A mensagem system nao pode estar vazia.")
        if not mensagem_usuario.strip():
            raise ValueError("A mensagem do usuario nao pode estar vazia.")
        if max_novos_tokens <= 0:
            raise ValueError("A quantidade de novos tokens deve ser positiva.")
        if self.tokenizer is None or self.modelo is None:
            raise RuntimeError("Nenhum modelo foi carregado para inferencia.")

        mensagens = [
            {"role": "system", "content": mensagem_system.strip()},
            {"role": "user", "content": mensagem_usuario.strip()},
        ]
        kwargs_template: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.chave_modelo.startswith("qwen"):
            kwargs_template["enable_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(
            mensagens,
            **kwargs_template,
        )
        entradas = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens_entrada,
        )
        dispositivo_modelo = next(self.modelo.parameters()).device
        entradas = entradas.to(dispositivo_modelo)

        with torch.inference_mode():
            tokens_gerados = self.modelo.generate(
                **entradas,
                max_new_tokens=max_novos_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        inicio_resposta = entradas["input_ids"].shape[1]
        tokens_resposta = tokens_gerados[0][inicio_resposta:]
        return self.tokenizer.decode(
            tokens_resposta,
            skip_special_tokens=True,
        ).strip()

    def _carregar_dataframe_fine_tuning_validado(self) -> pd.DataFrame:
        """Carrega e valida o dataset preparado antes de qualquer uso."""
        dataframe = self.servico_arquivo.gerar_dataframe(
            self.CAMINHO_ARQUIVO_FINE_TUNING
        ).copy()
        colunas_ausentes = [
            coluna
            for coluna in self.COLUNAS_DATASET_FINE_TUNING
            if coluna not in dataframe.columns
        ]
        if colunas_ausentes:
            raise ValueError(
                "Colunas necessarias ausentes no dataframe de fine-tuning: "
                + ", ".join(colunas_ausentes)
            )

        for coluna in self.COLUNAS_DATASET_FINE_TUNING:
            mascara_vazia = dataframe[coluna].isna() | (
                dataframe[coluna].astype(str).str.strip() == ""
            )
            if mascara_vazia.any():
                linhas = ", ".join(
                    str(indice + 2) for indice in dataframe.index[mascara_vazia]
                )
                raise ValueError(
                    f"Valores vazios na coluna {coluna}, linhas: {linhas}."
                )

        dataframe["id_exemplo"] = dataframe["id_exemplo"].map(
            self._normalizar_identificador
        )
        dataframe["split"] = (
            dataframe["split"].astype(str).str.strip().str.casefold()
        )
        for coluna in ("system", "user", "assistant"):
            dataframe[coluna] = dataframe[coluna].astype(str).str.strip()

        totais_tokens = pd.to_numeric(
            dataframe[self.COLUNA_TOTAL_TOKENS],
            errors="coerce",
        )
        totais_tokens_invalidos = (
            totais_tokens.isna()
            | (totais_tokens < 0)
            | (totais_tokens % 1 != 0)
        )
        if totais_tokens_invalidos.any():
            linhas = ", ".join(
                str(indice + 2)
                for indice in dataframe.index[totais_tokens_invalidos]
            )
            raise ValueError(
                f"Valores invalidos na coluna {self.COLUNA_TOTAL_TOKENS}, "
                f"linhas: {linhas}."
            )
        dataframe[self.COLUNA_TOTAL_TOKENS] = totais_tokens.astype("int64")

        splits_invalidos = sorted(
            set(dataframe["split"]) - set(self.SPLITS_ESPERADOS)
        )
        if splits_invalidos:
            raise ValueError(
                "Valores de split invalidos: " + ", ".join(splits_invalidos)
            )

        ids_duplicados = dataframe.loc[
            dataframe["id_exemplo"].duplicated(keep=False),
            "id_exemplo",
        ]
        if not ids_duplicados.empty:
            raise ValueError(
                "Identificadores repetidos entre os splits: "
                + ", ".join(sorted(set(ids_duplicados)))
            )

        conversas_duplicadas = dataframe.duplicated(
            subset=["system", "user", "assistant"],
            keep=False,
        )
        if conversas_duplicadas.any():
            raise ValueError(
                "Existem conversas repetidas entre os registros do dataset."
            )

        splits_ausentes = [
            nome_split
            for nome_split in self.SPLITS_ESPERADOS
            if not (dataframe["split"] == nome_split).any()
        ]
        if splits_ausentes:
            # Corrige arquivos gerados pela regra antiga quando a amostra tinha
            # menos de dez registros e a validacao recebia zero exemplos.
            dataframe["split"] = self._gerar_splits(dataframe)
            self.servico_arquivo.atualizar_excel(
                dataframe,
                self.CAMINHO_ARQUIVO_FINE_TUNING,
            )

        dataframe_elegivel = dataframe.loc[
            dataframe[self.COLUNA_TOTAL_TOKENS] < self.max_tokens_entrada
        ].copy()
        self.quantidade_registros_descartados_tokens = (
            len(dataframe) - len(dataframe_elegivel)
        )
        if len(dataframe_elegivel) < 3:
            raise ValueError(
                "O dataset deve possuir ao menos 3 registros com "
                f"{self.COLUNA_TOTAL_TOKENS} menor que "
                f"{self.max_tokens_entrada}. Registros elegiveis: "
                f"{len(dataframe_elegivel)}."
            )

        # Recalcula os splits depois do filtro para manter treino, validacao e
        # teste preenchidos somente com registros elegiveis.
        dataframe_elegivel["split"] = self._gerar_splits(dataframe_elegivel)
        return dataframe_elegivel

    def _calcular_estatisticas_tokens(
        self,
        datasets: DatasetDict,
    ) -> dict[str, dict[str, int | float]]:
        """Calcula estatisticas agregadas sem registrar o conteudo clinico."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer nao carregado para analisar os tokens.")

        estatisticas: dict[str, dict[str, int | float]] = {}
        for nome_split, dataset in datasets.items():
            quantidades_tokens = []
            for exemplo in dataset:
                mensagens = [*exemplo["prompt"], *exemplo["completion"]]
                tokens = self.tokenizer.apply_chat_template(
                    mensagens,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                ids_tokens = (
                    tokens["input_ids"]
                    if hasattr(tokens, "keys") and "input_ids" in tokens
                    else tokens
                )
                quantidades_tokens.append(len(ids_tokens))

            valores_ordenados = sorted(quantidades_tokens)
            indice_percentil_95 = max(
                ceil(len(valores_ordenados) * 0.95) - 1,
                0,
            )
            estatisticas[nome_split] = {
                "quantidade_exemplos": len(valores_ordenados),
                "tokens_minimos": valores_ordenados[0],
                "tokens_mediana": float(median(valores_ordenados)),
                "tokens_percentil_95": valores_ordenados[indice_percentil_95],
                "tokens_maximos": valores_ordenados[-1],
                "exemplos_acima_limite": sum(
                    quantidade > self.max_tokens_entrada
                    for quantidade in valores_ordenados
                ),
            }

        return estatisticas

    def _criar_configuracao_lora(self) -> LoraConfig:
        """Cria a configuracao inicial do adaptador LoRA (Unsloth Fast)."""
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=self.rank_lora,
            lora_alpha=32,
            # Dropout 0 e o caminho otimizado do Unsloth Fast.
            lora_dropout=0.0,
            bias="none",
            target_modules=list(self.MODULOS_LORA),
        )

    def _criar_configuracao_treinamento(
        self,
        max_passos: int | None,
    ) -> SFTConfig:
        """Cria a configuracao reproduzivel do treinamento supervisionado."""
        if max_passos is not None and max_passos <= 0:
            raise ValueError("A quantidade maxima de passos deve ser positiva.")

        return SFTConfig(
            output_dir=str(self.CAMINHO_CHECKPOINTS),
            learning_rate=1e-4,
            num_train_epochs=self.quantidade_epocas_fine_tuning,
            max_steps=max_passos if max_passos is not None else -1,
            per_device_train_batch_size=self.BATCH_TREINO,
            gradient_accumulation_steps=self.ACUMULACAO_GRADIENTE,
            per_device_eval_batch_size=self.BATCH_TREINO,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_strategy="steps",
            logging_steps=10,
            max_length=self.max_tokens_entrada,
            packing=True,
            eval_packing=True,
            completion_only_loss=True,
            seed=42,
            data_seed=42,
            use_cpu=False,
            fp16=self.DISPOSITIVO == "cuda" and not torch.cuda.is_bf16_supported(),
            bf16=self.DISPOSITIVO == "cuda" and torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            dataloader_pin_memory=self.DISPOSITIVO == "cuda",
            optim="paged_adamw_8bit",
            dataloader_num_workers=2,
            report_to="none",
            push_to_hub=False,
        )

    def _salvar_metricas_fine_tuning(
        self,
        metricas_treinamento: dict[str, Any],
        metricas_validacao_base: dict[str, Any],
        metricas_validacao: dict[str, Any],
        estatisticas_tokens: dict[str, dict[str, int | float]],
    ) -> Path:
        """Salva metricas e estatisticas agregadas em um relatorio TXT."""
        linhas = [
            f"modelo_base: {self.NOME_MODELO_BASE}",
            "metodo: SFT com Unsloth Fast + QLoRA (4-bit)",
            f"limite_registros: {self.limite_registros_fine_tuning}",
            f"lora_r: {self.rank_lora}",
            "lora_alpha: 32",
            "lora_dropout: 0.0",
            f"batch: {self.BATCH_TREINO}",
            f"gradient_accumulation: {self.ACUMULACAO_GRADIENTE}",
            "packing: True",
            f"atencao: {self._obter_implementacao_atencao()}",
            "taxa_aprendizado: 0.0001",
            f"epocas: {self.quantidade_epocas_fine_tuning}",
            f"max_tokens: {self.max_tokens_entrada}",
            "",
            "Metricas de treinamento:",
        ]
        linhas.extend(
            f"{chave}: {valor}"
            for chave, valor in sorted(metricas_treinamento.items())
        )
        linhas.append("")
        linhas.append("Metricas de validacao antes do fine-tuning:")
        linhas.extend(
            f"{chave}: {valor}"
            for chave, valor in sorted(metricas_validacao_base.items())
        )
        linhas.append("")
        linhas.append("Metricas de validacao:")
        linhas.extend(
            f"{chave}: {valor}"
            for chave, valor in sorted(metricas_validacao.items())
        )

        perda_validacao = metricas_validacao.get("eval_loss")
        if perda_validacao is not None:
            try:
                linhas.append(f"perplexidade: {exp(float(perda_validacao))}")
            except OverflowError:
                linhas.append("perplexidade: infinito")

        linhas.append("")
        linhas.append("Estatisticas de tokens:")
        for nome_split, estatisticas in estatisticas_tokens.items():
            linhas.append(f"[{nome_split}]")
            linhas.extend(
                f"{chave}: {valor}"
                for chave, valor in sorted(estatisticas.items())
            )

        caminho_relatorio = self.servico_arquivo.criar_arquivo_txt(
            "Metricas do fine-tuning",
            "\n".join(linhas),
            self.CAMINHO_RELATORIO_METRICAS,
        )
        if caminho_relatorio is None:
            raise OSError("Nao foi possivel salvar as metricas do fine-tuning.")
        return caminho_relatorio

    def _salvar_relatorio_tecnico_fine_tuning(
        self,
        metricas_treinamento: dict[str, Any],
        metricas_validacao_base: dict[str, Any],
        metricas_validacao: dict[str, Any],
        estatisticas_tokens: dict[str, dict[str, int | float]],
        historico_treinamento: list[dict[str, Any]],
    ) -> Path:
        """Gera um relatorio tecnico comparando o modelo antes e depois do SFT."""

        def obter_numero(metricas: dict[str, Any], chave: str) -> float | None:
            valor = metricas.get(chave)
            if valor is None or isinstance(valor, bool):
                return None
            try:
                numero = float(valor)
            except (TypeError, ValueError):
                return None
            return numero if isfinite(numero) else None

        perda_base = obter_numero(metricas_validacao_base, "eval_loss")
        perda_ajustada = obter_numero(metricas_validacao, "eval_loss")
        acuracia_base = obter_numero(
            metricas_validacao_base,
            "eval_mean_token_accuracy",
        )
        acuracia_ajustada = obter_numero(
            metricas_validacao,
            "eval_mean_token_accuracy",
        )

        reducao_perda_percentual = None
        if perda_base not in (None, 0.0) and perda_ajustada is not None:
            reducao_perda_percentual = (
                (perda_base - perda_ajustada) / perda_base * 100
            )

        variacao_acuracia = None
        if acuracia_base is not None and acuracia_ajustada is not None:
            variacao_acuracia = acuracia_ajustada - acuracia_base

        perplexidade_base = None
        perplexidade_ajustada = None
        try:
            if perda_base is not None:
                perplexidade_base = exp(perda_base)
            if perda_ajustada is not None:
                perplexidade_ajustada = exp(perda_ajustada)
        except OverflowError:
            pass

        registros_treino = [
            registro
            for registro in historico_treinamento
            if "loss" in registro and "eval_loss" not in registro
        ]
        perdas_treino = [
            float(registro["loss"])
            for registro in registros_treino
            if registro.get("loss") is not None
        ]
        normas_gradiente = [
            float(registro["grad_norm"])
            for registro in registros_treino
            if registro.get("grad_norm") is not None
        ]
        taxas_aprendizado = [
            float(registro["learning_rate"])
            for registro in registros_treino
            if registro.get("learning_rate") is not None
        ]

        perda_treino_media = obter_numero(metricas_treinamento, "train_loss")
        gap_generalizacao = None
        if perda_ajustada is not None and perda_treino_media is not None:
            gap_generalizacao = perda_ajustada - perda_treino_media

        total_truncados = sum(
            int(estatisticas.get("exemplos_acima_limite", 0))
            for estatisticas in estatisticas_tokens.values()
        )

        if perda_base is None or perda_ajustada is None:
            veredito = "INCONCLUSIVO"
            justificativa = (
                "Nao foi possivel comparar a perda de validacao antes e depois."
            )
        elif (
            reducao_perda_percentual is not None
            and reducao_perda_percentual >= 5.0
            and (variacao_acuracia is None or variacao_acuracia >= 0.0)
        ):
            veredito = "FUNCIONOU TECNICAMENTE"
            justificativa = (
                "A perda de validacao caiu pelo menos 5% sem reducao da "
                "acuracia de tokens. Ainda e necessaria avaliacao clinica."
            )
        elif perda_ajustada < perda_base:
            veredito = "INDICIO POSITIVO, MAS INCONCLUSIVO"
            justificativa = (
                "A perda de validacao diminuiu, mas o ganho foi inferior ao "
                "criterio de 5% ou a acuracia de tokens piorou."
            )
        else:
            veredito = "NAO FUNCIONOU NESTA EXECUCAO"
            justificativa = (
                "A perda de validacao nao melhorou em relacao ao modelo-base."
            )

        data_execucao = datetime.now().astimezone().isoformat(timespec="seconds")
        linhas: list[dict[str, Any]] = []

        def adicionar_metrica(
            secao: str,
            metrica: str,
            valor_base: Any = None,
            valor_ajustado: Any = None,
            variacao: Any = None,
            unidade: str = "",
            direcao_desejada: str = "",
            interpretacao: str = "",
        ) -> None:
            linhas.append(
                {
                    "data_execucao": data_execucao,
                    "secao": secao,
                    "metrica": metrica,
                    "modelo_base": valor_base,
                    "modelo_ajustado": valor_ajustado,
                    "variacao": variacao,
                    "unidade": unidade,
                    "direcao_desejada": direcao_desejada,
                    "interpretacao": interpretacao,
                }
            )

        adicionar_metrica(
            "resumo",
            "veredito_tecnico",
            valor_ajustado=veredito,
            interpretacao=justificativa,
        )
        adicionar_metrica(
            "validacao",
            "perda_validacao",
            perda_base,
            perda_ajustada,
            (
                None
                if perda_base is None or perda_ajustada is None
                else perda_ajustada - perda_base
            ),
            direcao_desejada="menor",
            interpretacao="Compara o mesmo split antes e depois do LoRA.",
        )
        adicionar_metrica(
            "validacao",
            "reducao_perda_validacao",
            valor_ajustado=reducao_perda_percentual,
            unidade="%",
            direcao_desejada="maior; referencia minima de 5%",
            interpretacao="Principal indicador automatico de aprendizado.",
        )
        adicionar_metrica(
            "validacao",
            "perplexidade",
            perplexidade_base,
            perplexidade_ajustada,
            (
                None
                if perplexidade_base is None or perplexidade_ajustada is None
                else perplexidade_ajustada - perplexidade_base
            ),
            direcao_desejada="menor",
            interpretacao=(
                "Derivada da perda; valores menores indicam maior "
                "previsibilidade."
            ),
        )
        adicionar_metrica(
            "validacao",
            "acuracia_media_tokens",
            acuracia_base,
            acuracia_ajustada,
            variacao_acuracia,
            unidade="proporcao",
            direcao_desejada="maior",
            interpretacao="Complementa a perda, mas nao mede correcao clinica.",
        )
        adicionar_metrica(
            "generalizacao",
            "gap_validacao_treino",
            valor_ajustado=gap_generalizacao,
            direcao_desejada="proximo de zero",
            interpretacao=(
                "Valores positivos elevados podem indicar sobreajuste; compare "
                "tambem a evolucao por epoca."
            ),
        )
        adicionar_metrica(
            "treinamento",
            "perda_primeiro_ultimo_passo",
            perdas_treino[0] if perdas_treino else None,
            perdas_treino[-1] if perdas_treino else None,
            (
                perdas_treino[-1] - perdas_treino[0]
                if len(perdas_treino) >= 2
                else None
            ),
            direcao_desejada="menor",
            interpretacao=(
                "Mostra se houve convergencia durante os passos registrados."
            ),
        )
        adicionar_metrica(
            "treinamento",
            "menor_perda_treino_registrada",
            valor_ajustado=min(perdas_treino) if perdas_treino else None,
            direcao_desejada="menor",
        )
        adicionar_metrica(
            "treinamento",
            "passos_otimizacao",
            valor_ajustado=obter_numero(
                metricas_treinamento,
                "passos_treinamento",
            ),
            unidade="passos",
            direcao_desejada="suficiente para convergir",
        )
        adicionar_metrica(
            "eficiencia",
            "tempo_treinamento",
            valor_ajustado=obter_numero(metricas_treinamento, "train_runtime"),
            unidade="segundos",
            direcao_desejada="menor para qualidade equivalente",
        )
        adicionar_metrica(
            "eficiencia",
            "amostras_por_segundo",
            valor_ajustado=obter_numero(
                metricas_treinamento,
                "train_samples_per_second",
            ),
            unidade="amostras/s",
            direcao_desejada="maior",
        )
        adicionar_metrica(
            "eficiencia",
            "passos_por_segundo",
            valor_ajustado=obter_numero(
                metricas_treinamento,
                "train_steps_per_second",
            ),
            unidade="passos/s",
            direcao_desejada="maior",
        )
        adicionar_metrica(
            "estabilidade",
            "norma_gradiente_media_maxima",
            sum(normas_gradiente) / len(normas_gradiente)
            if normas_gradiente
            else None,
            max(normas_gradiente) if normas_gradiente else None,
            interpretacao=(
                "Picos muito altos ou valores nao finitos sugerem instabilidade."
            ),
        )
        adicionar_metrica(
            "configuracao",
            "taxa_aprendizado_inicial_final",
            taxas_aprendizado[0] if taxas_aprendizado else None,
            taxas_aprendizado[-1] if taxas_aprendizado else None,
            direcao_desejada="decrescer conforme o scheduler",
        )
        adicionar_metrica(
            "configuracao",
            "epocas",
            valor_ajustado=self.quantidade_epocas_fine_tuning,
        )
        adicionar_metrica(
            "configuracao",
            "lora_rank",
            valor_ajustado=self.rank_lora,
        )
        adicionar_metrica(
            "configuracao",
            "lora_alpha_dropout",
            valor_base=32,
            valor_ajustado=0.0,
            interpretacao="Colunas representam alpha e dropout, respectivamente.",
        )
        adicionar_metrica(
            "configuracao",
            "modulos_lora",
            valor_ajustado=", ".join(self.MODULOS_LORA),
        )
        adicionar_metrica(
            "configuracao",
            "lote_efetivo",
            valor_ajustado=self.BATCH_TREINO * self.ACUMULACAO_GRADIENTE,
            unidade="exemplos por atualizacao",
        )
        adicionar_metrica(
            "configuracao",
            "packing",
            valor_ajustado=True,
        )
        adicionar_metrica(
            "configuracao",
            "limite_tokens",
            valor_ajustado=self.max_tokens_entrada,
            unidade="tokens",
        )
        adicionar_metrica(
            "configuracao",
            "parametros_treinaveis",
            valor_ajustado=obter_numero(
                metricas_treinamento,
                "parametros_treinaveis",
            ),
            unidade="parametros",
        )
        adicionar_metrica(
            "configuracao",
            "percentual_parametros_treinaveis",
            valor_ajustado=obter_numero(
                metricas_treinamento,
                "percentual_parametros_treinaveis",
            ),
            unidade="%",
        )
        adicionar_metrica(
            "tokens",
            "exemplos_acima_limite_total",
            valor_ajustado=total_truncados,
            unidade="exemplos",
            direcao_desejada="zero",
            interpretacao=(
                "Respostas truncadas prejudicam o aprendizado do formato "
                "completo."
            ),
        )
        adicionar_metrica(
            "tokens",
            "registros_descartados_por_limite",
            valor_ajustado=self.quantidade_registros_descartados_tokens,
            unidade="registros",
            direcao_desejada="baixo sem remover casos importantes",
            interpretacao=(
                "Registros completos excluidos antes dos splits por excederem "
                "o limite de tokens."
            ),
        )

        for nome_split, estatisticas in estatisticas_tokens.items():
            adicionar_metrica(
                f"dataset_{nome_split}",
                "quantidade_exemplos",
                valor_ajustado=estatisticas.get("quantidade_exemplos"),
                unidade="exemplos",
            )
            adicionar_metrica(
                f"dataset_{nome_split}",
                "tokens_mediana_p95_maximo",
                estatisticas.get("tokens_mediana"),
                estatisticas.get("tokens_percentil_95"),
                estatisticas.get("tokens_maximos"),
                unidade="tokens",
                interpretacao=(
                    "Colunas representam mediana, percentil 95 e maximo, "
                    "respectivamente."
                ),
            )

        adicionar_metrica(
            "criterios",
            "limite_do_veredito",
            valor_ajustado="reducao de eval_loss >= 5% e acuracia sem queda",
            interpretacao=(
                "O veredito e tecnico. Relevancia, seguranca e correcao clinica "
                "devem ser avaliadas nas inferencias do split de teste."
            ),
        )

        dataframe_relatorio = pd.DataFrame(linhas)
        return self.servico_arquivo.criar_excel(
            dataframe_relatorio,
            self.CAMINHO_RELATORIO_TECNICO,
        )

    def _validar_relatorio_inferencia(
        self,
        dataframe: pd.DataFrame,
        nome_coluna_resposta: str,
        nome_relatorio: str,
    ) -> pd.DataFrame:
        """Valida um relatorio antes de montar a comparacao final."""
        if dataframe.empty:
            raise ValueError(
                f"O relatorio de {nome_relatorio} nao possui registros."
            )

        colunas_necessarias = (
            "id_exemplo",
            "system",
            "user",
            nome_coluna_resposta,
        )
        colunas_ausentes = [
            coluna
            for coluna in colunas_necessarias
            if coluna not in dataframe.columns
        ]
        if colunas_ausentes:
            raise ValueError(
                f"Colunas ausentes no relatorio de {nome_relatorio}: "
                + ", ".join(colunas_ausentes)
            )

        dataframe = dataframe.copy()
        dataframe["id_exemplo"] = dataframe["id_exemplo"].map(
            self._normalizar_identificador
        )
        if (dataframe["id_exemplo"] == "").any():
            raise ValueError(
                f"Identificador vazio no relatorio de {nome_relatorio}."
            )
        if dataframe["id_exemplo"].duplicated().any():
            raise ValueError(
                f"Identificadores repetidos no relatorio de {nome_relatorio}."
            )

        for coluna in ("system", "user", nome_coluna_resposta):
            mascara_vazia = dataframe[coluna].isna() | (
                dataframe[coluna].astype(str).str.strip() == ""
            )
            if mascara_vazia.any():
                raise ValueError(
                    f"Valores vazios na coluna {coluna} do relatorio de "
                    f"{nome_relatorio}."
                )
            dataframe[coluna] = dataframe[coluna].astype(str).str.strip()
        return dataframe

    @staticmethod
    def _montar_mensagem_usuario(dataframe: pd.DataFrame) -> pd.Series:
        """Concatena os campos que formam a solicitacao enviada ao modelo."""
        valores = dataframe.fillna("").astype(str)
        return (
            "Papel do solicitante: "
            + valores["papel_solicitante"].str.strip()
            + "\nContexto da solicitacao: "
            + valores["contexto_solicitacao"].str.strip()
            + "\nProntuario: "
            + valores["prontuario_contexto_anonimizado"].str.strip()
            + "\nPergunta: "
            + valores["pergunta_original_anonimizado"].str.strip()
        )

    @staticmethod
    def _calcular_quantidades_splits(
        quantidade_registros: int,
    ) -> dict[str, int]:
        """Calcula a divisao 80/10/10 mantendo todos os splits preenchidos."""
        if quantidade_registros < 3:
            raise ValueError(
                "O dataset de fine-tuning deve possuir ao menos 3 registros "
                "validos para formar os splits de treino, validacao e teste. "
                "Reinicie o sistema, informe um percentual maior e execute "
                "novamente a opcao 0."
            )

        quantidade_treino = min(
            round(quantidade_registros * 0.8),
            quantidade_registros - 2,
        )
        quantidade_restante = quantidade_registros - quantidade_treino
        quantidade_validacao = quantidade_restante // 2
        quantidade_teste = quantidade_restante - quantidade_validacao
        return {
            "treino": quantidade_treino,
            "validacao": quantidade_validacao,
            "teste": quantidade_teste,
        }

    @classmethod
    def _gerar_splits(cls, dataframe: pd.DataFrame) -> pd.Series:
        """Distribui os exemplos em 80% treino, 10% validacao e 10% teste."""
        quantidades = cls._calcular_quantidades_splits(len(dataframe))
        splits = pd.Series("teste", index=dataframe.index, dtype="object")
        indices = dataframe.sample(
            frac=1,
            random_state=cls.SEMENTE_ALEATORIA,
        ).index
        quantidade_treino = quantidades["treino"]
        quantidade_validacao = quantidades["validacao"]

        splits.loc[indices[:quantidade_treino]] = "treino"
        splits.loc[
            indices[quantidade_treino : quantidade_treino + quantidade_validacao]
        ] = "validacao"
        return splits

    @staticmethod
    def _normalizar_identificador(identificador: object) -> str:
        """Normaliza IDs numericos lidos do Excel para permitir a comparacao."""
        if pd.isna(identificador):
            return ""
        if isinstance(identificador, float) and identificador.is_integer():
            return str(int(identificador))
        return str(identificador).strip()

    @classmethod
    def _validar_colunas_necessarias(cls, dataframe: pd.DataFrame) -> None:
        """Impede a preparacao quando alguma coluna de origem estiver ausente."""
        colunas_ausentes = [
            coluna
            for coluna in cls.COLUNAS_NECESSARIAS
            if coluna not in dataframe.columns
        ]
        if colunas_ausentes:
            raise ValueError(
                "Colunas necessarias ausentes no arquivo de auditoria: "
                + ", ".join(colunas_ausentes)
            )
