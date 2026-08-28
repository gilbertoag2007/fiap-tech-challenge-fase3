from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.services.arquivo_service import ArquivoService


class FineTuningService:
    """Prepara e salva os exemplos que serao usados no fine-tuning."""

    NOME_MODELO_BASE = "Qwen/Qwen3-0.6B"
    DISPOSITIVO = "cpu"
    MAX_TOKENS_ENTRADA = 512
    CAMINHO_ARQUIVO_AUDITORIA = Path(
        "app/data/processado/dados_medicos_auditoria.xlsx"
    )
    CAMINHO_ARQUIVO_FINE_TUNING = Path(
        "app/data/processado/dados_medicos_fine_tuning.xlsx"
    )
    CAMINHO_ARQUIVO_INFERENCIA_BASE = Path(
        "app/data/relatorios/inferencia_base.xlsx"
    )
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
    MENSAGEM_SYSTEM = (
        "Voce e um assistente especializado em apoio clinico. "
        "Responda de forma estruturada seguindo o padrao do hospital."
    )

    def __init__(self, servico_arquivo: ArquivoService | None = None) -> None:
        self.servico_arquivo = servico_arquivo or ArquivoService()
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.modelo: PreTrainedModel | None = None

    def carregar_modelo_base(self) -> None:
        """Carrega tokenizer e modelo do cache para execucao somente em CPU."""
        if self.tokenizer is not None and self.modelo is not None:
            return

        try:
            caminho_modelo = snapshot_download(
                repo_id=self.NOME_MODELO_BASE,
                local_files_only=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                caminho_modelo,
                local_files_only=True,
            )
            self.tokenizer.truncation_side = "left"
            self.modelo = AutoModelForCausalLM.from_pretrained(
                caminho_modelo,
                dtype=torch.float32,
                local_files_only=True,
            )
        except OSError as erro:
            raise FileNotFoundError(
                "Modelo base nao encontrado no cache local. Execute: "
                f"hf download {self.NOME_MODELO_BASE}"
            ) from erro

        self.modelo.to(self.DISPOSITIVO)
        self.modelo.eval()

    def realizar_inferencia_base(
        self,
        identificadores_registros: set[int | str],
        max_novos_tokens: int = 128,
    ) -> Path:
        """Executa a inferencia nos registros informados e salva o relatorio."""
        if not identificadores_registros:
            raise ValueError("Informe ao menos um identificador de registro.")

        dataframe = self.servico_arquivo.gerar_dataframe(
            self.CAMINHO_ARQUIVO_AUDITORIA
        )
        colunas_ausentes = [
            coluna
            for coluna in ("id", "pergunta_original_anonimizado")
            if coluna not in dataframe.columns
        ]
        if colunas_ausentes:
            raise ValueError(
                "Colunas necessarias ausentes no arquivo de auditoria: "
                + ", ".join(colunas_ausentes)
            )

        ids_solicitados = {
            self._normalizar_identificador(identificador)
            for identificador in identificadores_registros
        }
        ids_dataframe = dataframe["id"].map(
            self._normalizar_identificador
        )
        ids_ausentes = ids_solicitados - set(ids_dataframe)
        if ids_ausentes:
            raise ValueError(
                "Identificadores nao encontrados no dataframe: "
                + ", ".join(sorted(ids_ausentes))
            )

        registros = dataframe.loc[ids_dataframe.isin(ids_solicitados)]
        resultados_inferencia = []
        for _, registro in registros.iterrows():
            identificador = self._normalizar_identificador(
                registro["id"]
            )
            pergunta = str(
                registro["pergunta_original_anonimizado"]
            ).strip()
            resposta = self._gerar_resposta_base(
                pergunta,
                max_novos_tokens,
            )
            resultados_inferencia.append(
                {
                    "id": identificador,
                    "pergunta_inferencia_base": pergunta,
                    "resposta_inferencia_base": resposta,
                }
            )

        dataframe_inferencia = pd.DataFrame(
            resultados_inferencia,
            columns=(
                "id",
                "pergunta_inferencia_base",
                "resposta_inferencia_base",
            ),
        )
        caminho_arquivo = self.servico_arquivo.criar_excel(
            dataframe_inferencia,
            self.CAMINHO_ARQUIVO_INFERENCIA_BASE,
        )
        return caminho_arquivo

    def _gerar_resposta_base(
        self,
        mensagem_usuario: str,
        max_novos_tokens: int,
    ) -> str:
        """Gera uma resposta individual com o modelo antes do fine-tuning."""
        if not mensagem_usuario.strip():
            raise ValueError("A mensagem do usuario nao pode estar vazia.")
        if max_novos_tokens <= 0:
            raise ValueError("A quantidade de novos tokens deve ser positiva.")

        self.carregar_modelo_base()
        if self.tokenizer is None or self.modelo is None:
            raise RuntimeError("O modelo base nao foi carregado corretamente.")

        mensagens = [
            {"role": "system", "content": self.MENSAGEM_SYSTEM},
            {"role": "user", "content": mensagem_usuario.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(
            mensagens,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        entradas = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_TOKENS_ENTRADA,
        ).to(self.DISPOSITIVO)

        # Mantem a amostragem reproduzivel para comparar antes e depois do ajuste.
        torch.manual_seed(42)
        with torch.inference_mode():
            tokens_gerados = self.modelo.generate(
                **entradas,
                max_new_tokens=max_novos_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        inicio_resposta = entradas["input_ids"].shape[1]
        tokens_resposta = tokens_gerados[0][inicio_resposta:]
        return self.tokenizer.decode(
            tokens_resposta,
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _normalizar_identificador(identificador: object) -> str:
        """Normaliza IDs numericos lidos do Excel para permitir a comparacao."""
        if pd.isna(identificador):
            return ""
        if isinstance(identificador, float) and identificador.is_integer():
            return str(int(identificador))
        return str(identificador).strip()

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
        dataframe_fine_tuning["split"] = self._gerar_splits(
            dataframe_fine_tuning
        )

        self.servico_arquivo.criar_excel(
            dataframe_fine_tuning,
            self.CAMINHO_ARQUIVO_FINE_TUNING,
        )
        return dataframe_fine_tuning

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
    def _gerar_splits(dataframe: pd.DataFrame) -> pd.Series:
        """Distribui os exemplos em 80% treino, 10% validacao e 10% teste."""
        splits = pd.Series("teste", index=dataframe.index, dtype="object")
        indices = dataframe.sample(frac=1, random_state=42).index
        quantidade_treino = int(len(dataframe) * 0.8)
        quantidade_validacao = int(len(dataframe) * 0.1)

        splits.loc[indices[:quantidade_treino]] = "treino"
        splits.loc[
            indices[quantidade_treino : quantidade_treino + quantidade_validacao]
        ] = "validacao"
        return splits

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
