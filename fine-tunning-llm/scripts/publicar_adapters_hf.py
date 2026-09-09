"""Publica somente os arquivos mínimos dos adapters LoRA no Hugging Face."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi


RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "app" / "modelos"

ADAPTERS = {
    "llama": {
        "origem": MODELOS / "llama31_8b_instruct_lora_gpu",
        "base": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "nome": "fiap-medpt-llama31-8b-qlora",
        "titulo": "FIAP MedPT Llama 3.1 8B QLoRA",
        "licenca": "llama3.1",
        "amostra": "80% do conjunto preparado",
    },
    "qwen80": {
        "origem": (
            MODELOS
            / "qwen_06b_lora_gpu"
            / "qwen3_06b_lora_80pct_more_projections"
        ),
        "base": "Qwen/Qwen3-0.6B",
        "nome": "fiap-medpt-qwen3-06b-lora-80pct-qkvo",
        "titulo": "FIAP MedPT Qwen3 0.6B LoRA 80% q/k/v/o",
        "licenca": "apache-2.0",
        "amostra": "80% do conjunto preparado",
    },
}


def model_card(configuracao: dict[str, object]) -> str:
    return f"""---
base_model: {configuracao["base"]}
library_name: peft
license: {configuracao["licenca"]}
language:
- pt
pipeline_tag: text-generation
tags:
- medical
- academic
- lora
- fiap
---

# {configuracao["titulo"]}

Adapter LoRA acadêmico treinado sobre dados médicos anonimizados no Tech
Challenge da FIAP. Amostra de treinamento: {configuracao["amostra"]}.

## Uso

Carregue o modelo-base indicado no metadata e aplique este adapter com PEFT.
O modelo-base não está incluído neste repositório.

## Limitações e segurança

Este artefato é experimental, pode alucinar e não é um dispositivo médico.
Não deve diagnosticar, prescrever nem substituir avaliação profissional.
O projeto consumidor exige revisão humana (HITL) antes de liberar respostas.
"""


def preparar(origem: Path, destino: Path, configuracao: dict[str, object]) -> None:
    for nome in ("adapter_model.safetensors",):
        arquivo = origem / nome
        if not arquivo.exists():
            raise FileNotFoundError(arquivo)
        shutil.copy2(arquivo, destino / nome)

    config_adapter = json.loads(
        (origem / "adapter_config.json").read_text(encoding="utf-8")
    )
    config_adapter["base_model_name_or_path"] = configuracao["base"]
    (destino / "adapter_config.json").write_text(
        json.dumps(config_adapter, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destino / "README.md").write_text(
        model_card(configuracao),
        encoding="utf-8",
    )


def main() -> None:
    api = HfApi()
    identidade = api.whoami()
    namespace = os.getenv("HF_NAMESPACE") or identidade["name"]

    for chave, configuracao in ADAPTERS.items():
        repo_id = f"{namespace}/{configuracao['nome']}"
        with TemporaryDirectory(prefix=f"adapter-{chave}-") as temporario:
            destino = Path(temporario)
            preparar(Path(configuracao["origem"]), destino, configuracao)
            api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=destino,
                commit_message=f"Publica adapter {chave} sanitizado",
            )
        print(f"Publicado: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
