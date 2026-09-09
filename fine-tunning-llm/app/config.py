"""Configuração portátil compartilhada pelo CLI, API e contêineres."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ_PROJETO / ".env")
DIRETORIO_DADOS = Path(
    os.getenv("APP_DATA_DIR", RAIZ_PROJETO / "app" / "data")
).resolve()
DIRETORIO_MODELOS = Path(
    os.getenv("APP_MODELS_DIR", RAIZ_PROJETO / "app" / "modelos")
).resolve()
DIRETORIO_RELATORIOS = DIRETORIO_DADOS / "relatorios"
CAMINHO_AUDITORIA = Path(
    os.getenv(
        "APP_AUDIT_FILE",
        DIRETORIO_RELATORIOS / "auditoria_assistente.jsonl",
    )
).resolve()
CAMINHO_PRONTUARIOS = Path(
    os.getenv("APP_MEDICAL_DATA_FILE")
    or DIRETORIO_DADOS / "processado" / "dados_medicos_auditoria.xlsx"
).resolve()

HF_NAMESPACE = os.getenv("HF_NAMESPACE", "Hist3ry")
HF_REPO_LLAMA = os.getenv(
    "HF_REPO_LLAMA",
    f"{HF_NAMESPACE}/fiap-medpt-llama31-8b-qlora",
)
HF_REPO_QWEN80 = os.getenv(
    "HF_REPO_QWEN80",
    f"{HF_NAMESPACE}/fiap-medpt-qwen3-06b-lora-80pct-qkvo",
)

MODELO_PADRAO = os.getenv("APP_DEFAULT_MODEL", "llama")
CORS_ORIGINS = tuple(
    origem.strip()
    for origem in os.getenv(
        "APP_CORS_ORIGINS",
        "http://localhost:5173,http://localhost:8080",
    ).split(",")
    if origem.strip()
)
