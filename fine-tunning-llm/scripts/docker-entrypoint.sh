#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${APP_MODELS_DIR:-/models}"
if [[ ! -f "${APP_MODELS_DIR:-/models}/qwen3_06b_lora_cpu/adapter_config.json" ]]; then
  cp -a /app/app/modelos/qwen3_06b_lora_cpu "${APP_MODELS_DIR:-/models}/"
fi

exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 1
