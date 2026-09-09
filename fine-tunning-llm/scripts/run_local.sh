#!/usr/bin/env bash
# Sobe a API (FastAPI) e a interface (Vite) para desenvolvimento local.
# Portas configuraveis por ambiente: PORT_API (8000) e PORT_WEB (5173).
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

PORT_API="${PORT_API:-8000}"
PORT_WEB="${PORT_WEB:-5173}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"

erro() { printf '\nERRO: %s\n' "$1" >&2; exit 1; }

# 'wait -n' exige bash >= 4.3; sem ele usamos uma espera por sondagem.
BASH_TEM_WAIT_N=0
if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
  BASH_TEM_WAIT_N=1
fi

# ---------------------------------------------------------------------------
# Pre-condicoes
# ---------------------------------------------------------------------------
[[ -x .venv/bin/python ]] || erro "Ambiente não instalado. Execute ./scripts/setup_local.sh"

.venv/bin/python -c 'import uvicorn' 2>/dev/null \
  || erro "uvicorn não está instalado no .venv. Execute ./scripts/setup_local.sh"

[[ -d frontend/node_modules ]] \
  || erro "Dependências do frontend ausentes. Execute ./scripts/setup_local.sh"

# Teste de porta em bash puro: nem todo ambiente tem ss, lsof ou netstat.
# Se o bash tiver /dev/tcp desabilitado, a conexao falha e assumimos porta livre
# (degradacao segura: o uvicorn/vite reportam o conflito por conta propria).
porta_ocupada() {
  ( exec 3<>"/dev/tcp/127.0.0.1/$1" ) 2>/dev/null
}

for par in "API:$PORT_API:PORT_API" "frontend:$PORT_WEB:PORT_WEB"; do
  IFS=: read -r nome porta variavel <<<"$par"
  if porta_ocupada "$porta"; then
    erro "A porta $porta ($nome) já está em uso.
  Encerre o processo que a ocupa ou escolha outra: $variavel=<porta> ./scripts/run_local.sh
  Causa comum: uma execução anterior deixou processos ativos, ou a stack Docker está no ar
  (verifique com 'docker compose ps' na raiz do repositório)."
  fi
done

export PYTHONPATH="$RAIZ"

# Job control: cada processo em segundo plano vira líder do próprio grupo, o que
# permite encerrar também os filhos (o npm gera o vite como processo separado;
# matar apenas o npm deixaria o vite órfão segurando a porta).
set -m

# ---------------------------------------------------------------------------
# Processos
# ---------------------------------------------------------------------------
# Invocado como módulo, e não pelo console script .venv/bin/uvicorn: os console
# scripts gravam o caminho do interpretador no shebang, então renomear ou mover
# a pasta do projeto os quebra com "required file not found".
.venv/bin/python -m uvicorn app.api.main:app --host "$HOST_BIND" --port "$PORT_API" &
PID_API=$!

# --strictPort: falhar de forma visível se a porta estiver ocupada, em vez de
# subir em outra e contradizer a mensagem impressa abaixo.
npm --prefix frontend run dev -- --host "$HOST_BIND" --port "$PORT_WEB" --strictPort &
PID_FRONTEND=$!

encerrar() {
  trap - EXIT INT TERM
  local pid
  for pid in "${PID_API:-}" "${PID_FRONTEND:-}"; do
    [[ -n "$pid" ]] || continue
    # Mata o grupo inteiro; se o processo já morreu, tenta só o pid.
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
}
trap encerrar EXIT INT TERM

printf '\nFrontend: http://localhost:%s\n' "$PORT_WEB"
printf 'API:      http://localhost:%s/docs\n\n' "$PORT_API"

# Encerra tudo assim que qualquer um dos dois cair.
if (( BASH_TEM_WAIT_N )); then
  wait -n "$PID_API" "$PID_FRONTEND"
else
  while kill -0 "$PID_API" 2>/dev/null && kill -0 "$PID_FRONTEND" 2>/dev/null; do
    sleep 1
  done
fi
