#!/usr/bin/env bash
# Instala o ambiente local (venv Python + dependencias do frontend).
# Projetado para rodar em qualquer distribuicao Linux com bash, sem assumir
# gerenciador de pacotes, versao exata de Python ou ferramentas opcionais.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

PYTHON_MINIMO_MAIOR=3
PYTHON_MINIMO_MENOR=10
PYTHON_TESTADO="3.12"
NODE_MINIMO=18

erro() { printf '\nERRO: %s\n' "$1" >&2; exit 1; }
aviso() { printf 'AVISO: %s\n' "$1" >&2; }
etapa() { printf '\n==> %s\n' "$1"; }

# ---------------------------------------------------------------------------
# 1. Interpretador Python
# ---------------------------------------------------------------------------
versao_ok() {
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info >= ($PYTHON_MINIMO_MAIOR, $PYTHON_MINIMO_MENOR) else 1)" 2>/dev/null
}

# Só imprime o caminho e devolve status; as mensagens ficam no shell principal,
# porque um "exit" dentro de $( ) encerraria apenas o subshell.
encontrar_python() {
  local candidato
  # Prefere a versao testada; depois qualquer 3.x suportada.
  for candidato in python3.12 python3.13 python3.14 python3.11 python3.10 python3 python; do
    if command -v "$candidato" >/dev/null 2>&1 && versao_ok "$candidato"; then
      printf '%s' "$(command -v "$candidato")"
      return 0
    fi
  done
  return 1
}

etapa "Verificando o Python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  # Respeita um interpretador apontado explicitamente pelo usuario.
  command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || erro "PYTHON_BIN='$PYTHON_BIN' não foi encontrado no PATH."
  versao_ok "$PYTHON_BIN" \
    || erro "PYTHON_BIN='$PYTHON_BIN' é anterior ao Python ${PYTHON_MINIMO_MAIOR}.${PYTHON_MINIMO_MENOR} exigido."
  PYTHON="$(command -v "$PYTHON_BIN")"
else
  PYTHON="$(encontrar_python)" || erro \
"Nenhum Python >= ${PYTHON_MINIMO_MAIOR}.${PYTHON_MINIMO_MENOR} encontrado.
  Instale-o pelo gerenciador da sua distribuição, por exemplo:
    Debian/Ubuntu : sudo apt install python${PYTHON_TESTADO} python${PYTHON_TESTADO}-venv
    Fedora/RHEL   : sudo dnf install python${PYTHON_TESTADO}
    Arch          : sudo pacman -S python
  Ou aponte um interpretador existente: PYTHON_BIN=/caminho/python3 ./scripts/setup_local.sh"
fi

VERSAO_PYTHON="$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
echo "    $PYTHON (Python $VERSAO_PYTHON)"
case "$VERSAO_PYTHON" in
  "$PYTHON_TESTADO".*) ;;
  *) aviso "o projeto é validado no Python $PYTHON_TESTADO; $VERSAO_PYTHON deve funcionar, mas não foi testado." ;;
esac

# venv e pip precisam existir como modulos. No Debian/Ubuntu vêm em pacote
# separado, e a falha padrão do "python3 -m venv" é pouco informativa.
"$PYTHON" -c 'import venv' 2>/dev/null || erro \
"O módulo 'venv' não está disponível neste Python.
  Debian/Ubuntu: sudo apt install python3-venv
  Fedora/RHEL  : sudo dnf install python3-libs"
"$PYTHON" -c 'import ensurepip' 2>/dev/null || erro \
"O módulo 'ensurepip' não está disponível neste Python.
  Debian/Ubuntu: sudo apt install python3-venv
  (sem ele o venv é criado sem pip)"

# ---------------------------------------------------------------------------
# 2. Node.js e npm
# ---------------------------------------------------------------------------
etapa "Verificando o Node.js"
command -v node >/dev/null 2>&1 || erro \
"Node.js não encontrado (necessário >= $NODE_MINIMO para a interface).
  Instale pelo gerenciador da distribuição ou via https://nodejs.org
  Alternativa sem Node: use o Docker Compose na raiz do repositório."
command -v npm >/dev/null 2>&1 || erro "npm não encontrado (normalmente acompanha o Node.js)."

VERSAO_NODE="$(node --version | sed 's/^v//')"
MAIOR_NODE="${VERSAO_NODE%%.*}"
if [[ ! "$MAIOR_NODE" =~ ^[0-9]+$ ]] || (( MAIOR_NODE < NODE_MINIMO )); then
  erro "Node.js $VERSAO_NODE é anterior ao mínimo exigido ($NODE_MINIMO)."
fi
echo "    node $VERSAO_NODE / npm $(npm --version)"

# ---------------------------------------------------------------------------
# 3. Ambiente virtual
# ---------------------------------------------------------------------------
etapa "Preparando o ambiente virtual (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv || erro "Falha ao criar o ambiente virtual em .venv"
fi

# Os console scripts do venv (uvicorn, hf, accelerate...) gravam o caminho
# absoluto do interpretador no shebang. Se a pasta do projeto foi movida ou
# renomeada, eles quebram com "required file not found" — então reescrevemos
# qualquer shebang que aponte para um interpretador inexistente.
reparar_shebangs() {
  local reparados=0 script alvo
  for script in .venv/bin/*; do
    [[ -f "$script" && -r "$script" ]] || continue
    alvo="$(head -c 256 "$script" 2>/dev/null | sed -n '1s|^#!\([^ ]*\).*|\1|p')"
    [[ "$alvo" == *"/.venv/bin/python"* ]] || continue
    [[ -e "$alvo" ]] && continue
    sed -i "1s|^#!.*|#!$RAIZ/.venv/bin/python|" "$script"
    reparados=$((reparados + 1))
  done
  (( reparados > 0 )) && echo "    shebangs corrigidos em $reparados script(s) (projeto movido ou renomeado)"
  return 0
}
reparar_shebangs

VENV_PY="$RAIZ/.venv/bin/python"

# ---------------------------------------------------------------------------
# 4. Dependencias
# ---------------------------------------------------------------------------
# Sempre por "python -m", nunca pelos console scripts: imune a shebang quebrado.
etapa "Instalando dependências Python (pode demorar)"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

etapa "Instalando o modelo de português do spaCy"
"$VENV_PY" -m spacy download pt_core_news_sm

etapa "Instalando dependências do frontend"
npm --prefix frontend install

# ---------------------------------------------------------------------------
# 5. Diagnostico final
# ---------------------------------------------------------------------------
etapa "Diagnóstico"
"$VENV_PY" - <<'PY'
import importlib.util
import torch

print(f"    PyTorch {torch.__version__} | CUDA disponível: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("    Modo CPU: Qwen10/Qwen80 disponíveis; Llama exige GPU NVIDIA.")
if importlib.util.find_spec("uvicorn") is None:
    raise SystemExit("    ERRO: uvicorn não foi instalado corretamente.")
PY

printf '\nInstalação concluída. Execute: ./scripts/run_local.sh\n'
