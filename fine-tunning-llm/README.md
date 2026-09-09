# `fine-tunning-llm` — aplicação do Tech Challenge Fase 3

Este diretório contém o código da aplicação: pipeline de dados e fine-tuning
(`main.py`), assistente LangChain/LangGraph (`app/assistente`), API FastAPI
(`app/api`) e interface React (`frontend`).

> **A documentação completa do projeto — incluindo a matriz de atendimento ao
> enunciado, execução com Docker Compose, execução manual, arquitetura,
> avaliação do modelo e política de segurança — está no
> [README da raiz do repositório](../README.md).**
>
> Mantenha um único README canônico para evitar instruções divergentes.

## Atalhos

```bash
# Stack completa via Docker Compose (execute na RAIZ do repositório)
cd .. && docker compose up --build            # interface em http://localhost:8080

# Execução manual assistida (a partir deste diretório)
./scripts/setup_local.sh                      # cria .venv e instala dependências
./scripts/run_local.sh                        # FastAPI :8000 + Vite :5173

# Pipeline de dados e fine-tuning pelo menu Rich
python main.py                                # opções 0–12

# 2ª rodada da comparação de chatbots (Qwen vs. Llama COM LangGraph)
python scripts/comparar_chatbots_langgraph.py # exige GPU para o braço Llama
```

## Mapa rápido do código

| Caminho                               | Responsabilidade                                                     |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `main.py`                           | Menu Rich: qualidade, anonimização, dataset, treino, avaliação, HITL |
| `analisar_tokens.py`                | Diagnóstico da distribuição de tokens do dataset preparado           |
| `scripts/comparar_chatbots_langgraph.py` | Comparação comportamental Qwen vs. Llama em três braços (modelo cru, modelo com anti-repetição e pipeline LangGraph) |
| `app/services/arquivo_service.py`   | Leitura, amostragem e gravação atômica de Excel                     |
| `app/services/qualidade_service.py` | Duplicidades, campos ausentes e relatórios                            |
| `app/services/pii_service.py`       | Detecção e anonimização de PII/PHI (Presidio + spaCy)               |
| `app/services/fine_tuning_service.py` | Dataset, SFT/LoRA, inferências e comparação                        |
| `app/assistente/repositorio.py`     | Consulta controlada ao Excel anonimizado (allowlist, ID único)        |
| `app/assistente/ferramentas.py`     | Tool LangChain `buscar_prontuario`                                    |
| `app/assistente/modelo_chat.py`     | Adaptador `BaseChatModel` para o LoRA local                           |
| `app/assistente/chain.py`           | Prompt, LCEL, seções obrigatórias e aviso determinístico            |
| `app/assistente/fluxo.py`           | Grafo LangGraph, `interrupt`, HITL e leitura do `InMemorySaver`        |
| `app/assistente/auditoria.py`       | Log JSONL sanitizado                                                   |
| `app/api/main.py`                   | Endpoints REST, stream SSE e rota de checkpoints                       |
| `frontend/src/App.tsx`              | Chat, grafo animado, modal HITL e painel de logs do `InMemorySaver`    |

Detalhamento das etapas do pipeline: [`FLUXO_PIPELINE.md`](FLUXO_PIPELINE.md).

## Configuração local

```bash
cp .env.example .env
```

`app/config.py` carrega esse `.env` automaticamente nas execuções sem Docker.
No Docker, as variáveis vêm do `docker-compose.yml` na raiz (que também lê este
arquivo, se existir).
