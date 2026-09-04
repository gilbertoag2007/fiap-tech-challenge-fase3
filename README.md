# FIAP Tech Challenge — Fase 3

## Visão executiva

Este repositório entrega um demonstrador acadêmico local de apoio clínico: prepara dados tabulares, remove inconsistências, identifica e anonimiza PII/PHI, ajusta o `Qwen/Qwen3-0.6B` com LoRA e oferece uma consulta contextualizada sob revisão humana obrigatória. O assistente integra LangChain para compor o rascunho e LangGraph para impedir a liberação sem aprovação.

> **Aviso médico e de privacidade.** Não é um dispositivo médico e não deve ser usado para diagnóstico, prescrição, tratamento, triagem, prontuário real ou decisão clínica. Toda saída é probabilística, pode estar errada e requer profissional habilitado. Dados, modelos, adaptadores, checkpoints e logs reais ficam locais e fora do Git.

## Índice

- [Relatório técnico completo](fine-tunning-llm/README.md): requisitos, arquitetura, instalação, execução, fine-tuning, avaliação, segurança e roteiro de vídeo.
- [Diretrizes do subprojeto](fine-tunning-llm/AGENTS.md): regras de desenvolvimento, privacidade e validação.
- [Ponto de entrada](fine-tunning-llm/main.py): menu local de opções `0` a `12`.
- [Serviço de fine-tuning](fine-tunning-llm/app/services/fine_tuning_service.py), [chain](fine-tunning-llm/app/assistente/chain.py) e [fluxo LangGraph](fine-tunning-llm/app/assistente/fluxo.py): implementação principal.

## Escopo da entrega

| Eixo | Evidência no projeto |
| --- | --- |
| Fine-tuning | Dataset conversacional, split reprodutível, SFT/LoRA, inferência e comparação. |
| LangChain | Prompt estruturado, adaptador local do Qwen/LoRA e fontes determinísticas. |
| LangGraph | Estado, interrupção, decisão humana, aprovação/rejeição e checkpointer em memória. |
| Segurança | Anonimização com Presidio/spaCy, allowlist de campos, logs JSONL de metadados e interface Rich. |

## Início rápido

Execute os comandos dentro de [`fine-tunning-llm/`](fine-tunning-llm/). A sequência, os pré-requisitos dos artefatos e os cuidados para não expor dados clínicos estão detalhados no [relatório técnico](fine-tunning-llm/README.md#execução-pelo-menu-012).

```bash
cd fine-tunning-llm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download pt_core_news_sm
hf download Qwen/Qwen3-0.6B
python main.py
```

O download do modelo é uma preparação explícita do cache local; não há chave de API nem fallback remoto. Não versione qualquer resultado local produzido pelo pipeline.
