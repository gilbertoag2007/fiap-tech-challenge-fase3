# Diretrizes do Repositório

## Visão Geral
Este repositório implementa um pipeline em Python 3.12 para leitura de Excel, validação de qualidade, detecção de PHI/PII, anonimização e preparação de dados para fine-tuning de LLM.

## Estrutura Atual
- `main.py`: ponto de entrada local para executar o menu do pipeline.
- `app/services/`: regras de negócio para ingestão e qualidade de dados.
- `app/data/raw/`: arquivo original, que deve permanecer inalterado.
- `app/data/processed/`: arquivos tratados após validação e limpeza.
- `tests/`: testes automatizados, quando aplicáveis.

## Arquivos de Dados
- O arquivo original `app/data/raw/dados_medicos_base_V3.xlsx` deve permanecer inalterado.
- O arquivo tratado deve ser salvo em `app/data/processed/dados_medicos_base_V3_tratado.xlsx`.
- O relatório de valores ausentes atualmente gerado pelo fluxo fica em `app/data/processed/missing_values_report.txt`.

## Comandos Úteis
Use o ambiente virtual do projeto antes de executar os comandos.

- `python main.py`: executa o menu local do pipeline.
- `python -m pytest`: executa a suíte de testes.
- `python -m pip install -r requirements.txt`: instala dependências quando o arquivo estiver disponível.
- `python -m pip freeze > requirements.txt`: registra o ambiente atual após mudanças nas dependências.

## Estilo de Código
Siga Python com indentação de 4 espaços, nomes em `snake_case` para funções, módulos e variáveis, e `PascalCase` para classes e modelos.

Mantenha as etapas do pipeline pequenas e explícitas. Prefira nomes claros e orientados ao domínio, como `detect_phi_pii`, `anonymize_records` e `build_training_dataset`.

Se houver formatação ou lint, aplique de forma consistente em `app/` e `tests/`.

## Comentários
Os trechos de código adicionados devem ter comentários objetivos e explicativos quando o fluxo não estiver óbvio.

## Idioma
Os comentários, classes, métodos e variáveis devem estar em portugues-BR sempre que possível. Matenha em inglês apenas o que for convenção da linguagem.

## Testes
Use `pytest` para testes unitários e de integração.

- Nomeie os arquivos como `test_*.py`.
- Mantenha cada teste focado em um comportamento principal.
- Priorize cobertura para leitura e normalização do Excel, regras de qualidade, saída da anonimização e transições do fluxo.
- Ao testar arquivos, prefira fixtures em `tests/fixtures/` em vez do dataset real.

## Commits e Pull Requests
Use mensagens curtas, diretas e em português, descrevendo a mudança feita.

Os pull requests devem incluir:

- resumo objetivo da alteração
- etapa do pipeline impactada
- evidências de teste ou capturas de tela quando necessário
- observações sobre impacto em PHI/PII ou nos dados de treino

## Segurança
Nunca comite datasets brutos sensíveis nem chaves de API.

Os dados originais devem ficar em `app/data/raw/`, e os resultados tratados devem ser gravados em `app/data/processed/` ou `app/data/training/` somente após validação.
