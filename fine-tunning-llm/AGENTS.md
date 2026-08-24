# Diretrizes do Repositório

## Visão Geral
Este repositório implementa um pipeline em Python 3.12 para leitura de Excel, validação de qualidade, detecção de PHI/PII, anonimização e preparação de dados para fine-tuning de LLM.

## Estrutura Atual
- `main.py`: ponto de entrada local para executar o menu do pipeline.
- `app/services/`: regras de negócio para ingestão e qualidade de dados.
- `app/data/original/`: arquivo original, que deve permanecer inalterado.
- `app/data/processado/`: arquivos tratados após validação e limpeza.
- `app/data/relatorios/`: relatorios gerados durante analise e tratamento dos dados.

## Arquivos de Dados
- O arquivo original `app/data/original/dados_medicos_base.xlsx` deve permanecer inalterado.
- O arquivo tratado deve ser salvo em `app/data/processado/dados_medicos_auditoria.xlsx`.
- O relatórios gerados pelo fluxo fica em `app/data/relatorios/<nome do relatorio definido no sistema>.txt`.

## Comandos Úteis
Use o ambiente virtual do projeto antes de executar os comandos.

- `python main.py`: executa o menu local do pipeline.
- `python -m pip install -r requirements.txt`: instala dependências quando o arquivo estiver disponível.
- `python -m pip freeze > requirements.txt`: registra o ambiente atual após mudanças nas dependências.

## Estilo de Código
Siga Python com indentação de 4 espaços, nomes em `snake_case` para funções, módulos e variáveis, e `PascalCase` para classes e modelos.

Mantenha as etapas do pipeline pequenas, simples, objetivas e explícitas. Prefira nomes claros e orientados ao domínio, como `detect_phi_pii`, `anonymize_records` e `build_training_dataset`.

## Comentários
Os trechos de código adicionados devem ter comentários objetivos e explicativos quando o fluxo não estiver óbvio.

## Idioma
Os comentários, classes, métodos e variáveis devem estar em portugues-BR sempre que possível. Matenha em inglês apenas o que for convenção da linguagem.

## Testes
Não crie testes unitários para esse projeto, mas teste internamente as implementações antes de dados como concluída.

## Commits e Pull Requests
Use mensagens curtas, diretas e em português, descrevendo a mudança feita.

Os pull requests devem incluir:

- resumo objetivo da alteração
- etapa do pipeline impactada
- evidências de teste ou capturas de tela quando necessário
- observações sobre impacto em PHI/PII ou nos dados de treino

## Segurança
Nunca comite datasets brutos sensíveis nem chaves de API.

Não leia, abra ou inspecione nenhum arquivo ou pasta dentro de `app/data` durante a execução das tarefas do chatGPT.

## Implementação
Crie apenas o métodos e classes solicitadas no chat. Não crie métodos ou funcionalidades adicionais sem ser solicitado.

Sempre que for solicitado alteração em um método, analise os impactos nos métodos que chamam o método a ser alterado.

