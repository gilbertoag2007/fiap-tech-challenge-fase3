# Diretrizes do Repositório

## Visão Geral

Este repositório implementa um pipeline em Python 3.12 para leitura de Excel, validação de qualidade, detecção de PHI/PII, anonimização, preparação de dados, fine-tuning com LoRA e comparação entre as inferências do modelo-base e do modelo ajustado.

## Status Atual

Status revisado em 29/08/2026 a partir do código-fonte, sem leitura ou inspeção de `app/data`.

### Capacidades implementadas no código

1. Leitura do arquivo Excel original e geração de um `DataFrame`.
2. Identificação de registros repetidos e valores ausentes nas colunas monitoradas, com geração de relatórios TXT.
3. Remoção de registros repetidos e incompletos, com gravação do arquivo de auditoria.
4. Identificação de PII em 100% do dataframe com Microsoft Presidio e o modelo spaCy `pt_core_news_sm`.
5. Anonimização da pergunta original e do contexto de prontuário, incluindo tratamento específico de nome, CPF, telefone e data.
6. Preparação do dataframe de fine-tuning no formato de conversa `system`, `user` e `assistant`, com divisão reproduzível em 80% treino, 10% validação e 10% teste.
7. Inferência-base dos registros reservados no split `teste` usando as mensagens completas `system` e `user` com `Qwen/Qwen3-0.6B`, executada em CPU a partir do cache local do Hugging Face.
8. Validação dos campos, splits, identificadores, conversas repetidas e quantidade de tokens antes do treinamento.
9. Conversão dos registros em datasets conversacionais `prompt` e `completion` do Hugging Face.
10. Supervised Fine-Tuning com TRL e adaptador LoRA aplicado às projeções `q_proj` e `v_proj`.
11. Persistência local do adaptador, tokenizer, checkpoints e métricas de treino e validação.
12. Inferência com o modelo ajustado e comparação lado a lado com o modelo-base e a resposta esperada.

O menu de `main.py` apresenta essas funcionalidades como etapas 1 a 9; a identificação e a anonimização de PII fazem parte da etapa 4. A opção 0 executa somente as etapas 1 a 6, mantendo o treinamento como uma ação explícita.

### Pendências conhecidas

- O treinamento completo com o dataset real deve ser iniciado manualmente e pode ser demorado por utilizar CPU.
- A avaliação clínica das respostas comparadas continua sendo uma atividade manual e deve considerar estrutura, relevância, alucinação e exposição de PII.
- A execução isolada da opção 1 do menu chama `executar_etapa_1` sem o argumento obrigatório `caminho_arquivo`; o fluxo completo da opção 0 informa o caminho corretamente.
- Não existem testes automatizados por decisão do projeto. As validações devem ser executadas manualmente e sem inspecionar o conteúdo de `app/data` durante tarefas do ChatGPT.

## Estrutura Atual

- `main.py`: ponto de entrada local, menu e orquestração das nove etapas do pipeline.
- `app/services/arquivo_service.py`: leitura de Excel, criação de Excel/TXT e atualização atômica de arquivos Excel por meio de arquivo temporário.
- `app/services/qualidade_service.py`: análise e remoção de registros duplicados ou com campos monitorados ausentes.
- `app/services/pii_service.py`: detecção de PII com Presidio, reconhecedor de CPF e anonimização dos campos textuais.
- `app/services/fine_tuning_service.py`: preparação e validação dos datasets, treinamento LoRA, inferências e comparação dos resultados.
- `app/modelos/`: adaptadores e checkpoints locais, ignorados pelo Git.
- `app/data/original/`: arquivo original, que deve permanecer inalterado.
- `app/data/processado/`: arquivos de auditoria e de preparação para fine-tuning.
- `app/data/relatorios/`: relatórios de qualidade e resultado da inferência-base.
- `requirements.txt`: dependências do pipeline de dados, anonimização e LLM.

## Modelo e Formato de Fine-Tuning

- Modelo-base: `Qwen/Qwen3-0.6B`.
- Dispositivo configurado: CPU com `torch.float32`.
- Carregamento do modelo: somente a partir do cache local do Hugging Face.
- Limite do prompt: 512 tokens; a inferência-base gera, por padrão, até 128 novos tokens.
- Colunas do dataset preparado: `id_exemplo`, `system`, `user`, `assistant`, `especialidade_medica`, `tipo_pergunta` e `split`.
- A mensagem de usuário combina papel do solicitante, contexto da solicitação, prontuário anonimizado e pergunta anonimizada.
- Formato de treinamento: conversas `prompt`/`completion`, com perda calculada somente sobre a completion.
- Configuração LoRA inicial: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `q_proj` e `v_proj`.
- Configuração SFT inicial: 3 épocas, learning rate `1e-4`, lote 1, acumulação de gradiente 8 e seed 42.
- O split `teste` não participa do `SFTTrainer` e é usado apenas nas inferências comparativas.

## Arquivos de Dados e Relatórios

- O arquivo original `app/data/original/dados_medicos_base.xlsx` deve permanecer inalterado.
- O arquivo tratado deve ser salvo em `app/data/processado/dados_medicos_auditoria.xlsx`.
- O dataset preparado deve ser salvo em `app/data/processado/dados_medicos_fine_tuning.xlsx`.
- Os relatórios de qualidade são salvos em `app/data/relatorios/` no formato TXT.
- A inferência-base é salva em `app/data/relatorios/inferencia_base.xlsx`.
- O adaptador LoRA é salvo em `app/modelos/qwen3_06b_lora/`.
- As métricas são salvas em `app/data/relatorios/metricas_fine_tuning.txt`.
- A inferência ajustada é salva em `app/data/relatorios/inferencia_fine_tuning.xlsx`.
- A comparação final é salva em `app/data/relatorios/comparacao_inferencias.xlsx`.
- As colunas adicionadas pelo fluxo de PII incluem `entidades identificadas`, `possui_pii`, `pergunta_original_anonimizado` e `prontuario_contexto_anonimizado`, quando aplicável.

## Comandos Úteis

Use o ambiente virtual do projeto antes de executar os comandos.

- `python -m pip install -r requirements.txt`: instala as dependências do projeto.
- `python -m spacy download pt_core_news_sm`: instala o modelo de português usado pelo Presidio.
- `hf download Qwen/Qwen3-0.6B`: baixa o modelo-base para o cache local usado pela etapa de inferência.
- `python main.py`: executa o menu local do pipeline.
- `python -m compileall main.py app/services`: valida a sintaxe dos módulos sem executar o pipeline.
- `python -m pip freeze > requirements.txt`: registra o ambiente atual após mudanças nas dependências.

Use o comando atual `hf`; não use o comando descontinuado `huggingface-cli`.

## Dependências Principais

- Manipulação de dados e Excel: `pandas`, `openpyxl` e `pyarrow`.
- Detecção e anonimização: `presidio-analyzer`, `presidio-anonymizer` e spaCy.
- Modelos e fine-tuning: `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `trl` e `evaluate`.
- Componentes auxiliares já declarados: `pydantic`, `python-dotenv`, `langchain`, `langchain-openai`, `langgraph` e `scikit-learn`.

## Estilo de Código

Siga Python com indentação de 4 espaços, nomes em `snake_case` para funções, módulos e variáveis, e `PascalCase` para classes e modelos.

Mantenha as etapas do pipeline pequenas, simples, objetivas e explícitas. Prefira nomes claros e orientados ao domínio, como `detectar_phi_pii`, `anonimizar_registros` e `construir_dataset_treinamento`.

## Comentários

Os trechos de código adicionados devem ter comentários objetivos e explicativos quando o fluxo não estiver óbvio.

## Idioma

Os comentários, classes, métodos e variáveis devem estar em português-BR sempre que possível. Mantenha em inglês apenas o que for convenção da linguagem ou nome externo de API, biblioteca, modelo ou campo técnico.

## Testes

Não crie testes unitários para este projeto, mas teste internamente as implementações antes de dá-las como concluídas. Nunca use os arquivos reais de `app/data` para testes executados durante tarefas do ChatGPT.

## Commits e Pull Requests

Use mensagens curtas, diretas e em português, descrevendo a mudança feita.

Os pull requests devem incluir:

- resumo objetivo da alteração;
- etapa do pipeline impactada;
- evidências de teste ou capturas de tela, quando necessário;
- observações sobre impacto em PHI/PII ou nos dados de treino.

## Segurança

Nunca comite datasets brutos sensíveis, artefatos com PII/PHI nem chaves de API.

Não comite adaptadores, checkpoints ou modelos gerados pelo treinamento. Não publique esses artefatos no Hugging Face sem revisão e autorização explícitas.

Não leia, abra, liste ou inspecione nenhum arquivo ou pasta dentro de `app/data` durante a execução das tarefas do ChatGPT.

O arquivo original deve permanecer inalterado. Toda transformação deve ser gravada apenas nos caminhos de processamento ou relatórios definidos pelo sistema.

## Implementação

Crie apenas os métodos e as classes solicitados no chat. Não crie métodos ou funcionalidades adicionais sem solicitação.

Sempre que for solicitada alteração em um método, analise os impactos nos métodos que o chamam.

Preserve alterações locais do usuário e não modifique arquivos fora do escopo solicitado.
