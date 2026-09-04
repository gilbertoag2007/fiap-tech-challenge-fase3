# Diretrizes do Repositório

## Visão Geral

Este repositório implementa um pipeline em Python 3.12 para leitura de Excel, validação de qualidade, detecção e anonimização de PHI/PII, preparação de dados, fine-tuning com LoRA e comparação entre as inferências do modelo-base e do modelo ajustado.

A aplicação é executada pelo console. O menu, os avisos e os resumos das etapas utilizam a biblioteca Rich, com cores, painéis, tabelas, ícones e alternativa ASCII para terminais com codificação limitada.

## Status Atual

Status revisado em 02/09/2026 a partir do código-fonte, sem leitura ou inspeção de `app/data`.

### Capacidades implementadas no código

1. Seleção do percentual de registros no início da execução, com validação entre 0% e 100% e garantia mínima de três registros para os splits de treino, validação e teste.
2. Leitura do arquivo Excel original e geração de um `DataFrame` com amostragem reproduzível.
3. Identificação de registros repetidos e valores ausentes nas colunas monitoradas, com relatório Excel consolidado antes e depois do tratamento.
4. Remoção de registros repetidos e incompletos, com gravação atômica do arquivo de auditoria.
5. Identificação de PII em todo o dataframe já selecionado na etapa 2, sem uma segunda amostragem, com Microsoft Presidio, o modelo spaCy `pt_core_news_sm` e um reconhecedor específico de CPF.
6. Anonimização da pergunta original e do contexto de prontuário, incluindo tratamento específico de nome, CPF, telefone e data.
7. Preparação do dataframe de fine-tuning no formato de conversa `system`, `user` e `assistant`, com divisão reproduzível em 80% treino, 10% validação e 10% teste.
8. Validação dos campos obrigatórios, splits, identificadores, conversas repetidas e quantidade de tokens antes do treinamento.
9. Conversão dos registros em datasets conversacionais `prompt` e `completion` do Hugging Face, com perda calculada somente sobre a resposta esperada.
10. Inferência-base dos registros reservados no split `teste`, usando as mensagens completas `system` e `user` com `Qwen/Qwen3-0.6B`.
11. Supervised Fine-Tuning com TRL e adaptador LoRA aplicado às projeções `q_proj` e `v_proj`.
12. Persistência local do adaptador, tokenizer, checkpoints, métricas de treino e validação e relatório técnico do fine-tuning.
13. Inferência com o modelo ajustado e consolidação das respostas base, ajustada, esperada e avaliações manuais em um único arquivo.
14. Carregamento do modelo somente a partir do cache local do Hugging Face e execução configurada para CPU com `torch.float32`.

### Menu e ordem das etapas

O menu de `main.py` apresenta as etapas numeradas de 2 a 10:

- Etapa 2: leitura do arquivo Excel.
- Etapa 3: identificação de registros repetidos e ausentes.
- Etapa 4: tratamento das inconsistências.
- Etapa 5: identificação e tratamento de PII.
- Etapa 6: preparação do dataframe para fine-tuning.
- Etapa 7: inferência com o modelo-base.
- Etapa 8: fine-tuning com LoRA.
- Etapa 9: inferência com o modelo ajustado.
- Etapa 10: comparação das inferências.

Os atalhos do menu são:

- Opção 0: executa as etapas 2 a 6 para preparar os dados.
- Opção 1: executa as etapas 7 a 10 para treinar e avaliar.
- Opção 11: encerra a aplicação.

O treinamento permanece uma ação explícita: a opção 0 não executa inferências nem fine-tuning.

### Pendências e cuidados conhecidos

- O treinamento completo com o dataset real deve ser iniciado manualmente e pode ser demorado por utilizar CPU.
- A avaliação clínica das respostas comparadas continua sendo manual e deve considerar estrutura, relevância, alucinação e exposição de PII.
- A identificação de PII analisa todas as colunas configuradas, mas o fluxo integrado anonimiza especificamente `pergunta_original` e `prontuario_contexto`. Antes de usar dados reais no treinamento, é necessário validar se `papel_solicitante`, `contexto_solicitacao` e `resposta_estruturada` também podem conter PII, pois esses campos participam das conversas de treino.
- `especialidade_medica` e `tipo_pergunta` são mantidas como metadados no arquivo preparado, mas não compõem atualmente o `prompt` nem a `completion` usados pelo `SFTTrainer`.
- A coluna `total_okens_fine_tunning` contém uma grafia legada preservada no código e nos artefatos existentes. Uma eventual correção exige migração coordenada das referências e dos arquivos preparados.
- Não existem testes automatizados por decisão do projeto. As validações devem ser executadas manualmente e sem inspecionar o conteúdo de `app/data` durante tarefas do ChatGPT.

## Estrutura Atual

- `main.py`: ponto de entrada local, interface Rich e orquestração das etapas 2 a 10.
- `analisar_tokens.py`: diagnóstico manual da distribuição de tokens por split; depende dos arquivos preparados em `app/data`.
- `app/services/arquivo_service.py`: leitura e amostragem de Excel, criação de TXT e gravação atômica de arquivos Excel com uma ou várias abas por meio de arquivo temporário.
- `app/services/qualidade_service.py`: análise e remoção de registros duplicados ou com campos monitorados ausentes e geração do relatório consolidado de qualidade.
- `app/services/pii_service.py`: detecção de PII com Presidio, reconhecedor de CPF e anonimização dos campos textuais.
- `app/services/fine_tuning_service.py`: preparação e validação dos datasets, treinamento LoRA, inferências, métricas, relatório técnico e comparação dos resultados.
- `app/modelos/`: adaptadores e checkpoints locais, ignorados pelo Git.
- `app/data/original/`: arquivo original, que deve permanecer inalterado.
- `app/data/processado/`: arquivos de auditoria e de preparação para fine-tuning.
- `app/data/relatorios/`: relatórios de qualidade, treinamento, inferências e comparação.
- `requirements.txt`: dependências do pipeline de dados, anonimização, interface de console e LLM.

## Modelo e Formato de Fine-Tuning

- Modelo-base: `Qwen/Qwen3-0.6B`.
- Dispositivo configurado: CPU com `torch.float32`.
- Carregamento do modelo: somente a partir do cache local do Hugging Face.
- Limite do prompt: 512 tokens.
- Inferências base e ajustada: até 384 novos tokens por padrão e limite padrão de três registros do split `teste`.
- Colunas do dataset preparado: `id_exemplo`, `system`, `user`, `assistant`, `especialidade_medica`, `tipo_pergunta`, `total_okens_fine_tunning` e `split`.
- A mensagem de usuário combina papel do solicitante, contexto da solicitação, prontuário anonimizado e pergunta anonimizada.
- Formato de treinamento: conversas `prompt`/`completion`, com perda calculada somente sobre a `completion`.
- Divisão dos dados: 80% treino, 10% validação e 10% teste, com seed 42 e pelo menos um registro em cada split.
- Registros cujo total de tokens não seja inferior ao limite configurado são removidos antes da divisão dos dados.
- Configuração LoRA inicial: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `q_proj` e `v_proj`.
- Configuração SFT inicial: 3 épocas, learning rate `1e-4`, lote 1, acumulação de gradiente 8 e seed 42.
- O split `teste` não participa do `SFTTrainer` e é usado apenas nas inferências comparativas.

## Arquivos de Dados e Relatórios

- O arquivo original `app/data/original/dados_medicos_base.xlsx` deve permanecer inalterado.
- O arquivo tratado deve ser salvo em `app/data/processado/dados_medicos_auditoria.xlsx`.
- O dataset preparado deve ser salvo em `app/data/processado/dados_medicos_fine_tuning.xlsx`.
- A qualidade é consolidada em `app/data/relatorios/relatorio_qualidade.xlsx`. A etapa 3 inicia o relatório e a etapa 4 o substitui pela visão completa de antes e depois do tratamento.
  - A aba `resumo` contém `tipo_inconsistencia`, `quantidade_antes`, `registros_removidos`, `quantidade_depois` e `resultado`.
  - A aba `ocorrencias` contém `momento`, `tipo_inconsistencia`, `linha_excel`, `id_registro`, `coluna` e `descricao`.
  - `quantidade_antes` e `quantidade_depois` contabilizam ocorrências; `registros_removidos` contabiliza linhas excluídas. Os valores podem diferir porque todas as cópias de uma duplicidade são registradas, mas uma é preservada, ou porque uma linha possui mais de um campo ausente.
- As inferências e a comparação são consolidadas em `app/data/relatorios/avaliacao_inferencias.xlsx`. A etapa 7 inicia um novo ciclo com a resposta-base, a etapa 9 inclui a resposta ajustada e limpa avaliações manuais anteriores, e a etapa 10 valida a consistência do mesmo arquivo.
  - Colunas de referência e respostas: `id_exemplo`, `split`, `system`, `user`, `resposta_esperada`, `resposta_inferencia_base` e `resposta_inferencia_fine_tuning`.
  - Colunas de avaliação manual: `avaliacao_estrutura`, `avaliacao_relevancia_clinica`, `avaliacao_alucinacao`, `avaliacao_exposicao_pii` e `observacoes`.
- Os relatórios separados de qualidade e de inferência usados anteriormente não fazem mais parte do fluxo atual.
- O adaptador LoRA é salvo em `app/modelos/qwen3_06b_lora/`.
- As métricas são salvas em `app/data/relatorios/metricas_fine_tuning.txt`.
- O relatório técnico é salvo em `app/data/relatorios/relatorio_tecnico_fine_tuning.xlsx`.
- As colunas adicionadas pelo fluxo de PII incluem `entidades identificadas`, `possui_pii`, `pergunta_original_anonimizado` e `prontuario_contexto_anonimizado`, quando aplicável.

## Comandos Úteis

Use o ambiente virtual do projeto antes de executar os comandos.

- `python -m pip install -r requirements.txt`: instala as dependências do projeto.
- `python -m spacy download pt_core_news_sm`: instala o modelo de português usado pelo Presidio.
- `hf download Qwen/Qwen3-0.6B`: baixa o modelo-base para o cache local usado pelas etapas de inferência e treinamento.
- `python main.py`: executa o menu local do pipeline.
- `python analisar_tokens.py`: executa manualmente o diagnóstico de tokens usando o dataset preparado.
- `python -m compileall main.py analisar_tokens.py app/services`: valida a sintaxe dos módulos sem executar o pipeline.
- `python -m pip freeze > requirements.txt`: registra o ambiente atual após mudanças nas dependências.

Use o comando atual `hf`; não use o comando descontinuado `huggingface-cli`.

Não execute `analisar_tokens.py` durante tarefas do ChatGPT, pois o script carrega arquivos dentro de `app/data`.

## Dependências Principais

- Manipulação de dados e Excel: `pandas`, `openpyxl` e `pyarrow`.
- Interface de console: `rich`.
- Detecção e anonimização: `presidio-analyzer`, `presidio-anonymizer` e spaCy.
- Modelos e fine-tuning: `torch`, `transformers`, `datasets`, `huggingface-hub`, `accelerate`, `peft`, `trl` e `evaluate`.
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
