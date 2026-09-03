# Assistente médico com LangChain e LangGraph

## Contexto

O projeto já prepara dados médicos, anonimiza PII/PHI, realiza fine-tuning do
Qwen3-0.6B com LoRA e compara inferências. A próxima etapa deve transformar o
modelo ajustado em um assistente médico demonstrável, capaz de contextualizar
perguntas com registros estruturados e de organizar um fluxo seguro e
auditável.

Esta especificação cobre somente a integração com LangChain e LangGraph. Ela
não altera o treinamento, a anonimização nem os arquivos de dados existentes.

## Objetivos

- Integrar o Qwen3-0.6B com o adaptador LoRA local a uma chain do LangChain.
- Consultar o Excel anonimizado produzido pelo pipeline por identificador do
  registro, usando uma abstração de repositório substituível.
- Contextualizar a pergunta com os campos clínicos estruturados disponíveis.
- Coordenar consulta, geração, validação e revisão humana em um `StateGraph`.
- Impedir que um rascunho clínico seja apresentado como resposta final antes
  da decisão humana.
- Informar as fontes estruturadas usadas para compor o contexto.
- Manter um log de auditoria sem pergunta, prontuário, resposta ou outro texto
  clínico.
- Disponibilizar o fluxo no menu de terminal existente e documentá-lo no
  README.

## Fora de escopo

- Criar uma interface web ou uma API HTTP.
- Substituir o Excel anonimizado por um banco SQL.
- Usar OpenAI ou qualquer outro provedor remoto de LLM.
- Prescrever, executar condutas ou atualizar prontuários automaticamente.
- Treinar ou baixar o modelo durante a execução do assistente.
- Criar testes unitários versionados, conforme a diretriz do repositório.

## Restrições

- O modelo e o adaptador LoRA devem ser carregados apenas do ambiente local.
- A fonte clínica deve ser o arquivo de auditoria anonimizado já produzido pelo
  pipeline.
- Nenhuma chave de API deve ser necessária.
- O desenvolvimento e as verificações não podem abrir nem usar arquivos reais
  dentro de `app/data`.
- Logs não podem conter PII/PHI nem texto clínico, mesmo anonimizado.
- LangChain e LangGraph devem usar APIs estáveis da série 1.x.
- Nomes, mensagens e comentários próprios do projeto devem permanecer em
  português-BR sempre que possível.

## Arquitetura

### Repositório de prontuários

`RepositorioProntuarios` representa a fronteira de acesso aos dados
estruturados. A implementação `RepositorioProntuariosExcel` recebe um
`ArquivoService` e um caminho configurável, carrega o arquivo anonimizado sob
demanda e localiza uma única linha pelo campo `id`.

O resultado da consulta é um `RegistroClinico`, contendo o identificador, um
dicionário somente com campos permitidos e a lista dos nomes desses campos. A
lista permitida é `prontuario_contexto_anonimizado`, `hipotese_clinica`,
`diagnostico_confirmado`, `exames_relevantes`, `medicamentos_utilizados`,
`alergias`, `diagnosticos_anteriores` e `especialidade_medica`. Somente `id` e
`prontuario_contexto_anonimizado` são obrigatórios; campos opcionais
inexistentes ou vazios são descartados. O repositório nunca inclui as colunas
originais não anonimizadas de pergunta ou prontuário. A abstração permite que
a validação interna use um repositório sintético em memória sem acessar
`app/data` e que uma futura versão troque o Excel por SQLite.

### Adaptador LangChain para o modelo local

`ModeloChatQwenLocal` adapta a geração já oferecida por `FineTuningService` à
interface `BaseChatModel`. O adaptador converte mensagens LangChain em uma
mensagem de sistema e uma mensagem de usuário, carrega o modelo ajustado
somente quando necessário e devolve um `AIMessage` dentro de `ChatResult`.

O adaptador não implementa tool calling nem acesso remoto. Seu objetivo é
expor o Qwen ajustado como um chat model padrão, permitindo composição,
callbacks e rastreamento do LangChain sem duplicar a lógica de inferência.

### Chain clínica

`AssistenteChain` compõe:

1. um `ChatPromptTemplate` com política de apoio clínico e campos obrigatórios;
2. o `ModeloChatQwenLocal` ou outro `BaseChatModel` injetado;
3. um parser para texto simples;
4. uma normalização determinística que adiciona as fontes e o aviso de revisão.

O prompt exige as seções `Resposta`, `Considerações clínicas`,
`Conduta/Orientação` e `Limitações`. Ele informa que a saída é um rascunho,
proíbe afirmar que uma prescrição foi executada e limita a fundamentação ao
contexto estruturado fornecido. As fontes são produzidas pelo código a partir
dos nomes dos campos consultados, não pela LLM.

### Grafo do assistente

`FluxoAssistenteMedico` usa um estado tipado e os seguintes nós:

1. `validar_entrada`: normaliza o identificador e rejeita pergunta vazia;
2. `consultar_registro`: busca o registro anonimizado e registra as fontes;
3. `gerar_rascunho`: invoca a chain do LangChain;
4. `validar_seguranca`: verifica presença de resposta, fontes, limitações e
   aviso de revisão, registrando alertas determinísticos;
5. `solicitar_revisao_humana`: chama `interrupt` com um payload serializável
   que contém o rascunho, as fontes e os alertas;
6. `finalizar_aprovacao`: libera a resposta com situação `aprovada`;
7. `finalizar_rejeicao`: termina com situação `rejeitada`, sem liberar o
   rascunho como resposta final.

O grafo usa arestas condicionais após a revisão humana. Ele é compilado com
`InMemorySaver`, adequado à demonstração local, e cada execução recebe um
`thread_id` igual ao identificador da execução. A interface do serviço permite
iniciar o grafo, identificar a interrupção e retomá-lo com `Command(resume=...)`.

## Contratos

### Solicitação

`SolicitacaoAssistente` contém:

- `id_registro: str`;
- `pergunta_clinica: str`;
- `id_execucao: str`, gerado automaticamente quando não informado.

### Estado do grafo

`EstadoAssistente` mantém somente valores serializáveis:

- entrada normalizada;
- contexto clínico e nomes dos campos-fonte;
- rascunho;
- alertas de segurança;
- decisão e observação humanas;
- situação da execução;
- resposta final.

### Revisão pendente

`RevisaoPendente` é devolvida somente quando o grafo está interrompido e
contém `id_execucao`, `id_registro`, `rascunho`, `fontes`, `alertas` e `aviso`.
Esse contrato é destinado ao profissional que fará a revisão.

### Resposta final

`RespostaAssistente` contém:

- `id_execucao` e `id_registro`;
- `situacao`, com valor `aprovada` ou `rejeitada`;
- `resposta`, preenchida somente quando aprovada;
- `fontes` e `alertas`;
- `aviso`, informando que o assistente não substitui a decisão clínica.

Uma `RespostaAssistente` rejeitada não contém o rascunho. O serviço não expõe
o estado interno completo do grafo depois da decisão.

## Segurança e validação humana

A resposta da LLM é sempre um rascunho. Mesmo que a validação determinística
não encontre alertas, o grafo obrigatoriamente interrompe antes da finalização.
O revisor informa uma decisão estruturada com `aprovado: bool` e uma observação
opcional. Somente `aprovado: true` copia o rascunho para `resposta`.

O payload da interrupção deve ser exibido apenas no terminal da sessão do
revisor. Ele não é gravado no log de auditoria. O fluxo não possui nó capaz de
alterar prontuários, emitir receitas ou acionar procedimentos.

## Auditoria

`ServicoAuditoriaAssistente` grava eventos JSON Lines em
`app/data/relatorios/auditoria_assistente.jsonl` durante o uso real. Cada evento
contém somente:

- data e hora em UTC;
- identificador aleatório da execução;
- nome da etapa;
- situação;
- nomes dos campos-fonte;
- códigos dos alertas;
- decisão humana, quando disponível;
- tipo da exceção, em caso de falha.

Identificador do registro, pergunta, contexto, rascunho, resposta e observação
humana nunca são gravados. O caminho é injetável para que a validação use um
diretório temporário.

## Integração com o terminal

O menu ganha uma opção `11` para o assistente LangChain/LangGraph e a opção de
sair passa a ser `12`. O fluxo solicita o identificador e a pergunta, executa o
grafo até a interrupção, exibe rascunho, fontes e alertas ao revisor e solicita
aprovação ou rejeição. Depois da retomada, exibe apenas a resposta final
aprovada ou informa que a recomendação foi rejeitada.

Erros esperados são apresentados em linguagem clara e devolvem o usuário ao
menu sem encerrar o programa.

## Tratamento de erros

- Pergunta vazia ou identificador inválido: `ValueError` antes da consulta.
- Registro ausente ou duplicado: erro de domínio com identificação do problema.
- Colunas anonimizadas obrigatórias ausentes: erro de configuração dos dados.
- Modelo-base ou adaptador LoRA ausente: mensagem com a etapa que deve ser
  executada previamente.
- Falha de geração: evento de auditoria apenas com o tipo da exceção e nova
  mensagem de domínio sem conteúdo clínico.
- Decisão humana malformada: a retomada é recusada e nenhuma resposta é
  liberada.

## Documentação

O README da raiz será ampliado com visão geral do pipeline e apontará para as
instruções do assistente. Um README específico do projeto Python documentará
instalação, pré-requisitos, execução, exemplo seguro com dados sintéticos,
arquitetura, diagrama Mermaid, política de logs e limitações.

## Verificação

Não serão adicionados testes unitários ao repositório. A implementação será
verificada com dados sintéticos e dependências injetadas, fora de `app/data`,
cobrindo:

- criação e composição da chain;
- consulta de registro existente e erro de registro ausente;
- pausa obrigatória antes da resposta final;
- retomada com aprovação e com rejeição;
- presença das fontes e do aviso de limitação;
- rejeição de entradas e decisões inválidas;
- ausência de texto clínico no JSONL de auditoria;
- compilação de todos os módulos Python.

A validação do modelo real se limita à construção do adaptador e às interfaces,
pois baixar, treinar ou executar o Qwen não faz parte desta mudança.

## Critérios de aceitação

- O código importa e usa LangChain e LangGraph de forma funcional, não apenas
  declarativa.
- O modelo ajustado local pode ser invocado através de um `BaseChatModel`.
- O grafo consulta o registro, gera um rascunho, valida e sempre pausa para
  revisão humana.
- Uma rejeição nunca contém resposta final.
- Uma aprovação contém resposta final, fontes e aviso médico.
- O log é detalhado o suficiente para reconstruir as etapas e não contém texto
  clínico.
- O menu permite demonstrar o ciclo completo de aprovação e rejeição.
- O README contém instruções completas e um diagrama do fluxo.
- As verificações sintéticas e a compilação terminam sem erros.
