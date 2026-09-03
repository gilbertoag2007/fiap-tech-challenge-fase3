---
base_model: Qwen/Qwen3-0.6B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- pt
license: apache-2.0
tags:
- peft
- lora
- sft
- transformers
- trl
- qwen3
- medical
- portuguese
---

# Qwen3-0.6B Medical PT-BR LoRA

Adaptador LoRA experimental para o modelo
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), ajustado por
Supervised Fine-Tuning (SFT) para geração de respostas estruturadas em
português brasileiro a partir de perguntas e contextos médicos.

Este repositório contém apenas o adaptador PEFT/LoRA e o tokenizer utilizado
no treinamento. Os pesos do modelo-base não estão incluídos e são necessários
para executar a inferência.

> **Aviso:** este modelo é um artefato acadêmico e experimental. Ele não foi
> validado como dispositivo médico e não deve ser usado para diagnóstico,
> prescrição, definição de tratamento, triagem de emergência ou qualquer
> decisão clínica sem revisão de um profissional de saúde qualificado.

## Detalhes do modelo

| Propriedade | Valor |
|---|---|
| Modelo-base | `Qwen/Qwen3-0.6B` |
| Tipo | Adaptador PEFT/LoRA para modelo causal de linguagem |
| Idioma principal | Português brasileiro |
| Tarefa | Geração de texto em contexto médico |
| Método de ajuste | Supervised Fine-Tuning (SFT) com LoRA |
| Frameworks | Transformers, PEFT e TRL |
| Dispositivo de treinamento | CPU |
| Precisão de treinamento | `torch.float32` |
| Licença | Apache 2.0 |

## Uso pretendido

O adaptador foi desenvolvido para fins acadêmicos, incluindo:

- estudo de fine-tuning eficiente de modelos de linguagem;
- geração experimental de respostas estruturadas a partir de contexto médico;
- comparação entre as respostas do modelo-base e do modelo ajustado;
- análise manual de estrutura, relevância clínica, alucinação e exposição de
  informações pessoais.

O modelo deve receber um contexto suficiente para responder. Ele não substitui
fontes clínicas atualizadas, protocolos institucionais nem avaliação humana.

## Usos não recomendados

Não utilize este modelo para:

- tomar decisões clínicas de forma autônoma;
- diagnosticar doenças ou recomendar medicamentos e tratamentos;
- atender emergências;
- processar ou divulgar dados identificáveis de pacientes;
- produzir conteúdo médico apresentado como garantidamente correto;
- operar em produção sem validações técnicas, clínicas, éticas e de segurança.

## Dados de treinamento

O experimento utilizou uma amostra de 10% do conjunto de dados disponível no
projeto. Depois das etapas de preparação e filtragem, foram usados 1.303
exemplos, divididos de forma reproduzível com semente 42:

| Split | Exemplos | Proporção aproximada |
|---|---:|---:|
| Treino | 1.042 | 80% |
| Validação | 130 | 10% |
| Teste | 131 | 10% |

O conjunto de dados não é distribuído com este adaptador. O pipeline do projeto
inclui detecção e anonimização de PII em campos textuais selecionados. Esse
tratamento reduz riscos, mas não constitui garantia absoluta contra presença ou
memorização de PII/PHI. Uma revisão específica de privacidade é necessária antes
de disponibilizar o adaptador publicamente.

Os registros com quantidade de tokens igual ou superior ao limite configurado
foram removidos antes da divisão dos dados. Essa filtragem pode reduzir a
representação de casos mais longos ou complexos.

## Formato de treinamento

Cada exemplo foi convertido em uma conversa contendo:

1. mensagem `system` com as instruções da tarefa;
2. mensagem `user` com papel do solicitante, contexto da solicitação, contexto
   médico anonimizado e pergunta anonimizada;
3. mensagem `assistant` com a resposta esperada.

O treinamento utilizou pares conversacionais `prompt` e `completion`, com a
perda calculada somente sobre a resposta esperada (`completion_only_loss=True`).
O modo de raciocínio do Qwen3 foi desativado com `enable_thinking=False`.

## Configuração do fine-tuning

| Hiperparâmetro | Valor |
|---|---:|
| Épocas | 3 |
| Learning rate | `1e-4` |
| Tamanho do lote por dispositivo | 1 |
| Acumulação de gradiente | 8 |
| Comprimento máximo | 512 tokens |
| Otimizador | `adamw_torch` |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0,05 |
| Módulos-alvo | `q_proj`, `v_proj` |
| Bias | `none` |
| Semente | 42 |

Foram ajustados 2.293.760 dos 598.344.000 parâmetros do modelo, equivalentes a
aproximadamente 0,383% do total.

## Resultados

Os resultados abaixo foram medidos no split de validação do experimento com 10%
dos dados:

| Métrica | Modelo-base | Adaptador LoRA |
|---|---:|---:|
| Loss de validação | 2,5002 | 0,6778 |
| Perplexidade | aproximadamente 12,18 | 1,97 |
| Acurácia média por token | 52,12% | 86,41% |
| Entropia | 1,5878 | 0,6783 |

Outras métricas do treinamento:

| Métrica | Valor |
|---|---:|
| Loss médio de treinamento | 0,7582 |
| Melhor loss de validação | 0,6778 |
| Passos de treinamento | 393 |
| Tempo de treinamento | aproximadamente 10h29min |

Essas métricas avaliam predição de tokens e não comprovam correção clínica. A
qualidade das respostas deve ser avaliada separadamente por profissionais
qualificados, considerando relevância clínica, alucinação, estrutura e eventual
exposição de PII/PHI.

## Distribuição de tokens

| Split | Mediana | Percentil 95 | Máximo |
|---|---:|---:|---:|
| Treino | 450 | 503 | 511 |
| Validação | 458 | 498 | 508 |
| Teste | 447 | 500 | 508 |

Nenhum dos exemplos selecionados ultrapassou o limite de 512 tokens.

## Como usar

Instale as dependências:

```bash
python -m pip install "transformers>=4.56.2" "peft>=0.20.0" torch
```

Substitua `<USUARIO_OU_ORGANIZACAO>` pelo namespace do repositório no Hugging
Face Hub:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


modelo_base_id = "Qwen/Qwen3-0.6B"
adaptador_id = "<USUARIO_OU_ORGANIZACAO>/qwen3-0.6b-medical-ptbr-lora"

tokenizer = AutoTokenizer.from_pretrained(adaptador_id)
modelo_base = AutoModelForCausalLM.from_pretrained(
    modelo_base_id,
    torch_dtype=torch.float32,
)
modelo = PeftModel.from_pretrained(modelo_base, adaptador_id)
modelo.eval()

mensagens = [
    {
        "role": "system",
        "content": "Responda de forma estruturada usando apenas o contexto fornecido.",
    },
    {
        "role": "user",
        "content": "Insira aqui um contexto médico sem dados identificáveis e a pergunta.",
    },
]

prompt = tokenizer.apply_chat_template(
    mensagens,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
entradas = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

with torch.inference_mode():
    tokens_gerados = modelo.generate(
        **entradas,
        max_new_tokens=384,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

inicio_resposta = entradas["input_ids"].shape[1]
resposta = tokenizer.decode(
    tokens_gerados[0][inicio_resposta:],
    skip_special_tokens=True,
)
print(resposta.strip())
```

Como este repositório contém somente o adaptador, o carregamento também baixa
ou utiliza do cache os pesos do modelo-base `Qwen/Qwen3-0.6B`.

## Limitações e riscos

- O treinamento utilizou somente uma amostra de 10% dos dados disponíveis.
- O desempenho não foi confirmado em benchmarks clínicos externos.
- Acurácia por token não representa acurácia médica ou factual.
- O modelo pode alucinar, omitir informações importantes ou apresentar
  recomendações incorretas com linguagem convincente.
- O limite de 512 tokens pode prejudicar contextos extensos.
- A filtragem por comprimento pode introduzir viés em favor de casos menores.
- O modelo pode reproduzir vieses presentes nos dados e no modelo-base.
- A anonimização automatizada pode falhar; nunca forneça PII/PHI sem controles
  adicionais de privacidade.
- O comportamento fora do português brasileiro e do formato de treinamento não
  foi avaliado.

## Considerações éticas e clínicas

Todo uso envolvendo pessoas reais deve preservar confidencialidade, finalidade,
necessidade e controle de acesso. As respostas precisam ser revisadas por um
profissional habilitado. Antes de qualquer implantação, recomenda-se executar
avaliação clínica cega, análise de privacidade, testes de segurança, investigação
de vieses e monitoramento contínuo.

## Modelo-base e licença

Este adaptador foi treinado sobre o
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), distribuído sob a
licença Apache 2.0. O usuário também é responsável por verificar direitos,
restrições e requisitos aplicáveis aos dados e ao cenário em que pretende usar
este artefato.

## Citação

Caso este artefato seja utilizado em trabalho acadêmico, cite o projeto e o
modelo-base Qwen3. Preencha os dados abaixo antes da publicação:

```bibtex
@misc{qwen3_medical_ptbr_lora,
  title        = {Qwen3-0.6B Medical PT-BR LoRA},
  author       = {NOME DO AUTOR OU EQUIPE},
  year         = {2026},
  howpublished = {Hugging Face Hub},
  url          = {URL DO REPOSITORIO}
}
```

## Contato

Preencha antes da publicação:

- Responsável: `NOME DO AUTOR OU EQUIPE`
- Contato: `EMAIL OU URL DO PROJETO`
