# Comparacao chatbot: Qwen 0.6B (80% q/k/v/o) vs Llama 3.1 QLoRA

Gerado em: 2026-09-04T14:31:26

> **1a rodada — modelos isolados.** Medido sem LangChain, sem LangGraph e sem `repetition_penalty`/`no_repeat_ngram_size`. Para o comportamento do assistente entregue, ver `comparacao_chatbots_langgraph_20q.md`.

## Resumo automatico

| Modelo | Media secoes (0-4) | Media chars |
|---|---:|---:|
| Qwen3-0.6B 80% (q/k/v/o) | 3.0 | 1003 |
| Llama 3.1 8B Instruct QLoRA | 4.0 | 917 |

## P1 — Soja e puberdade precoce

### Pergunta

```
Papel do solicitante: Medico(a)
Contexto da solicitacao: Diagnostico
Prontuario: Paciente do Sexo Nao informado. Idade: 7. Exames/Resultado: Nao informado.
Pergunta: O suco de soja por um periodo pode causar puberdade precoce em criancas?
```

### Qwen3-0.6B 80% (q/k/v/o)

_secoes=4/4 | chars=851_

Resposta:
O suco de soja não causa puberdade precoce. O suco de soja é uma fonte de proteína e pode ser usado como um complemento alimentar para ajudar a prevenir a puberdade precoce. Consulte um nutricionista para saber como usar o suco de soja de forma saudável.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Puberdade Precoce.

Conduta/Orientação:
Recomenda-se avaliação por Nutricionista para correlação dos achados e definição individualizada da conduta. A confirmação diagnóstica deve ser baseada em história clínica, exame físico e exames complementares quando indicados.

Limitações:
Não foram informados exames ou resultados complementares. Não há informação sobre idade pessoal. Não há informação sobre respectivo status diagnóstico, confirmação diagnóstica ou avaliação individualizada da conduta.

### Llama 3.1 8B Instruct QLoRA

_secoes=4/4 | chars=863_

Resposta:
Não, o suco de soja não causa puberdade precoce. A puberdade precoce é um fenômeno complexo, com muitos fatores envolvidos, incluindo genéticos, hormonais e ambientais. O ideal é que você consulte um endocrinologista pediátrico para avaliar a causa da puberdade precoce e orientar a conduta.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Puberdade Precoce.

Conduta/Orientação:
Recomenda-se avaliação por Endocrinologista pediátrico para correlação dos achados e definição individualizada da conduta. A confirmação diagnóstica deve ser baseada em história clínica, exame físico e exames complementares quando indicados.

Limitações:
Não foram informados exames ou resultados complementares. Não há informação sobre tratamento ou terapia indicada. O registro está correlacionado à hipótese de Puberdade Precoce.

## P2 — Vacina COVID e leucopenia

### Pergunta

```
Papel do solicitante: Medico(a)
Contexto da solicitacao: Conduta e Tratamento
Prontuario: Paciente do Sexo Feminino. Idade: 12. Diagnosticos anteriores: leucopenia em acompanhamento.
Pergunta: Adolescente com leucopenia pode receber vacina contra COVID-19?
```

### Qwen3-0.6B 80% (q/k/v/o)

_secoes=2/4 | chars=1111_

Resposta:
Sim, pode. A vacina contra o COVID-19 é recomendada para todos os grupos de risco, incluindo adolescentes. A leucopenia, ou deficiência de leucócitos, é uma condição que pode aumentar o risco de infecções graves, como a COVID-19. Portanto, é importante que você siga as orientações do seu médico e siga as medidas de prevenção recomendadas para a vacinação contra o COVID-19. Se você tem leucopenia, é importante que você siga as orientações do seu médico e siga as medidas de prevenção recomendadas para a vacinação contra o COVID-19. Se você tem leucopenia, é importante que você siga as orientações do seu médico e siga as medidas de prevenção recomendadas para a vacinação contra o COVID-19. Se você tem leucopenia, é importante que você siga as orientações do seu médico e siga as medidas de prevenção recomendadas para a vacinação contra o COVID-19.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Leucopenia. A condição é apresentada no registro como diagnóstico confirmado. O histórico de leucopenia em acompanhamento pode ser relevante para a interpre

### Llama 3.1 8B Instruct QLoRA

_secoes=4/4 | chars=999_

Resposta:
Sim, pode. A vacina não interfere na leucopenia.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Leucopenia. Os diagnósticos anteriores de leucopenia em acompanhamento devem ser considerados na avaliação da condição atual.

Conduta/Orientação:
Recomenda-se avaliação por Hematologista pediátrico para correlação dos achados e definição individualizada da conduta. Qualquer início, suspensão ou ajuste de tratamento deve ser realizado após avaliação clínica e orientação do profissional responsável. Levar a lista de medicamentos, doses e horários de uso pode auxiliar na revisão terapêutica.

Limitações:
Não há informação sobre limitações de uso ou interações possíveis em relação à pergunta feita. Não há informação sobre se os medicamentos mencionados estão sendo usados atualmente. Não foram fornecidos dados sobre tolerabilidade ou efeitos adversos. A orientação remota não substitui anamnese, exame físico e avaliação clínica individualizada.

## P3 — Hipotireoidismo e drenagem linfatica

### Pergunta

```
Papel do solicitante: Medico(a)
Contexto da solicitacao: Procedimentos
Prontuario: Paciente refere hipotireoidismo em tratamento.
Pergunta: Paciente com hipotireoidismo pode fazer drenagem linfatica?
```

### Qwen3-0.6B 80% (q/k/v/o)

_secoes=4/4 | chars=828_

Resposta:
Sim, pode sim. A drenagem linfática é uma técnica de tratamento de hipotireoidismo. A drenagem linfática é uma técnica de tratamento de hipotireoidismo. A drenagem linfática é uma técnica de tratamento de hipotireoidismo. A drenagem linfática é uma técnica de tratamento de hipotireoidismo.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Hipotireoidismo.

Conduta/Orientação:
Recomenda-se avaliação por Endocrinologista para correlação dos achados e definição individualizada da conduta. Qualquer início, suspensão ou ajuste de tratamento deve ser realizado após avaliação clínica e orientação do profissional responsável.

Limitações:
Não há informação de limitação respectiva na resposta. A orientação remota não substitui anamnese, avaliação clínica e avaliação presencial.

### Llama 3.1 8B Instruct QLoRA

_secoes=4/4 | chars=1027_

Resposta:
Pode sim, mas é importante que o médico que irá realizar a drenagem esteja ciente do seu diagnóstico e do seu tratamento.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Hipotireoidismo. Os medicamentos em uso devem ser considerados na avaliação de resposta terapêutica, possíveis efeitos adversos e interações. O histórico de hipotireoidismo em tratamento é informação relevante para a avaliação atual.

Conduta/Orientação:
Recomenda-se avaliação por Endocrinologista para correlação dos achados e definição individualizada da conduta. Qualquer início, suspensão ou ajuste de tratamento deve ser realizado após avaliação clínica e orientação do profissional responsável. Levar a lista de medicamentos, doses e horários de uso para a consulta.

Limitações:
Não há informação sobre limitações de confiabilidade ou validade da informação. Não há informação sobre confirmação diagnóstica. O status de confirmação diagnóstica deve ser considerado na interpretação do quadro atual.

## P4 — Dor toracica

### Pergunta

```
Papel do solicitante: Medico(a)
Contexto da solicitacao: Diagnostico
Prontuario: Paciente adulto com dor no peito ha 2 horas, sudorese e falta de ar. Exames nao informados.
Pergunta: Como orientar esse caso de dor toracica aguda?
```

### Qwen3-0.6B 80% (q/k/v/o)

_secoes=1/4 | chars=1099_

Resposta:
A dor torácica aguda pode ser causada por vários fatores, como a dor de peito, dor de abdome, dor de estômago, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor de cabeça, dor

### Llama 3.1 8B Instruct QLoRA

_secoes=4/4 | chars=678_

Resposta:
O ideal é que o paciente seja avaliado por um cardiologista para que possa ser feito o diagnóstico correto e a conduta adequada.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Dor No Peito. Os medicamentos informados são: Furosemida, Amlodipino.

Conduta/Orientação:
Recomenda-se avaliação por Cardiologista para correlação dos achados e definição individualizada da conduta. A confirmação diagnóstica deve ser baseada em história clínica, exame físico e exames complementares quando indicados.

Limitações:
Não há informação sobre exames realizados. Não há informação sobre confirmação diagnóstica. O registro está inacabado.

## P5 — Pos-colecistectomia

### Pergunta

```
Papel do solicitante: Medico(a)
Contexto da solicitacao: Procedimentos
Prontuario: Paciente 19 anos, cirurgia de emergencia de vesicula ha poucos dias, com nauseas, vomitos e dor abdominal.
Pergunta: Solicita-se orientacao sobre esse quadro pos-operatorio.
```

### Qwen3-0.6B 80% (q/k/v/o)

_secoes=4/4 | chars=1124_

Resposta:
O quadro de refluxo gastroesofágico pode ser devido a uma cirurgia de emergência, como a cirurgia de uma vesícula hiperestática. O tratamento é devido a uma abordagem multidisciplinar, que inclui o acompanhamento com o cirurgião, o gastroenterologista e o nutricionista. O refluxo pode ser tratado com medicações, como antialérgicos, antipruriginantes, antihistamínicos, antialérgicos e antipruriginantes. O acompanhamento com o cirurgião é fundamental para o controle do refluxo e a prevenção de complicações.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Refluxo gastroesofágico. A condição é apresentada no registro como distâctica que ha de ser considerada antes de qualquer orientação terapêutica.

Conduta/Orientação:
Recomenda-se avaliação por Cirurgião do aparelho digestivo para correlação dos achados e definição individualizada da conduta. Qualquer início, suspensão ou ajuste de tratamento deve ser realizado após avaliação clínica e orientação do profissional responsável.

Limitações:
Não há informação sobre tratamento ou complicações associadas à condição

### Llama 3.1 8B Instruct QLoRA

_secoes=4/4 | chars=1016_

Resposta:
O melhor a ser realizado, neste caso é consultar um gastroenterologistahepato de sua confiança, a fim de esclarecer detalhadamente cada dúvida.

Considerações clínicas:
O caso deve ser interpretado no contexto da hipótese clínica de Cálculos na vesícula biliar. Os medicamentos em uso devem ser considerados na avaliação de resposta terapêutica, interações e possíveis efeitos adversos.

Conduta/Orientação:
Recomenda-se avaliação por Gastroenterologista para correlação dos achados e definição individualizada da conduta. A confirmação diagnóstica deve ser baseada em história clínica, exame físico e exames complementares quando indicados. Levar a lista de medicamentos, doses e horários de uso pode auxiliar na revisão terapêutica.

Limitações:
Não há informação sobre limitações de uso, interações medicamentosas ou orientações específicas para o paciente. Não há informação sobre confirmação do diagnóstico. A orientação remota não substitui anamnese, exame físico e avaliação clínica individualizada.
