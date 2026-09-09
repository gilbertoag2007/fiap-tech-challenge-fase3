# Relatório técnico — Tech Challenge Fase 3 (Grupo 88)

Fonte LaTeX do relatório técnico exigido no enunciado, seguindo o esqueleto  (capa, resumo, sumário, seções numeradas,
tabelas, listings, figuras e referências).

| Arquivo                                | Conteúdo                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `relatorio_tech_challenge_fase3.tex` | Documento LaTeX completo e autocontido                                     |
| `relatorio_tech_challenge_fase3.pdf` | PDF de referência já compilado (26 páginas), gerado a partir do`.tex` |

## Compilar no Overleaf

1. Acesse [https://overleaf.com](https://overleaf.com) → **New Project** → **Upload Project** (ou
   **Blank Project** e faça upload do `.tex`).
2. Envie apenas `relatorio_tech_challenge_fase3.tex` — o documento é
   autocontido e **não depende de imagens externas**: todos os diagramas são
   desenhados em TikZ e o gráfico de métricas em pgfplots.
3. Em **Menu → Compiler**, selecione **pdfLaTeX**.
4. Compile **duas vezes** para que o sumário e as referências cruzadas
   (`\ref`) sejam resolvidos.

## Compilar localmente

```bash
pdflatex relatorio_tech_challenge_fase3.tex
pdflatex relatorio_tech_challenge_fase3.tex   # segunda passagem: sumário
```

Pacotes utilizados (todos presentes na distribuição padrão do Overleaf e do
TeX Live completo): `babel` (português), `geometry`, `fancyhdr`, `booktabs`,
`tabularx`, `enumitem`, `listings`, `xcolor`, `tikz` (bibliotecas
`arrows.meta`, `positioning`, `calc`, `fit`, `backgrounds`), `pgfplots`,
`amsmath`, `hyperref`.

## Estrutura do documento

| Seção | Conteúdo                                                                             |
| ------: | ------------------------------------------------------------------------------------- |
|       1 | Introdução, problema, objetivo e matriz de requisitos atendidos                     |
|       2 | Descrição do*dataset* (AKCIT/MedPT) e características dos dados                  |
|       3 | *Preprocessing*, curadoria e anonimização — **Figura 1**                   |
|       4 | Explicação do processo de*fine-tuning* (SFT + LoRA)                               |
|       5 | Descrição do assistente —**Figura 2** e **Figura 3** (fluxo LangChain) |
|       6 | Segurança, validação e explicabilidade                                             |
|       7 | Avaliação do modelo e análise dos resultados —**Figura 4**                  |
|       8 | Discussão crítica, limitações e conclusão                                        |
|      — | Referências                                                                          |

O conteúdo é derivado do [README do repositório](../README.md); ao alterar um,
mantenha o outro coerente.
