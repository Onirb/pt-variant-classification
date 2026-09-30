# Protocolo v2 - comparações internas para robustez de domínio

## Pergunta

O modelo final v1 depende demais de termos contextuais, entidades ou marcas de
formato? Duas intervenções predefinidas serão comparadas apenas no ambiente de
desenvolvimento interno.

## Candidatos

1. `masked_char_word_linear_svm`: a SVM v1 recebe textos com URLs, e-mails,
   menções, hashtags, números e identificadores substituídos por marcadores
   genéricos. Não é NER completo; é uma redução determinística de pistas de
   contexto facilmente identificáveis.
2. `bertimbau_mean_pool_linear_svm`: embeddings mean-pooling do encoder
   `neuralmind/bert-base-portuguese-cased` na revisão
   `94d69c95f98f7d5b2a8700c420230ae10def0baa`, seguidos por `LinearSVC`.

## Regra de seleção

- A divisão continua sendo estratificada dentro do treino público, com os
  textos locais somente no lado de treino.
- O conjunto público da competição, DSL-TL PT_dev e PtBrVId valid não entram
  em treino, seleção, limiar ou comparação desses candidatos.
- O smoke test do transformer mede apenas compatibilidade, tempo e memória;
  não é resultado científico.
- Qualquer comparação externa posterior precisará de um quarto conjunto,
  ainda não observado por estas escolhas.

## Limitações antecipadas

O encoder escolhido é um modelo base em português e sua pré-formação pode não
ser neutra entre pt-BR e pt-PT. A comparação mede ganho prático no protocolo
interno, não demonstra por si só robustez entre variedades ou domínios.
