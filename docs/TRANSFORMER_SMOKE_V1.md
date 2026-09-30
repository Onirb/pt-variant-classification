# Smoke test v1 - BERTimbau embeddings em CPU

## Objetivo

Verificar se o encoder pode ser usado localmente antes de executar uma
comparação completa. O teste não é elegível como resultado de seleção porque
utilizou uma amostra pequena do desenvolvimento interno.

## Configuração

| Item | Valor |
| --- | --- |
| Modelo | `neuralmind/bert-base-portuguese-cased` |
| Revisão | `94d69c95f98f7d5b2a8700c420230ae10def0baa` |
| Dispositivo | CPU |
| Parâmetros | 108.923.136 |
| Pesos estimados float32 | 415,51 MB |
| Lote | 16 |
| Máximo de tokens | 256 |
| Threads | 8 |
| Textos de treino / validação | 400 / 200 |

## Medição local

- Tempo total: **215,694 s**.
- Pico observado do processo: aproximadamente **1,2 GB** de RAM.
- F1 macro da amostra: **0,980000**.

O F1 do smoke test não é comparável diretamente ao experimento principal: a
amostra é pequena, balanceada e não foi desenhada para seleção estatística.

## Decisão operacional

Uma execução completa com 14.764 textos, mantendo lote 16 e 256 tokens, é
estimada em aproximadamente 80 a 90 minutos no CPU atual. A alternativa
eficiente é uma comparação amostral previamente fixada, ou executar o
fine-tuning/embedding completo em GPU remota. Nenhuma dessas alternativas foi
iniciada por este teste.
