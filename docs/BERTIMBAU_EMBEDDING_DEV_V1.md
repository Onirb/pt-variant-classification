# BERTimbau embeddings + SVM - validação interna v1

## Resultado consolidado: três rodadas completas

O encoder `neuralmind/bert-base-portuguese-cased` gera embeddings por
mean-pooling, seguidos por `LinearSVC`. As três sementes usam o mesmo
protocolo interno, sem consulta ao teste público ou a qualquer fonte externa.

| Modelo / semente | F1 macro |
| --- | ---: |
| SVM mascarada (média: 7, 42, 2026) | 0,967036 |
| BERTimbau + SVM (7) | 0,965322 |
| BERTimbau + SVM (42) | 0,969615 |
| BERTimbau + SVM (2026) | 0,976006 |
| **BERTimbau + SVM (média)** | **0,970314** |

Configuração por rodada: 12.420 textos de treino, 2.344 de validação, lote 16,
máximo de 256 tokens e oito threads de CPU. Os tempos foram 24,2, 28,3 e
30,5 min, respectivamente. A amplitude entre as sementes foi 0,010684 F1.

## Interpretação

O BERTimbau + SVM superou a SVM mascarada em 0,003278 F1 macro médio. Em uma
das sementes (7), a SVM mascarada foi marginalmente superior (0,000433), mas
as outras duas apresentaram ganho de 0,005995 e 0,004273. Portanto, ele atende
ao critério operacional previamente definido: ganho médio de pelo menos 0,002,
sem queda isolada material.

**Decisão:** BERTimbau + SVM é promovido a candidato principal para o modelo
final. A decisão vale apenas para o desenvolvimento interno; os testes público
e externos seguem congelados e não serão reutilizados para ajuste.
