# Track B B1 — baselines explicativos v1

**Status:** B1.1, B1.2 e B1.3 concluídos em 30 de setembro de 2026. O teste
final DSL-TL não foi baixado, lido ou usado.

## Protocolo comum

Os três experimentos usam a mesma visão preparada do WMT24++:

- 1.877 textos, 939 PT-PT e 938 PT-BR;
- três sementes: 7, 42 e 2026;
- StratifiedGroupKFold com cinco folds;
- uma validação por semente;
- nenhum document_id aparece ao mesmo tempo em treino e validação;
- domínio e classe compõem a estratificação, portanto cada validação contém
  literary, news, social e speech.

O primeiro piloto B1.1 não tinha essa estratificação por domínio e permanece
salvo apenas como registro. A comparação oficial abaixo usa B1.1 v2.

## Resultados agregados

| Experimento | F1 macro médio | Desvio entre sementes | Hipótese |
| --- | ---: | ---: | --- |
| B1.1 TF-IDF char+word mais LinearSVC | 0,797927 | 0,009782 | Referência lexical e ortográfica |
| B1.2 B1.1 com mascaramento estrutural | 0,803765 | 0,010356 | Reduzir pistas de URLs, números e formatos |
| B1.3 BERTimbau congelado mais LinearSVC | 0,815561 | 0,013612 | Representação contextual melhora generalização |

B1.2 melhora B1.1 em 0,005838 de F1 macro. B1.3 melhora B1.2 em 0,011796.
Essas comparações usam exatamente as mesmas sementes, dados e grupos.

## F1 macro médio por domínio

| Experimento | literary | news | social | speech |
| --- | ---: | ---: | ---: | ---: |
| B1.1 | 0,885487 | 0,664910 | 0,762869 | 0,885991 |
| B1.2 | 0,888545 | 0,670245 | 0,769964 | 0,893617 |
| B1.3 | 0,885710 | 0,844933 | 0,739907 | 0,931806 |

Leitura inicial:

- O mascaramento produz ganho modesto e estável, principalmente em social.
- Os embeddings congelados trazem o maior ganho agregado, muito forte em news
  e speech.
- Social continua como o domínio mais difícil para B1.3. Isso impede concluir
  que a melhora agregada representa avanço uniforme; será investigado em B2.

## Custo e artefatos

B1.3 foi executado localmente em CPU, com quatro threads, lote de oito e
checkpoints de 128 textos. A extração e as três avaliações levaram 188,593
segundos. O modelo usado foi neuralmind/bert-base-portuguese-cased, revisão
94d69c95f98f7d5b2a8700c420230ae10def0baa, com mean pooling ponderado pela
attention mask.

Os relatórios e predições locais estão em:

- runs/track_b/b1_1_char_word_svm_v2
- runs/track_b/b1_2_masked_char_word_svm_v1
- runs/track_b/b1_3_bertimbau_frozen_svm_v1

## Próximo passo e decisão

B2 pode começar já: análise estratificada dos falsos positivos, falsos
negativos e margens, com foco no domínio social e em possíveis entidades,
formatos e pistas de proveniência.

B1.4, fine-tuning supervisionado do BERTimbau, não será iniciado
automaticamente. Ele requer uma escolha explícita de custo: executar em CPU,
mais lento e com maior carga local, ou planejar acesso a GPU. A decisão não
impede B2.
