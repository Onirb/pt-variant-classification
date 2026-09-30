# Artefato final BERTimbau + LinearSVC v1

## Escopo

Este é o ajuste final do candidato selecionado por validação interna. Ele foi
feito uma única vez sobre todo o corpus de desenvolvimento e **não** consultou
o teste público nem os conjuntos externos DSL-TL e PtBrVId.

## Artefatos locais

Os arquivos estão em `runs/bertimbau_final_v1/` e são ignorados pelo Git:

| Arquivo | Conteúdo |
| --- | --- |
| `linear_svc.joblib` | LinearSVC ajustada sobre embeddings de todos os textos de desenvolvimento |
| `manifest.json` | Modelo, revisão, hash do corpus, configuração e tempo do ajuste |

O encoder não é duplicado: a inferência exige o cache local do BERTimbau na
revisão fixada `94d69c95f98f7d5b2a8700c420230ae10def0baa`.

## Configuração registrada

- Corpus: 14.764 textos (7.625 PT-BR, 7.139 PT-PT).
- Fontes de desenvolvimento: 11.717 textos `cc4051/pt_vid:train` e 3.047 de
  `PT_train.tsv`; 420 rótulos locais genéricos `PT` permaneceram excluídos.
- Encoder: `neuralmind/bert-base-portuguese-cased`, mean-pooling.
- Classificador: `LinearSVC(C=1.0, random_state=42)`.
- Execução conservadora: CPU, 4 threads, lote 8, blocos de checkpoint de 128
  textos.
- Duração desta execução: 2.368,247 s (39,5 min).
- Hash SHA-256 do corpus: `fae21001cf58a1d3ab85b4326880dc1670f08369148e6ed0ce6b301094e80215`.

## Validação operacional

A CLI foi exercitada com uma frase pt-BR e uma pt-PT, retornando as classes
esperadas. Isso valida carregamento e inferência local; não é uma nova métrica
de desempenho e não substitui avaliação independente.

## Limite científico

Este ajuste final fecha a seleção interna. Uma nova fonte externa ainda não
observada é necessária para a avaliação final de generalização e só deve ser
usada uma vez, sem retreinamento posterior orientado por seu resultado.
