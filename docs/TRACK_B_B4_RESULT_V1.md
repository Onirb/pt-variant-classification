# Track B B4 — avaliação externa final, congelada

**Estado em 30/09/2026: concluído; seleção encerrada.** Uma única inferência no
teste final DSL-TL português binário, com o artefato B3 e regra argmax congelados.
Não houve troca de checkpoint, calibração, retreinamento, commit ou publicação.

## Resultado principal

| Métrica | Teste oficial binário | Sensibilidade sem sobreposição histórica |
| --- | ---: | ---: |
| Textos | 436 | 435 |
| F1 macro | **0,762398** | 0,761644 |
| Accuracy | 0,768349 | 0,767816 |

| Classe real | N | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| PT-PT | 137 | 0,578261 | **0,970803** | 0,724796 |
| PT-BR | 299 | **0,980583** | 0,675585 | 0,800000 |

Matriz de confusão, linhas reais e colunas previstas:

| Real → prevista | PT-PT | PT-BR |
| --- | ---: | ---: |
| PT-PT | 133 | 4 |
| PT-BR | 97 | 202 |

335 acertos e 101 erros. Dos erros, 97 são textos brasileiros classificados
como europeus. A diferença `recall(PT-BR) − recall(PT-PT)` é **−0,295218**:
29,52 pontos percentuais a favor do recall PT-PT. A predominância de previsão
PT-PT neste conjunto contrasta com a assimetria observada no desenvolvimento B1.4,
que favorecia o recall PT-BR. Não afirmar que a meta de equilíbrio foi atingida.

## Interpretação científica

A média B1.4 de desenvolvimento era 0,877068; o F1 externo é 0,762398.
São populações diferentes e não uma comparação pareada: traduções/pós-edições
WMT24++ versus textos jornalísticos naturais com rótulos humanos DSL-TL.
O resultado mostra capacidade de discriminação externa nesta amostra, mas não
reproduz o desempenho médio interno nem o equilíbrio desejado entre variedades.

A mudança de direção da assimetria é **compatível com sensibilidade ao domínio
ou à construção do corpus**, mas esta avaliação não isola qual mecanismo a causa.
Não atribuir causalmente o efeito a tópico, comprimento, entidades ou tokenizer
sem experimento independente. Não há resultado Track A neste mesmo teste final,
portanto não alegar superioridade externa sobre Track A ou reproduzir diretamente
os números multilíngues do artigo DSL-TL.

## Recuperação rastreável do gold oficial

Revisão dos autores: `44a083029be0c2fa7f304323908e808215c8eee1`.
O teste oficial possui 1.290 textos sem rótulos nem IDs; as anotações portuguesas
completas estão em `PT_withFeatures.tsv`. Foram recuperados os 495 IDs ausentes
do treino/dev oficial, com correspondência exata ao teste multilíngue:

- 4.458 IDs de treino/dev coincidem em ID, texto e rótulo com o arquivo de gold.
- Os 495 IDs restantes são únicos e pertencem ao teste oficial por texto exato.
- 299 PT-BR, 137 PT-PT e 59 `PT` (ambos/nenhum).
- Os 59 casos ambíguos foram excluídos pelo protocolo binário, não pelos erros
  ou pela confiança do modelo.
- Marcadores anotados não foram usados como entrada; somente o texto bruto.

Essa recuperação local foi feita antes da inferência. Não envolve rótulos
gerados por modelo ou país de origem inferido. Fontes e hashes:
`data/external/dsl_tl_test_v1/manifest.json`.

## Auditoria pré-inferência

Os limiares foram registrados em [TRACK_B_B4_PROTOCOL_V1.md](TRACK_B_B4_PROTOCOL_V1.md)
antes das previsões. Igualdade normalizada, char 5-grams por cosseno e word
5-grams por Jaccard; regras lexicais, não detector semântico universal.

- Ajuste real Track B, 1.877 textos: **zero hits** nos três métodos.
- Importação WMT24++ já vista, 1.920 textos: zero hits.
- Teste público histórico, DSL-TL dev, PtBrVId valid e FRMT histórico: zero hits.
- Um único ID de teste (`174435`) tem quase-duplicata no desenvolvimento Track A
  e igualdade normalizada no PT_train oficial. A igualdade de texto não implica
  igualdade de IDs: esse caso pertence oficialmente ao teste.

A sensibilidade de 435 textos exclui esse ID por regra pré-definida e reutiliza
as mesmas previsões. O F1 permanece praticamente igual (0,761644); não há
evidência de que esse único caso explique o resultado agregado. A sensibilidade
não constitui outro teste independente. Nenhum resultado FRMT foi usado para
escolher pesos ou limiar nesta etapa.

## Execução e congelamento

- CPU, quatro threads; 436 previsões persistidas por ID, sem repetições.
- Tempo somado de inferência: 36,18 s; mediana por texto: 84,33 ms.
- Nenhum dos 436 textos foi truncado pelo limite de 256 tokens.
- RSS de 434,08 MiB foi amostrado **após liberar o classificador**; não é pico de
  RAM durante inferência. A checagem funcional B3 registrou outra amostra, com
  o modelo ativo; não interpretar a diferença como uma redução de memória.
- 30 testes locais passaram, incluindo recuperação do gold, rótulos ignorados
  pela auditoria, retomada por prefixo de IDs e guarda contra segunda inferência.

Artefatos locais ignorados pelo Git:

- `runs/track_b/b4_pre_inference_audit_v1/report.json`
- `runs/track_b/b4_pre_inference_audit_v1/frozen_membership.parquet`
- `runs/track_b/b4_dsl_tl_test_v1/execution_state.json`
- `runs/track_b/b4_dsl_tl_test_v1/predictions.jsonl`
- `runs/track_b/b4_dsl_tl_test_v1/report.json`
- `runs/track_b/b4_dsl_tl_test_v1/freeze_manifest.json`

Hash do teste recuperado:
`cc630d763ebb11ccf00f954a3cd5f8aa2df1ca8164254f26c27f0f6f0741af93`.
Hash da auditoria pré-inferência:
`4b051f23427cac0c684ceb048049392fb4709ea9f7443aa3e287d63ad030a1fa`.
Hash do manifesto do modelo:
`965dc4e16bce6aab3a6fe800ccfcdc2951184590aa03ffc6644f2cea5b2bbeea`.
Hashes do protocolo, script, previsões e relatório estão no estado de execução
e no manifesto de congelamento. Relatórios e previsões foram conferidos por
recontagem independente da matriz de confusão e do F1 macro.

## Limites e continuidade permitida

Este ciclo não cumpriu todos os critérios de sucesso do Track B: a assimetria
permanece, em sentido oposto, e não há comprovação de robustez em todos os domínios.
DSL-TL test pertence à mesma família do DSL-TL dev já consultado no Track A,
portanto é uma partição oficial não usada na seleção B, não uma coleta totalmente
independente. A auditoria não verifica contaminação no pré-treino, paráfrases ou
dependência de artigos/origens além dos dados fornecidos. A checagem final de
quase-duplicatas não substitui uma auditoria interna retroativa das validações B1.

Pode-se consolidar o relatório de portfólio e analisar erros descritivamente,
sem escolher novos parâmetros a partir deste teste. Qualquer novo ciclo de
melhoria exigirá validação apropriada e **outro teste ainda não observado**;
este resultado não volta a ser critério de seleção.

## Fontes

Resultados: relatório congelado e previsões locais acima; custos de importação
e execução do artefato em [TRACK_B_FINAL_MODEL_V1.md](TRACK_B_FINAL_MODEL_V1.md).
Desenvolvimento: [TRACK_B_B1_4_RESULT_V1.md](TRACK_B_B1_4_RESULT_V1.md).
Natureza dos rótulos e dados oficiais:
[Language Variety Identification with True Labels](https://aclanthology.org/2024.lrec-main.882/)
e [repositório dos autores](https://github.com/LanguageTechnologyLab/DSL-TL/tree/44a083029be0c2fa7f304323908e808215c8eee1).
