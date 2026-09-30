# Track B B1.4 — resultado Colab auditado

Concluído e auditado em 30 de setembro de 2026. Fonte: ZIP fornecido pelo
usuário, track_b_b1_4_results_v1.zip. SHA-256:
1f5fb1dfe14f80f5fa45e89450802b8bed7dec90bd6e514e97380f8a97a8d12c.

## Validação da execução

As sementes 7, 42 e 2026 concluíram três épocas e 279 passos cada. O hash do
pacote remoto corresponde exatamente ao pacote local. Cada validação contém
398 textos; treino contém 1.479. Identidades, textos, classes, domínio e
segmentos foram comparados com as predições salvas de B1.1, B1.2 e B1.3.
Todas as validações correspondem. F1, accuracy, matriz de confusão e métricas
por domínio/classe foram recalculados a partir das predições e conferidos.
O teste curto foi excluído da comparação científica.

## Comparação do desenvolvimento

| Método | F1 macro médio | Desvio populacional entre sementes |
| --- | ---: | ---: |
| B1.1 char+word SVM | 0,797927 | 0,009782 |
| B1.2 SVM mascarado | 0,803765 | 0,010356 |
| B1.3 BERTimbau congelado + SVM | 0,815561 | 0,013612 |
| B1.4 BERTimbau fine-tuning | 0,877068 | 0,003049 |

B1.4 melhora B1.3 em 6,1507 pontos percentuais de F1 macro. Por semente,
o F1 macro B1.4 foi 0,873554 (7), 0,880989 (42) e 0,876659 (2026).
O desvio pequeno descreve somente estas três execuções; não é intervalo de
confiança nem prova de estabilidade fora do corpus.

| Domínio | B1.3 | B1.4 | Diferença em pontos percentuais |
| --- | ---: | ---: | ---: |
| literary | 0,885710 | 0,944425 | +5,87 |
| news | 0,844933 | 0,915961 | +7,10 |
| social | 0,739907 | 0,809132 | +6,92 |
| speech | 0,931806 | 0,946942 | +1,51 |

As médias melhoram nos quatro domínios. O menor F1 médio de domínio é social,
com 0,809132; este resultado também supera o social de B1.2 (0,769964).
Isso não significa que toda semente/domínio individual melhorou.

## Assimetria das classes

| Recall médio | B1.3 | B1.4 |
| --- | ---: | ---: |
| PT-PT | 0,785594 | 0,807370 |
| PT-BR | 0,845896 | 0,948074 |
| Diferença absoluta | 0,060302 | 0,140704 |

Ambos os recalls melhoraram, mas PT-BR melhorou muito mais. A diferença passou
de 6,03 para 14,07 pontos percentuais. Portanto, B1.4 é o candidato com melhor
desempenho agregado, mas não cumpriu a meta de reduzir a assimetria.
No agregado de observações entre sementes, a matriz de confusão é
[[482, 115], [31, 566]], com linhas reais/colunas previstas em ordem PT-PT,
PT-BR. As sementes contêm sobreposição de documentos: estes totais não são
uma amostra independente de 1.194 textos distintos.

Textos de até 40 caracteres: 45 erros em 128 observações (35,16%), frente
a 46,09% em B1.3. Houve melhora, embora a dificuldade de trechos curtos persista.

## Custo efetivamente observado

- GPU: Tesla T4, 14,56 GiB disponibilizados pelo runtime.
- Tempo somado de treino das três sementes: 438,08 segundos (7 min 18 s).
  A medida inclui validação/checkpoints dentro do treino; não é o tempo total
  de instalação, download, avaliação posterior, exportação ou sessão Colab.
- Pico de memória GPU alocada pelo PyTorch: 2,079 GiB.
- Maior memória GPU reservada pelo PyTorch: 2,289 GiB.
- Python 3.13.15; PyTorch 2.11.0+cu128; CUDA 12.8;
  Transformers 4.57.1; Accelerate 1.11.0.

As métricas de memória são do allocator PyTorch, não um monitor integral de
todos os processos do runtime. O experimento cabe confortavelmente na GPU
observada. Disponibilidade futura de T4 gratuita não é garantida.

## Estado dos artefatos e próximo passo

Relatórios e predições importados e preservados em
runs/track_b/b1_4_colab_import_v1. Auditoria reproduzível em
audit_and_comparison.json; importador em scripts/import_track_b_colab_results.py.

O ZIP não contém pesos. Os modelos por semente permanecem no Drive:
MyDrive/pt-variant-track-b/b1_4_colab_v1/full/seed_{7,42,2026}/model.
São modelos ajustados em partições de desenvolvimento, não o modelo final
treinado em todos os 1.877 textos.

B1.4 pode seguir como candidato principal para B3, com a assimetria documentada.
Antes de B4, é necessário congelar a regra de seleção e o orçamento de épocas
do ajuste final, produzir o artefato em todo o desenvolvimento e validar sua
inferência. A escolha do checkpoint usou validação (passo 100 na semente 7;
250 nas sementes 42 e 2026), portanto não se deve confundir o F1 de
desenvolvimento com desempenho externo confirmado. DSL-TL test continua
reservado.
