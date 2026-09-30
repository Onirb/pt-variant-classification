# Track B — plano de generalização entre variantes do português

## Objetivo

Investigar, de forma separada do Track A, por que o classificador distingue bem
os dados internos, mas tende a chamar ambas as traduções regionais de pt-BR no
FRMT. O objetivo não é perseguir a métrica FRMT: é criar um modelo mais robusto
a domínio e testar essa hipótese em uma fonte final que ainda não tenha sido
observada.

## Perguntas científicas

1. O viés para pt-BR vem do corpus de desenvolvimento, do formato de tradução
   ou da representação do encoder?
2. Técnicas de redução de pistas espúrias (entidades, URLs, formatos e termos de
   origem) melhoram a generalização sem prejudicar a separação linguística?
3. Fine-tuning supervisionado do BERTimbau supera embeddings congelados quando
   a validação ocorre em outro domínio?
4. Há ganho consistente em textos naturalmente produzidos e em traduções, ou a
   melhoria ocorre apenas em um tipo de dado?

## Regras de isolamento

- FRMT é somente registro histórico de Track A: proibido em treino, validação,
  calibração e escolha de modelo do Track B.
- Os artefatos do Track A são somente baselines de comparação; não serão
  sobrescritos.
- Track B terá diretórios próprios em `data/track_b/` e `runs/track_b/`, ambos
  ignorados pelo Git quando contiverem dados ou modelos.
- A fonte final de Track B deve ser escolhida e congelada **antes** de qualquer
  experimento de modelagem.

## Fase B0 — desenho e dados

1. Pesquisar duas novas fontes independentes: uma para desenvolvimento
   multidomínio e outra para teste final, com licenças e proveniência auditáveis.
2. Definir qual é a natureza de cada rótulo: humano, profissional por variante
   ou *silver* por metadados.
3. Medir duplicatas normalizadas e quase-duplicatas entre todas as fontes.
4. Fixar uma divisão por fonte/domínio, nunca uma divisão aleatória de textos do
   mesmo acervo.
5. Publicar o protocolo antes de treinar.

**Decisão necessária:** escolher a fonte de desenvolvimento e a fonte final
após a pesquisa B0. Nenhum dado novo deve ser baixado antes dessa decisão.

**Recomendação B0:** WMT24++ (`en-pt_BR` e `en-pt_PT`) para
desenvolvimento e DSL-TL `test` para avaliação final. A justificativa,
riscos e protocolo pré-registrado estão em
[`TRACK_B_B0_RECOMMENDATION.md`](TRACK_B_B0_RECOMMENDATION.md). Nenhum dado
novo deve ser baixado antes da aprovação explícita dessa escolha.

## Fase B1 — baselines explicativos

Executar os métodos abaixo na mesma divisão cruzada por domínio:

| Experimento | Hipótese testada | Custo esperado |
| --- | --- | --- |
| B1.1 SVM char+word original | Referência de domínio | baixo, CPU |
| B1.2 SVM com mascaramento ampliado | Reduzir dependência de pistas de origem | baixo, CPU |
| B1.3 BERTimbau congelado + SVM | Separar efeito da representação | médio, CPU |
| B1.4 BERTimbau com fine-tuning | Adaptar representações à tarefa | alto; idealmente GPU |

Cada linha deve ser repetida em pelo menos três sementes quando houver
aleatoriedade. A promoção exige ganho médio e ausência de regressão material no
pior domínio, não apenas uma média agregada maior.

**Estado atual:** B1.1, B1.2 e B1.3 foram concluídos. Consulte
[TRACK_B_B1_RESULT_V1.md](TRACK_B_B1_RESULT_V1.md). B1.4 foi concluído no
Colab e auditado em [TRACK_B_B1_4_RESULT_V1.md](TRACK_B_B1_4_RESULT_V1.md):
F1 macro médio 0,877068, melhoria média nos quatro domínios e treino somado
de 7min18s na T4. A diferença de recall entre PT-BR e PT-PT aumentou para
14,07 pontos percentuais; essa meta de redução da assimetria permanece aberta.

## Fase B2 — análises de erro

Para cada candidato, produzir:

- matriz de confusão por domínio e classe;
- distribuição das margens, sem chamá-las de probabilidades;
- amostra estratificada de falsos positivos e falsos negativos;
- auditoria de entidades, URLs, números, marcas e nomes geográficos;
- análise de pares paralelos apenas se a fonte os contiver, sem misturá-los com
  textos independentes na mesma interpretação estatística.

O objetivo é diferenciar erro linguístico de atalho de domínio.

**Estado atual:** concluída para B1.1--B1.3 em
[TRACK_B_B2_ERROR_ANALYSIS_V1.md](TRACK_B_B2_ERROR_ANALYSIS_V1.md). O domínio
social, textos curtos e a assimetria contra PT-PT são os principais pontos de
atenção.

## Fase B3 — seleção e artefato

1. Escolher o vencedor somente pelas fontes de desenvolvimento definidas em B0.
2. Ajustar uma única vez no conjunto de desenvolvimento completo.
3. Salvar manifesto, hashes, revisão do encoder, dependências e custo de
   execução.
4. Testar CLI/API local para entrada vazia, Unicode, texto longo e saída
   determinística.

**Estado atual:** B3 concluída. B1.4 ajustado em todos os 1.877 textos por
318 passos, semente 42; pesos importados e inferência offline verificada.
Os 18 testes locais passaram. Treino na T4: 3min06s; inferência CPU de textos
curtos: medianas entre 30 e 39 ms. Resultado e limites:
[TRACK_B_FINAL_MODEL_V1.md](TRACK_B_FINAL_MODEL_V1.md). Protocolo congelado:
[TRACK_B_FINAL_SELECTION_V1.md](TRACK_B_FINAL_SELECTION_V1.md).

## Fase B4 — avaliação final

1. Auditar sobreposição contra todos os dados de Track B antes da inferência.
2. Executar o teste externo ainda não observado uma única vez.
3. Reportar o resultado por domínio, classe e natureza do rótulo.
4. Encerrar a seleção; resultados negativos são documentação científica, não
   motivo para reabrir o teste.

**Estado atual:** concluída em 30/09/2026, com seleção encerrada. Auditoria
pré-inferência sem sobreposição com o ajuste Track B; teste oficial binário
de 436 textos com F1 macro 0,762398 e accuracy 0,768349. A sensibilidade em
435 textos produziu F1 0,761644, reutilizando as mesmas previsões. Recall
PT-PT 0,970803 e PT-BR 0,675585: a assimetria permanece e mudou de direção.
O ciclo foi executado, mas não atingiu todos os critérios científicos de sucesso.
Resultado e limites: [TRACK_B_B4_RESULT_V1.md](TRACK_B_B4_RESULT_V1.md).
Protocolo e recuperação do gold oficial:
[TRACK_B_B4_PROTOCOL_V1.md](TRACK_B_B4_PROTOCOL_V1.md).

## Critérios de sucesso

O Track B será considerado avanço apenas se:

- melhorar de modo consistente sobre o baseline em desenvolvimento cruzado por
  domínio;
- reduzir a assimetria entre recalls pt-BR e pt-PT;
- não depender de uma única fonte, tópico ou pista de proveniência;
- registrar uma avaliação final independente sem ajuste posterior.

## Recursos e ordem recomendada

Começar por B0 e B1.1–B1.3 em CPU, pois são baratos e permitem entender o
efeito de domínio antes de treinar redes. Só considerar B1.4 após demonstrar
que os baselines controlados não resolvem o problema; para fine-tuning, uma GPU
NVIDIA ou serviço temporário de GPU será mais adequado que a RX 580 local.
