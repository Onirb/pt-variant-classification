# Avaliação externa v1 - DSL-TL PT_dev

## Propósito

Esta avaliação verifica se o modelo final v1 generaliza para textos que não
fazem parte da competição original. O conjunto escolhido é `PT_dev` do DSL-TL,
um recurso oficial com rótulos humanos para variedades do português.

Ele é usado **somente para avaliação**. Não entra em treinamento, escolha de
parâmetros nem reclassificação de candidatos.

## Integridade da avaliação

| Verificação | Resultado |
| --- | ---: |
| Linhas brutas recebidas | 991 |
| PT-BR | 588 |
| PT-PT | 269 |
| PT genérico, excluído da tarefa binária | 134 |
| Linhas avaliadas | 857 |
| Duplicatas textuais normalizadas com o treino | 0 |

O arquivo local usado tem SHA-256
`ad4daca01bd290f1b147dbcd7f7edc106806ce0015b0dc03759289e08a5d9364`.

## Resultado

| Métrica | Valor |
| --- | ---: |
| Acurácia | 0,767795 |
| F1 macro | 0,727371 |
| F1 PT-PT | 0,622391 |
| F1 PT-BR | 0,832350 |

Matriz de confusão (linhas = classe real; colunas = predição):

| Real \ Predito | PT-PT | PT-BR |
| --- | ---: | ---: |
| PT-PT | 164 | 105 |
| PT-BR | 94 | 494 |

O desempenho é mais baixo que a referência do teste público da competição
(F1 macro 0,970001). Isso é evidência de sensibilidade a domínio e ao critério
de rotulagem: no DSL-TL, os casos sem marcador confiável podem ser marcados
como `PT` e são excluídos da avaliação binária.

Esse resultado não é uma nova seleção de modelo. Ele é um diagnóstico externo
para orientar a próxima pesquisa: analisar os erros, separar textos ambíguos e
comparar modelos por robustez entre domínios.

## Diagnóstico dos erros

Dos 857 textos binários, 199 foram classificados incorretamente. A margem é a
distância até a fronteira da SVM, e não uma probabilidade calibrada. Ainda
assim, ela ajuda a caracterizar incerteza:

| Grupo | Linhas | Mediana de caracteres | Mediana de margem absoluta |
| --- | ---: | ---: | ---: |
| Acertos totais | 658 | - | 0,641129 |
| Erros totais | 199 | - | 0,243074 |
| PT-PT classificado como PT-BR | 105 | 186,0 | 0,237275 |
| PT-BR classificado como PT-PT | 94 | 232,0 | 0,266797 |

Os erros ficam bem mais perto da fronteira de decisão. Textos pt-PT corretos
também têm margem menor que os pt-BR corretos, sugerindo que a separação atual
é menos estável para a variedade europeia neste domínio. Isso sustenta uma
etapa futura de abstenção ou encaminhamento de casos incertos, mas não autoriza
alterar o limiar sem uma nova validação independente.

## Reprodução

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_dsl_tl_external
$env:HF_HOME = "$PWD\.hf_cache"
.\.venv\Scripts\python.exe -m scripts.evaluate_dsl_tl_external
```

O conjunto externo é salvo em `data/external/` e o relatório executável em
`runs/dsl_tl_external_v1/`; ambos ficam fora do Git.
