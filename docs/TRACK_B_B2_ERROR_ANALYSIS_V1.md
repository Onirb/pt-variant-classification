# Track B B2 — análise de erros v1

**Status:** concluída em 30 de setembro de 2026. Esta análise só leu as
predições B1 existentes; não treinou modelos, não alterou os dados e não
acessou o teste final DSL-TL.

## Escopo

Foram analisadas 1.194 observações de validação por experimento: três
validações documentais, cada uma com 398 textos. As observações de sementes
diferentes podem conter documentos repetidos; por isso os totais servem para
comparação entre métodos, não como uma nova amostra independente.

## Taxa de erro agregada

| Experimento | Erros | Taxa de erro |
| --- | ---: | ---: |
| B1.1 char+word | 241 | 0,201843 |
| B1.2 mascarado | 234 | 0,195980 |
| B1.3 BERTimbau congelado | 220 | 0,184255 |

## Diagnóstico do B1.3

| Corte | Taxa de erro |
| --- | ---: |
| literary | 0,114198 |
| news | 0,154762 |
| social | 0,259649 |
| speech | 0,068182 |
| PT-PT real | 0,214405 |
| PT-BR real | 0,154104 |
| Até 40 caracteres | 0,460938 |
| 41 a 120 caracteres | 0,224824 |
| 121 a 300 caracteres | 0,144654 |
| Mais de 300 caracteres | 0,059190 |

### Interpretação

1. O ganho agregado do B1.3 não é uniforme: ele reduz bastante os erros em
   news e speech, mas social continua o domínio mais difícil.
2. Há assimetria residual: o modelo erra mais PT-PT real e tende a predizê-lo
   como PT-BR. Isso é compatível com o viés visto no Track A, mas não prova a
   mesma causa.
3. Textos curtos são a maior limitação. Em frases de até 40 caracteres, há
   pouca evidência ortográfica ou sintática para distinguir as variedades.
4. Menções e hashtags têm 29,6% de erro no B1.3, contra 17,9% na ausência
   delas. É um sinal a investigar, não uma conclusão: existem apenas 54
   observações desse tipo.
5. URLs, e-mails e identificadores em maiúscula são raros demais para uma
   conclusão causal.

## Margens

A média da margem absoluta do LinearSVC foi 2,189253 nos acertos e 0,950222
nos erros. Isso indica que erros tendem a estar mais próximos da fronteira de
decisão. A margem não é probabilidade calibrada e não será apresentada como
confiança sem uma etapa explícita de calibração.

## Consequência para a seleção

B1.3 é o melhor candidato até aqui em F1 macro agregado, mas B1.2 ainda é
competitivo em social. Portanto, a decisão não deve depender só da média:
antes da avaliação final, será preciso comparar o possível B1.4 com ambos,
considerando a pior performance por domínio e a assimetria entre PT-PT e PT-BR.

O arquivo local de erros do B1.3 está em:

runs/track_b/b2_error_analysis_v1/b1_3_errors.parquet
