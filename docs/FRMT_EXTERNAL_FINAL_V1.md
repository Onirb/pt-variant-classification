# Avaliação externa final - FRMT test

## Protocolo

Esta avaliação foi executada uma única vez depois de congelar e ajustar o
artefato BERTimbau + LinearSVC. Não houve mudança de dados, hiperparâmetros,
limiar, calibração ou novo ajuste após a leitura do resultado.

- Fonte: [FRMT](https://github.com/google-research/google-research/tree/master/frmt),
  revisão `d36068b845da4c2b24927fee2cea1e6ef98dadda`, CC BY-SA 3.0.
- Partição: somente `test`, traduções profissionais em pt-BR e pt-PT.
- Corpus: 5.232 textos equilibrados, 2.616 por classe.
- Sobreposição textual normalizada com desenvolvimento, teste público, DSL-TL e
  PtBrVId: **zero** em todos os quatro casos.
- Artefato avaliado: `runs/bertimbau_final_v1/`, cujo manifesto é identificado
  pelo SHA-256 `95841eaa591f0bbebfebba5c3b545814723dd90728da04cb1baefcd8ffb15e9c`.

Os hashes individuais dos seis TSVs, o hash do conjunto elegível e a matriz de
confusão completa estão em `runs/frmt_external_final_v1/report.json`.

## Resultado global

| Medida | Resultado |
| --- | ---: |
| Accuracy | 0,624618 |
| F1 macro | **0,608864** |
| Recall PT-PT | 0,423930 |
| Recall PT-BR | 0,825306 |

Matriz de confusão, linhas reais e colunas previstas (`PT-PT`, `PT-BR`):

```text
[[1109, 1507],
 [ 457, 2159]]
```

O conjunto é balanceado, portanto a assimetria não vem da frequência das
classes: o modelo tendeu a prever PT-BR para muitas traduções de ambas as
variedades.

## Resultado por bucket

| Bucket | Textos | Accuracy | F1 macro |
| --- | ---: | ---: | ---: |
| `entity` | 1.970 | 0,603046 | 0,588693 |
| `lexical` | 1.748 | 0,642449 | 0,626519 |
| `random` | 1.514 | 0,632100 | 0,614784 |

Mesmo `lexical`, planejado para conter diferenças regionais explícitas, não
resolveu a assimetria. Isso é evidência contra uma explicação baseada apenas em
textos pouco distintivos.

## Análise por pares de tradução

O FRMT contém versões pt-BR e pt-PT da mesma sentença inglesa. Entre 2.608
pares unívocos, o comportamento foi:

| Padrão de previsões no par | Pares | Proporção |
| --- | ---: | ---: |
| Ambas previstas como PT-BR | 1.440 | 55,2% |
| Ambas previstas como PT-PT | 391 | 15,0% |
| Ambas corretas | 712 | 27,3% |
| Ambas invertidas | 65 | 2,5% |

Quatro grupos tinham a mesma frase inglesa repetida no arquivo e não foram
tratados como pares unívocos. A conclusão é que a transferência para traduções
paralelas regionais é limitada: em cerca de 70% dos pares válidos, o modelo não
separou as duas variantes e atribuiu a mesma classe às duas traduções.

## Interpretação e limite

O resultado não invalida as medições anteriores — cada uma mede domínios e
formas de rotulagem diferentes — mas impede apresentar o modelo como robusto
para todas as formas de pt-BR/pt-PT. Ele é forte no protocolo interno e teve
resultados diferentes em DSL-TL, PtBrVId e FRMT, o que reforça que identificação
de variedade é sensível ao domínio, à origem e ao estilo dos textos.

O FRMT é texto de tradução profissional, não escrita espontânea. Assim, ele não
é uma estimativa universal de uso real; ainda assim, por ser independente,
equilibrado e pareado, é uma evidência forte de que o modelo final possui viés
para PT-BR nesse cenário.

Nenhuma ação de modelagem posterior deve usar FRMT para selecionar parâmetros.
Qualquer continuação deve abrir um **Track B** claramente separado, com novos
dados de desenvolvimento e um teste independente ainda não observado.
