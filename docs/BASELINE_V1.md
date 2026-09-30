# Baseline reproduzível v1

## Pergunta

É possível distinguir pt-BR de pt-PT com um método simples, transparente e
avaliado em uma divisão que não participou do treino?

## Configuração

- treino: 11.717 exemplos públicos de `cc4051/pt_vid` mais 3.047 linhas
  inequivocamente rotuladas do `PT_train.tsv`;
- exclusão registrada: 420 linhas locais com rótulo genérico `PT`;
- teste: 2.570 exemplos públicos de `cc4051/pt_vid`, nunca usados no treino;
- vetorização: TF-IDF de n-gramas de caracteres de 2 a 5;
- classificador: regressão logística (`C=4`, `random_state=42`).

## Resultado

| Métrica | Valor |
|---|---:|
| Accuracy | 0,957588 |
| F1 macro | 0,957547 |
| F1 ponderado | 0,957598 |
| F1 pt-PT | 0,958852 |
| F1 pt-BR | 0,956242 |

O artefato e o relatório detalhado ficam apenas em
`runs/tfidf_char_baseline_v1/`; são deliberadamente ignorados pelo Git.

## Interpretação

Variação ortográfica e morfológica contém sinal suficiente para uma solução
baseada em caracteres atingir desempenho alto. Qualquer LSTM, CNN ou modelo
transformer posterior deve ser comparado contra esta baseline no mesmo teste e
justificar aumento de custo, latência ou complexidade.
