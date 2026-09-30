# Auditoria do legado

## Dados

O notebook original combina o treino de `cc4051/pt_vid` com
`data/PT_train.tsv`. A fonte pública tem rótulos inteiros, que foram validados
por exemplos linguísticos e pelo mapeamento do notebook: `0 = PT-PT` e
`1 = PT-BR`.

O TSV local contém três rótulos: `PT-PT`, `PT-BR` e `PT`. As linhas `PT` não
distinguem variante e o notebook as descartava por `map_label(...)=None` sem
registrar essa decisão. A reprodução mantém a exclusão, mas a informa no
relatório `runs/corpus_audit.json`.

O teste público não é combinado ao treino. Ele é preservado para avaliação
final, evitando medir desempenho no mesmo conjunto usado para ajuste.

## Pesos salvos

`models/modelLTSM.pth` e `models/modelCNN.pth` contêm ambos o mesmo conjunto de
chaves de uma LSTM: embedding `(110, 140)`, duas camadas LSTM e saída binária.
O arquivo chamado `modelCNN.pth` não contém camadas convolucionais.

Os pesos não são adequados para inferência confiável porque o notebook constrói
o vocabulário com `list(set(corpus.lower()))` e não salva a ordem resultante nem
o tokenizer. Como o índice de cada caractere no embedding depende dessa ordem,
não é possível recuperar a associação caractere -> índice apenas a partir dos
pesos.

## Consequência

Os resultados históricos permanecem como referência acadêmica. A nova linha de
base será treinada de forma determinística, com vocabulário, configuração,
semente, divisão e métricas salvos junto ao artefato.
