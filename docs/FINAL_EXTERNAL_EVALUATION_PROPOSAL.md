# Proposta de avaliação externa final - FRMT

## Decisão proposta

Usar exclusivamente a partição **test** do subconjunto português do
[FRMT](https://github.com/google-research/google-research/tree/master/frmt)
como a quarta e última avaliação externa do modelo final BERTimbau + LinearSVC.
Nenhum arquivo do FRMT será usado para treino, seleção de hiperparâmetros,
calibração ou definição de limiar.

## Por que FRMT

O FRMT contém traduções profissionais de sentenças da Wikipedia em inglês para
português do Brasil e de Portugal. Seu repositório informa 2.616 sentenças de
teste emparelhadas com cada variante, isto é, até 5.232 textos de classificação
após transformar cada tradução em uma instância com rótulo `PT-BR` ou `PT-PT`.
Os textos são separados por documento entre `exemplar`, `dev` e `test`.

Para esta aplicação, ele é complementar às avaliações já congeladas:

| Avaliação | Natureza do rótulo | Domínio principal |
| --- | --- | --- |
| DSL-TL | Anotação humana por falantes nativos | Jornalístico, texto naturalmente produzido |
| PtBrVId | *Silver labels* por metadados de origem | Seis domínios |
| FRMT proposto | Tradução profissional encomendada para cada variante | Tradução/Wikipedia, pares semânticos alinhados |

O FRMT também foi usado como conjunto de teste por pesquisa recente de
identificação de variedades em português. Isso torna a comparação metodológica
mais clara, sem reutilizar os conjuntos que já foram observados.

## Limites importantes

- Trata-se de texto traduzido, não de produção espontânea; portanto não mede
  sozinho a generalização para todos os registros naturais.
- O bucket `lexical` foi deliberadamente construído com termos que diferem entre
  as regiões. Avaliá-lo isoladamente superestimaria a capacidade geral.
- As duas traduções de uma mesma frase em inglês não são exemplos
  estatisticamente independentes. Por isso serão apresentados resultados
  agregados e também por par/tripla de origem quando aplicável.

O resultado deve ser reportado separadamente para `lexical`, `entity` e
`random`, além do resultado global. O conjunto `random` será a leitura mais
próxima de uma generalização não direcionada dentro desse benchmark.

## Protocolo congelado antes do download

1. Baixar somente os TSVs portugueses da partição `test`.
2. Extrair a coluna de tradução e criar rótulos pelo sufixo oficial do arquivo:
   `pt-BR` ou `pt-PT`.
3. Normalizar texto para auditoria e remover qualquer duplicata exata contra:
   corpus de desenvolvimento, teste público histórico, DSL-TL e PtBrVId.
4. Registrar linhas removidas, hashes, licença, revisão/origem e tamanhos por
   bucket e por classe.
5. Executar o modelo final uma única vez, sem alteração posterior do modelo.
6. Reportar accuracy, F1 macro, F1 por classe, matriz de confusão e métricas por
   bucket. A margem não será recalibrada com FRMT.
7. Se o conjunto ficar vazio ou houver sobreposição material, parar e registrar
   a limitação em vez de substituir dados silenciosamente.

## Fontes

- Repositório e estatísticas do FRMT:
  <https://github.com/google-research/google-research/tree/master/frmt>
- Artigo do benchmark:
  <https://arxiv.org/abs/2210.00193>
- Anúncio técnico do Google Research:
  <https://research.google/blog/frmt-a-benchmark-for-few-shot-region-aware-machine-translation/>
- Uso de FRMT como conjunto de teste em trabalho recente de variedades
  portuguesas:
  <https://ojs.aaai.org/index.php/AAAI/article/download/34704/36859>
