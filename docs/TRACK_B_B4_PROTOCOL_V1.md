# Track B B4 — execução final e auditoria antes da inferência

Estado: protocolo operacional registrado antes de previsões no teste final.
Não há novo candidato, limiar ou treinamento nesta etapa.

## Fonte oficial e recuperação de rótulos

O repositório [DSL-TL](https://github.com/LanguageTechnologyLab/DSL-TL)
fornece `DSL-TL-test.tsv` sem rótulos nem IDs, com 1.290 textos dos três idiomas.
Na mesma revisão há `PT_withFeatures.tsv`, com 4.953 IDs, textos, marcadores e
`New.Gold.Label`, além dos 3.467 IDs de treino e 991 IDs de desenvolvimento.
Revisão fixada: `44a083029be0c2fa7f304323908e808215c8eee1`.

Recuperação determinística, sem inferir idioma/variante por origem ou pelo modelo:

1. Exigir que os 4.458 IDs de treino/dev tenham exatamente os mesmos textos e
   rótulos em `PT_withFeatures.tsv`.
2. Selecionar os 495 IDs restantes; nenhum pertence ao treino/dev oficial.
3. Exigir correspondência exata e única desses textos ao teste multilíngue.
4. Preservar ID, número da linha oficial e rótulo humano. Não usar os marcadores
   linguísticos como entrada do classificador.
5. Excluir os 59 rótulos `PT` (ambos/nenhum), conforme o protocolo binário inicial.
   Restam 299 PT-BR e 137 PT-PT, 436 textos.

Existe uma frase que corresponde a dois IDs no arquivo completo de anotações.
A desambiguação é feita pela pertença oficial aos splits, **não** pelo primeiro
match de texto ou pela preferência de rótulo. Todos os bytes e hashes de fontes
e da visão recuperada serão registrados. A recuperação é uma derivação local
do teste oficial, não um novo conjunto anotado por nós.

Fonte metodológica: [Zampieri et al., 2024](https://aclanthology.org/2024.lrec-main.882/)
e a documentação dos autores na revisão fixada. O corpus é jornalístico,
com anotação humana. A exclusão dos rótulos ambíguos corresponde à tarefa binária.

## Auditoria lexical congelada antes das previsões

Normalização: Unicode NFKC, casefold e espaços normalizados, preservando acentos.
Comparar todos os textos elegíveis com o ajuste Track B exato, importação bruta
WMT já vista, desenvolvimento/teste histórico, DSL-TL dev, PtBrVId valid,
FRMT histórico e PT_train oficial. Ler textos históricos para auditoria não
autoriza usar resultados FRMT para escolha de modelo.

- Igualdade exata normalizada: sempre sinalizada.
- Quase-duplicatas: cosseno de conjuntos binários de char 5-grams >= 0,90,
  exigindo pelo menos 80 caracteres nos dois textos.
- Complemento: Jaccard de conjuntos de word 5-grams >= 0,80, exigindo pelo menos
  seis shingles únicos nos dois textos.

São limiares operacionais conservadores, escolhidos e registrados **sem olhar
predições ou métricas finais**; não existe limiar universal validado para este
corpus. As matrizes esparsas examinam os pares, em blocos de 32 consultas.
A indexação lexical usa textos não rotulados e não altera o tokenizer ou os
pesos do classificador. Implementação apoiada na documentação de
[TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
e [similaridade cosseno](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

Paráfrases e contaminação do pré-treino não são medidas por essa auditoria.
Não declarar independência total apenas por ausência de duplicatas lexicais.

## Coortes e regra de segurança

- Primária: os 436 textos binários oficiais, preservando a seleção original.
- Sensibilidade pré-definida: o subconjunto sem qualquer sinal lexical contra
  fontes anteriores. Reutiliza as mesmas previsões; não envolve segundo treino,
  segunda inferência ou filtragem por erros/confiança.
- Havendo sinal de sobreposição com o **corpus efetivamente usado no ajuste
  Track B**, interromper antes da inferência e revisar. Sobreposição histórica
  é reportada e refletida na coorte de sensibilidade, sem esconder o resultado
  na população oficial.

IDs das coortes, regras e hashes são congelados no relatório pré-inferência.
Isso não faz da sensibilidade outro teste independente; seu perfil pode diferir
do teste completo, especialmente se as exclusões não forem equilibradas.

## Execução única e encerramento

Verificar hashes dos dados, da auditoria e do artefato B3. Usar somente o texto,
em CPU com quatro threads e decisão argmax existente. Registrar a execução
antes da primeira previsão. Gravar previsões por ID para retomada de uma
interrupção, sem refazer os itens já persistidos. Execução concluída não pode
ser repetida ou sobrescrita.

Reportar F1 macro, accuracy, precision/recall/F1 por classe, diferença entre
recalls e matriz de confusão com ordem PT-PT, PT-BR. Domínio único jornalístico;
rótulos humanos. Não inventar métricas de outros domínios ou tratar logits como
probabilidade. Depois de ver os resultados, encerrar a seleção e congelar o
teste: resultado negativo não autoriza buscar outro checkpoint ou calibrar
usando esse teste.
