# Modelo final v1 - SVM com n-gramas de caracteres e palavras

## Resultado consolidado

O modelo escolhido foi uma `LinearSVC` treinada com a combinação de n-gramas
de caracteres, n-gramas de palavras e normalização TF-IDF.

No teste público `cc4051/pt_vid`, composto por 2.570 textos, ele alcançou:

| Métrica | Resultado |
| --- | ---: |
| Acurácia | 0,970039 |
| F1 macro | 0,970001 |
| F1 ponderado | 0,970043 |
| F1 PT-PT | 0,971064 |
| F1 PT-BR | 0,968939 |

O treinamento final reuniu 14.764 textos: 11.717 da partição pública de treino
e 3.047 textos locais com rótulo inequívoco. Outros 420 textos locais com
rótulo genérico `PT` foram excluídos.

## Como o modelo foi escolhido

O teste público **não foi usado para selecionar** a família, os parâmetros ou a
representação do modelo final. A seleção ocorreu com uma validação estratificada
interna na partição pública de treino e foi repetida com três sementes:

| Semente | F1 macro de validação |
| --- | ---: |
| 7 | 0,962771 |
| 42 | 0,961904 |
| 2026 | 0,967454 |
| Média | 0,964043 |

Após essa confirmação, o vencedor foi ajustado uma vez com todo o corpus de
desenvolvimento e avaliado no teste público.

## Limite metodológico a partir daqui

Uma linha de base preliminar já havia consultado esse mesmo teste público antes
do protocolo atual. Além disso, esta avaliação final expôs a métrica do SVM.
Portanto, o conjunto deve agora ser tratado como **congelado para comparação**:
novos ajustes, modelos neurais ou transformers não devem ser escolhidos com
base nele.

As próximas comparações devem usar somente validação interna ou uma fonte
externa, preferencialmente com domínio diferente. O teste público permanece
como referência histórica do modelo final v1, não como painel de otimização.

## Reprodução

No PowerShell, a partir da raiz do repositório:

```powershell
$env:HF_HOME = "$PWD\.hf_cache"
.\.venv\Scripts\python.exe -m scripts.audit_corpus
.\.venv\Scripts\python.exe -m scripts.run_classical_search
.\.venv\Scripts\python.exe -m scripts.validate_winner
.\.venv\Scripts\python.exe -m scripts.train_final_svm
```

Os artefatos ficam em `runs/`, fora do versionamento. O treino final grava
`runs/final_char_word_svm_v1/report.json`.
