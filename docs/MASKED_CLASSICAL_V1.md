# Comparação interna v1 - mascaramento determinístico

## Hipótese

Reduzir pistas estruturais de contexto pode tornar o classificador menos
dependente de URLs, números, e-mails, menções, hashtags e identificadores.
Esta versão não executa NER: apenas substitui esses padrões por marcadores
determinísticos antes do TF-IDF.

## Protocolo

- Modelo base: TF-IDF de caracteres e palavras + `LinearSVC(C=1.0)`.
- Mesmas três sementes de validação: 7, 42 e 2026.
- Treino público com dados locais aceitos no lado de treino.
- Nenhum conjunto público final, DSL-TL ou PtBrVId foi consultado.

## Resultado

| Semente | SVM original | SVM mascarada |
| --- | ---: | ---: |
| 7 | 0,962771 | 0,965755 |
| 42 | 0,961904 | 0,963620 |
| 2026 | 0,967454 | 0,971733 |
| **Média** | **0,964043** | **0,967036** |

O ganho médio é de **0,002993 F1 macro**. Ele é pequeno, mas aparece em todas
as sementes. Isso justifica manter a hipótese no próximo conjunto de
desenvolvimento independente; não justifica promover o modelo nem reavaliar os
testes congelados.

## Reprodução

```powershell
$env:HF_HOME = "$PWD\.hf_cache"
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m scripts.run_masked_classical_dev
```
