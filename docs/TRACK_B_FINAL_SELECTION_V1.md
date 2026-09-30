# Track B — seleção e ajuste final v1

## Estado e finalidade

B1.4 foi escolhido como candidato para avaliação final, não como demonstração
de que todos os objetivos do Track B foram cumpridos. O próximo ajuste usa todo
o desenvolvimento público (1.877 textos, 170 documentos). Os três modelos de
validação permanecem separados; nenhum deles é reaproveitado como ponto inicial.

**Estado em 30/09/2026: treino final concluído e pesos importados/verificados em
CPU. Avaliação externa posteriormente concluída e congelada.** Consulte
[TRACK_B_B4_RESULT_V1.md](TRACK_B_B4_RESULT_V1.md). Resultado B3, medições e proveniência em
[TRACK_B_FINAL_MODEL_V1.md](TRACK_B_FINAL_MODEL_V1.md).

## Evidência da seleção

O F1 macro médio em três sementes passou de 0,815561 (B1.3) para 0,877068
(B1.4). O F1 médio melhorou nos quatro domínios; social continua o mais difícil.
A diferença entre recall PT-BR e PT-PT, entretanto, aumentou de 6,03 para 14,07
pontos percentuais. Essa limitação permanece explícita: não se altera o limiar
ou o objetivo de treino para escondê-la.

Fontes locais: [auditoria B1.4](TRACK_B_B1_4_RESULT_V1.md),
`runs/track_b/b1_4_colab_import_v1/audit_and_comparison.json` e os três
`full/seed_*/report.json`. As validações são agrupadas por documento, com
estratificação de domínio e classe; não são validações leave-one-domain-out.

## Duração fixada antes do treino final

| Semente de validação | Melhor checkpoint | Passos/época | Épocas equivalentes |
| --- | ---: | ---: | ---: |
| 7 | 100 | 93 | 1,075269 |
| 42 | 250 | 93 | 2,688172 |
| 2026 | 250 | 93 | 2,688172 |

A regra escolhida é a **mediana das épocas equivalentes dos melhores checkpoints**.
É uma decisão de engenharia registrada, não uma garantia de duração ótima.
No corpus completo, batch efetivo 16 produz `ceil(1877/16) = 118` passos/época.
O orçamento final é `ceil(118 × 250/93) = 318` passos, aproximadamente 2,69 épocas.
Com um orçamento fixo, o artefato é o estado ao final desses 318 passos; não há
validação ou escolha adicional de checkpoint durante este ajuste.

- Encoder: `neuralmind/bert-base-portuguese-cased`.
- Revisão: `94d69c95f98f7d5b2a8700c420230ae10def0baa`.
- Inicialização: encoder original e cabeça de classificação nova.
- Semente final: 42, referência fixa, não escolha da semente com maior F1.
- Uma GPU; batch 8, acumulação 2, comprimento máximo 256 tokens.
- AdamW, learning rate 2e-5, weight decay 0,01, warmup 10%, FP16.
- Todos os parâmetros são ajustados; decisão por argmax dos dois logits.
- Logits e suas diferenças não são probabilidades calibradas.
- Checkpoints a cada 50 passos; retomada do último checkpoint disponível.

O funcionamento de `max_steps`, retomada e `save_pretrained` segue a
[documentação do Trainer, versão 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).

## Executar no Colab

1. Abra `colab/Track_B_Final_Colab.ipynb` no Colab e selecione GPU.
2. Envie `runs/track_b/colab_final_bundle_v1/track_b_final_package_v1.zip`.
3. Execute as células em ordem. Não é necessário repetir os três experimentos.
4. Baixe `track_b_final_receipt_v1.json` e `track_b_final_model_v1.zip`.

Drive: `MyDrive/pt-variant-track-b/final_colab_v1`. Os experimentos anteriores
não são sobrescritos. O modelo completo deve ocupar aproximadamente 400–450 MB;
o tamanho real constará do recibo. GPU gratuita depende da disponibilidade e dos
[limites do Colab](https://research.google.com/colaboratory/faq.html).

## Importação e inferência local

O importador confere recibo, hash e tamanho do ZIP, arquivos, revisão do encoder,
corpus, seleção congelada e conclusão dos 318 passos. Extrai para uma pasta
pendente e só registra o modelo definitivo após recarregar os pesos em CPU e
comparar a inferência das sondas com os escores exportados. Não sobrescreve
artefatos diferentes. Em falha, preserva a pasta pendente para diagnóstico.

```powershell
# Execute from your cloned repository root.
.\.venv\Scripts\python.exe -m scripts.import_track_b_final_model `
  "$env:USERPROFILE\Downloads\track_b_final_model_v1.zip" `
  "$env:USERPROFILE\Downloads\track_b_final_receipt_v1.json"
.\.venv\Scripts\python.exe -m scripts.predict_track_b --verify
.\.venv\Scripts\python.exe -m scripts.predict_track_b "Estou a estudar este problema."
```

Destino: `runs/track_b/final_model_v1`. A inferência usa somente arquivos locais,
CPU com quatro threads, safetensors, limite de tokens sinalizado e entrada vazia
rejeitada. Track A não é modificado.

## Limites e próxima etapa

As sondas de exportação testam funcionamento e consistência, não qualidade ou
acurácia. Os testes locais usam uma rede sintética minúscula e não representam
um novo treino ou resultado do BERTimbau. Em 30/09/2026, os 18 testes locais
passaram em modo offline, incluindo retomada de checkpoint, importação,
rejeição de pesos alterados, Unicode e truncamento. Nessa preparação o teste
DSL-TL ainda não havia sido consultado. A auditoria e a avaliação única foram
concluídas posteriormente em B4, com a seleção encerrada. FRMT não foi usado
para nova seleção. Qualquer novo ciclo exigirá outro teste não observado.

O ganho em desenvolvimento não demonstra, sozinho, generalização externa,
equilíbrio entre recalls ou independência de pistas de fonte.
