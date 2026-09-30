# Track B — modelo final importado e verificado

## Estado em 30/09/2026

**B3 concluída:** ajuste final no Colab, importação dos pesos e checagem funcional
em Windows/CPU. Na importação B3 não houve avaliação do teste reservado.
**Atualização: B4 concluída posteriormente no mesmo dia**, com F1 externo
0,762398; veja [TRACK_B_B4_RESULT_V1.md](TRACK_B_B4_RESULT_V1.md).
Não houve alteração nos artefatos do Track A, novo treino
local, calibração de limiar, commit ou publicação.

## Proveniência e protocolo conferidos

- Base: `neuralmind/bert-base-portuguese-cased`.
- Revisão: `94d69c95f98f7d5b2a8700c420230ae10def0baa`.
- Corpus: 1.877 textos de 170 documentos; hash do Parquet empacotado:
  `544ae5c4b4f9c75499f38a962f28119fc43b1fb984eca3fd468c2c6d79e996f9`.
- Semente 42, batch 8 × acumulação 2, limite de 256 tokens, uma GPU, FP16.
- Orçamento fixado: 318 passos; execução concluída: 318 passos.
- Épocas reportadas pelo Trainer: 2,697872; a regra do orçamento era aproximadamente
  2,69 épocas. O critério de conclusão é o orçamento de 318 passos; esse valor
  reportado não altera o protocolo congelado.
- Decisão: argmax dos logits, sem probabilidades calibradas.

A seleção exportada coincide com a seleção local congelada. O ZIP contém pesos
em safetensors, tokenizer, configuração, seleção e manifesto; não contém corpus
ou otimizador. Recibo e manifesto do ZIP coincidem. Todos os hashes internos,
hash e tamanho do ZIP, revisão do encoder e hash do corpus foram conferidos.

| Registro | SHA-256 |
| --- | --- |
| ZIP recebido | `b6b42776961d4ef0d38e0c82d7e2d34a9d353c5c6ef27c967832b3ae10ba0d2e` |
| Recibo recebido | `34efb7c2ef70bd9027ef4f6be85c123a84d01c9bd6f762242a1008d05e0e9332` |
| Manifesto importado | `965dc4e16bce6aab3a6fe800ccfcdc2951184590aa03ffc6644f2cea5b2bbeea` |
| Pesos safetensors | `825373a6676493c0e15273ed902081f7f463e8bc97e5dfbc6375d3f1b7e02e0c` |

Destino: `runs/track_b/final_model_v1`. A pasta `.pending` não permaneceu:
o registro definitivo ocorreu somente após verificação em CPU.

## Custo observado

Colab, conforme recibo e manifesto com integridade conferida:

- Tesla T4; treino medido em 185,873 s, aproximadamente **3min06s**.
- Pico de memória GPU alocada: 2,076 GiB; reservada: 2,285 GiB.
- ZIP: 404.652.232 bytes, aproximadamente 385,9 MiB.
- Artefato local completo: 436.616.374 bytes, aproximadamente 416,4 MiB.

Windows, inferência offline com quatro threads:

- Carregar pesos e conferir hashes: 0,730 s, sem incluir importação inicial das
  bibliotecas e inicialização do processo.
- Três textos curtos: medianas de 30,41 ms, 29,73 ms e 39,25 ms, cada um com sete
  repetições após aquecimento. São medições locais, não SLA ou benchmark geral.
- Texto de 14.099 caracteres: 314,89 ms; truncamento sinalizado.
- RSS do processo após as verificações: 760,23 MiB, incluindo bibliotecas.
  É uma amostra ao final, **não um pico de RAM monitorado continuamente**. RSS
  imediatamente após carregar era menor por mapeamento e materialização tardia
  dos pesos; não interpretar a diferença inicial como consumo total do modelo.

Relatório das medições e ambiente:
`runs/track_b/final_model_verification_v1/report.json`.
Ambiente local: Python 3.12.10, PyTorch 2.14.0+cpu, Transformers 4.57.1,
NumPy 2.5.3 e psutil 7.2.2. Ambiente de origem registrado no manifesto:
Python 3.13.15, PyTorch 2.11.0+cu128 e CUDA 12.8.

## O que foi testado

As três sondas Unicode reproduziram os logits exportados no Colab dentro da
tolerância numérica (`rtol=atol=1e-4`). A inferência foi determinística nas sete
repetições de cada sonda. Entradas vazias e tipos inválidos foram rejeitados;
texto longo sinalizou o corte em 256 tokens. O CLI de verificação e uma previsão
real foram executados em modo offline. A suíte local de 18 testes também passou.

Estas sondas **não são um conjunto de avaliação rotulado**. A frase neutra da
terceira sonda foi classificada PT-PT; isso não demonstra que uma variante seja
objetivamente identificável nessa frase. Não converter sondas em acurácia ou
alegar que a assimetria entre recalls foi resolvida.

## Usar e consultar

```powershell
# Execute from your cloned repository root.
.\.venv\Scripts\python.exe -m scripts.predict_track_b --verify
.\.venv\Scripts\python.exe -m scripts.predict_track_b "Estou estudando este problema."
```

Para repetir a medição, use `scripts.verify_track_b_final_model` com outro
`--output-dir`, preservando o relatório existente. Dependências opcionais para
testes e medição estão em `requirements-track-b-cpu.txt`.

## Próxima etapa e limites científicos

A auditoria de sobreposição normalizada e quase-duplicatas foi registrada antes
da inferência. A avaliação final foi executada uma única vez e congelada, sem
reabrir a seleção. O teste DSL-TL agora está observado e não poderá servir para
ajustar ou escolher outro modelo neste ciclo.

O F1 macro 0,877068 continua sendo a média das três **validações B1.4**, não
uma métrica do modelo final em teste externo. A diferença entre recalls
de 14,07 pontos percentuais era a limitação observada em desenvolvimento;
no teste ela passou a 29,52 pontos no sentido oposto, favorecendo PT-PT.

Fontes: manifesto local dos pesos; relatório funcional; seleção congelada em
[TRACK_B_FINAL_SELECTION_V1.md](TRACK_B_FINAL_SELECTION_V1.md); auditoria das
validações em [TRACK_B_B1_4_RESULT_V1.md](TRACK_B_B1_4_RESULT_V1.md).
