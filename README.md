# Portuguese Variant Classification

Reproducible experiments and an offline command-line classifier for **European
Portuguese (PT-PT)** and **Brazilian Portuguese (PT-BR)**.

Originally developed during a master's-level Machine Learning course for the
[Portuguese Variant Identification Kaggle challenge](https://www.kaggle.com/competitions/portuguese-variant-identification),
the project now separates its academic legacy, reproducible baselines and an
independent generalization experiment. This is an **educational portfolio
project**, not a production-ready language identification service.

## Current status

- **Track A is closed:** classical baselines and frozen BERTimbau embeddings
  were evaluated, including an external FRMT check that exposed a PT-BR bias.
- **Track B is closed:** BERTimbau was fine-tuned on 1,877 WMT24++ texts and
  evaluated once on the official DSL-TL Portuguese binary test.
- Final Track B **macro F1: 0.762398** on 436 texts; **accuracy: 76.83%**.
- Offline CPU inference, artifact integrity checks, checkpoint recovery and
  **35 synthetic local tests** are implemented.
- No post-test threshold tuning, checkpoint search or retraining was performed.

## Results and limitations

These scores belong to different populations and must **not** be compared as
if they were measured on the same test set.

| Track / model | Evaluation | Macro F1 |
| --- | --- | ---: |
| A: character + word TF-IDF / LinearSVC | historical public test | 0.970001 |
| A: frozen BERTimbau / LinearSVC | internal validation, mean of 3 seeds | 0.970314 |
| A: frozen BERTimbau / LinearSVC | FRMT external final test, 5,232 texts | 0.608864 |
| B: fine-tuned BERTimbau | document-grouped WMT24++ validation, mean of 3 seeds | 0.877068 |
| B: final fine-tuned BERTimbau | official DSL-TL binary test, 436 texts | **0.762398** |

Final Track B metrics:

| Human label | Texts | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| PT-PT | 137 | 0.578261 | **0.970803** | 0.724796 |
| PT-BR | 299 | **0.980583** | 0.675585 | 0.800000 |

The model misclassified 97 Brazilian texts as European and four European texts
as Brazilian. **The class-recall balance objective was not achieved.** Removing
one historically overlapping example in a pre-defined sensitivity analysis
leaves macro F1 at 0.761644 (435 texts), using the same predictions.

The final test contains natural journalistic texts, whereas Track B development
uses translations/post-edits. The observed asymmetry changes direction between
development and test. This is compatible with domain/corpus sensitivity, but
does not identify a causal mechanism or demonstrate robustness in every domain.

See the [final report](docs/TRACK_B_B4_RESULT_V1.md),
[pre-inference protocol](docs/TRACK_B_B4_PROTOCOL_V1.md) and
[Track A closure](docs/TRACK_A_CLOSED.md). All observed tests are frozen for
selection; another improvement cycle requires another unseen final test.

## Architecture

- Python backend modules and command-line entry points; no hosted API required.
- Track A: character/word TF-IDF baselines and frozen BERTimbau + LinearSVC.
- Track B: `neuralmind/bert-base-portuguese-cased`, fine-tuned with Transformers
  on a Colab T4; local inference runs on CPU with four threads.
- Local model artifacts include safetensors weights, tokenizer, hashes, source
  revision and the frozen selection protocol.
- Outputs are labels and raw decision scores, **not calibrated probabilities**.
- Final test auditing checks normalized equality and lexical near-duplicates
  before prediction. These checks do not detect all paraphrases or pretraining
  contamination.

## Install and test on Windows

Validated locally with Python 3.12.10 on Windows 11. Run from the repository root:

```powershell
git clone https://github.com/Onirb/pt-variant-classification.git
cd pt-variant-classification
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-track-b-cpu.txt
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Package installation needs network access; the tests do not. They create
synthetic corpora, bundles and tiny random networks in temporary directories.
They do **not** need local `runs/`, downloaded corpora or the real BERTimbau
weights, and do not retrain or re-evaluate the frozen scientific experiments.
See [testing and reproduction](docs/TESTING_AND_REPRODUCTION.md).

## Offline inference

The modern trained weights are **not distributed in Git**. A fresh clone can
run the tests and inspect the protocols, but needs a completed local model
artifact before it can classify text.

After generating/importing the final artifact into `runs/track_b/final_model_v1`:

```powershell
.\.venv\Scripts\python.exe -m scripts.predict_track_b --verify
.\.venv\Scripts\python.exe -m scripts.predict_track_b "Estou a estudar este problema."
.\.venv\Scripts\python.exe -m scripts.predict_track_b "Estou estudando esse problema."
```

The classifier rejects empty input, supports Unicode, signals truncation beyond
256 tokens and checks artifact hashes before loading. Once the artifact is
available, inference does not require an online model provider.

Artifact preparation, import and costs are documented in
[final selection](docs/TRACK_B_FINAL_SELECTION_V1.md) and
[local model verification](docs/TRACK_B_FINAL_MODEL_V1.md).
The final Colab notebook is [Track_B_Final_Colab.ipynb](colab/Track_B_Final_Colab.ipynb);
the three-seed validation notebook is [Track_B_B1_4_Colab.ipynb](colab/Track_B_B1_4_Colab.ipynb).
They require the locally generated upload packages described in the reports.

## Data provenance

- Track A: [`cc4051/pt_vid`](https://huggingface.co/datasets/cc4051/pt_vid) and
  the repository's historical `data/PT_train.tsv`.
- Track B development: [`google/wmt24pp`](https://huggingface.co/datasets/google/wmt24pp),
  configurations `en-pt_BR` and `en-pt_PT`; document-grouped splits, invalid
  source filtering and conflicting-label removal.
- Final Track B test: [official DSL-TL](https://github.com/LanguageTechnologyLab/DSL-TL),
  reconstructed through exact official ID/text checks against the human
  annotations; 59 ambiguous `PT` labels excluded for the binary task.
- Other Track A checks: [PtBrVId](docs/EXTERNAL_PTBRVID_V1.md) and
  [FRMT](docs/FRMT_EXTERNAL_FINAL_V1.md).

Source revisions, hashes and exclusions are recorded in the reports. Dataset
and pretrained-model terms remain those of their respective authors; newly
downloaded corpora and modern weights are not republished in this repository.

## Repository layout

```text
src/        Data preparation, model helpers, inference and lexical audit
scripts/    Explicit download, training, import and evaluation commands
tests/      Offline synthetic unit/integration tests
colab/      GPU training notebooks and scripts
docs/       Protocols, results, limits and experiment history
notebooks/  Original academic notebook, preserved as legacy
models/     Original academic weights, already tracked in the legacy repository
runs/       Generated artifacts and detailed predictions (local, ignored)
```

The historical files are preserved, not recommended for inference. The legacy
weight called `modelCNN.pth` contains LSTM parameters, and the original
character vocabulary was not saved deterministically. Historical CNN/LSTM
scores are therefore not presented as verified current results; see
[legacy audit](docs/LEGACY_AUDIT.md).

Next maintenance tasks are listed in [NEXT_STEPS.md](docs/NEXT_STEPS.md).
