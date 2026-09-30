"""Avalia o modelo v1 no PtBrVId após excluir qualquer sobreposição textual."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.data import load_corpus
from src.external import PTBRVID_VALID_PATH, file_sha256, load_ptbrvid_valid, normalized_texts


MODEL_PATH = Path("runs/final_char_word_svm_v1/model.joblib")
OUTPUT_PATH = Path("runs/ptbrvid_external_v1/report.json")


def metrics(frame, prediction) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "accuracy": round(float(accuracy_score(frame["label"], prediction)), 6),
        "macro_f1": round(float(f1_score(frame["label"], prediction, average="macro")), 6),
        "classification_report": classification_report(
            frame["label"], prediction, target_names=["PT-PT", "PT-BR"], output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    if not PTBRVID_VALID_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Execute a aquisição PtBrVId e o treino final antes da avaliação.")
    corpus, public_test, _ = load_corpus()
    external = load_ptbrvid_valid()
    train_texts = normalized_texts(corpus)
    public_test_texts = normalized_texts(public_test)
    normalized = external["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    overlaps_train = normalized.isin(train_texts)
    overlaps_public_test = normalized.isin(public_test_texts)
    eligible = external.loc[~(overlaps_train | overlaps_public_test)].copy()
    if eligible.empty:
        raise ValueError("Não restaram textos independentes para a avaliação externa.")

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(eligible["text"])
    by_domain: dict[str, object] = {}
    for domain, frame in eligible.groupby("domain", sort=True):
        by_domain[domain] = metrics(frame, model.predict(frame["text"]))
    report = {
        "evaluation": "external_ptbrvid_valid_v1",
        "rule": "PtBrVId is evaluation-only; it must not be used for model selection or training.",
        "source": {
            "dataset": "liaad/PtBrVId",
            "revision": "910745e06ee2a66e64c3cd958b56728c28abd5dc",
            "path": str(PTBRVID_VALID_PATH),
            "sha256": file_sha256(PTBRVID_VALID_PATH),
            "canonical_label_mapping": {"0": "PT-PT", "1": "PT-BR"},
            "raw_label_correction": {"web": {"0": "PT-BR", "1": "PT-PT"}, "other_domains": {"0": "PT-PT", "1": "PT-BR"}},
        },
        "integrity": {
            "raw_rows": int(len(external)),
            "normalized_text_overlap_with_training": int(overlaps_train.sum()),
            "normalized_text_overlap_with_public_test": int(overlaps_public_test.sum()),
            "eligible_rows": int(len(eligible)),
        },
        "overall": metrics(eligible, predictions),
        "by_domain": by_domain,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
