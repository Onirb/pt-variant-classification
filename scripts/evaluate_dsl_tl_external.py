"""Avalia o modelo final v1 no DSL-TL PT_dev, sem o usar para ajuste."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import load_corpus
from src.external import DSL_TL_PT_DEV_PATH, file_sha256, load_dsl_tl_pt_dev, normalized_texts


MODEL_PATH = Path("runs/final_char_word_svm_v1/model.joblib")
OUTPUT_PATH = Path("runs/dsl_tl_external_v1/report.json")


def main() -> None:
    if not DSL_TL_PT_DEV_PATH.exists():
        raise FileNotFoundError(
            f"Conjunto externo ausente: {DSL_TL_PT_DEV_PATH}. "
            "Execute python -m scripts.fetch_dsl_tl_external primeiro."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo final ausente: {MODEL_PATH}.")

    corpus, _, _ = load_corpus()
    external, raw_label_counts = load_dsl_tl_pt_dev()
    train_overlap = normalized_texts(corpus) & normalized_texts(external)

    model = joblib.load(MODEL_PATH)
    prediction = model.predict(external["text"])
    report = {
        "evaluation": "external_dsl_tl_pt_dev_binary_only",
        "rule": "The external data is evaluation-only and must not be used for model selection or training.",
        "source": {
            "url": "https://github.com/LanguageTechnologyLab/DSL-TL",
            "path": str(DSL_TL_PT_DEV_PATH),
            "sha256": file_sha256(DSL_TL_PT_DEV_PATH),
        },
        "rows": {
            "raw": int(sum(raw_label_counts.values())),
            "raw_label_counts": raw_label_counts,
            "binary_evaluated": int(len(external)),
            "excluded_generic_pt": int(raw_label_counts.get("PT", 0)),
            "normalized_text_overlap_with_training": int(len(train_overlap)),
        },
        "metrics": {
            "accuracy": round(float(accuracy_score(external["label"], prediction)), 6),
            "macro_f1": round(float(f1_score(external["label"], prediction, average="macro")), 6),
            "confusion_matrix": {
                "labels": ["PT-PT", "PT-BR"],
                "rows_actual_columns_predicted": confusion_matrix(external["label"], prediction, labels=[0, 1]).tolist(),
            },
            "classification_report": classification_report(
                external["label"], prediction, target_names=["PT-PT", "PT-BR"], output_dict=True, zero_division=0
            ),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
