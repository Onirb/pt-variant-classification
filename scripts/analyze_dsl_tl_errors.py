"""Produz um diagnóstico local dos erros do DSL-TL, sem retreinar modelos."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.external import DSL_TL_PT_DEV_PATH, load_dsl_tl_pt_dev


MODEL_PATH = Path("runs/final_char_word_svm_v1/model.joblib")
OUTPUT_DIR = Path("runs/dsl_tl_external_v1/error_analysis")
LABELS = {0: "PT-PT", 1: "PT-BR"}


def main() -> None:
    if not DSL_TL_PT_DEV_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Execute a aquisição externa e o treino final antes desta análise.")

    external, _ = load_dsl_tl_pt_dev()
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(external["text"])
    margins = model.decision_function(external["text"])

    rows = external.assign(
        actual=external["label"].map(LABELS),
        predicted=pd.Series(predictions, index=external.index).map(LABELS),
        correct=external["label"].eq(predictions),
        margin=margins,
        absolute_margin=pd.Series(margins, index=external.index).abs(),
        characters=external["text"].str.len(),
    )
    grouped = (
        rows.groupby(["actual", "correct"], observed=True)
        .agg(rows=("id", "size"), median_characters=("characters", "median"), median_absolute_margin=("absolute_margin", "median"))
        .reset_index()
    )
    summary = {
        "interpretation": "Margin is distance to the SVM decision boundary, not a calibrated probability.",
        "overall": {
            "rows": int(len(rows)),
            "errors": int((~rows["correct"]).sum()),
            "median_absolute_margin_correct": round(float(rows.loc[rows["correct"], "absolute_margin"].median()), 6),
            "median_absolute_margin_incorrect": round(float(rows.loc[~rows["correct"], "absolute_margin"].median()), 6),
        },
        "by_actual_class_and_correctness": grouped.to_dict(orient="records"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.loc[~rows["correct"], ["id", "text", "actual", "predicted", "margin", "absolute_margin", "characters"]].to_csv(
        OUTPUT_DIR / "misclassified_rows.csv", index=False
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
