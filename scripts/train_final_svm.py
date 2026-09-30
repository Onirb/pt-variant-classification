"""Treina o candidato selecionado e mede uma vez no teste público congelado."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.data import ID_TO_LABEL, load_corpus
from src.experiments import candidate_factories


def main() -> None:
    train, test, summary = load_corpus(cache_dir=os.environ.get("HF_HOME"))
    model_name = "char_word_linear_svm"
    model = candidate_factories()[model_name]()
    model.fit(train["text"], train["label"])
    predictions = model.predict(test["text"])
    payload = {
        "model": model_name,
        "selection_evidence": "winner_validation_v1; three internal seeds; public test not used in selection",
        "corpus": summary.to_dict(),
        "test": {
            "rows": len(test),
            "accuracy": round(float(accuracy_score(test["label"], predictions)), 6),
            "macro_f1": round(float(f1_score(test["label"], predictions, average="macro")), 6),
            "weighted_f1": round(float(f1_score(test["label"], predictions, average="weighted")), 6),
            "classification_report": classification_report(
                test["label"], predictions, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]], output_dict=True, zero_division=0
            ),
        },
    }
    output = Path("runs/final_char_word_svm_v1")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
