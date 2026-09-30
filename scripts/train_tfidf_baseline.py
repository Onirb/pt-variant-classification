"""Treina e registra uma linha de base de n-gramas de caracteres."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from src.baseline import detailed_report, fit_and_evaluate
from src.data import load_corpus


def main() -> None:
    train, test, corpus_summary = load_corpus(cache_dir=os.environ.get("HF_HOME"))
    pipeline, report = fit_and_evaluate(train, test)
    predictions = pipeline.predict(test["text"])
    output = Path("runs/tfidf_char_baseline_v1")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "model.joblib")
    payload = {
        "model": "TF-IDF character 2-5 grams + LogisticRegression",
        "corpus": corpus_summary.to_dict(),
        "metrics": report.to_dict(),
        "classification_report": detailed_report(test, predictions),
    }
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
