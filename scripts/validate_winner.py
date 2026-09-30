"""Confirma estabilidade do melhor candidato sem usar o teste público."""

from __future__ import annotations

import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.data import load_corpus
from src.experiments import candidate_factories, development_split


def main() -> None:
    corpus, _, summary = load_corpus(cache_dir=os.environ.get("HF_HOME"))
    factory = candidate_factories()["char_word_linear_svm"]
    rows: list[dict[str, float | int]] = []
    for seed in (7, 42, 2026):
        train, validation = development_split(corpus, seed=seed)
        model = factory()
        model.fit(train["text"], train["label"])
        predictions = model.predict(validation["text"])
        rows.append(
            {
                "seed": seed,
                "accuracy": round(float(accuracy_score(validation["label"], predictions)), 6),
                "macro_f1": round(float(f1_score(validation["label"], predictions, average="macro")), 6),
            }
        )
    scores = [float(row["macro_f1"]) for row in rows]
    payload = {
        "candidate": "char_word_linear_svm",
        "protocol": "three public-train stratified validations; public test untouched",
        "corpus": summary.to_dict(),
        "runs": rows,
        "macro_f1_mean": round(sum(scores) / len(scores), 6),
        "macro_f1_min": round(min(scores), 6),
        "macro_f1_max": round(max(scores), 6),
    }
    output = Path("runs/winner_validation_v1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
