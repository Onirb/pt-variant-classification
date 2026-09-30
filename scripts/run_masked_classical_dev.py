"""Compara SVM original versus mascaramento determinístico no desenvolvimento."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.data import load_corpus
from src.experiments import development_split
from src.text_transform import mask_contextual_tokens


OUTPUT_PATH = Path("runs/masked_classical_v1/report.json")
SEEDS = (7, 42, 2026)


def make_model(*, masked: bool) -> Pipeline:
    transform = mask_contextual_tokens if masked else None
    return Pipeline([
        ("features", FeatureUnion([
            ("characters", TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True, preprocessor=transform)),
            ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, preprocessor=transform)),
        ])),
        ("classifier", LinearSVC(C=1.0, random_state=42)),
    ])


def evaluate(*, masked: bool) -> list[dict[str, float | int]]:
    corpus, _, _ = load_corpus()
    results: list[dict[str, float | int]] = []
    for seed in SEEDS:
        train, validation = development_split(corpus, seed=seed)
        model = make_model(masked=masked)
        model.fit(train["text"], train["label"])
        prediction = model.predict(validation["text"])
        results.append({
            "seed": seed,
            "rows_train": int(len(train)),
            "rows_validation": int(len(validation)),
            "accuracy": round(float(accuracy_score(validation["label"], prediction)), 6),
            "macro_f1": round(float(f1_score(validation["label"], prediction, average="macro")), 6),
        })
    return results


def main() -> None:
    report = {
        "scope": "internal-development-only; no public or external test was used",
        "baseline_original": evaluate(masked=False),
        "masked_candidate": evaluate(masked=True),
    }
    for key in ("baseline_original", "masked_candidate"):
        scores = [item["macro_f1"] for item in report[key]]
        report[f"{key}_mean_macro_f1"] = round(sum(scores) / len(scores), 6)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
