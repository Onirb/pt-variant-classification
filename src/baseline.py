"""Linhas de base reproduzíveis para classificação pt-BR versus pt-PT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from .data import ID_TO_LABEL


@dataclass(frozen=True)
class BaselineReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_char_tfidf_pipeline() -> Pipeline:
    """Modelo explicável baseado em padrões ortográficos de caracteres."""
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents=None,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def fit_and_evaluate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Pipeline, BaselineReport]:
    pipeline = build_char_tfidf_pipeline()
    pipeline.fit(train["text"], train["label"])
    predictions = pipeline.predict(test["text"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test["label"], predictions, labels=[0, 1], zero_division=0
    )
    report = BaselineReport(
        accuracy=round(float(accuracy_score(test["label"], predictions)), 6),
        macro_f1=round(float(f1_score(test["label"], predictions, average="macro")), 6),
        weighted_f1=round(float(f1_score(test["label"], predictions, average="weighted")), 6),
        per_class={
            ID_TO_LABEL[label]: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
            }
            for index, label in enumerate((0, 1))
        },
    )
    return pipeline, report


def detailed_report(test: pd.DataFrame, predictions: list[int]) -> dict[str, Any]:
    return classification_report(
        test["label"], predictions, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]], output_dict=True, zero_division=0
    )
