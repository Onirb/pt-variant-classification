"""Protocolo de desenvolvimento para comparar classificadores textuais."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .data import ID_TO_LABEL


PUBLIC_TRAIN_SOURCE = "cc4051/pt_vid:train"


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    fit_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def development_split(corpus: pd.DataFrame, *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria validação a partir do treino público; dados locais ficam no treino."""
    public = corpus.loc[corpus["source"].eq(PUBLIC_TRAIN_SOURCE)].copy()
    local = corpus.loc[~corpus["source"].eq(PUBLIC_TRAIN_SOURCE)].copy()
    public_train, validation = train_test_split(
        public,
        test_size=0.20,
        random_state=seed,
        stratify=public["label"],
    )
    return pd.concat([public_train, local], ignore_index=True), validation.reset_index(drop=True)


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=4.0, max_iter=1_000, random_state=42, solver="liblinear")


def _linear_svm(c: float = 1.0) -> LinearSVC:
    return LinearSVC(C=c, random_state=42)


def candidate_factories() -> dict[str, Callable[[], Pipeline]]:
    """Candidatos baratos, complementares e adequados a texto de variantes."""
    return {
        "char_2_5_logreg": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ("classifier", _logistic()),
        ]),
        "char_3_6_logreg": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=2, sublinear_tf=True)),
            ("classifier", _logistic()),
        ]),
        "char_wb_3_6_logreg": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=2, sublinear_tf=True)),
            ("classifier", _logistic()),
        ]),
        "char_word_logreg": lambda: Pipeline([
            ("features", FeatureUnion([
                ("characters", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
                ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ])),
            ("classifier", _logistic()),
        ]),
        "char_2_5_linear_svm": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ("classifier", _linear_svm()),
        ]),
        "char_2_6_linear_svm_c0_5": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True)),
            ("classifier", _linear_svm(0.5)),
        ]),
        "char_2_6_linear_svm_c1": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True)),
            ("classifier", _linear_svm(1.0)),
        ]),
        "char_2_6_linear_svm_c2": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True)),
            ("classifier", _linear_svm(2.0)),
        ]),
        "char_3_7_linear_svm": lambda: Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(3, 7), min_df=2, sublinear_tf=True)),
            ("classifier", _linear_svm()),
        ]),
        "char_word_linear_svm": lambda: Pipeline([
            ("features", FeatureUnion([
                ("characters", TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True)),
                ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ])),
            ("classifier", _linear_svm()),
        ]),
    }


def run_candidates(train: pd.DataFrame, validation: pd.DataFrame) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for name, factory in candidate_factories().items():
        model = factory()
        started = perf_counter()
        model.fit(train["text"], train["label"])
        predictions = model.predict(validation["text"])
        elapsed = perf_counter() - started
        _, _, per_class_f1, _ = precision_recall_fscore_support(
            validation["label"], predictions, labels=[0, 1], zero_division=0
        )
        results.append(
            ExperimentResult(
                name=name,
                accuracy=round(float(accuracy_score(validation["label"], predictions)), 6),
                macro_f1=round(float(f1_score(validation["label"], predictions, average="macro")), 6),
                per_class_f1={ID_TO_LABEL[label]: round(float(per_class_f1[index]), 6) for index, label in enumerate((0, 1))},
                fit_seconds=round(elapsed, 3),
            )
        )
    return sorted(results, key=lambda result: result.macro_f1, reverse=True)
