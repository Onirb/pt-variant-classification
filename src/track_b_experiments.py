"""Experimentos controlados do Track B, com separação estrita por documento."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .data import ID_TO_LABEL
from .text_transform import mask_contextual_tokens


TRACK_B_SEEDS = (7, 42, 2026)


@dataclass(frozen=True)
class DocumentSplit:
    seed: int
    train: pd.DataFrame
    validation: pd.DataFrame


def document_split(frame: pd.DataFrame, *, seed: int, n_splits: int = 5) -> DocumentSplit:
    """Cria validação por documento, estratificada por domínio e classe."""
    required = {"text", "label", "domain", "document_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para a divisão documental: {sorted(missing)}")
    if frame["document_id"].isna().any() or frame["document_id"].astype(str).str.strip().eq("").any():
        raise ValueError("document_id ausente ou vazio.")
    strata = frame["domain"].astype(str) + "::" + frame["label"].astype(str)
    if strata.value_counts().min() < n_splits:
        raise ValueError("Há poucos documentos para estratificar todos os domínios e classes.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, validation_index = next(
        splitter.split(frame["text"], strata, groups=frame["document_id"])
    )
    train = frame.iloc[train_index].reset_index(drop=True)
    validation = frame.iloc[validation_index].reset_index(drop=True)
    shared_documents = set(train["document_id"]).intersection(validation["document_id"])
    if shared_documents:
        raise AssertionError("A divisão vazou documentos entre treino e validação.")
    if set(train["label"]) != {0, 1} or set(validation["label"]) != {0, 1}:
        raise ValueError("Uma divisão documental perdeu uma das classes binárias.")
    if set(train["domain"]) != set(frame["domain"]) or set(validation["domain"]) != set(frame["domain"]):
        raise ValueError("Uma divisão documental perdeu um domínio.")
    return DocumentSplit(seed=seed, train=train, validation=validation)


def build_char_word_svm() -> Pipeline:
    """Baseline B1.1: TF-IDF char 2--6 + palavra 1--2 e LinearSVC."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "characters",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(2, 6),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents=None,
                            ),
                        ),
                        (
                            "words",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents=None,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LinearSVC(C=1.0, random_state=42)),
        ]
    )


def _masked_lowercase(text: str) -> str:
    return mask_contextual_tokens(text).lower()


def build_masked_char_word_svm() -> Pipeline:
    """B1.2: mesma arquitetura do B1.1 com pistas estruturais mascaradas."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "characters",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(2, 6),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents=None,
                                preprocessor=_masked_lowercase,
                            ),
                        ),
                        (
                            "words",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents=None,
                                preprocessor=_masked_lowercase,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LinearSVC(C=1.0, random_state=42)),
        ]
    )


def evaluate_split(
    split: DocumentSplit, model_factory: Callable[[], Pipeline]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Treina um candidato e devolve métricas e predições de uma divisão."""
    model = model_factory()
    started = perf_counter()
    model.fit(split.train["text"], split.train["label"])
    predictions = model.predict(split.validation["text"])
    margins = model.decision_function(split.validation["text"])
    elapsed = perf_counter() - started
    precision, recall, f1, support = precision_recall_fscore_support(
        split.validation["label"], predictions, labels=[0, 1], zero_division=0
    )
    prediction_frame = split.validation.loc[
        :, ["text", "label", "domain", "document_id", "segment_id", "lp"]
    ].copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["margin"] = margins
    domain_metrics: dict[str, dict[str, float | int]] = {}
    for domain, domain_frame in prediction_frame.groupby("domain", sort=True):
        domain_metrics[str(domain)] = {
            "rows": int(len(domain_frame)),
            "macro_f1": round(
                float(f1_score(domain_frame["label"], domain_frame["prediction"], average="macro")), 6
            ),
            "accuracy": round(float(accuracy_score(domain_frame["label"], domain_frame["prediction"])), 6),
        }
    report = {
        "seed": split.seed,
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "train_documents": int(split.train["document_id"].nunique()),
        "validation_documents": int(split.validation["document_id"].nunique()),
        "fit_seconds": round(elapsed, 3),
        "accuracy": round(float(accuracy_score(split.validation["label"], predictions)), 6),
        "macro_f1": round(float(f1_score(split.validation["label"], predictions, average="macro")), 6),
        "per_class": {
            ID_TO_LABEL[label]: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, label in enumerate((0, 1))
        },
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(
            split.validation["label"], predictions, labels=[0, 1]
        ).tolist(),
        "per_domain": domain_metrics,
    }
    return report, prediction_frame


def evaluate_b1_1_split(split: DocumentSplit) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compatibilidade explícita do baseline B1.1."""
    return evaluate_split(split, build_char_word_svm)


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume médias e dispersão entre as divisões já pré-especificadas."""
    scores = pd.DataFrame(
        [{"seed": report["seed"], "macro_f1": report["macro_f1"], "accuracy": report["accuracy"]} for report in reports]
    )
    return {
        "seeds": list(TRACK_B_SEEDS),
        "macro_f1_mean": round(float(scores["macro_f1"].mean()), 6),
        "macro_f1_std": round(float(scores["macro_f1"].std(ddof=0)), 6),
        "accuracy_mean": round(float(scores["accuracy"].mean()), 6),
        "accuracy_std": round(float(scores["accuracy"].std(ddof=0)), 6),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escreve resultado novo sem sobrescrever uma execução já registrada."""
    if path.exists():
        raise FileExistsError(f"Resultado já existe: {path}. Não será sobrescrito automaticamente.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
