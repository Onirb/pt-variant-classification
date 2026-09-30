"""Analisa erros e margens dos baselines Track B sem retreinar modelos."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.track_b_experiments import write_json


INPUTS = {
    "B1.1": Path("runs/track_b/b1_1_char_word_svm_v2/predictions.parquet"),
    "B1.2": Path("runs/track_b/b1_2_masked_char_word_svm_v1/predictions.parquet"),
    "B1.3": Path("runs/track_b/b1_3_bertimbau_frozen_svm_v1/predictions.parquet"),
}
RUN_DIR = Path("runs/track_b/b2_error_analysis_v1")

URL_OR_EMAIL = re.compile(r"(?:https?://|www\.|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)", re.IGNORECASE)
MENTION_OR_HASHTAG = re.compile(r"(?<!\w)[@#][\w_]+")
NUMBER = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
UPPER_IDENTIFIER = re.compile(r"\b[A-Z]{2,}[\d_-]{2,}\b")


def group_error_summary(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    grouped = frame.groupby(column, dropna=False)
    rows = []
    for value, subset in grouped:
        rows.append(
            {
                column: str(value),
                "observations": int(len(subset)),
                "errors": int(subset["is_error"].sum()),
                "error_rate": round(float(subset["is_error"].mean()), 6),
                "mean_abs_margin": round(float(subset["abs_margin"].mean()), 6),
            }
        )
    return rows


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["is_error"] = enriched["label"].ne(enriched["prediction"])
    enriched["abs_margin"] = enriched["margin"].abs()
    enriched["length_chars"] = enriched["text"].str.len()
    enriched["length_bucket"] = pd.cut(
        enriched["length_chars"],
        bins=[-1, 40, 120, 300, float("inf")],
        labels=["0-40", "41-120", "121-300", "301+"],
    ).astype(str)
    enriched["has_url_or_email"] = enriched["text"].str.contains(URL_OR_EMAIL, na=False)
    enriched["has_mention_or_hashtag"] = enriched["text"].str.contains(MENTION_OR_HASHTAG, na=False)
    enriched["has_number"] = enriched["text"].str.contains(NUMBER, na=False)
    enriched["has_upper_identifier"] = enriched["text"].str.contains(UPPER_IDENTIFIER, na=False)
    return enriched


def analyze(name: str, path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    frame = enrich(pd.read_parquet(path))
    flags = ["has_url_or_email", "has_mention_or_hashtag", "has_number", "has_upper_identifier"]
    report = {
        "experiment": name,
        "observations": int(len(frame)),
        "errors": int(frame["is_error"].sum()),
        "error_rate": round(float(frame["is_error"].mean()), 6),
        "errors_by_domain": group_error_summary(frame, "domain"),
        "errors_by_actual_class": group_error_summary(frame, "label"),
        "errors_by_length": group_error_summary(frame, "length_bucket"),
        "errors_by_structural_pattern": {
            flag: {
                "present": {
                    "observations": int(frame[flag].sum()),
                    "error_rate": round(float(frame.loc[frame[flag], "is_error"].mean()), 6)
                    if frame[flag].any()
                    else None,
                },
                "absent": {
                    "observations": int((~frame[flag]).sum()),
                    "error_rate": round(float(frame.loc[~frame[flag], "is_error"].mean()), 6),
                },
            }
            for flag in flags
        },
        "mean_abs_margin": {
            "correct": round(float(frame.loc[~frame["is_error"], "abs_margin"].mean()), 6),
            "error": round(float(frame.loc[frame["is_error"], "abs_margin"].mean()), 6),
        },
    }
    return report, frame


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"Análise B2 já existe: {RUN_DIR}. Não será sobrescrita automaticamente.")
    all_reports = {}
    b13_frame = None
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Predições ausentes para {name}: {path}")
        report, frame = analyze(name, path)
        all_reports[name] = report
        if name == "B1.3":
            b13_frame = frame
    assert b13_frame is not None
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    errors = b13_frame.loc[
        b13_frame["is_error"],
        [
            "seed",
            "text",
            "label",
            "prediction",
            "margin",
            "domain",
            "document_id",
            "segment_id",
            "length_chars",
            "has_url_or_email",
            "has_mention_or_hashtag",
            "has_number",
            "has_upper_identifier",
        ],
    ].copy()
    errors.to_parquet(RUN_DIR / "b1_3_errors.parquet", index=False)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "analysis only; no model, source data or final test was changed",
        "margin_note": "LinearSVC margins are signed decision values, not calibrated probabilities.",
        "reports": all_reports,
        "artifacts": {"b1_3_error_rows": str(RUN_DIR / "b1_3_errors.parquet")},
    }
    write_json(RUN_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
