"""Executa B1.1 do Track B sem consultar DSL-TL test."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.track_b import TRACK_B_PREPARED_PATH, TRACK_B_PREPARATION_MANIFEST_PATH
from src.track_b_experiments import (
    TRACK_B_SEEDS,
    aggregate_reports,
    document_split,
    evaluate_b1_1_split,
    write_json,
)


RUN_DIR = Path("runs/track_b/b1_1_char_word_svm_v2")


def main() -> None:
    if not TRACK_B_PREPARED_PATH.exists():
        raise FileNotFoundError(f"Dados preparados ausentes: {TRACK_B_PREPARED_PATH}")
    if RUN_DIR.exists():
        raise FileExistsError(f"Execução já existe: {RUN_DIR}. Não será sobrescrita automaticamente.")
    frame = pd.read_parquet(TRACK_B_PREPARED_PATH)
    reports = []
    prediction_frames = []
    for seed in TRACK_B_SEEDS:
        split = document_split(frame, seed=seed)
        report, predictions = evaluate_b1_1_split(split)
        reports.append(report)
        predictions.insert(0, "seed", seed)
        prediction_frames.append(predictions)
        print(f"Seed {seed}: macro F1 {report['macro_f1']:.6f}", flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(RUN_DIR / "predictions.parquet", index=False)
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "B1.1 char+word TF-IDF with LinearSVC",
        "selection_rule": "report only; no final test was accessed",
        "input_data": str(TRACK_B_PREPARED_PATH),
        "input_preparation_manifest": str(TRACK_B_PREPARATION_MANIFEST_PATH),
        "split_rule": "StratifiedGroupKFold, 5 folds, first fold, strata domain+label, grouped by document_id",
        "reports": reports,
        "aggregate": aggregate_reports(reports),
    }
    write_json(RUN_DIR / "report.json", payload)
    print(payload["aggregate"])


if __name__ == "__main__":
    main()
