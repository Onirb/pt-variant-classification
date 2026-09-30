"""Executa B1.3: embeddings BERTimbau congelados mais LinearSVC."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

from src.data import ID_TO_LABEL
from src.embeddings import BERTIMBAU_MODEL, BERTIMBAU_REVISION, load_bertimbau, mean_pool_embeddings
from src.track_b import TRACK_B_PREPARED_PATH, TRACK_B_PREPARATION_MANIFEST_PATH
from src.track_b_experiments import TRACK_B_SEEDS, aggregate_reports, document_split, write_json


RUN_DIR = Path("runs/track_b/b1_3_bertimbau_frozen_svm_v1")
WORK_DIR = Path("runs/track_b/b1_3_bertimbau_frozen_svm_v1_work")
CHUNK_SIZE = 128
BATCH_SIZE = 8
MAX_LENGTH = 256
THREADS = 4


def atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def corpus_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, ["text", "label", "document_id", "domain"]].itertuples(index=False):
        digest.update("\u241f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_or_create_progress(expected: dict[str, object]) -> dict[str, object]:
    path = WORK_DIR / "progress.json"
    if path.exists():
        progress = json.loads(path.read_text(encoding="utf-8"))
        keys = ("corpus_sha256", "rows", "chunk_size", "batch_size", "max_length", "model_revision")
        mismatch = [key for key in keys if progress.get(key) != expected.get(key)]
        if mismatch:
            raise ValueError(f"Checkpoint incompatível: {', '.join(mismatch)}")
        return progress
    atomic_json_write(path, expected)
    return expected


def embed_with_checkpoints(frame: pd.DataFrame) -> np.ndarray:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    expected: dict[str, object] = {
        "schema_version": "1.0",
        "corpus_sha256": corpus_sha256(frame),
        "rows": int(len(frame)),
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "threads": THREADS,
        "model_revision": BERTIMBAU_REVISION,
        "completed_chunks": [],
    }
    progress = load_or_create_progress(expected)
    completed = {int(index) for index in progress.get("completed_chunks", [])}
    starts = list(range(0, len(frame), CHUNK_SIZE))
    torch.set_num_threads(THREADS)
    tokenizer, encoder = load_bertimbau(cache_dir=".hf_cache/bertimbau")
    for chunk_index, start in enumerate(starts):
        end = min(start + CHUNK_SIZE, len(frame))
        output = WORK_DIR / f"embeddings_{chunk_index:05d}.npy"
        if chunk_index in completed and output.exists():
            continue
        embeddings = mean_pool_embeddings(
            frame.iloc[start:end]["text"],
            tokenizer=tokenizer,
            model=encoder,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        temporary = output.with_suffix(".tmp.npy")
        np.save(temporary, embeddings)
        temporary.replace(output)
        completed.add(chunk_index)
        progress["completed_chunks"] = sorted(completed)
        atomic_json_write(WORK_DIR / "progress.json", progress)
        print(f"Checkpoint {chunk_index + 1}/{len(starts)} salvo ({end}/{len(frame)} textos).", flush=True)
    return np.concatenate(
        [np.load(WORK_DIR / f"embeddings_{index:05d}.npy") for index in range(len(starts))],
        axis=0,
    )


def evaluate_split(frame: pd.DataFrame, embeddings: np.ndarray, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    split = document_split(frame, seed=seed)
    validation_documents = set(split.validation["document_id"])
    validation_mask = frame["document_id"].isin(validation_documents).to_numpy()
    train_mask = ~validation_mask
    model = LinearSVC(C=1.0, random_state=42)
    started = perf_counter()
    model.fit(embeddings[train_mask], frame.loc[train_mask, "label"])
    predictions = model.predict(embeddings[validation_mask])
    margins = model.decision_function(embeddings[validation_mask])
    validation = frame.loc[validation_mask].copy().reset_index(drop=True)
    precision, recall, f1, support = precision_recall_fscore_support(
        validation["label"], predictions, labels=[0, 1], zero_division=0
    )
    prediction_frame = validation.loc[:, ["text", "label", "domain", "document_id", "segment_id", "lp"]].copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["margin"] = margins
    per_domain = {}
    for domain, domain_frame in prediction_frame.groupby("domain", sort=True):
        per_domain[str(domain)] = {
            "rows": int(len(domain_frame)),
            "macro_f1": round(float(f1_score(domain_frame["label"], domain_frame["prediction"], average="macro")), 6),
            "accuracy": round(float(accuracy_score(domain_frame["label"], domain_frame["prediction"])), 6),
        }
    report = {
        "seed": seed,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "train_documents": int(frame.loc[train_mask, "document_id"].nunique()),
        "validation_documents": int(frame.loc[validation_mask, "document_id"].nunique()),
        "fit_seconds": round(perf_counter() - started, 3),
        "accuracy": round(float(accuracy_score(validation["label"], predictions)), 6),
        "macro_f1": round(float(f1_score(validation["label"], predictions, average="macro")), 6),
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
            validation["label"], predictions, labels=[0, 1]
        ).tolist(),
        "per_domain": per_domain,
    }
    return report, prediction_frame


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"Execução já existe: {RUN_DIR}. Não será sobrescrita automaticamente.")
    frame = pd.read_parquet(TRACK_B_PREPARED_PATH).reset_index(drop=True)
    started = perf_counter()
    embeddings = embed_with_checkpoints(frame)
    reports = []
    prediction_frames = []
    for seed in TRACK_B_SEEDS:
        report, predictions = evaluate_split(frame, embeddings, seed)
        reports.append(report)
        predictions.insert(0, "seed", seed)
        prediction_frames.append(predictions)
        print(f"Seed {seed}: macro F1 {report['macro_f1']:.6f}", flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    np.save(RUN_DIR / "embeddings.npy", embeddings)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(RUN_DIR / "predictions.parquet", index=False)
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "B1.3 frozen BERTimbau mean pooling plus LinearSVC",
        "selection_rule": "report only; no final test was accessed",
        "input_data": str(TRACK_B_PREPARED_PATH),
        "input_preparation_manifest": str(TRACK_B_PREPARATION_MANIFEST_PATH),
        "model": BERTIMBAU_MODEL,
        "model_revision": BERTIMBAU_REVISION,
        "pooling": "attention-mask weighted mean over last_hidden_state",
        "settings": {
            "device": "cpu",
            "threads": THREADS,
            "batch_size": BATCH_SIZE,
            "chunk_size": CHUNK_SIZE,
            "max_length": MAX_LENGTH,
        },
        "embedding_dimension": int(embeddings.shape[1]),
        "split_rule": "StratifiedGroupKFold, 5 folds, first fold, strata domain+label, grouped by document_id",
        "reports": reports,
        "aggregate": aggregate_reports(reports),
        "seconds": round(perf_counter() - started, 3),
    }
    write_json(RUN_DIR / "report.json", payload)
    print(payload["aggregate"])


if __name__ == "__main__":
    main()
