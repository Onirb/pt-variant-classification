"""Avaliação externa final e única do artefato BERTimbau no FRMT test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import load_corpus
from src.embeddings import load_bertimbau, mean_pool_embeddings
from src.external import (
    DSL_TL_PT_DEV_PATH,
    FRMT_SOURCE_REVISION,
    FRMT_TEST_DIR,
    PTBRVID_VALID_PATH,
    file_sha256,
    load_dsl_tl_pt_dev,
    load_frmt_test,
    load_ptbrvid_valid,
    normalized_texts,
)


MODEL_DIR = Path("runs/bertimbau_final_v1")
OUTPUT_DIR = Path("runs/frmt_external_final_v1")
WORK_DIR = Path("runs/frmt_external_final_v1_work")
CHUNK_SIZE = 128
BATCH_SIZE = 8
THREADS = 4


def atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def dataset_fingerprint(frame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, ["text", "label", "bucket", "source_file"]].itertuples(index=False):
        digest.update("\u241f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def metric_block(frame, prediction) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "accuracy": round(float(accuracy_score(frame["label"], prediction)), 6),
        "macro_f1": round(float(f1_score(frame["label"], prediction, average="macro")), 6),
        "confusion_matrix": confusion_matrix(frame["label"], prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            frame["label"], prediction, target_names=["PT-PT", "PT-BR"], output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    report_path = OUTPUT_DIR / "report.json"
    if report_path.exists():
        raise FileExistsError("A avaliação FRMT final já foi registrada; não será repetida automaticamente.")
    manifest_path = MODEL_DIR / "manifest.json"
    classifier_path = MODEL_DIR / "linear_svc.joblib"
    if not manifest_path.is_file() or not classifier_path.is_file():
        raise FileNotFoundError("Artefato final BERTimbau ausente.")

    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus, public_test, _ = load_corpus()
    external = load_frmt_test()
    blocked_sources = {
        "development": normalized_texts(corpus),
        "public_test": normalized_texts(public_test),
        "dsl_tl": normalized_texts(load_dsl_tl_pt_dev()[0]) if DSL_TL_PT_DEV_PATH.exists() else set(),
        "ptbrvid": normalized_texts(load_ptbrvid_valid()) if PTBRVID_VALID_PATH.exists() else set(),
    }
    normalized = external["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    overlap_masks = {name: normalized.isin(values) for name, values in blocked_sources.items()}
    blocked = np.logical_or.reduce([mask.to_numpy() for mask in overlap_masks.values()])
    eligible = external.loc[~blocked].reset_index(drop=True)
    removed = int(blocked.sum())
    audit = {
        "raw_rows": int(len(external)),
        "removed_due_to_normalized_overlap": removed,
        "eligible_rows": int(len(eligible)),
        "overlap_by_prior_set": {name: int(mask.sum()) for name, mask in overlap_masks.items()},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json_write(OUTPUT_DIR / "audit.json", audit)
    if eligible.empty or removed / len(external) > 0.01:
        raise ValueError("Sobreposição FRMT material ou conjunto elegível vazio; métricas não foram calculadas.")

    fingerprint = dataset_fingerprint(eligible)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = WORK_DIR / "progress.json"
    expected = {"fingerprint": fingerprint, "rows": int(len(eligible)), "chunk_size": CHUNK_SIZE, "completed_chunks": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(key) != expected[key] for key in ("fingerprint", "rows", "chunk_size")):
            raise ValueError("Checkpoint FRMT incompatível; métricas não foram calculadas.")
    else:
        progress = expected
        atomic_json_write(progress_path, progress)
    completed = {int(value) for value in progress.get("completed_chunks", [])}

    torch.set_num_threads(THREADS)
    classifier = joblib.load(classifier_path)
    tokenizer, encoder = load_bertimbau(cache_dir=".hf_cache/bertimbau")
    starts = list(range(0, len(eligible), CHUNK_SIZE))
    for chunk_index, start in enumerate(starts):
        end = min(start + CHUNK_SIZE, len(eligible))
        output = WORK_DIR / f"predictions_{chunk_index:05d}.npy"
        if chunk_index in completed and output.is_file():
            continue
        embeddings = mean_pool_embeddings(
            eligible.iloc[start:end]["text"], tokenizer=tokenizer, model=encoder, batch_size=BATCH_SIZE, max_length=256
        )
        prediction = classifier.predict(embeddings)
        temporary = output.with_suffix(".tmp.npy")
        np.save(temporary, prediction)
        temporary.replace(output)
        completed.add(chunk_index)
        progress["completed_chunks"] = sorted(completed)
        atomic_json_write(progress_path, progress)
        print(f"Checkpoint FRMT {chunk_index + 1}/{len(starts)} salvo ({end}/{len(eligible)} textos).", flush=True)

    predictions = np.concatenate([np.load(WORK_DIR / f"predictions_{index:05d}.npy") for index in range(len(starts))])
    report = {
        "evaluation": "external_frmt_test_final_v1",
        "rule": "FRMT test was used once, after final model selection; no model parameter was changed.",
        "source": {
            "repository": "https://github.com/google-research/google-research/tree/master/frmt",
            "revision": FRMT_SOURCE_REVISION,
            "license": "CC BY-SA 3.0",
            "files": {path.name: file_sha256(path) for path in sorted(FRMT_TEST_DIR.glob("*.tsv"))},
        },
        "model": {
            "manifest_sha256": file_sha256(manifest_path),
            "model": model_manifest["model"],
            "revision": model_manifest["revision"],
            "settings": model_manifest["settings"],
        },
        "integrity": {**audit, "eligible_fingerprint": fingerprint},
        "overall": metric_block(eligible, predictions),
        "by_bucket": {
            bucket: metric_block(frame, predictions[frame.index])
            for bucket, frame in eligible.groupby("bucket", sort=True)
        },
    }
    atomic_json_write(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
