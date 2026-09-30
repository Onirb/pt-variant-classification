"""Treino final BERTimbau + LinearSVC com checkpoints locais e retomada."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch
from sklearn.svm import LinearSVC

from src.data import load_corpus
from src.embeddings import BERTIMBAU_MODEL, BERTIMBAU_REVISION, load_bertimbau, mean_pool_embeddings


def corpus_sha256(corpus) -> str:
    """Impressão estável do conteúdo, rótulos e origem usados no ajuste."""
    digest = hashlib.sha256()
    for row in corpus.loc[:, ["text", "label", "source"]].itertuples(index=False):
        digest.update("\u241f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_or_create_progress(work_dir: Path, expected: dict[str, object]) -> dict[str, object]:
    progress_path = work_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        keys = ("corpus_sha256", "rows", "chunk_size", "batch_size", "max_length", "model_revision")
        mismatches = [key for key in keys if progress.get(key) != expected.get(key)]
        if mismatches:
            raise ValueError(f"Checkpoint incompatível com a configuração atual: {', '.join(mismatches)}")
        return progress
    atomic_json_write(progress_path, expected)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128, help="Textos por checkpoint de embedding.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/bertimbau_final_v1"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/bertimbau_final_v1_work"))
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Artefato final já existe: {args.output_dir}. Não sobrescrevendo.")
    if args.batch_size < 1 or args.chunk_size < 1 or args.threads < 1:
        raise ValueError("batch-size, chunk-size e threads devem ser positivos.")

    torch.set_num_threads(args.threads)
    corpus, _, summary = load_corpus()
    digest = corpus_sha256(corpus)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    expected_progress: dict[str, object] = {
        "schema_version": 1,
        "corpus_sha256": digest,
        "rows": int(len(corpus)),
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "model_revision": BERTIMBAU_REVISION,
        "completed_chunks": [],
    }
    progress = load_or_create_progress(args.work_dir, expected_progress)
    completed = {int(index) for index in progress.get("completed_chunks", [])}
    starts = list(range(0, len(corpus), args.chunk_size))

    started = perf_counter()
    tokenizer, encoder = load_bertimbau(cache_dir=".hf_cache/bertimbau")
    weights_mb = sum(parameter.numel() for parameter in encoder.parameters()) * 4 / (1024**2)
    for chunk_index, start in enumerate(starts):
        end = min(start + args.chunk_size, len(corpus))
        output = args.work_dir / f"embeddings_{chunk_index:05d}.npy"
        if chunk_index in completed and output.is_file():
            continue
        embeddings = mean_pool_embeddings(
            corpus.iloc[start:end]["text"],
            tokenizer=tokenizer,
            model=encoder,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        temporary = output.with_suffix(".tmp.npy")
        np.save(temporary, embeddings)
        temporary.replace(output)
        completed.add(chunk_index)
        progress["completed_chunks"] = sorted(completed)
        atomic_json_write(args.work_dir / "progress.json", progress)
        print(f"Checkpoint {chunk_index + 1}/{len(starts)} salvo ({end}/{len(corpus)} textos).", flush=True)

    embedded = np.concatenate(
        [np.load(args.work_dir / f"embeddings_{index:05d}.npy") for index in range(len(starts))], axis=0
    )
    classifier = LinearSVC(C=1.0, random_state=42)
    classifier.fit(embedded, corpus["label"])
    elapsed = perf_counter() - started

    staging = args.output_dir.with_name(f"{args.output_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    joblib.dump(classifier, staging / "linear_svc.joblib")
    manifest = {
        "scope": "final-fit-on-development-corpus-only; no public or external test was used",
        "model": BERTIMBAU_MODEL,
        "revision": BERTIMBAU_REVISION,
        "pooling": "mean over encoder last_hidden_state, attention-mask weighted",
        "classifier": {"type": "LinearSVC", "C": 1.0, "random_state": 42},
        "device": "cpu",
        "settings": {"batch_size": args.batch_size, "chunk_size": args.chunk_size, "max_length": args.max_length, "threads": args.threads},
        "corpus": {"sha256": digest, "summary": summary.to_dict()},
        "rows_fit": int(len(corpus)),
        "embedding_dimension": int(embedded.shape[1]),
        "approx_model_weights_mb_float32": round(weights_mb, 2),
        "seconds_this_run": round(elapsed, 3),
        "artifacts": {"classifier": "linear_svc.joblib"},
    }
    atomic_json_write(staging / "manifest.json", manifest)
    staging.replace(args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
