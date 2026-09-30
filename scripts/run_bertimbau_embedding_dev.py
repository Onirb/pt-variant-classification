"""Smoke test ou comparação interna de BERTimbau mean-pooling + SVM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from src.data import load_corpus
from src.embeddings import BERTIMBAU_MODEL, BERTIMBAU_REVISION, load_bertimbau, mean_pool_embeddings
from src.experiments import development_split


def balanced_sample(frame, per_class: int, seed: int):
    return (
        frame.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=seed)
        .sort_index()
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Usa 200 textos por classe no treino e 100 na validação.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    corpus, _, _ = load_corpus()
    train, validation = development_split(corpus, seed=args.seed)
    if args.smoke:
        train = balanced_sample(train, per_class=200, seed=args.seed)
        validation = balanced_sample(validation, per_class=100, seed=args.seed)

    started = perf_counter()
    tokenizer, encoder = load_bertimbau(cache_dir=".hf_cache/bertimbau")
    weights_mb = sum(parameter.numel() for parameter in encoder.parameters()) * 4 / (1024**2)
    embedded_train = mean_pool_embeddings(
        train["text"], tokenizer=tokenizer, model=encoder, batch_size=args.batch_size, max_length=args.max_length
    )
    embedded_validation = mean_pool_embeddings(
        validation["text"], tokenizer=tokenizer, model=encoder, batch_size=args.batch_size, max_length=args.max_length
    )
    classifier = LinearSVC(C=1.0, random_state=42)
    classifier.fit(embedded_train, train["label"])
    prediction = classifier.predict(embedded_validation)
    elapsed = perf_counter() - started
    report = {
        "scope": "internal-development-only; no public or external test was used",
        "mode": "smoke" if args.smoke else "full",
        "model": BERTIMBAU_MODEL,
        "revision": BERTIMBAU_REVISION,
        "device": "cpu",
        "parameters": int(sum(parameter.numel() for parameter in encoder.parameters())),
        "approx_model_weights_mb_float32": round(weights_mb, 2),
        "settings": {"batch_size": args.batch_size, "max_length": args.max_length, "threads": args.threads},
        "seed": args.seed,
        "rows": {"train": int(len(train)), "validation": int(len(validation))},
        "metrics": {
            "accuracy": round(float(accuracy_score(validation["label"], prediction)), 6),
            "macro_f1": round(float(f1_score(validation["label"], prediction, average="macro")), 6),
        },
        "seconds": round(elapsed, 3),
    }
    output = (
        Path(f"runs/bertimbau_embedding_smoke_v1/seed_{args.seed}.json")
        if args.smoke
        else Path("runs/bertimbau_embedding_dev_v1/report.json")
        if args.seed == 42
        else Path(f"runs/bertimbau_embedding_dev_v1/seed_{args.seed}.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
