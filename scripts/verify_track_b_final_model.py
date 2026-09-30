"""Checagem funcional e custo CPU do artefato real; não lê o teste reservado."""

import argparse
import importlib.metadata
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import psutil

from src.fine_tuned_inference import DEFAULT_ARTIFACT, FineTunedVariantClassifier, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/track_b/final_model_verification_v1"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Relatório já existe; use outro output-dir: {args.output_dir}")
    process = psutil.Process()
    before_rss = process.memory_info().rss
    started = perf_counter()
    classifier = FineTunedVariantClassifier(args.artifact_dir, threads=args.threads)
    load_seconds = perf_counter() - started
    loaded_rss = process.memory_info().rss
    probe_result = classifier.verify_probes()
    texts = classifier.manifest["probe_texts"]
    examples = []
    for text in texts:
        initial = classifier.predict(text)
        durations = []
        for _ in range(7):
            started = perf_counter()
            result = classifier.predict(text)
            durations.append((perf_counter() - started) * 1000)
            if result != initial:
                raise ValueError("Saída não determinística nas sondas repetidas.")
        examples.append({"text": text, "prediction": initial.to_dict(), "repetitions": 7,
                         "median_ms": statistics.median(durations), "min_ms": min(durations),
                         "max_ms": max(durations)})
    for invalid, error_type in (("  ", ValueError), (None, TypeError)):
        try:
            classifier.predict(invalid)
        except error_type:
            pass
        else:
            raise ValueError("Entrada inválida não foi rejeitada.")
    started = perf_counter()
    long_result = classifier.predict("Estou a estudar esta questão de classificação. " * 300)
    long_ms = (perf_counter() - started) * 1000
    if not long_result.was_truncated:
        raise ValueError("Texto longo não sinalizou truncamento.")
    final_rss = process.memory_info().rss
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "functional probes only; not labeled evaluation; reserved test untouched",
        "artifact": str(args.artifact_dir), "manifest_sha256": sha256_file(args.artifact_dir / "manifest.json"),
        "rows_fit": classifier.manifest["rows_fit"], "global_steps": classifier.manifest["global_steps"],
        "probes": probe_result, "empty_input_rejected": True, "invalid_type_rejected": True,
        "repeated_predictions_deterministic": True, "long_input": {
            "prediction": long_result.to_dict(), "milliseconds": long_ms},
        "examples": examples, "load_seconds_including_hash_verification": load_seconds,
        "rss_mib_before_model_load": before_rss / 1024**2,
        "rss_mib_after_model_load": loaded_rss / 1024**2,
        "rss_mib_after_checks": final_rss / 1024**2,
        "rss_mib_load_delta": (loaded_rss - before_rss) / 1024**2,
        "memory_note": "sampled process RSS, not continuously monitored peak; includes libraries",
        "cpu_threads": args.threads, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "psutil")},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
