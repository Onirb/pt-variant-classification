"""Audita resultados Colab contra o pacote original e registra a comparação."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from colab.train_b1_4 import PROTOCOL, SEEDS, load_inputs


PACKAGE = Path("runs/track_b/colab_b1_4_bundle_v1/track_b_colab")
DESTINATION = Path("runs/track_b/b1_4_colab_import_v1")
BASELINES = {
    "B1.1": Path("runs/track_b/b1_1_char_word_svm_v2"),
    "B1.2": Path("runs/track_b/b1_2_masked_char_word_svm_v1"),
    "B1.3": Path("runs/track_b/b1_3_bertimbau_frozen_svm_v1"),
}
KEYS = ["text", "label", "domain", "document_id", "segment_id", "lp"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equal_number(actual, expected, description, tolerance=1e-9):
    if not np.isfinite(actual) or abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"Métrica divergente ({description}): {actual} versus {expected}")


def sorted_keys(frame):
    return frame[KEYS].sort_values(KEYS).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    if DESTINATION.exists():
        raise FileExistsError(f"Importação já registrada: {DESTINATION}")
    corpus, splits, manifest = load_inputs(PACKAGE)
    package_hash = hashlib.sha256((PACKAGE / "package_manifest.json").read_bytes()).hexdigest()
    baseline_reports = {name: read_json(path / "report.json") for name, path in BASELINES.items()}
    baseline_predictions = {name: pd.read_parquet(path / "predictions.parquet") for name, path in BASELINES.items()}
    full_reports = []
    audit = []
    payloads = {}
    predicted_frames = []
    with ZipFile(args.zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP corrompido.")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contém entradas repetidas.")
        aggregate = json.loads(archive.read("full/aggregate.json"))
        for seed in SEEDS:
            report_name = f"full/seed_{seed}/report.json"
            report = json.loads(archive.read(report_name))
            config = json.loads(archive.read(f"full/seed_{seed}/run_config.json"))
            if report["protocol"] != PROTOCOL or report["package_sha256"] != package_hash:
                raise ValueError(f"Seed {seed}: protocolo ou pacote diverge do preparado.")
            if config != {key: report[key] for key in ("protocol", "seed", "smoke", "package_sha256")}:
                raise ValueError(f"Seed {seed}: configuração não corresponde ao relatório.")
            if report["seed"] != seed or report["smoke"] or report["global_steps"] != 279:
                raise ValueError(f"Seed {seed}: execução incompleta ou inesperada.")
            equal_number(report["training_metrics"]["epoch"], 3, "épocas")
            expected = corpus.iloc[splits[str(seed)]["validation_indices"]]
            prediction = pd.read_parquet(BytesIO(archive.read(f"full/seed_{seed}/predictions.parquet")))
            if len(prediction) != len(expected) or prediction["seed"].unique().tolist() != [seed]:
                raise ValueError(f"Seed {seed}: contagens ou sementes inválidas.")
            pd.testing.assert_frame_equal(sorted_keys(prediction), sorted_keys(expected), check_dtype=False)
            if not prediction.prediction.isin([0, 1]).all():
                raise ValueError("Predição fora da tarefa binária.")
            equal_number(f1_score(prediction.label, prediction.prediction, average="macro"), report["macro_f1"], "F1 macro")
            equal_number(accuracy_score(prediction.label, prediction.prediction), report["accuracy"], "accuracy")
            confusion = confusion_matrix(prediction.label, prediction.prediction, labels=[0, 1]).tolist()
            if confusion != report["confusion_matrix_rows_actual_columns_predicted"]:
                raise ValueError("Matriz de confusão divergente.")
            precision, recall, f1, support = precision_recall_fscore_support(prediction.label, prediction.prediction, labels=[0, 1], zero_division=0)
            for label, name in enumerate(("PT-PT", "PT-BR")):
                for field, value in (("precision", precision[label]), ("recall", recall[label]), ("f1", f1[label]), ("support", support[label])):
                    equal_number(value, report["per_class"][name][field], field)
            for domain, part in prediction.groupby("domain"):
                equal_number(f1_score(part.label, part.prediction, average="macro"), report["per_domain"][domain]["macro_f1"], "F1 domínio")
                equal_number(accuracy_score(part.label, part.prediction), report["per_domain"][domain]["accuracy"], "accuracy domínio")
                equal_number(len(part), report["per_domain"][domain]["rows"], "linhas domínio")
            for name, baseline in baseline_predictions.items():
                original = baseline.loc[baseline.seed.eq(seed)]
                pd.testing.assert_frame_equal(sorted_keys(prediction), sorted_keys(original), check_dtype=False)
            if report not in aggregate["reports"]:
                raise ValueError("Relatório individual difere do agregado.")
            audit.append({"seed": seed, "rows": len(prediction), "package_matches": True,
                          "split_matches_all_baselines": True, "metrics_recomputed": True})
            full_reports.append(report)
            predicted_frames.append(prediction)
        scores = np.array([report["macro_f1"] for report in full_reports])
        equal_number(scores.mean(), aggregate["macro_f1_mean"], "média")
        equal_number(scores.std(), aggregate["macro_f1_std"], "desvio")
        # Extrair apenas os artefatos conhecidos; nenhum arquivo do ZIP é executado.
        allowed = ["full/aggregate.json"]
        for group, seeds in (("full", SEEDS), ("smoke", (42,))):
            for seed in seeds:
                allowed.extend(f"{group}/seed_{seed}/{name}" for name in ("report.json", "run_config.json", "predictions.parquet"))
        payloads = {name: archive.read(name) for name in allowed}

    comparison = {}
    for name, source in {**baseline_reports, "B1.4": {"reports": full_reports}}.items():
        reports = source["reports"]
        scores = np.array([report["macro_f1"] for report in reports])
        recalls = {label: float(np.mean([report["per_class"][label]["recall"] for report in reports]))
                   for label in ("PT-PT", "PT-BR")}
        comparison[name] = {
            "macro_f1_mean": float(scores.mean()), "macro_f1_std": float(scores.std()),
            "per_domain_mean": {domain: float(np.mean([report["per_domain"][domain]["macro_f1"] for report in reports]))
                                for domain in ("literary", "news", "social", "speech")},
            "recall_mean": recalls, "absolute_recall_gap": abs(recalls["PT-BR"] - recalls["PT-PT"]),
        }
    combined = pd.concat(predicted_frames, ignore_index=True)
    short = combined.loc[combined.text.str.len().le(40)]
    summary = {
        "source_zip_sha256": hashlib.sha256(args.zip_path.read_bytes()).hexdigest(),
        "package_sha256": package_hash, "audit": audit,
        "comparison": comparison,
        "b1_4_training_seconds": sum(report["training_seconds_this_session"] for report in full_reports),
        "gpu": sorted({report["gpu"] for report in full_reports}),
        "maximum_allocated_gpu_gib": max(report["peak_allocated_gpu_gib"] for report in full_reports),
        "maximum_reserved_gpu_gib": max(report["peak_reserved_gpu_gib"] for report in full_reports),
        "short_text_validation_observations": len(short),
        "short_text_errors": int(short.label.ne(short.prediction).sum()),
        "short_text_error_rate": float(short.label.ne(short.prediction).mean()),
        "limitations": [
            "Validation checkpoint selection; metrics are development estimates.",
            "Seed validations overlap; observations are not independent.",
            "Four domains appear in both training and validation; not leave-one-domain-out.",
            "Recall asymmetry increases despite aggregate and per-domain improvements.",
            "ZIP has predictions and reports only; saved model weights remain in Colab/Drive.",
        ],
    }
    DESTINATION.mkdir(parents=True)
    for name, data in payloads.items():
        path = DESTINATION / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (DESTINATION / "audit_and_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
