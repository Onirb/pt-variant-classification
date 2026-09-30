"""Uma avaliação final, com previsões persistidas e resultado congelado."""

import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
import psutil

from src.fine_tuned_inference import DEFAULT_ARTIFACT, FineTunedVariantClassifier, sha256_file
from src.track_b_final_test import AUDIT_DIR, DATA_DIR, FINAL_DIR, TEST_PATH, LEXICAL_RULES, metrics


def validate_saved_predictions(rows, frame):
    if len(rows) > len(frame):
        raise ValueError("Mais previsões que textos elegíveis.")
    for index, record in enumerate(rows):
        expected = frame.iloc[index]
        if record["id"] != str(expected.id) or record["actual"] != int(expected.label):
            raise ValueError("Previsões salvas não são um prefixo da população congelada.")
        if record["predicted"] not in (0, 1) or not math.isfinite(record["logit_difference_br_minus_pt"]):
            raise ValueError("Previsão salva inválida.")


def verify_completed_output(root):
    freeze = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    for name, expected in freeze["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != expected:
            raise ValueError("Avaliação congelada foi alterada.")
    return json.loads((root / "report.json").read_text(encoding="utf-8"))


def main():
    if (FINAL_DIR / "freeze_manifest.json").exists():
        report = verify_completed_output(FINAL_DIR)
        print(json.dumps({"status": "already complete; no inference repeated", "primary": report["primary"]}, indent=2))
        return
    audit_path = AUDIT_DIR / "report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["ready_for_inference"] or audit["fit_overlap_ids"] or audit["lexical_rules"] != LEXICAL_RULES:
        raise ValueError("Auditoria não autoriza inferência final.")
    model_hash = sha256_file(DEFAULT_ARTIFACT / "manifest.json")
    if model_hash != audit["model_manifest_sha256"] or sha256_file(TEST_PATH) != audit["data_sha256"]:
        raise ValueError("Modelo ou corpus diferem da auditoria pré-inferência.")
    source_manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    for name, evidence in source_manifest["sources"].items():
        if sha256_file(DATA_DIR / name) != evidence["sha256"]:
            raise ValueError("Fonte oficial alterada.")
    frame = pd.read_parquet(TEST_PATH)
    frame = frame.loc[frame.label.notna()].reset_index(drop=True)
    membership = pd.read_parquet(AUDIT_DIR / "frozen_membership.parquet")
    if frame[["id", "official_test_line", "raw_label"]].to_dict("records") != membership.to_dict("records"):
        raise ValueError("População final difere dos IDs e rótulos congelados.")
    config = {"model_manifest_sha256": model_hash, "test_sha256": audit["data_sha256"],
              "audit_sha256": sha256_file(audit_path), "population_rows": len(frame),
              "protocol_sha256": sha256_file(Path("docs/TRACK_B_B4_PROTOCOL_V1.md")),
              "evaluation_script_sha256": sha256_file(Path(__file__)), "cpu_threads": 4,
              "decision_rule": "argmax; no threshold calibration"}
    state_path = FINAL_DIR / "execution_state.json"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8"))["configuration"] != config:
            raise ValueError("Execução iniciada com outro protocolo; não repetir.")
    else:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(),
                                         "configuration": config}, indent=2) + "\n", encoding="utf-8")
    prediction_path = FINAL_DIR / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()] if prediction_path.exists() else []
    validate_saved_predictions(rows, frame)
    if len(rows) < len(frame):
        classifier = FineTunedVariantClassifier(DEFAULT_ARTIFACT, threads=4)
        classifier.verify_probes()
        with prediction_path.open("a", encoding="utf-8") as stream:
            for index in range(len(rows), len(frame)):
                item = frame.iloc[index]
                started = perf_counter()
                prediction = classifier.predict(str(item.text))
                elapsed = perf_counter() - started
                record = {"id": str(item.id), "official_test_line": int(item.official_test_line),
                          "actual": int(item.label), "predicted": 0 if prediction.label == "PT-PT" else 1,
                          "logit_difference_br_minus_pt": prediction.logit_difference_br_minus_pt,
                          "was_truncated": prediction.was_truncated, "characters": prediction.characters,
                          "inference_ms": elapsed * 1000}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()  # IDs já persistidos não serão inferidos novamente na retomada.
                rows.append(record)
                if (index + 1) % 50 == 0 or index + 1 == len(frame):
                    print(f"Previsões persistidas: {index + 1}/{len(frame)}", flush=True)
        del classifier
    validate_saved_predictions(rows, frame)
    excluded = set(audit["previously_seen_overlap_ids"])
    clean = [record for record in rows if record["id"] not in excluded]
    primary = metrics([record["actual"] for record in rows], [record["predicted"] for record in rows])
    sensitivity = metrics([record["actual"] for record in clean], [record["predicted"] for record in clean])
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(), "configuration": config,
        "scope": "single fixed candidate on official binary DSL-TL test; selection ended",
        "source_revision": source_manifest["revision"], "domain": "journalistic",
        "label_nature": "official human New.Gold.Label recovered through exact ID/text checks",
        "ambiguous_labels_excluded": 59, "primary": primary,
        "clean_sensitivity": sensitivity, "sensitivity_excluded_ids": sorted(excluded),
        "fit_contamination_ids": audit["fit_overlap_ids"],
        "truncated_rows": sum(record["was_truncated"] for record in rows),
        "inference_seconds_total": sum(record["inference_ms"] for record in rows) / 1000,
        "median_inference_ms": statistics.median(record["inference_ms"] for record in rows),
        "process_rss_mib_after_inference": psutil.Process().memory_info().rss / 1024**2,
        "python": platform.python_version(),
        "interpretation_limits": ["same benchmark family as DSL-TL dev already used in Track A",
            "one natural journalistic domain; not evidence for all domains",
            "lexical near-duplicate heuristics do not audit pretraining or paraphrases",
            "class imbalance: 299 PT-BR and 137 PT-PT; report macro F1 and per-class recall",
            "sensibility subset reuses predictions and is not another independent test",
            "no threshold/model/epoch adjustment authorized from this final test"],
    }
    report_path = FINAL_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    freeze = {"sha256": {path.name: sha256_file(path) for path in (state_path, prediction_path, report_path)},
              "selection_closed": True, "no_second_inference": True}
    (FINAL_DIR / "freeze_manifest.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
