"""Audita fontes anteriores e congela coortes antes da primeira inferência final."""

import json
from datetime import datetime, timezone

import pandas as pd

from src.fine_tuned_inference import DEFAULT_ARTIFACT, sha256_file, verify_manifest
from src.track_b import _historic_frames, TRACK_B_DATA_PATH
from src.track_b_final_test import AUDIT_DIR, DATA_DIR, TEST_PATH, LEXICAL_RULES, fingerprint, lexical_overlap


FIT_DATA = "runs/track_b/colab_final_bundle_v1/track_b_final/data/development.parquet"


def main():
    from pathlib import Path
    if AUDIT_DIR.exists():
        raise FileExistsError("Auditoria final já existe; não alterar coortes congeladas.")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(TEST_PATH) != manifest["data_sha256"]:
        raise ValueError("Teste recuperado alterado.")
    for name, evidence in manifest["sources"].items():
        if sha256_file(DATA_DIR / name) != evidence["sha256"]:
            raise ValueError(f"Fonte alterada: {name}")
    model_manifest = verify_manifest(DEFAULT_ARTIFACT)
    if sha256_file(Path(FIT_DATA)) != model_manifest["development_data_sha256"]:
        raise ValueError("Corpus de ajuste difere do manifesto do modelo.")
    test = pd.read_parquet(TEST_PATH)
    binary = test.loc[test.label.notna()].copy().reset_index(drop=True)
    references = []
    sources = [("track_b_fit", pd.read_parquet(FIT_DATA)),
               ("track_b_raw_seen", pd.read_parquet(TRACK_B_DATA_PATH))] + _historic_frames()
    source_evidence = {}
    for name, frame in sources:
        reference = pd.DataFrame({"id": frame["id"].astype(str) if "id" in frame else [str(i) for i in range(len(frame))],
                                  "text": frame.text.astype(str), "source": name})
        source_evidence[name] = {"rows": len(reference), "text_fingerprint": fingerprint(reference)}
        references.append(reference)
    # Fonte original PT_train inteira também detecta a duplicata do benchmark.
    train = pd.read_csv(DATA_DIR / "PT_train.tsv", sep="\t", header=None,
                        names=["id", "text", "raw_label"], quoting=3, dtype=str)
    source = "dsl_tl_official_train"
    reference = train[["id", "text"]].copy()
    reference["source"] = source
    source_evidence[source] = {"rows": len(reference), "text_fingerprint": fingerprint(reference)}
    references.append(reference)
    hits = lexical_overlap(binary, pd.concat(references, ignore_index=True))
    fit_ids = sorted({hit["test_id"] for hit in hits if hit["reference_source"] == "track_b_fit"})
    previously_seen_ids = sorted({hit["test_id"] for hit in hits})
    by_source = {}
    for source in source_evidence:
        by_source[source] = {method: len({hit["test_id"] for hit in hits
            if hit["reference_source"] == source and hit["method"] == method})
            for method in ("exact_normalized", "word5_jaccard", "char5_cosine")}
    AUDIT_DIR.mkdir(parents=True)
    binary[["id", "official_test_line", "raw_label"]].to_parquet(AUDIT_DIR / "frozen_membership.parquet", index=False)
    audit = {"generated_at": datetime.now(timezone.utc).isoformat(), "predictions_made": False,
             "data_sha256": manifest["data_sha256"], "model_manifest_sha256": sha256_file(DEFAULT_ARTIFACT / "manifest.json"),
             "lexical_rules": LEXICAL_RULES, "references": source_evidence, "by_source": by_source,
             "fit_overlap_ids": fit_ids, "previously_seen_overlap_ids": previously_seen_ids,
             "official_binary_rows": len(binary), "clean_sensitivity_rows": len(binary) - len(previously_seen_ids),
             "primary": "official binary test; preserve membership; report all fit contamination if present",
             "sensitivity": "same predictions on cohort without any lexical hit against prior sources; no metric-based filtering",
             "ready_for_inference": not fit_ids, "hits": hits}
    (AUDIT_DIR / "report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "hits"}, ensure_ascii=False, indent=2))
    if fit_ids:
        raise ValueError("Sobreposição com ajuste Track B: revisar antes da avaliação final.")


if __name__ == "__main__":
    main()
