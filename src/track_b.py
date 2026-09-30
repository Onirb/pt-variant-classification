"""Importação rastreável e auditoria de dados do Track B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from .data import LABEL_TO_ID, load_corpus
from .external import (
    DSL_TL_PT_DEV_PATH,
    FRMT_TEST_DIR,
    PTBRVID_VALID_PATH,
    load_dsl_tl_pt_dev,
    load_frmt_test,
    load_ptbrvid_valid,
    normalized_texts,
)


WMT24PP_DATASET = "google/wmt24pp"
WMT24PP_REVISION = "fd7405c06494bc66a57b25f55d217a72f96e60dc"
WMT24PP_CONFIGS = {"en-pt_BR": LABEL_TO_ID["PT-BR"], "en-pt_PT": LABEL_TO_ID["PT-PT"]}
TRACK_B_DIR = Path("data/track_b/wmt24pp_development_v1")
TRACK_B_DATA_PATH = TRACK_B_DIR / "development.parquet"
TRACK_B_MANIFEST_PATH = TRACK_B_DIR / "manifest.json"
TRACK_B_PREPARED_PATH = TRACK_B_DIR / "development_prepared.parquet"
TRACK_B_PREPARATION_MANIFEST_PATH = TRACK_B_DIR / "preparation_manifest.json"
REQUIRED_COLUMNS = (
    "text",
    "label",
    "lp",
    "domain",
    "document_id",
    "segment_id",
    "source_en",
    "original_target",
)


@dataclass(frozen=True)
class AuditResult:
    source: str
    rows: int
    normalized_overlap: int | None
    status: str


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _dataset_revision() -> str:
    # Reproduce the recorded corpus, not a moving Hub branch.
    return WMT24PP_REVISION


def build_wmt24pp_development(*, cache_dir: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carrega as duas variantes WMT24++, removendo fontes marcadas inválidas."""
    revision = _dataset_revision()
    frames: list[pd.DataFrame] = []
    raw_rows: dict[str, int] = {}
    removed_bad_source: dict[str, int] = {}

    for config, label in WMT24PP_CONFIGS.items():
        dataset = load_dataset(
            WMT24PP_DATASET,
            name=config,
            split="train",
            revision=revision,
            cache_dir=cache_dir,
        )
        frame = dataset.to_pandas()
        expected = {
            "lp",
            "domain",
            "document_id",
            "segment_id",
            "is_bad_source",
            "source",
            "target",
            "original_target",
        }
        missing = expected.difference(frame.columns)
        if missing:
            raise ValueError(f"{config}: colunas ausentes: {sorted(missing)}")
        raw_rows[config] = len(frame)
        valid = frame.loc[~frame["is_bad_source"].astype(bool)].copy()
        removed_bad_source[config] = len(frame) - len(valid)
        valid = valid.rename(columns={"target": "text", "source": "source_en"})
        valid["label"] = label
        valid = valid.loc[:, REQUIRED_COLUMNS]
        valid["text"] = valid["text"].fillna("").astype(str).str.strip()
        valid = valid.loc[valid["text"].ne("")].copy()
        frames.append(valid)

    combined = pd.concat(frames, ignore_index=True)
    duplicate_texts = int(combined["text"].duplicated(keep=False).sum())
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": WMT24PP_DATASET,
        "revision": revision,
        "configs": list(WMT24PP_CONFIGS),
        "split": "train",
        "text_field": "target",
        "excluded": {"is_bad_source": True, "empty_target": True},
        "raw_rows_by_config": raw_rows,
        "removed_bad_source_by_config": removed_bad_source,
        "rows_after_filter": int(len(combined)),
        "label_counts": {
            "PT-PT": int((combined["label"] == LABEL_TO_ID["PT-PT"]).sum()),
            "PT-BR": int((combined["label"] == LABEL_TO_ID["PT-BR"]).sum()),
        },
        "domain_counts": {str(key): int(value) for key, value in combined["domain"].value_counts().sort_index().items()},
        "document_count": int(combined["document_id"].nunique()),
        "texts_in_duplicate_groups": duplicate_texts,
    }
    return combined.reset_index(drop=True), manifest


def save_wmt24pp_development(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    destination: Path = TRACK_B_DATA_PATH,
    manifest_path: Path = TRACK_B_MANIFEST_PATH,
) -> None:
    """Persiste importação uma vez; não sobrescreve artefato congelado."""
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Importação Track B já existe em {destination.parent}; não será sobrescrita automaticamente."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    final_manifest = dict(manifest)
    final_manifest["data_file"] = str(destination)
    final_manifest["data_sha256"] = _sha256_file(destination)
    manifest_path.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_wmt24pp_development(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove textos com rótulos opostos sem alterar a importação bruta.

    Um texto final associado a PT-BR e PT-PT não fornece evidência para a
    classificação binária. A regra é aplicada antes da divisão documental e
    não consulta métricas nem modelos.
    """
    labels_by_text = frame.groupby("text")["label"].nunique()
    conflicting_texts = set(labels_by_text.loc[labels_by_text.gt(1)].index)
    conflict_mask = frame["text"].isin(conflicting_texts)
    prepared = frame.loc[~conflict_mask].copy()
    before_deduplication = len(prepared)
    prepared = prepared.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input_file": str(TRACK_B_DATA_PATH),
        "rule": "drop every text with more than one binary label; then deduplicate exact same-label text",
        "input_rows": int(len(frame)),
        "conflicting_text_groups": int(len(conflicting_texts)),
        "rows_removed_for_conflicting_labels": int(conflict_mask.sum()),
        "same_label_duplicates_removed": int(before_deduplication - len(prepared)),
        "rows_after_preparation": int(len(prepared)),
        "label_counts": {
            "PT-PT": int((prepared["label"] == LABEL_TO_ID["PT-PT"]).sum()),
            "PT-BR": int((prepared["label"] == LABEL_TO_ID["PT-BR"]).sum()),
        },
        "domain_counts": {
            str(key): int(value) for key, value in prepared["domain"].value_counts().sort_index().items()
        },
    }
    return prepared, manifest


def save_prepared_wmt24pp_development(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    destination: Path = TRACK_B_PREPARED_PATH,
    manifest_path: Path = TRACK_B_PREPARATION_MANIFEST_PATH,
) -> None:
    """Salva a visão preparada uma única vez e registra hash de integridade."""
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Visão preparada Track B já existe em {destination.parent}; não será sobrescrita automaticamente."
        )
    frame.to_parquet(destination, index=False)
    final_manifest = dict(manifest)
    final_manifest["data_file"] = str(destination)
    final_manifest["data_sha256"] = _sha256_file(destination)
    manifest_path.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _historic_frames() -> list[tuple[str, pd.DataFrame]]:
    """Carrega fontes já presentes; ausência vira registro, não download implícito."""
    frames: list[tuple[str, pd.DataFrame]] = []
    development, public_test, _ = load_corpus(cache_dir=".hf_cache")
    frames.extend([("track_a_development", development), ("public_test", public_test)])
    if DSL_TL_PT_DEV_PATH.exists():
        frames.append(("dsl_tl_pt_dev", load_dsl_tl_pt_dev()[0]))
    if PTBRVID_VALID_PATH.exists():
        frames.append(("ptbrvid_valid", load_ptbrvid_valid()))
    if FRMT_TEST_DIR.exists():
        frames.append(("frmt_test_frozen", load_frmt_test()))
    return frames


def audit_against_historic_data(frame: pd.DataFrame) -> list[AuditResult]:
    """Mede sobreposição textual normalizada sem alterar nenhuma fonte histórica."""
    candidate_texts = normalized_texts(frame)
    results: list[AuditResult] = []
    for source, historic in _historic_frames():
        overlap = len(candidate_texts.intersection(normalized_texts(historic)))
        results.append(
            AuditResult(
                source=source,
                rows=int(len(historic)),
                normalized_overlap=overlap,
                status="ok" if overlap == 0 else "review_required",
            )
        )
    for source, path in (
        ("dsl_tl_pt_dev", DSL_TL_PT_DEV_PATH),
        ("ptbrvid_valid", PTBRVID_VALID_PATH),
        ("frmt_test_frozen", FRMT_TEST_DIR),
    ):
        if not path.exists():
            results.append(AuditResult(source=source, rows=0, normalized_overlap=None, status="not_available"))
    return results


def write_audit_report(
    results: list[AuditResult],
    *,
    destination: Path = TRACK_B_DIR / "overlap_audit.json",
) -> None:
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison": "lowercase + whitespace normalization, exact match only",
        "results": [asdict(result) for result in results],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
