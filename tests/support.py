"""Synthetic fixtures: no external corpus, model download or private runs needed."""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from scripts import prepare_track_b_final_colab as selection_module


def fixture_selection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for seed, checkpoint in ((7, 100), (42, 250), (2026, 250)):
            path = root / "full" / f"seed_{seed}" / "report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"seed": seed, "best_model_checkpoint": f"checkpoint-{checkpoint}",
                "global_steps": 279, "training_metrics": {"epoch": 3.0}}), encoding="utf-8")
        with patch.object(selection_module, "RESULTS", root):
            return selection_module.selection_from_results()


def build_test_bundle(directory, *, final=False):
    directory = Path(directory)
    package = directory / ("track_b_final" if final else "track_b_colab")
    (package / "data").mkdir(parents=True)
    frame = pd.DataFrame({
        "text": [f"synthetic fixture sentence {index}" for index in range(1877)],
        "label": [index % 2 for index in range(1877)],
        "document_id": [f"fixture_document_{index // 2}" for index in range(1877)],
        "domain": [["news", "social", "literary", "speech"][(index // 2) % 4] for index in range(1877)],
        "segment_id": list(range(1877)),
        "lp": ["en-pt_PT" if index % 2 == 0 else "en-pt_BR" for index in range(1877)],
    })
    frame.to_parquet(package / "data/development.parquet", index=False)
    splits = {str(seed): {"train_indices": list(range(398, 1877)),
                         "validation_indices": list(range(398))} for seed in (7, 42, 2026)}
    (package / "data/splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (package / "data/preparation_manifest.json").write_text('{"scope": "synthetic test fixture"}', encoding="utf-8")
    (package / "requirements-colab.txt").write_text("# synthetic fixture; not installed\n", encoding="utf-8")
    (package / "train_b1_4.py").write_text("# synthetic package member; never executed\n", encoding="utf-8")
    if final:
        (package / "selection.json").write_text(json.dumps(fixture_selection()), encoding="utf-8")
        (package / "train_track_b_final.py").write_text("# synthetic package member; never executed\n", encoding="utf-8")
    manifest = {"rows": 1877, "seeds": [7, 42, 2026], "scope": "synthetic test fixture only",
                "sha256": {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in package.rglob("*") if path.is_file()}}
    (package / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive_path = directory / "synthetic_bundle.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in package.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    return archive_path
