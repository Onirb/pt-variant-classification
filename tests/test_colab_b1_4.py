"""Integridade, isolamento e retomada; não treina o BERTimbau localmente."""

import hashlib
import json
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from zipfile import ZipFile

import torch
from transformers import BertConfig, BertForSequenceClassification, Trainer, default_data_collator

from colab.train_b1_4 import load_inputs, training_arguments, compute_metrics
from tests.support import build_test_bundle


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = build_test_bundle(Path(self.temporary.name) / "bundle")
        with ZipFile(self.bundle) as archive:
            archive.extractall(self.temporary.name)
        self.root = Path(self.temporary.name) / "track_b_colab"

    def test_frozen_splits_cover_every_row_without_shared_documents(self):
        frame, splits, manifest = load_inputs(self.root)
        self.assertEqual(len(frame), 1877)
        for seed in (7, 42, 2026):
            self.assertEqual(len(splits[str(seed)]["validation_indices"]), 398)
        self.assertEqual(len(manifest["sha256"]), 5)

    def test_tampered_data_is_rejected(self):
        with (self.root / "data/development.parquet").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "alterado"):
            load_inputs(self.root)

    def test_document_leak_is_rejected_even_with_valid_file_checksum(self):
        split_path = self.root / "data/splits.json"
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        train = splits["7"]["train_indices"]
        validation = splits["7"]["validation_indices"]
        # Troca um membro de um documento entre os lados, preservando índices únicos.
        train[0], validation[0] = validation[0], train[0]
        split_path.write_text(json.dumps(splits), encoding="utf-8")
        manifest_path = self.root / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sha256"]["data/splits.json"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "documento compartilhado"):
            load_inputs(self.root)

    def test_notebook_cells_are_valid_python(self):
        notebook = json.loads(Path("colab/Track_B_B1_4_Colab.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "colab-cell", "exec")

    def test_upload_cell_accepts_colab_renamed_zip(self):
        notebook = json.loads(Path("colab/Track_B_B1_4_Colab.ipynb").read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and "files.upload()" in "".join(cell["source"]))
        code = "".join(cell["source"])
        extraction = Path(self.temporary.name) / "renamed-upload"
        code = code.replace("Path('/content/pt_variant_b1_4')", "Path(" + repr(str(extraction)) + ")")
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.files = types.SimpleNamespace(upload=lambda: {
            "track_b_colab_package_v1 (2).zip": self.bundle.read_bytes()
        })
        google.colab = colab
        with patch.dict("sys.modules", {"google": google, "google.colab": colab}):
            namespace = {}
            exec(compile(code, "renamed-upload-cell", "exec"), namespace)
        self.assertEqual(namespace["manifest"]["rows"], 1877)
        self.assertTrue((namespace["PACKAGE_ROOT"] / "data/development.parquet").is_file())


class TinyTrainingCheckpointTests(unittest.TestCase):
    def test_training_evaluation_and_checkpoint_resume(self):
        torch.set_num_threads(2)
        config = BertConfig(vocab_size=128, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32, num_labels=2)
        samples = [{"input_ids": [2, 7 + label, 4, 5, 3],
                    "attention_mask": [1] * 5, "labels": label}
                   for _ in range(16) for label in (0, 1)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = training_arguments(output, 42, smoke=True, cpu_test=True)
            args.disable_tqdm = True
            args.logging_strategy = "no"
            args.save_steps = 5
            args.eval_steps = 5
            trainer = Trainer(model=BertForSequenceClassification(config), args=args,
                              train_dataset=samples, eval_dataset=samples,
                              data_collator=default_data_collator, compute_metrics=compute_metrics)
            trainer.train()
            checkpoint = output / "checkpoint-5"
            self.assertTrue((checkpoint / "optimizer.pt").is_file())
            self.assertTrue((checkpoint / "rng_state.pth").is_file())
            self.assertTrue((checkpoint / "scheduler.pt").is_file())
            resumed = Trainer(model=BertForSequenceClassification(config), args=args,
                              train_dataset=samples, eval_dataset=samples,
                              data_collator=default_data_collator, compute_metrics=compute_metrics)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 10)
            self.assertIn("eval_macro_f1", resumed.evaluate())


if __name__ == "__main__":
    unittest.main()
