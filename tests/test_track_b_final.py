"""Testes pequenos e offline; não treinam o BERTimbau completo no Windows."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, Trainer, default_data_collator

from colab.train_track_b_final import final_training_arguments
from tests.support import build_test_bundle, fixture_selection
from scripts import import_track_b_final_model as importer
from src.fine_tuned_inference import FineTunedVariantClassifier, LABEL_MAP, sha256_file


class FinalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = build_test_bundle(self.temporary.name, final=True)

    def test_budget_uses_median_development_checkpoints(self):
        selection = fixture_selection()
        self.assertEqual([row["best_step"] for row in selection["evidence"]], [100, 250, 250])
        self.assertAlmostEqual(selection["median_epoch_fraction"], 250 / 93)
        self.assertEqual(selection["final_protocol"]["max_steps"], 318)
        self.assertEqual(selection["final_protocol"]["seed"], 42)
        self.assertEqual(selection["final_protocol"]["rows"], 1877)

    def test_final_training_has_no_validation_or_best_model_selection(self):
        torch.set_num_threads(2)
        selection = fixture_selection()
        with tempfile.TemporaryDirectory() as directory:
            args = final_training_arguments(Path(directory), selection, cpu_test=True)
            self.assertEqual(args.eval_strategy.value, "no")
            self.assertFalse(args.load_best_model_at_end)
            self.assertEqual(args.max_steps, 318)
            args.max_steps = 4
            args.save_steps = 2
            args.logging_strategy = "no"
            args.disable_tqdm = True
            model = BertForSequenceClassification(BertConfig(
                vocab_size=16, hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16, num_labels=2))
            samples = [{"input_ids": [2, 6 + label, 3], "attention_mask": [1, 1, 1], "labels": label}
                       for _ in range(10) for label in (0, 1)]
            trainer = Trainer(model=model, args=args, train_dataset=samples,
                              data_collator=default_data_collator)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 4)
            self.assertIsNone(trainer.eval_dataset)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            self.assertTrue((Path(directory) / "checkpoint-4" / "model.safetensors").is_file())

    def test_notebook_syntax_and_renamed_upload(self):
        notebook = json.loads(Path("colab/Track_B_Final_Colab.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "final-colab-cell", "exec")
        cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                    and "files.upload()" in "".join(cell["source"]))
        with tempfile.TemporaryDirectory() as directory:
            code = "".join(cell["source"]).replace("Path('/content/pt_variant_final')", "Path(" + repr(directory) + ")")
            google = types.ModuleType("google")
            colab = types.ModuleType("google.colab")
            colab.files = types.SimpleNamespace(upload=lambda: {"track_b_final_package_v1 (2).zip": self.bundle.read_bytes()})
            google.colab = colab
            with patch.dict("sys.modules", {"google": google, "google.colab": colab}):
                namespace = {}
                exec(compile(code, "final-upload-cell", "exec"), namespace)
            self.assertEqual(namespace["selection"]["final_protocol"]["max_steps"], 318)
            self.assertEqual(namespace["manifest"]["rows"], 1877)


class OfflineArtifactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "artifact"
        model_dir = self.artifact / "model"
        model_dir.mkdir(parents=True)
        vocab = self.root / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                    "estou", "a", "estudar", "hoje", "ação", "amanhã", "."]) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
        model = BertForSequenceClassification(BertConfig(
            vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, num_labels=2,
            id2label=LABEL_MAP, label2id={value: key for key, value in LABEL_MAP.items()})).eval()
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        self.selection = fixture_selection()
        (self.artifact / "selection.json").write_text(json.dumps(self.selection), encoding="utf-8")
        texts = ["Estou a estudar hoje.", "Ação amanhã."]
        with torch.inference_mode():
            logits = model(**tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")).logits.tolist()
        self.manifest = {
            "artifact_type": "track_b_fine_tuned_classifier", "model": self.selection["model"],
            "revision": self.selection["revision"], "final_protocol": self.selection["final_protocol"],
            "global_steps": 318, "rows_fit": 1877, "development_data_sha256": "test-corpus-hash",
            "probe_texts": texts, "cpu_reload_probe_logits": logits,
            "sha256": {path.relative_to(self.artifact).as_posix(): sha256_file(path)
                       for path in self.artifact.rglob("*") if path.is_file()},
        }
        (self.artifact / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_offline_unicode_deterministic_and_truncation(self):
        classifier = FineTunedVariantClassifier(self.artifact, threads=2)
        self.assertTrue(classifier.verify_probes()["probe_logits_match"])
        first = classifier.predict("Ação amanhã.")
        self.assertEqual(first, classifier.predict("Ação amanhã."))
        self.assertIn(first.label, LABEL_MAP.values())
        self.assertFalse(first.was_truncated)
        self.assertTrue(classifier.predict("estudar " * 300).was_truncated)
        self.assertIn("not calibrated", first.score_interpretation)

    def test_empty_and_invalid_inputs(self):
        classifier = FineTunedVariantClassifier(self.artifact, threads=2)
        with self.assertRaises(ValueError):
            classifier.predict("  ")
        with self.assertRaises(TypeError):
            classifier.predict(None)

    def test_modified_weights_are_rejected_before_loading(self):
        with (self.artifact / "model/model.safetensors").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "alterado"):
            FineTunedVariantClassifier(self.artifact, threads=2)

    def test_incomplete_training_is_rejected(self):
        self.manifest["global_steps"] = 317
        (self.artifact / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incompleto"):
            FineTunedVariantClassifier(self.artifact, threads=2)

    def test_import_checks_receipt_and_publishes_only_verified_artifact(self):
        package = self.root / "package"
        package.mkdir()
        selection_path = package / "selection.json"
        selection_path.write_text(json.dumps(self.selection), encoding="utf-8")
        (package / "package_manifest.json").write_text(json.dumps(
            {"sha256": {"data/development.parquet": "test-corpus-hash"}}), encoding="utf-8")
        zip_path = self.root / "synthetic_model.zip"
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in self.artifact.rglob("*"):
                if path.is_file():
                    archive.write(path, "track_b_final_v1/" + path.relative_to(self.artifact).as_posix())
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps({"manifest": self.manifest,
            "artifact_zip_sha256": sha256_file(zip_path), "artifact_zip_bytes": zip_path.stat().st_size}), encoding="utf-8")
        destination = self.root / "imported"
        with patch.object(importer, "SELECTION", selection_path), patch.object(importer, "DEFAULT_ARTIFACT", destination), \
                patch("sys.argv", ["importer", str(zip_path), str(receipt_path)]):
            importer.main()
            self.assertTrue(destination.is_dir())
            self.assertFalse(destination.with_name("imported.pending").exists())
            importer.main()  # Repetir valida sem sobrescrever.
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifact_zip_sha256"] = "invalid"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Hash"):
                importer.main()


if __name__ == "__main__":
    unittest.main()
