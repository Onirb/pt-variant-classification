"""Guarda de retomada e congelamento com artefatos sintéticos, não o teste real."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.evaluate_track_b_final_test import validate_saved_predictions, verify_completed_output
from scripts import evaluate_track_b_final_test as evaluator
from src.fine_tuned_inference import sha256_file


class ExecutionGuards(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"id": ["a", "b"], "label": [0, 1]})
        self.rows = [{"id": "a", "actual": 0, "predicted": 1, "logit_difference_br_minus_pt": 1.0}]

    def test_valid_partial_prefix_can_resume(self):
        validate_saved_predictions(self.rows, self.frame)

    def test_wrong_order_is_rejected(self):
        self.rows[0]["id"] = "b"
        with self.assertRaisesRegex(ValueError, "prefixo"):
            validate_saved_predictions(self.rows, self.frame)

    def test_nonfinite_prediction_is_rejected(self):
        self.rows[0]["logit_difference_br_minus_pt"] = float("nan")
        with self.assertRaisesRegex(ValueError, "inválida"):
            validate_saved_predictions(self.rows, self.frame)

    def test_frozen_report_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text('{"primary": {"macro_f1": 0.5}}', encoding="utf-8")
            (root / "freeze_manifest.json").write_text(json.dumps({"sha256": {"report.json": sha256_file(report)}}), encoding="utf-8")
            self.assertEqual(verify_completed_output(root)["primary"]["macro_f1"], 0.5)
            report.write_text('{"primary": {"macro_f1": 0.9}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alterada"):
                verify_completed_output(root)

    def test_completed_run_does_not_load_or_run_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text('{"primary": {"macro_f1": 0.5}}', encoding="utf-8")
            (root / "freeze_manifest.json").write_text(json.dumps({"sha256": {"report.json": sha256_file(report)}}), encoding="utf-8")
            with patch.object(evaluator, "FINAL_DIR", root), patch.object(
                    evaluator, "FineTunedVariantClassifier", side_effect=AssertionError("Não repetir inferência")) as model:
                evaluator.main()
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
