"""Testes sintéticos do gold oficial e auditoria; não fazem inferência no teste."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.track_b_final_test import normalize, recover_test, lexical_overlap, metrics


class GoldRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "PT_withFeatures.tsv").write_text(
            "Id\tInstance\tMarkers\tNew.Gold.Label\n"
            "1\tmesmo texto\t[]\tPT-BR\n"
            "2\tdev texto\t[]\tPT-PT\n"
            "3\tmesmo texto\t[]\tPT-BR\n"
            "4\tambíguo\t[]\tPT\n", encoding="utf-8")
        (self.root / "PT_train.tsv").write_text("1\tmesmo texto\tPT-BR\n", encoding="utf-8")
        (self.root / "PT_dev.tsv").write_text("2\tdev texto\tPT-PT\n", encoding="utf-8")
        (self.root / "DSL-TL-test.tsv").write_text("outro idioma\nambíguo\nmesmo texto\n", encoding="utf-8")

    def test_recovery_uses_ids_not_arbitrary_first_text_match(self):
        frame, report = recover_test(self.root, expected_portuguese_rows=2)
        self.assertEqual(frame.id.tolist(), ["4", "3"])
        self.assertEqual(frame.official_test_line.tolist(), [2, 3])
        self.assertEqual(frame.loc[frame.label.notna(), "label"].tolist(), [1])
        self.assertEqual(report["known_train_dev_ids"], 2)

    def test_inconsistent_gold_is_rejected(self):
        (self.root / "PT_train.tsv").write_text("1\tmesmo texto\tPT-PT\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "não corresponde"):
            recover_test(self.root, expected_portuguese_rows=2)

    def test_missing_official_test_member_is_rejected(self):
        (self.root / "DSL-TL-test.tsv").write_text("mesmo texto\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "não cobre"):
            recover_test(self.root, expected_portuguese_rows=2)

    def test_duplicate_test_sentences_are_rejected(self):
        (self.root / "DSL-TL-test.tsv").write_text("mesmo texto\nmesmo texto\nambíguo\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicados"):
            recover_test(self.root, expected_portuguese_rows=2)


class LexicalAuditTests(unittest.TestCase):
    def test_unicode_case_and_whitespace_equality(self):
        self.assertEqual(normalize("  Ａção\nAMANHÃ  "), "ação amanhã")

    def test_exact_and_near_matches_ignore_labels(self):
        text = "A análise desta amostra científica considera diferentes condições de observação e registra os procedimentos necessários para documentar resultados reproduzíveis de maneira transparente."
        query = pd.DataFrame({"id": ["q1", "q2", "q3"], "text": ["  AÇÃO amanhã ", text, "texto inteiramente diferente"], "label": [0, 1, 1]})
        references = pd.DataFrame({"id": ["r1", "r2"], "source": ["fixture", "fixture"],
                                   "text": ["ação AMANHÃ", text[:-1] + "!"]})
        hits = lexical_overlap(query, references)
        self.assertEqual({hit["test_id"] for hit in hits}, {"q1", "q2"})
        self.assertIn("exact_normalized", {hit["method"] for hit in hits})
        self.assertIn("char5_cosine", {hit["method"] for hit in hits})
        self.assertIn("word5_jaccard", {hit["method"] for hit in hits})
        query.label = 1 - query.label
        self.assertEqual(hits, lexical_overlap(query, references))

    def test_known_metrics_and_confusion_order(self):
        result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["confusion_matrix_rows_actual_columns_predicted"], [[1, 1], [0, 2]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["recall_gap_br_minus_pt"], 0.5)


if __name__ == "__main__":
    unittest.main()
