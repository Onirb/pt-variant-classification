"""Testes locais das garantias de isolamento do Track B."""

import unittest

import pandas as pd

from src.track_b_experiments import document_split


class DocumentSplitTests(unittest.TestCase):
    def test_documents_do_not_cross_split(self):
        frame = pd.DataFrame(
            {
                "text": [f"texto {document} {label}" for document in range(10) for label in (0, 1)],
                "label": [label for _ in range(10) for label in (0, 1)],
                "domain": [f"dominio-{document % 2}" for document in range(10) for _ in (0, 1)],
                "document_id": [f"doc-{document}" for document in range(10) for _ in (0, 1)],
            }
        )
        split = document_split(frame, seed=42)
        self.assertFalse(set(split.train["document_id"]) & set(split.validation["document_id"]))
        self.assertEqual(set(split.train["label"]), {0, 1})
        self.assertEqual(set(split.validation["label"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
