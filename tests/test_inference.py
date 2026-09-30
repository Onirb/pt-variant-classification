"""Testes unitários que não carregam o encoder nem artefatos grandes."""

import unittest

from src.inference import validate_text


class ValidateTextTests(unittest.TestCase):
    def test_trims_valid_text(self):
        self.assertEqual(validate_text("  Olá, tudo bem?  "), "Olá, tudo bem?")

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            validate_text("   ")

    def test_rejects_non_string(self):
        with self.assertRaises(TypeError):
            validate_text(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
