"""Synthetic credential checks; no real tokens or credentials in fixtures."""

import unittest

from scripts.audit_publication import secret_locations, forbidden_new_path
from src.track_b import _dataset_revision


class PublicationChecks(unittest.TestCase):
    def test_known_token_shapes_are_detected_without_returning_values(self):
        sample = "gh" + "p_" + "A" * 36 + "\n" + "h" + "f_" + "B" * 32
        findings = secret_locations(sample)
        self.assertEqual({item["kind"] for item in findings}, {"github_token", "huggingface_token"})
        self.assertTrue(all(set(item) == {"kind", "line"} for item in findings))

    def test_hashes_and_placeholders_are_not_tokens(self):
        self.assertEqual(secret_locations("SHA256=" + "a" * 64 + "\nAPI_KEY=<your-key>"), [])

    def test_private_key_header_is_detected(self):
        sample = "-----" + "BEGIN " + "RSA PRIVATE " + "KEY-----"
        self.assertEqual(secret_locations(sample)[0]["kind"], "private_key")

    def test_artifacts_and_local_credentials_are_blocked_as_new_files(self):
        for path in ("runs/report.json", "data/external/test.tsv", ".env", ".aws/config", "model.safetensors"):
            self.assertTrue(forbidden_new_path(path))
        self.assertFalse(forbidden_new_path("docs/RESULT.md"))
        self.assertFalse(forbidden_new_path(".env.example"))

    def test_wmt_source_revision_is_frozen_without_network(self):
        self.assertEqual(_dataset_revision(), "fd7405c06494bc66a57b25f55d217a72f96e60dc")


if __name__ == "__main__":
    unittest.main()
