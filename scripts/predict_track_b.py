"""CLI offline do artefato final Track B, mantendo o CLI do Track A disponível."""

import argparse
import json
from pathlib import Path

from src.fine_tuned_inference import DEFAULT_ARTIFACT, FineTunedVariantClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.text is None and not args.verify:
        parser.error("Informe um texto ou use --verify.")
    classifier = FineTunedVariantClassifier(args.artifact_dir)
    if args.verify:
        print(json.dumps(classifier.verify_probes(), indent=2))
    if args.text is not None:
        print(json.dumps(classifier.predict(args.text).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
