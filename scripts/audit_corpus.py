"""Gera um resumo local do corpus usado na reprodução."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.data import ID_TO_LABEL, load_corpus


def main() -> None:
    cache_dir = os.environ.get("HF_HOME")
    train, test, summary = load_corpus(cache_dir=cache_dir)
    report = {
        "dataset": "cc4051/pt_vid",
        "train": summary.to_dict(),
        "test": {
            "rows": len(test),
            "label_counts": {
                ID_TO_LABEL[key]: int(value)
                for key, value in test["label"].value_counts().sort_index().items()
            },
        },
    }
    output = Path("runs/corpus_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
