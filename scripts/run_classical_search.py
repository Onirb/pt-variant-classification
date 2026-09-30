"""Executa a busca clássica sem consultar o teste público congelado."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.data import load_corpus
from src.experiments import development_split, run_candidates


def main() -> None:
    corpus, _, summary = load_corpus(cache_dir=os.environ.get("HF_HOME"))
    train, validation = development_split(corpus)
    results = run_candidates(train, validation)
    payload = {
        "protocol": "public-train stratified validation, seed=42; public test untouched",
        "corpus": summary.to_dict(),
        "development_train_rows": len(train),
        "development_validation_rows": len(validation),
        "results": [result.to_dict() for result in results],
    }
    output = Path("runs/classical_search_v2")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
