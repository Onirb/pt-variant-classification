"""Importa anotações oficiais fixadas; nunca chama modelo ou faz previsões."""

import json
from datetime import datetime, timezone
from urllib.request import urlopen

from src.fine_tuned_inference import sha256_file
from src.track_b_final_test import DATA_DIR, REVISION, TEST_PATH, recover_test


FILES = {
    "DSL-TL-test.tsv": "DSL-TL-Corpus/Test-DSL-TL/DSL-TL-test.tsv",
    "PT_withFeatures.tsv": "DSL-TL-Corpus/Features-DSL-TL/PT-Features/PT_withFeatures.tsv",
    "PT_train.tsv": "DSL-TL-Corpus/PT-DSL-TL/PT_train.tsv",
    "PT_dev.tsv": "DSL-TL-Corpus/PT-DSL-TL/PT_dev.tsv",
    "corpus_README.md": "DSL-TL-Corpus/README.md",
    "test_README.md": "DSL-TL-Corpus/Test-DSL-TL/README.md",
}


def main():
    if DATA_DIR.exists():
        raise FileExistsError("Importação final já existe; não sobrescrever.")
    DATA_DIR.mkdir(parents=True)
    sources = {}
    for name, remote_path in FILES.items():
        url = f"https://raw.githubusercontent.com/LanguageTechnologyLab/DSL-TL/{REVISION}/{remote_path}"
        with urlopen(url, timeout=30) as response:
            payload = response.read()
        target = DATA_DIR / name
        target.write_bytes(payload)
        sources[name] = {"url": url, "bytes": len(payload), "sha256": sha256_file(target)}
    frame, recovery = recover_test(DATA_DIR)
    if len(frame.loc[frame.label.notna()]) != 436:
        raise ValueError("Tamanho da tarefa binária inesperado.")
    frame.to_parquet(TEST_PATH, index=False)
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "revision": REVISION,
                "sources": sources, "recovery": recovery, "binary_rows": 436,
                "binary_exclusion": "raw_label PT; both/neither; no predicted labels",
                "data_sha256": sha256_file(TEST_PATH), "predictions_made": False}
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
