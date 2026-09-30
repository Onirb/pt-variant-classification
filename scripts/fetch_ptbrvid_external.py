"""Baixa a amostra externa multidomínio PtBrVId, com revisão fixa."""

from __future__ import annotations

from src.external import (
    PTBRVID_DATASET,
    PTBRVID_REVISION,
    PTBRVID_VALID_PATH,
    download_ptbrvid_valid,
    file_sha256,
    load_ptbrvid_valid,
)


def main() -> None:
    path = download_ptbrvid_valid()
    frame = load_ptbrvid_valid(path)
    print(
        {
            "path": str(path),
            "dataset": PTBRVID_DATASET,
            "revision": PTBRVID_REVISION,
            "rows": int(len(frame)),
            "domains": frame["domain"].value_counts().sort_index().to_dict(),
            "labels": frame["label"].value_counts().sort_index().to_dict(),
            "sha256": file_sha256(path),
        }
    )


if __name__ == "__main__":
    main()
