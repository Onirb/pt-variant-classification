"""Gera a visão de desenvolvimento do Track B sem rótulos contraditórios."""

from __future__ import annotations

import json

import pandas as pd

from src.track_b import (
    TRACK_B_DATA_PATH,
    TRACK_B_PREPARED_PATH,
    TRACK_B_PREPARATION_MANIFEST_PATH,
    prepare_wmt24pp_development,
    save_prepared_wmt24pp_development,
)


def main() -> None:
    if not TRACK_B_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Importação WMT24++ ausente: {TRACK_B_DATA_PATH}. Execute fetch_track_b_wmt24pp primeiro."
        )
    frame = pd.read_parquet(TRACK_B_DATA_PATH)
    prepared, manifest = prepare_wmt24pp_development(frame)
    save_prepared_wmt24pp_development(prepared, manifest)
    print(
        json.dumps(
            {
                "data_path": str(TRACK_B_PREPARED_PATH),
                "manifest_path": str(TRACK_B_PREPARATION_MANIFEST_PATH),
                **manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
