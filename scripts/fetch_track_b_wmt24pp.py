"""Baixa, congela e audita o conjunto de desenvolvimento aprovado para o Track B."""

from __future__ import annotations

import json

from src.track_b import (
    TRACK_B_DATA_PATH,
    TRACK_B_MANIFEST_PATH,
    audit_against_historic_data,
    build_wmt24pp_development,
    save_wmt24pp_development,
    write_audit_report,
)


def main() -> None:
    frame, manifest = build_wmt24pp_development(cache_dir=".hf_cache")
    save_wmt24pp_development(frame, manifest)
    audit = audit_against_historic_data(frame)
    write_audit_report(audit)
    print(
        json.dumps(
            {
                "data_path": str(TRACK_B_DATA_PATH),
                "manifest_path": str(TRACK_B_MANIFEST_PATH),
                "rows": len(frame),
                "revision": manifest["revision"],
                "audit": [result.__dict__ for result in audit],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
