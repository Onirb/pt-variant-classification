"""Baixa o conjunto externo DSL-TL PT_dev para avaliação local."""

from __future__ import annotations

from src.external import DSL_TL_PT_DEV_PATH, DSL_TL_PT_DEV_URL, download_dsl_tl_pt_dev, file_sha256


def main() -> None:
    path = download_dsl_tl_pt_dev()
    print(
        {
            "path": str(path),
            "source": DSL_TL_PT_DEV_URL,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    )


if __name__ == "__main__":
    main()
