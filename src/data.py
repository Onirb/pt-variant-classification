"""Aquisição e validação determinística do corpus pt-BR versus pt-PT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset


REMOTE_DATASET = "cc4051/pt_vid"
LABEL_TO_ID = {"PT-PT": 0, "PT-BR": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class CorpusSummary:
    """Resumo rastreável de uma divisão preparada."""

    rows: int
    label_counts: dict[str, int]
    sources: dict[str, int]
    excluded_local_labels: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {source}: {sorted(missing)}")

    cleaned = frame.loc[:, ["text", "label"]].copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned.loc[cleaned["text"].ne("")].copy()
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="raise").astype(int)

    invalid = set(cleaned["label"].unique()).difference(ID_TO_LABEL)
    if invalid:
        raise ValueError(f"Rótulos inválidos em {source}: {sorted(invalid)}")

    cleaned["source"] = source
    return cleaned.reset_index(drop=True)


def load_corpus(
    local_tsv: Path = Path("data/PT_train.tsv"),
    *,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, CorpusSummary]:
    """Combina treino público e TSV local, preservando o teste público intacto.

    Linhas locais rotuladas apenas como ``PT`` são ambíguas e são excluídas;
    essa exclusão é registrada no resumo retornado.
    """

    remote = load_dataset(REMOTE_DATASET, cache_dir=cache_dir)
    remote_train = _clean(remote["train"].to_pandas(), source="cc4051/pt_vid:train")
    remote_test = _clean(remote["test"].to_pandas(), source="cc4051/pt_vid:test")

    raw_local = pd.read_csv(local_tsv, sep="\t", header=None, names=["text", "label"])
    raw_local["label"] = raw_local["label"].astype(str).str.strip()
    excluded = raw_local.loc[~raw_local["label"].isin(LABEL_TO_ID), "label"].value_counts().to_dict()
    local = raw_local.loc[raw_local["label"].isin(LABEL_TO_ID)].copy()
    local["label"] = local["label"].map(LABEL_TO_ID)
    local = _clean(local, source="PT_train.tsv")

    train = pd.concat([remote_train, local], ignore_index=True)
    summary = CorpusSummary(
        rows=len(train),
        label_counts={ID_TO_LABEL[key]: int(value) for key, value in train["label"].value_counts().sort_index().items()},
        sources={str(key): int(value) for key, value in train["source"].value_counts().items()},
        excluded_local_labels={str(key): int(value) for key, value in excluded.items()},
    )
    return train, remote_test, summary
