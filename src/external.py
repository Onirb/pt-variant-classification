"""Aquisição e auditoria de conjuntos externos para avaliação."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen

import pandas as pd

from .data import LABEL_TO_ID


DSL_TL_PT_DEV_URL = (
    "https://raw.githubusercontent.com/LanguageTechnologyLab/DSL-TL/main/"
    "DSL-TL-Corpus/PT-DSL-TL/PT_dev.tsv"
)
DSL_TL_PT_DEV_PATH = Path("data/external/dsl_tl/PT_dev.tsv")
PTBRVID_DATASET = "liaad/PtBrVId"
PTBRVID_REVISION = "910745e06ee2a66e64c3cd958b56728c28abd5dc"
PTBRVID_CONFIGS = ("journalistic", "legal", "literature", "politics", "social_media", "web")
PTBRVID_VALID_PATH = Path("data/external/ptbrvid_valid_v1/combined.tsv")
FRMT_TEST_DIR = Path("data/external/frmt_test_raw")
FRMT_SOURCE_REVISION = "d36068b845da4c2b24927fee2cea1e6ef98dadda"
FRMT_BUCKETS = ("entity", "lexical", "random")
PTBRVID_RAW_TO_CANONICAL = {
    "journalistic": {0: 0, 1: 1},
    "legal": {0: 0, 1: 1},
    "literature": {0: 0, 1: 1},
    "politics": {0: 0, 1: 1},
    "social_media": {0: 0, 1: 1},
    "web": {0: 1, 1: 0},
}


def download_dsl_tl_pt_dev(destination: Path = DSL_TL_PT_DEV_PATH) -> Path:
    """Baixa a partição oficial externa sem substituir arquivos existentes."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DSL_TL_PT_DEV_URL, timeout=60) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def load_dsl_tl_pt_dev(path: Path = DSL_TL_PT_DEV_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    """Carrega PT_dev e mantém somente a tarefa binária pt-BR versus pt-PT."""
    raw = pd.read_csv(path, sep="\t", header=None, names=["id", "text", "raw_label"])
    raw["text"] = raw["text"].fillna("").astype(str).str.strip()
    raw = raw.loc[raw["text"].ne("")].copy()
    counts = {str(label): int(count) for label, count in raw["raw_label"].value_counts().items()}
    binary = raw.loc[raw["raw_label"].isin(LABEL_TO_ID)].copy()
    binary["label"] = binary["raw_label"].map(LABEL_TO_ID).astype(int)
    return binary.reset_index(drop=True), counts


def normalized_texts(frame: pd.DataFrame, column: str = "text") -> set[str]:
    return set(
        frame[column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def download_ptbrvid_valid(destination: Path = PTBRVID_VALID_PATH) -> Path:
    """Baixa seis Parquet de validação na revisão pública fixada do PtBrVId."""
    if destination.exists():
        return destination
    frames: list[pd.DataFrame] = []
    for config in PTBRVID_CONFIGS:
        url = (
            f"https://huggingface.co/datasets/{PTBRVID_DATASET}/resolve/"
            f"{PTBRVID_REVISION}/{config}/valid-00000-of-00001.parquet"
        )
        with urlopen(url, timeout=120) as response:
            frame = pd.read_parquet(BytesIO(response.read()))[["text", "label"]].copy()
        if len(frame) != 1_000:
            raise RuntimeError(f"PtBrVId {config}: esperadas 1000 linhas, recebidas {len(frame)}.")
        frame.insert(0, "domain", config)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, sep="\t", index=False)
    return destination


def load_ptbrvid_valid(path: Path = PTBRVID_VALID_PATH) -> pd.DataFrame:
    """Carrega PtBrVId e normaliza a codificação de rótulos por domínio."""
    frame = pd.read_csv(path, sep="\t")
    expected = {"domain", "text", "label"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Colunas inválidas em {path}: {list(frame.columns)}")
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["text"].ne("")].copy()
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("PtBrVId possui rótulos fora de 0/1.")
    frame["raw_label"] = frame["label"].astype(int)
    frame["label"] = frame.apply(
        lambda row: PTBRVID_RAW_TO_CANONICAL[str(row["domain"])][int(row["raw_label"])], axis=1
    )
    return frame.reset_index(drop=True)


def load_frmt_test(directory: Path = FRMT_TEST_DIR) -> pd.DataFrame:
    """Carrega somente o teste FRMT em pt-BR e pt-PT, mantendo o bucket."""
    frames: list[pd.DataFrame] = []
    expected_counts = {"entity": 985, "lexical": 874, "random": 757}
    for bucket in FRMT_BUCKETS:
        for variety, label in (("BR", LABEL_TO_ID["PT-BR"]), ("PT", LABEL_TO_ID["PT-PT"])):
            path = directory / f"{bucket}_bucket__pt_{bucket}_test_en_pt-{variety}.tsv"
            if not path.is_file():
                raise FileNotFoundError(f"Arquivo FRMT ausente: {path}")
            # FRMT é TSV de uma linha por segmento. Alguns segmentos contêm
            # aspas não balanceadas, portanto o parser CSV padrão juntaria ou
            # descartaria linhas válidas. Separar no primeiro TAB preserva o
            # formato descrito pelo próprio avaliador oficial (cut -f2).
            rows: list[tuple[str, str]] = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                source_en, separator, text = line.partition("\t")
                if not separator or not source_en or not text:
                    raise ValueError(f"FRMT {path.name}: linha inválida {line_number}.")
                rows.append((source_en, text))
            frame = pd.DataFrame(rows, columns=["source_en", "text"])
            if len(frame) != expected_counts[bucket]:
                raise ValueError(f"FRMT {bucket} pt-{variety}: esperado {expected_counts[bucket]} textos, recebido {len(frame)}.")
            frame["bucket"] = bucket
            frame["label"] = label
            frame["source_file"] = path.name
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)
