"""Inferência offline do modelo final ajustado do Track B."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_ARTIFACT = Path("runs/track_b/final_model_v1")
LABEL_MAP = {0: "PT-PT", 1: "PT-BR"}


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(artifact: Path) -> dict:
    path = artifact / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifesto do modelo final ausente: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "track_b_fine_tuned_classifier":
        raise ValueError("Artefato não é um classificador final Track B.")
    if manifest["global_steps"] != manifest["final_protocol"]["max_steps"]:
        raise ValueError("Ajuste final incompleto.")
    declared = set(manifest["sha256"])
    actual = {file.relative_to(artifact).as_posix() for file in artifact.rglob("*")
              if file.is_file() and file.name != "manifest.json"}
    if declared != actual or "model/config.json" not in declared or "selection.json" not in declared:
        raise ValueError("Arquivos do artefato diferem do manifesto.")
    for name, expected in manifest["sha256"].items():
        candidate = (artifact / name).resolve()
        if not candidate.is_relative_to(artifact.resolve()) or not candidate.is_file():
            raise ValueError(f"Arquivo ausente ou caminho inválido: {name}")
        if sha256_file(candidate) != expected:
            raise ValueError(f"Arquivo alterado: {name}")
    return manifest


@dataclass(frozen=True)
class FineTunedPrediction:
    label: str
    logit_difference_br_minus_pt: float
    characters: int
    was_truncated: bool
    score_interpretation: str = "raw logit difference; not calibrated probability"

    def to_dict(self):
        return asdict(self)


class FineTunedVariantClassifier:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT, *, threads: int = 4):
        if threads < 1:
            raise ValueError("threads deve ser positivo.")
        self.artifact_dir = Path(artifact_dir)
        self.manifest = verify_manifest(self.artifact_dir)
        self.max_length = int(self.manifest["final_protocol"]["max_length"])
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(self.artifact_dir / "model", local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.artifact_dir / "model", local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        self.model.to("cpu").eval()
        if self.model.config.id2label != LABEL_MAP:
            raise ValueError("Mapa de classes incompatível.")

    def predict(self, text: str) -> FineTunedPrediction:
        if not isinstance(text, str):
            raise TypeError("O texto deve ser uma string.")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Informe um texto não vazio.")
        # tokenize não aplica truncamento e permite sinalizar a limitação ao usuário.
        token_count = len(self.tokenizer.tokenize(cleaned)) + self.tokenizer.num_special_tokens_to_add(pair=False)
        encoded = self.tokenizer(cleaned, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0].float().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("O modelo retornou escores não finitos.")
        return FineTunedPrediction(
            label=LABEL_MAP[int(logits.argmax())],
            logit_difference_br_minus_pt=round(float(logits[1] - logits[0]), 6),
            characters=len(cleaned), was_truncated=token_count > self.max_length,
        )

    def verify_probes(self) -> dict:
        texts = self.manifest["probe_texts"]
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**encoded).logits.float().numpy()
        expected = np.array(self.manifest["cpu_reload_probe_logits"])
        if not np.allclose(logits, expected, rtol=1e-4, atol=1e-4):
            raise ValueError("Inferência local difere das sondas exportadas; revisar antes do teste final.")
        return {"probe_logits_match": True, "probe_count": len(texts)}
