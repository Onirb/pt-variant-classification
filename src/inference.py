"""Inferência local para o artefato final de identificação de variante."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from src.data import ID_TO_LABEL
from src.embeddings import load_bertimbau, mean_pool_embeddings


@dataclass(frozen=True)
class Prediction:
    """Resultado local; ``margin`` não é uma probabilidade calibrada."""

    label: str
    margin: float
    low_margin: bool
    characters: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_text(text: str) -> str:
    """Normaliza a entrada e recusa textos vazios antes de chamar o modelo."""

    if not isinstance(text, str):
        raise TypeError("O texto deve ser uma string.")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Informe um texto não vazio para classificar.")
    return cleaned


class LocalVariantClassifier:
    """Carrega somente artefatos locais e executa uma classificação por vez."""

    def __init__(self, artifact_dir: Path = Path("runs/bertimbau_final_v1")) -> None:
        manifest_path = artifact_dir / "manifest.json"
        classifier_path = artifact_dir / "linear_svc.joblib"
        if not manifest_path.is_file() or not classifier_path.is_file():
            raise FileNotFoundError(
                "Artefato final ausente. Execute `python -m scripts.train_final_bertimbau` primeiro."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.classifier = joblib.load(classifier_path)
        self.tokenizer, self.encoder = load_bertimbau(cache_dir=".hf_cache/bertimbau")

    def predict(self, text: str) -> Prediction:
        cleaned = validate_text(text)
        max_length = int(self.manifest["settings"]["max_length"])
        embedding = mean_pool_embeddings(
            [cleaned], tokenizer=self.tokenizer, model=self.encoder, batch_size=1, max_length=max_length
        )
        predicted_id = int(self.classifier.predict(embedding)[0])
        margin = float(np.ravel(self.classifier.decision_function(embedding))[0])
        return Prediction(
            label=ID_TO_LABEL[predicted_id],
            margin=round(margin, 6),
            low_margin=abs(margin) < 0.25,
            characters=len(cleaned),
        )
