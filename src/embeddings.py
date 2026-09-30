"""Extração local de embeddings de encoder para comparações de classificação."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


BERTIMBAU_MODEL = "neuralmind/bert-base-portuguese-cased"
BERTIMBAU_REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"


def load_bertimbau(*, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(BERTIMBAU_MODEL, revision=BERTIMBAU_REVISION, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(BERTIMBAU_MODEL, revision=BERTIMBAU_REVISION, cache_dir=cache_dir)
    model.eval()
    return tokenizer, model


def mean_pool_embeddings(
    texts: Iterable[str],
    *,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    values = list(map(str, texts))
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            encoded = tokenizer(
                values[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            chunks.append(pooled.cpu().numpy())
    return np.concatenate(chunks, axis=0)
