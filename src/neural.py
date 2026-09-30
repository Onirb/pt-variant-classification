"""Modelos neurais de caracteres com artefatos reprodutíveis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class CharacterTokenizer:
    vocabulary: tuple[str, ...]
    max_length: int

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, text: str) -> list[int]:
        mapping = self.token_to_id
        return [mapping.get(character, self.unk_id) for character in text.lower()[: self.max_length]] or [self.unk_id]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def fit(cls, texts: Iterable[str], *, max_length: int) -> "CharacterTokenizer":
        characters = sorted({character for text in texts for character in text.lower()})
        return cls(vocabulary=(PAD_TOKEN, UNK_TOKEN, *characters), max_length=max_length)


class CharacterDataset(Dataset[tuple[list[int], int]]):
    def __init__(self, texts: Iterable[str], labels: Iterable[int], tokenizer: CharacterTokenizer) -> None:
        self.samples = [(tokenizer.encode(text), int(label)) for text, label in zip(texts, labels, strict=True)]
        self.pad_id = tokenizer.pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.samples[index]

    def collate(self, batch: list[tuple[list[int], int]]) -> tuple[Tensor, Tensor]:
        width = max(len(tokens) for tokens, _ in batch)
        features = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.empty(len(batch), dtype=torch.long)
        for row, (tokens, label) in enumerate(batch):
            features[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            labels[row] = label
        return features, labels


class CharacterCNN(nn.Module):
    """CNN de caracteres com filtros de tamanhos distintos e max pooling global."""

    def __init__(self, vocabulary_size: int, *, embedding_dim: int = 96, channels: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dim, channels, kernel_size=kernel) for kernel in (3, 4, 5)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 3, 2)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        features = [torch.amax(torch.relu(convolution(embedded)), dim=2) for convolution in self.convolutions]
        return self.classifier(self.dropout(torch.cat(features, dim=1)))
