"""Transformações determinísticas de texto para comparações controladas."""

from __future__ import annotations

import re


URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
MENTION = re.compile(r"(?<!\w)@[\w_]+")
HASHTAG = re.compile(r"(?<!\w)#[\w_]+")
NUMBER = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
IDENTIFIER = re.compile(r"\b[A-Z]{2,}[\d_-]{2,}\b")
WHITESPACE = re.compile(r"\s+")


def mask_contextual_tokens(text: str) -> str:
    """Oculta padrões estruturais sem realizar reconhecimento de entidades."""
    value = str(text)
    value = URL.sub(" URLTOKEN ", value)
    value = EMAIL.sub(" EMAILTOKEN ", value)
    value = MENTION.sub(" MENTIONTOKEN ", value)
    value = HASHTAG.sub(" HASHTAGTOKEN ", value)
    value = IDENTIFIER.sub(" IDENTIFIERTOKEN ", value)
    value = NUMBER.sub(" NUMBERTOKEN ", value)
    return WHITESPACE.sub(" ", value).strip()
