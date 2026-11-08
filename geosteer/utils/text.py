"""Text normalisation used for duplicate detection and matching."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize_for_dedup(text: str) -> str:
    """Casefold, NFKC, drop punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _PUNCT.sub(" ", text)
    return normalize_whitespace(text)
