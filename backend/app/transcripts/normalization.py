"""Deterministic, multilingual-safe text normalization (Goal 09).

Design guarantees
-----------------
* **Unicode NFC** canonical composition. NFC is idempotent and does NOT fold
  compatibility characters (unlike NFKC), so it never changes meaning — safe
  for Bangla combining marks as well as Latin accents.
* **Whitespace / line-break handling**: any run of Unicode whitespace
  (spaces, tabs, newlines, form feeds, Unicode spaces) collapses to a single
  ASCII space; leading/trailing whitespace is then trimmed. Within a segment,
  line breaks become a single space so each segment is one logical line.
* **Punctuation and all non-whitespace characters are preserved** (after NFC).
  Semantic meaning of sentences is unchanged; words are not altered.
* **No English-specific processing**: no lowercasing, no ASCII folding, no
  stemming, no stop-word removal, no language detection. Works identically for
  Bangla, English, and mixed / multilingual transcripts (Bangla has no case).
* **Original preserved separately**: :func:`normalize_text` returns a
  :class:`NormalizedText` carrying BOTH ``original`` (verbatim) and
  ``normalized``. It never overwrites the input.
* **Pure & deterministic**: no randomness, no locale, no DB, no clock, no I/O.
  Applying it twice yields the same ``normalized`` value (idempotent).

The normalization layer is independent of the input contract and of any
persistence concern; the ingestion service composes it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ``\s`` on a Python ``str`` pattern is Unicode-aware by default, so this also
# matches non-ASCII whitespace (e.g. NBSP) without any language-specific rules.
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedText:
    """The verbatim ``original`` text and its deterministic ``normalized`` form."""

    original: str
    normalized: str


def normalize_text(text: str) -> NormalizedText:
    """Return both the original and a deterministically normalized form.

    Steps: (1) Unicode NFC composition, (2) collapse whitespace runs to one
    space, (3) trim. Punctuation, casing, and non-whitespace content survive.
    """
    composed = unicodedata.normalize("NFC", text)
    normalized = _WHITESPACE_RUN.sub(" ", composed).strip()
    return NormalizedText(original=text, normalized=normalized)


__all__ = ["NormalizedText", "normalize_text"]
