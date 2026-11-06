"""Goal 09 — deterministic text-normalization tests.

Pure functions; multilingual (Bangla / English / mixed), punctuation-safe, and
idempotent. No database, no network.
"""
import unicodedata

from app.transcripts.normalization import NormalizedText, normalize_text


# --- Whitespace handling ------------------------------------------------------


def test_trims_leading_trailing_whitespace() -> None:
    assert normalize_text("   hello   ").normalized == "hello"


def test_collapses_repeated_spaces() -> None:
    assert normalize_text("a    b     c").normalized == "a b c"


def test_collapses_newlines_and_mixed_whitespace() -> None:
    assert normalize_text("a\n\nb\t\t c\r\n d").normalized == "a b c d"


def test_collapses_unicode_whitespace() -> None:
    assert normalize_text("a\u00a0\u00a0b").normalized == "a b"


# --- Unicode normalization ----------------------------------------------------


def test_unicode_nfc_composition() -> None:
    # "e" + combining acute -> precomposed "é" (NFC), deterministically.
    result = normalize_text("Cafe\u0301")
    assert result.normalized == "Caf\u00e9"
    assert result.normalized == unicodedata.normalize("NFC", "Cafe\u0301")


def test_uses_nfc_not_nfkc_preserves_meaning() -> None:
    # NFKC would fold the full-width "Ａ" to "A"; NFC keeps it. We choose NFC
    # so normalization never changes the semantic characters.
    result = normalize_text("\uff21")
    assert result.normalized == "\uff21"
    assert unicodedata.normalize("NFKC", "\uff21") == "A"


# --- Multilingual safety ------------------------------------------------------


def test_bangla_text_is_preserved() -> None:
    text = "আমি  বাংলায়   কথা  বলি  ।"
    result = normalize_text(text)
    assert result.normalized == "আমি বাংলায় কথা বলি ।"  # দাঁড়ি (।) preserved
    assert result.original == text


def test_english_text_preserves_case_and_punctuation() -> None:
    text = "  Hello,   World!  "
    assert normalize_text(text).normalized == "Hello, World!"


def test_mixed_bangla_english_text() -> None:
    text = "  ModernBERT 和一个   বাংলা   মডেল  "
    assert normalize_text(text).normalized == "ModernBERT 和一个 বাংলা মডেল"


def test_no_lowercasing() -> None:
    assert normalize_text("Hello WORLD").normalized == "Hello WORLD"


def test_punctuation_is_preserved() -> None:
    text = "Wait ,  what ?!  (yes)  ...  "
    assert normalize_text(text).normalized == "Wait , what ?! (yes) ..."


# --- Original vs normalized, determinism, idempotency -------------------------


def test_original_is_preserved_separately() -> None:
    original = "  A  B\n C  "
    result = normalize_text(original)
    assert isinstance(result, NormalizedText)
    assert result.original == original  # verbatim, never overwritten
    assert result.normalized == "A B C"
    assert result.original != result.normalized


def test_deterministic() -> None:
    text = "  Café  বাংলা \n x  "
    assert [normalize_text(text).normalized for _ in range(5)] == [
        normalize_text(text).normalized
    ] * 5


def test_idempotent() -> None:
    for text in ("  a   b  ", "Cafe\u0301 x", "আমি  বলি ।", "Hello, World!"):
        once = normalize_text(text).normalized
        assert normalize_text(once).normalized == once


def test_blank_input_normalizes_to_empty_string() -> None:
    result = normalize_text("   \n\t  ")
    assert result.normalized == ""
    assert result.original == "   \n\t  "
