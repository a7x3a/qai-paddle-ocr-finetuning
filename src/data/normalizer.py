"""Unicode and Kurdish-specific text normalization module.

Provides high-performance, deterministic normalization for Kurdish (Sorani and Kurmanji),
Arabic script, and Latin script datasets. Ensures preservation of critical morphological
markers such as Zero-Width Non-Joiner (ZWNJ), handles script unification, and removes
corrupting control characters.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, Pattern

# Zero-Width Non-Joiner constant
ZWNJ: Final[str] = "\u200c"

# Control characters and unprintable whitespace to remove
CONTROL_CHARS_PATTERN: Final[Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f\r\n\t]")

# Tatweel (Kashida) character used for justification/elongation
TATWEEL_PATTERN: Final[Pattern[str]] = re.compile(r"\u0640")

# Arabic Harakat / Diacritics (Fathatan, Dammatan, Kasratan, Fatha, Damma, Kasra, Shadda, Sukun)
ARABIC_DIACRITICS_PATTERN: Final[Pattern[str]] = re.compile(r"[\u064b-\u0652\u0670]")

# Multi-space collapse pattern (excluding ZWNJ)
MULTI_SPACE_PATTERN: Final[Pattern[str]] = re.compile(r"[^\S\u200c]+")


class KurdishTextNormalizer:
    """Production-grade text normalizer for Kurdish OCR tasks.
    
    Handles both Sorani (Arabic-based script) and Kurmanji (Latin-based script),
    unifying character variants while strictly preserving essential morphological
    boundaries (ZWNJ) and rejecting noise/control tokens.
    """

    # Mapping table for Arabic script variants to standard Kurdish (Sorani) characters
    SORANI_CHAR_MAP: Final[dict[str, str]] = {
        # Standard Arabic Kaf to Kurdish/Persian Keheh
        "\u0643": "\u06a9",  # ك -> ک
        # Standard Arabic Yeh to Kurdish Yeh (dotless in final/isolated form)
        "\u064a": "\u06cc",  # ي -> ی
        # Arabic Teh Marbuta to Kurdish Ae / Heh
        "\u0629": "\u06d5",  # ة -> ە
        # Arabic Alef with Madda / Hamza normalization to standard Alef forms
        "\u0622": "\u0622",  # آ preserved
        "\u0623": "\u0627",  # أ -> ا (common normalization in OCR)
        "\u0625": "\u0627",  # إ -> ا
        "\u0671": "\u0627",  # ٱ -> ا
        # Persian/Kurdish Heh Goal variants to standard Kurdish Ae
        "\u06c1": "\u06d5",  # ہ -> ە
        "\u06be": "\u06be",  # ھ (Kurdish Heh Do-cheshme) preserved
        # Kurdish unique consonants and vowels check
        "\u067e": "\u067e",  # پ (Pe)
        "\u0686": "\u0686",  # چ (Tche)
        "\u0698": "\u0698",  # ژ (Zhe)
        "\u06af": "\u06af",  # گ (Gaf)
        "\u0695": "\u0695",  # ڕ (Re with small V below)
        "\u06b5": "\u06b5",  # ڵ (Lam with small V)
        "\u06c6": "\u06c6",  # ۆ (Vav with small V)
        "\u06ce": "\u06ce",  # ێ (Yeh with small V)
        "\u06d5": "\u06d5",  # ە (Ae)
        # Eastern Arabic Numerals to standard Arabic-Indic digits or standard digits
        "\u06f0": "0", "\u06f1": "1", "\u06f2": "2", "\u06f3": "3", "\u06f4": "4",
        "\u06f5": "5", "\u06f6": "6", "\u06f7": "7", "\u06f8": "8", "\u06f9": "9",
        "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
        "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    }

    def __init__(
        self,
        preserve_zwnj: bool = True,
        remove_tatweel: bool = True,
        remove_diacritics: bool = True,
        unify_sorani_glyphs: bool = True,
    ) -> None:
        """Initialize Kurdish normalizer configuration.

        Args:
            preserve_zwnj: If True, preserves \\u200c (Zero-Width Non-Joiner).
            remove_tatweel: If True, removes \\u0640 (kashida/tatweel elongation).
            remove_diacritics: If True, strips Arabic harakat diacritics.
            unify_sorani_glyphs: If True, unifies Arabic kaf/yeh/teh marbuta to Kurdish standards.
        """
        self.preserve_zwnj = preserve_zwnj
        self.remove_tatweel = remove_tatweel
        self.remove_diacritics = remove_diacritics
        self.unify_sorani_glyphs = unify_sorani_glyphs

    def normalize(self, text: str | None) -> str:
        """Apply comprehensive deterministic Unicode and Kurdish normalization.

        Steps:
            1. Handle null / empty inputs gracefully.
            2. Mask ZWNJ if preservation is requested before Unicode NFKC normalization.
            3. Apply standard unicodedata.normalize('NFKC').
            4. Restore ZWNJ safely.
            5. Remove unprintable/control characters (tabs, newlines, null bytes).
            6. Optionally strip tatweel / kashida elongation.
            7. Optionally strip Arabic diacritics (harakat).
            8. Unify Sorani Arabic-script glyphs if configured.
            9. Collapse multiple consecutive whitespace characters into a single space.
            10. Strip leading and trailing whitespace.

        Args:
            text: Raw input string.

        Returns:
            Normalized, clean UTF-8 string.
        """
        if not text:
            return ""

        raw = str(text)

        # Step 2: Protect ZWNJ during NFKC decomposition if requested
        zwnj_placeholder = "___KURDISH_ZWNJ_TOKEN___"
        if self.preserve_zwnj and ZWNJ in raw:
            raw = raw.replace(ZWNJ, zwnj_placeholder)

        # Step 3: Unicode NFKC normalization (handles ligature decomposition and Latin accents)
        normalized = unicodedata.normalize("NFKC", raw)

        # Step 4: Restore ZWNJ
        if self.preserve_zwnj and zwnj_placeholder in normalized:
            normalized = normalized.replace(zwnj_placeholder, ZWNJ)
        elif not self.preserve_zwnj:
            normalized = normalized.replace(ZWNJ, "")

        # Step 5: Clean control and non-printable characters
        normalized = CONTROL_CHARS_PATTERN.sub("", normalized)

        # Step 6: Remove tatweel (kashida)
        if self.remove_tatweel:
            normalized = TATWEEL_PATTERN.sub("", normalized)

        # Step 7: Remove diacritics if configured
        if self.remove_diacritics:
            normalized = ARABIC_DIACRITICS_PATTERN.sub("", normalized)

        # Step 8: Unify Sorani glyph variants
        if self.unify_sorani_glyphs:
            chars = [self.SORANI_CHAR_MAP.get(char, char) for char in normalized]
            normalized = "".join(chars)

        # Step 9: Collapse whitespace runs while keeping ZWNJ intact
        normalized = MULTI_SPACE_PATTERN.sub(" ", normalized)

        # Step 10: Strip boundary whitespace
        normalized = normalized.strip()

        # Remove redundant ZWNJs at start or end of text
        normalized = normalized.strip(ZWNJ)

        return normalized


def normalize_kurdish_text(
    text: str | None,
    preserve_zwnj: bool = True,
    remove_tatweel: bool = True,
    remove_diacritics: bool = True,
    unify_sorani_glyphs: bool = True,
) -> str:
    """Functional convenience interface for Kurdish text normalization.

    Args:
        text: Raw input text.
        preserve_zwnj: Whether to retain Zero-Width Non-Joiner characters.
        remove_tatweel: Whether to remove tatweel/kashida elongations.
        remove_diacritics: Whether to remove Arabic harakat diacritics.
        unify_sorani_glyphs: Whether to map standard Arabic kaf/yeh to Kurdish equivalents.

    Returns:
        Pristine normalized UTF-8 string.
    """
    normalizer = KurdishTextNormalizer(
        preserve_zwnj=preserve_zwnj,
        remove_tatweel=remove_tatweel,
        remove_diacritics=remove_diacritics,
        unify_sorani_glyphs=unify_sorani_glyphs,
    )
    return normalizer.normalize(text)
