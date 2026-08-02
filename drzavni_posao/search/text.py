from __future__ import annotations

import re
import unicodedata


# Serbian Cyrillic has a one-to-one Latin spelling except for three digraphs.
# Mapping characters independently also handles all-uppercase text correctly.
CYRILLIC_TO_LATIN = {
    "А": "A",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Ђ": "Đ",
    "Е": "E",
    "Ж": "Ž",
    "З": "Z",
    "И": "I",
    "Ј": "J",
    "К": "K",
    "Л": "L",
    "Љ": "LJ",
    "М": "M",
    "Н": "N",
    "Њ": "NJ",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "Ћ": "Ć",
    "У": "U",
    "Ф": "F",
    "Х": "H",
    "Ц": "C",
    "Ч": "Č",
    "Џ": "DŽ",
    "Ш": "Š",
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ђ": "đ",
    "е": "e",
    "ж": "ž",
    "з": "z",
    "и": "i",
    "ј": "j",
    "к": "k",
    "л": "l",
    "љ": "lj",
    "м": "m",
    "н": "n",
    "њ": "nj",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ћ": "ć",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "č",
    "џ": "dž",
    "ш": "š",
}

DIACRITIC_FOLD = str.maketrans(
    {
        "č": "c",
        "ć": "c",
        "š": "s",
        "ž": "z",
        "đ": "dj",
        "Č": "C",
        "Ć": "C",
        "Š": "S",
        "Ž": "Z",
        "Đ": "Dj",
    }
)

CHARACTER_DISPLAY_NAMES = {
    "boskic": "Boškić",
    "cvarkov": "Đorđe Čvarkov",
    "torbica": "Dragan Torbica",
}

CHARACTER_ALIASES: dict[str, tuple[str, ...]] = {
    "boskic": (
        "boškić",
        "boskic",
        "boške",
        "boske",
        "boško",
        "bosko",
        "božkić",
        "bozkic",
    ),
    "cvarkov": (
        "čvarkov",
        "cvarkov",
        "čvarko",
        "cvarko",
        "đorđe čvarkov",
        "djordje cvarkov",
        "đorđe",
        "djordje",
    ),
    "torbica": ("torbica", "torbice", "dragan torbica", "dragan"),
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def transliterate_serbian(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return "".join(
        CYRILLIC_TO_LATIN.get(character, character) for character in normalized
    )


def semantic_text(text: str) -> str:
    """Light normalization for multilingual embedding input."""
    value = transliterate_serbian(text)
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def basic_tokens(text: str) -> list[str]:
    value = semantic_text(text).casefold().translate(DIACRITIC_FOLD)
    return _TOKEN_PATTERN.findall(value)


def _alias_patterns() -> list[tuple[tuple[str, ...], str]]:
    patterns: list[tuple[tuple[str, ...], str]] = []
    for canonical, aliases in CHARACTER_ALIASES.items():
        for alias in aliases:
            patterns.append((tuple(basic_tokens(alias)), canonical))
    return sorted(patterns, key=lambda item: (-len(item[0]), item[0]))


_SORTED_ALIAS_PATTERNS = _alias_patterns()


def canonicalize_alias_tokens(tokens: list[str]) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(tokens):
        for pattern, canonical in _SORTED_ALIAS_PATTERNS:
            width = len(pattern)
            if tuple(tokens[index : index + width]) == pattern:
                output.append(canonical)
                index += width
                break
        else:
            output.append(tokens[index])
            index += 1
    return output


def normalize_text(text: str, *, apply_aliases: bool = True) -> str:
    tokens = basic_tokens(text)
    if apply_aliases:
        tokens = canonicalize_alias_tokens(tokens)
    return " ".join(tokens)


# This is intentionally conservative.  Original tokens remain in the stream;
# stems are extra recall features, so a bad stem cannot remove an exact match.
_SERBIAN_SUFFIXES = tuple(
    sorted(
        {
            "ovima",
            "evima",
            "anjima",
            "enjima",
            "acijama",
            "iranje",
            "anje",
            "enje",
            "ovima",
            "evima",
            "ima",
            "ama",
            "anje",
            "enje",
            "osti",
            "skog",
            "skoj",
            "skim",
            "skih",
            "nog",
            "nom",
            "nih",
            "nja",
            "nje",
            "ati",
            "iti",
            "ovati",
            "uje",
            "anje",
        },
        key=len,
        reverse=True,
    )
)


def light_serbian_stem(token: str) -> str:
    if token in CHARACTER_DISPLAY_NAMES or len(token) < 6:
        return token
    for suffix in _SERBIAN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, *, include_stems: bool = True) -> list[str]:
    base = normalize_text(text).split()
    if not include_stems:
        return base
    result: list[str] = []
    for token in base:
        result.append(token)
        stem = light_serbian_stem(token)
        if stem != token:
            result.append(f"__stem_{stem}")
    return result
