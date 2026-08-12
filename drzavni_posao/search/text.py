from __future__ import annotations

import re

from ..constants import transliterate_serbian

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
