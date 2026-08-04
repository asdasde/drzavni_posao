from __future__ import annotations

from dataclasses import dataclass

from .text import (
    CHARACTER_DISPLAY_NAMES,
    normalize_text,
    semantic_text,
    tokenize,
)


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    original: str
    semantic: str
    normalized: str
    lexical_text: str
    lexical_tokens: tuple[str, ...]
    characters: tuple[str, ...]


def process_query(query: str) -> ProcessedQuery:
    normalized = normalize_text(query)
    base_tokens = normalized.split()
    characters = tuple(
        canonical for canonical in CHARACTER_DISPLAY_NAMES if canonical in base_tokens
    )

    return ProcessedQuery(
        original=query,
        semantic=semantic_text(query),
        normalized=normalized,
        lexical_text=normalized,
        lexical_tokens=tuple(tokenize(normalized)),
        characters=characters,
    )


def display_name(character_key: str) -> str:
    return CHARACTER_DISPLAY_NAMES.get(character_key, character_key)
