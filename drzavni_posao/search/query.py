from __future__ import annotations

from dataclasses import dataclass

from .text import (
    CHARACTER_DISPLAY_NAMES,
    basic_tokens,
    normalize_text,
    semantic_text,
    tokenize,
)


# Controlled, deliberately small bilingual expansion. It is applied only to
# lexical queries, never directly to passage text sent to a dense encoder.
QUERY_EXPANSION_GROUPS: dict[str, tuple[str, ...]] = {
    "elections": (
        "election",
        "elections",
        "vote",
        "voting",
        "izbor",
        "izbori",
        "izborima",
        "glas",
        "glasanje",
        "glasa",
        "stranka",
        "partija",
        "kandidat",
        "kandidati",
        "listić",
        "listic",
        "biranje",
    ),
    "argument": (
        "argue",
        "argued",
        "arguing",
        "argument",
        "fight",
        "fighting",
        "svađa",
        "svadja",
        "rasprava",
        "prepirka",
        "vika",
        "viču",
        "vicu",
        "konflikt",
        "posvađali",
        "posvadjali",
    ),
    "money": (
        "money",
        "cash",
        "salary",
        "pay",
        "novac",
        "pare",
        "plata",
        "plaćanje",
        "placanje",
        "dinara",
        "evra",
        "kredit",
        "dug",
    ),
    "work": (
        "work",
        "job",
        "office",
        "boss",
        "posao",
        "rad",
        "firma",
        "kancelarija",
        "šef",
        "sef",
        "direktor",
        "prekovremeno",
    ),
    "computer": (
        "computer",
        "internet",
        "email",
        "technology",
        "kompjuter",
        "računar",
        "racunar",
        "mejl",
        "elektronski",
        "tehnologija",
    ),
    "health": (
        "health",
        "sick",
        "illness",
        "doctor",
        "zdravlje",
        "bolest",
        "bolestan",
        "doktor",
        "lek",
        "povreda",
        "prehlada",
    ),
}

DENSE_QUERY_EXPANSIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "elections": (
        ("election", "elections", "vote", "voting"),
        "izbori glasanje stranka kandidat",
    ),
    "argument": (
        ("argue", "argued", "arguing", "argument", "fight", "fighting"),
        "svadja rasprava prepirka konflikt",
    ),
    "money": (("money", "cash", "salary", "pay"), "novac pare plata dug kredit"),
    "work": (
        ("work", "job", "office", "boss"),
        "posao kancelarija sef direktor firma",
    ),
    "computer": (
        ("computer", "internet", "email", "technology"),
        "kompjuter racunar internet mejl",
    ),
    "health": (
        ("health", "sick", "illness", "doctor"),
        "zdravlje bolest doktor lek povreda",
    ),
}


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    original: str
    semantic: str
    normalized: str
    lexical_text: str
    lexical_tokens: tuple[str, ...]
    characters: tuple[str, ...]
    topic_terms: tuple[str, ...]
    expansion_groups: tuple[str, ...]


def process_query(query: str, *, expand: bool = True) -> ProcessedQuery:
    normalized = normalize_text(query)
    base_tokens = normalized.split()
    characters = tuple(
        canonical for canonical in CHARACTER_DISPLAY_NAMES if canonical in base_tokens
    )

    expanded_tokens = list(base_tokens)
    topic_terms: list[str] = []
    groups: list[str] = []
    dense_additions: list[str] = []
    if expand:
        token_set = set(basic_tokens(query)) | set(base_tokens)
        for group_name, terms in QUERY_EXPANSION_GROUPS.items():
            normalized_terms = {
                token
                for term in terms
                for token in normalize_text(term, apply_aliases=False).split()
            }
            if token_set.intersection(normalized_terms):
                groups.append(group_name)
                topic_terms.extend(sorted(normalized_terms))
                expanded_tokens.extend(sorted(normalized_terms))
                dense_config = DENSE_QUERY_EXPANSIONS.get(group_name)
                if dense_config is not None:
                    english_triggers, serbian_bridge = dense_config
                    if token_set.intersection(english_triggers):
                        dense_additions.append(serbian_bridge)

    lexical_text = " ".join(dict.fromkeys(expanded_tokens))
    dense_semantic = semantic_text(query)
    if dense_additions:
        dense_semantic = f"{dense_semantic}. {' '.join(dense_additions)}"
    return ProcessedQuery(
        original=query,
        semantic=dense_semantic,
        normalized=normalized,
        lexical_text=lexical_text,
        lexical_tokens=tuple(tokenize(lexical_text)),
        characters=characters,
        topic_terms=tuple(dict.fromkeys(topic_terms)),
        expansion_groups=tuple(groups),
    )


def display_name(character_key: str) -> str:
    return CHARACTER_DISPLAY_NAMES.get(character_key, character_key)
