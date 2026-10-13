import unicodedata

from ...transcription.constants import transliterate_serbian


LETTERS = "abcdefghijklmnopqrstuvwxyzčćđšž"
ALPHABET = frozenset(LETTERS + LETTERS.upper() + "0123456789" + " ,.?!'-%*")

_REPLACEMENTS = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "’": "'",
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "ß": "ss",
        "ı": "i",
        "œ": "oe",
        "Œ": "Oe",
        "ħ": "h",
        "Ħ": "H",
        "\xad": None,
        "„": None,
        "“": None,
        "”": None,
        '"': None,
        "«": None,
        "»": None,
        ":": None,
        "&": None,
        "+": None,
        "_": None,
    }
)


def normalize_characters(text: str) -> str:
    """Map known foreign characters onto the alphabet and collapse whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_REPLACEMENTS)
    return " ".join("".join(map(_strip_foreign_diacritic, text)).split())


def clean_dialogue_text(text: str) -> tuple[str | None, str | None]:
    """Return ``(text, None)``, or ``(None, reason)`` when the segment is rejected."""
    text = normalize_characters(text)
    if not text:
        return None, "empty"
    if any(_is_cyrillic(character) for character in text):
        return None, "cyrillic"
    if not ALPHABET.issuperset(text):
        return None, "unsupported_character"
    return text, None


def clean_premise_text(text: str) -> str:
    text = normalize_characters(transliterate_serbian(text))
    return " ".join("".join(c for c in text if c in ALPHABET).split())


def _strip_foreign_diacritic(character: str) -> str:
    if character in ALPHABET:
        return character
    base, *marks = unicodedata.normalize("NFD", character)
    if marks and base.isascii() and all(unicodedata.combining(m) for m in marks):
        return base
    return character


def _is_cyrillic(character: str) -> bool:
    return "CYRILLIC" in unicodedata.name(character, "")
