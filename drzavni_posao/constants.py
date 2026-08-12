import unicodedata


SR: int = 16_000  # WhisperX audio sample rate
ANONYMOUS_LABEL = "[anonymous]"

# Serbian Cyrillic → Latin. Whisper flips script mid-transcript; the Transcriber
# normalizes to Latin once. Digraphs (љ/њ/џ) map to multi-char values, which
# str.translate handles. Context-free uppercase digraphs default to title case;
# transliterate_serbian preserves their all-caps form when appropriate.
CYRILLIC_TO_LATIN = str.maketrans(
    {
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
        "Љ": "Lj",
        "М": "M",
        "Н": "N",
        "Њ": "Nj",
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
        "Џ": "Dž",
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
)

_UPPERCASE_DIGRAPHS = {
    "Љ": ("Lj", "LJ"),
    "Њ": ("Nj", "NJ"),
    "Џ": ("Dž", "DŽ"),
}


def transliterate_serbian(text: str) -> str:
    """Transliterate Serbian Cyrillic while preserving digraph casing."""
    normalized = unicodedata.normalize("NFC", text)
    output: list[str] = []
    for index, character in enumerate(normalized):
        digraph = _UPPERCASE_DIGRAPHS.get(character)
        if digraph is None:
            output.append(character.translate(CYRILLIC_TO_LATIN))
            continue
        next_character = normalized[index + 1 : index + 2]
        title_case, upper_case = digraph
        output.append(
            title_case if next_character and next_character.islower() else upper_case
        )
    return "".join(output)
