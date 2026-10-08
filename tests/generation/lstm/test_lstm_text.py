import unittest

from drzavni_posao.generation.lstm.text import (
    ALPHABET,
    clean_dialogue_text,
    clean_premise_text,
)


class DialogueCleaningTests(unittest.TestCase):
    def test_keeps_serbian_letters_digits_and_censored_words(self):
        text = "Đorđe, šta ćeš sa 1750 dinara? Ma p*** kol'ko je 70%!"

        self.assertEqual(clean_dialogue_text(text), (text, None))

    def test_strips_foreign_diacritics_but_not_serbian_ones(self):
        self.assertEqual(
            clean_dialogue_text("Café Müller, łódź, čvarci"),
            ("Cafe Muller, lodz, čvarci", None),
        )

    def test_maps_and_deletes_rare_punctuation(self):
        self.assertEqual(
            clean_dialogue_text("„Ajde“ – reče: «bre» & to"),
            ("Ajde - reče bre to", None),
        )

    def test_rejects_cyrillic_segments(self):
        self.assertEqual(clean_dialogue_text("Да, брате."), (None, "cyrillic"))

    def test_rejects_noise_segments(self):
        self.assertEqual(clean_dialogue_text("🎵"), (None, "unsupported_character"))
        self.assertEqual(
            clean_dialogue_text("ne zna\ufffd"), (None, "unsupported_character")
        )

    def test_rejects_segments_empty_after_cleaning(self):
        self.assertEqual(clean_dialogue_text(" „“ "), (None, "empty"))


class PremiseCleaningTests(unittest.TestCase):
    def test_transliterates_cyrillic_and_drops_unknown_characters(self):
        cleaned = clean_premise_text("Torbica — „šef“ arhive\xad, сада ન jasno.")

        self.assertEqual(cleaned, "Torbica - šef arhive, sada jasno.")
        self.assertTrue(ALPHABET.issuperset(cleaned))


if __name__ == "__main__":
    unittest.main()
