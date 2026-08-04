import unittest
import unicodedata

from drzavni_posao.search.query import process_query
from drzavni_posao.search.text import normalize_text, transliterate_serbian


class PreprocessTests(unittest.TestCase):
    def test_cyrillic_uppercase_digraphs(self) -> None:
        self.assertEqual(
            transliterate_serbian("ЉУБАВ ЊЕГОШ ЏЕЗ"),
            "LJUBAV NJEGOŠ DŽEZ",
        )

    def test_nfc_and_nfd_normalize_identically(self) -> None:
        text = "Boškić i Čvarkov"
        self.assertEqual(
            normalize_text(text),
            normalize_text(unicodedata.normalize("NFD", text)),
        )

    def test_mixed_script_diacritics_and_aliases(self) -> None:
        self.assertEqual(
            normalize_text("БОШКИЋ, Boške i Čvarko!"),
            "boskic boskic i cvarkov",
        )

    def test_alias_normalization_is_idempotent_and_token_bounded(self) -> None:
        once = normalize_text("Djordje Cvarkov i Boskic")
        self.assertEqual(once, "cvarkov i boskic")
        self.assertEqual(normalize_text(once), once)
        self.assertEqual(normalize_text("mikrodraganov"), "mikrodraganov")

    def test_query_processing_does_not_inject_corpus_specific_terms(self) -> None:
        query = process_query("Boškić was arguing about elections")
        self.assertNotIn("izbori", query.lexical_tokens)
        self.assertNotIn("svadja", query.lexical_tokens)
        self.assertEqual(query.semantic, "Boškić was arguing about elections")
        self.assertEqual(query.characters, ("boskic",))


if __name__ == "__main__":
    unittest.main()
