import unittest

from drzavni_posao.generation.lstm.bpe import BpeVocabulary
from drzavni_posao.generation.lstm.dataset import (
    SPECIAL_TOKENS,
    CharVocabulary,
    encode_episode,
)


TEXTS = ["Ma daj, Torbice.", "Ma nemoj, Đorđe!", "Daj mi to.", "Ma to je to."] * 20


class BpeVocabularyTests(unittest.TestCase):
    def setUp(self):
        self.vocab = BpeVocabulary.train(TEXTS, vocab_size=120)

    def test_special_tokens_share_ids_with_characters(self):
        characters = CharVocabulary()

        for token in SPECIAL_TOKENS:
            self.assertEqual(self.vocab.ids[token], characters.ids[token])

    def test_merges_shorten_text_and_round_trip(self):
        text = "Ma daj, Đorđe!"
        ids = self.vocab.encode(text)

        self.assertLess(len(ids), len(text))
        self.assertEqual(self.vocab.decode(ids), f" {text}")

    def test_episode_stream_keeps_tags_and_newlines_atomic(self):
        episode = {
            "turns": [
                {"speaker": "Boskic", "text": "Ma daj."},
                {"speaker": "Dragan Torbica", "text": "Daj mi to."},
            ]
        }

        ids, scored = encode_episode(episode, self.vocab)

        self.assertEqual(
            self.vocab.decode(ids), "<start>[Boskic] Ma daj.\n[Dragan Torbica] Daj mi to.\n"
        )
        self.assertEqual(
            self.vocab.decode(i for i, s in zip(ids, scored) if s), " Ma daj.\n Daj mi to.\n"
        )


if __name__ == "__main__":
    unittest.main()
