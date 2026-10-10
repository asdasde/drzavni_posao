import unittest

import numpy as np

from drzavni_posao.generation.lstm.dataset import CharVocabulary, encode_episode, encode_turn
from drzavni_posao.generation.lstm.ngram import (
    KneserNeyLM,
    SpeakerMixtureLM,
    max_order,
    mixture_bpc,
)


CORPUS = [[5, 6, 7, 5, 6, 8, 5, 6, 7, 9], [6, 7, 5, 6, 7, 7, 8]]
VOCAB_SIZE = 10


class KneserNeyTests(unittest.TestCase):
    def test_distribution_sums_to_one_for_seen_and_unseen_histories(self):
        model = KneserNeyLM(3, VOCAB_SIZE).fit(CORPUS)

        for history in ([5, 6], [9, 9], []):
            with self.subTest(history=history):
                self.assertAlmostEqual(model.next_token_probability(history).sum(), 1.0)

    def test_unseen_tokens_keep_nonzero_probability(self):
        model = KneserNeyLM(3, VOCAB_SIZE).fit(CORPUS)

        self.assertTrue((model.next_token_probability([5, 6]) > 0).all())

    def test_frequent_continuation_is_most_likely(self):
        model = KneserNeyLM(3, VOCAB_SIZE).fit(CORPUS)

        self.assertEqual(model.next_token_probability([5, 6]).argmax(), 7)

    def test_only_the_last_order_minus_one_tokens_matter(self):
        model = KneserNeyLM(3, VOCAB_SIZE).fit(CORPUS)

        np.testing.assert_allclose(
            model.next_token_probability([9, 8, 5, 6]), model.next_token_probability([5, 6])
        )

    def test_generation_stops_at_the_stop_token(self):
        model = KneserNeyLM(3, VOCAB_SIZE).fit(CORPUS)
        rng = np.random.default_rng(0)

        ids = model.generate([5], stop=9, rng=rng, max_tokens=1000)

        self.assertEqual(ids[-1], 9)
        self.assertNotIn(9, ids[:-1])

    def test_generation_respects_max_tokens(self):
        model = KneserNeyLM(3, VOCAB_SIZE).fit(CORPUS)
        rng = np.random.default_rng(0)

        self.assertEqual(len(model.generate([5], stop=-1, rng=rng, max_tokens=4)), 4)

    def test_higher_order_fits_training_text_better(self):
        masks = [[True] * len(sequence) for sequence in CORPUS]
        bits = [
            -KneserNeyLM(order, VOCAB_SIZE).fit(CORPUS).log_probs(CORPUS, masks).mean()
            for order in (1, 2, 3)
        ]

        self.assertTrue(np.all(np.diff(bits) < 0))

    def test_max_order_keeps_codes_in_int64(self):
        self.assertEqual(max_order(88), 9)
        self.assertEqual(max_order(4096), 5)
        with self.assertRaisesRegex(ValueError, "order"):
            KneserNeyLM(max_order(VOCAB_SIZE) + 1, VOCAB_SIZE)


def styled_episode():
    # After "a" each speaker always continues differently, which an order-2 model
    # can only tell apart once the speaker tag has left its history.
    return {
        "turns": [
            {"speaker": "Boskic", "text": "ab ab ab ab"},
            {"speaker": "Dragan Torbica", "text": "ac ac ac ac"},
            {"speaker": "Djordje Cvarkov", "text": "ad ad ad ad"},
        ]
    }


class SpeakerMixtureTests(unittest.TestCase):
    def setUp(self):
        self.vocab = CharVocabulary()
        self.model = SpeakerMixtureLM(2, self.vocab).fit([styled_episode()] * 3)

    def test_components_cover_every_scored_position(self):
        episode_probs, speaker_probs = self.model.component_probs([styled_episode()])
        scored = sum(encode_episode(styled_episode(), self.vocab)[1])

        self.assertEqual(episode_probs.shape, (scored,))
        self.assertEqual(speaker_probs.shape, (scored,))
        self.assertTrue(((0 < speaker_probs) & (speaker_probs <= 1)).all())

    def test_speaker_models_remember_who_is_talking(self):
        components = self.model.component_probs([styled_episode()])
        characters = sum(encode_episode(styled_episode(), self.vocab)[1])

        self.assertLess(
            mixture_bpc(components, 1.0, characters), mixture_bpc(components, 0.0, characters)
        )

    def test_generation_follows_the_speaker_of_the_current_turn(self):
        self.model.weight = 1.0
        rng = np.random.default_rng(0)
        earlier_turn = encode_turn(
            {"speaker": "Boskic", "text": "ab ab"}, self.vocab
        )[0]
        prompt = [*earlier_turn, self.vocab.ids["[Dragan Torbica]"], self.vocab.ids["a"]]

        ids = self.model.generate(prompt, self.vocab.ids["\n"], rng, temperature=0.1, max_tokens=4)

        self.assertEqual(self.vocab.decode(ids), "c ac")


class EncodeEpisodeTests(unittest.TestCase):
    def test_streams_turns_and_scores_only_text_and_newlines(self):
        vocab = CharVocabulary()
        episode = {
            "turns": [
                {"speaker": "Boskic", "text": "Da."},
                {"speaker": "Dragan Torbica", "text": "Ne."},
            ]
        }

        ids, scored = encode_episode(episode, vocab)

        self.assertEqual(vocab.decode(ids), "<start>[Boskic] Da.\n[Dragan Torbica] Ne.\n")
        self.assertEqual(vocab.decode(i for i, s in zip(ids, scored) if s), "Da.\nNe.\n")


if __name__ == "__main__":
    unittest.main()
