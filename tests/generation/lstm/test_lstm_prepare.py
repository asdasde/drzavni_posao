import unittest
from collections import Counter

from drzavni_posao.generation.lstm.prepare import build_turns, split_turn
from drzavni_posao.transcription.transcript import Transcript


def segment(speaker, text):
    return {"speaker": speaker, "text": text}


class BuildTurnsTests(unittest.TestCase):
    def test_merges_same_speaker_segments(self):
        transcript = Transcript(
            [
                segment("Dragan Torbica", "Šta je, Boškić,"),
                segment("Dragan Torbica", " malo vurnica?"),
                segment("Boskic", "Ma jok."),
            ]
        )

        self.assertEqual(
            build_turns(transcript, max_turn_chars=300, stats=Counter()),
            [
                {"speaker": "Dragan Torbica", "text": "Šta je, Boškić, malo vurnica?"},
                {"speaker": "Boskic", "text": "Ma jok."},
            ],
        )

    def test_drops_anonymous_after_merging_so_interrupted_turns_stay_apart(self):
        stats = Counter()
        transcript = Transcript(
            [
                segment("Djordje Cvarkov", "Mora se isključiti."),
                segment("[anonymous]", "Dobro."),
                segment("Djordje Cvarkov", "Isključuj."),
            ]
        )

        self.assertEqual(
            build_turns(transcript, max_turn_chars=300, stats=stats),
            [
                {"speaker": "Djordje Cvarkov", "text": "Mora se isključiti."},
                {"speaker": "Djordje Cvarkov", "text": "Isključuj."},
            ],
        )
        self.assertEqual(stats, Counter({"[anonymous]": 1}))

    def test_rejected_segment_does_not_merge_neighbouring_turns(self):
        stats = Counter()
        transcript = Transcript(
            [
                segment("Boskic", "Prva."),
                segment("Dragan Torbica", "Да."),
                segment("Boskic", "Druga."),
            ]
        )

        self.assertEqual(
            [t["text"] for t in build_turns(transcript, max_turn_chars=300, stats=stats)],
            ["Prva.", "Druga."],
        )
        self.assertEqual(stats, Counter({"cyrillic": 1}))

    def test_keeps_remaining_text_of_partly_rejected_turn(self):
        transcript = Transcript(
            [
                segment("Boskic", "Dobro jutro."),
                segment("Boskic", "🎵"),
                segment("Boskic", "Kafa?"),
            ]
        )

        self.assertEqual(
            build_turns(transcript, max_turn_chars=300, stats=Counter()),
            [{"speaker": "Boskic", "text": "Dobro jutro. Kafa?"}],
        )

    def test_splits_long_turns_into_same_speaker_turns(self):
        transcript = Transcript(
            [segment("Boskic", "Prva rečenica ovde. Druga rečenica tamo.")]
        )

        self.assertEqual(
            build_turns(transcript, max_turn_chars=25, stats=Counter()),
            [
                {"speaker": "Boskic", "text": "Prva rečenica ovde."},
                {"speaker": "Boskic", "text": "Druga rečenica tamo."},
            ],
        )


class SplitTurnTests(unittest.TestCase):
    def test_short_turn_is_unchanged(self):
        self.assertEqual(split_turn("Kratko.", 300), ["Kratko."])

    def test_packs_sentences_up_to_the_limit(self):
        self.assertEqual(
            split_turn("Jedan. Dva. Tri. Četiri. Pet.", 12),
            ["Jedan. Dva.", "Tri. Četiri.", "Pet."],
        )

    def test_splits_overlong_sentence_at_word_boundaries(self):
        chunks = split_turn("aaa bbb ccc ddd eee", 8)

        self.assertEqual(chunks, ["aaa bbb", "ccc ddd", "eee"])

    def test_hard_cuts_a_single_word_longer_than_the_limit(self):
        self.assertEqual(split_turn("abcdefghij", 4), ["abcd", "efgh", "ij"])


if __name__ == "__main__":
    unittest.main()
