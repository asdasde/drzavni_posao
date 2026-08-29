import unittest
from datetime import date
from types import SimpleNamespace

from drzavni_posao.generation.dataset import (
    TranscriptQualityError,
    assign_folds,
    build_pair,
    select_splits,
)
from drzavni_posao.transcript import Transcript


class GenerationDatasetTests(unittest.TestCase):
    def test_builds_pair_from_premise_and_episode(self):
        episode = SimpleNamespace(
            key="330",
            transcript=Transcript(
                [
                    {
                        "start": 0,
                        "end": 1,
                        "speaker": "Djordje Cvarkov",
                        "text": "Dobro.",
                    },
                    {
                        "start": 1,
                        "end": 2,
                        "speaker": "Djordje Cvarkov",
                        "text": "Šta ćemo?",
                    },
                    {
                        "start": 2,
                        "end": 3,
                        "speaker": "Boskic",
                        "text": "Ne znam.",
                    },
                ]
            ),
        )

        self.assertEqual(
            build_pair(episode, "Razgovaraju o problemu.", check_quality=False),
            {
                "episode_key": "330",
                "premise": "Razgovaraju o problemu.",
                "dialogue": ("[Djordje Cvarkov] Dobro. Šta ćemo?\n[Boskic] Ne znam."),
            },
        )

    def test_removes_anonymous_turns(self):
        episode = self.quality_episode(include_anonymous=True)

        pair = build_pair(episode, "Razgovaraju o problemu.")

        self.assertNotIn("[[anonymous]]", pair["dialogue"])
        self.assertIn("[Djordje Cvarkov]", pair["dialogue"])
        self.assertIn("[Boskic]", pair["dialogue"])

    def test_keeps_structurally_sound_low_confidence_transcript(self):
        episode = self.quality_episode(word_score=0.1)

        pair = build_pair(episode, "Razgovaraju o problemu.")

        self.assertTrue(pair["dialogue"])

    def test_rejects_transcript_with_too_many_unknown_speaker_words(self):
        episode = self.quality_episode()
        episode.transcript.segments.append(
            {
                "start": 300,
                "end": 310,
                "speaker": "SPEAKER_00",
                "text": self.unique_words("unknown", 20),
            }
        )

        with self.assertRaises(TranscriptQualityError) as context:
            build_pair(episode, "Razgovaraju o problemu.")

        self.assertIn("unknown_speakers_gt_2pct", context.exception.reasons)

    def test_rejects_structurally_broken_transcript(self):
        episode = SimpleNamespace(
            key="328",
            transcript=Transcript(
                [
                    {
                        "start": 60,
                        "end": 64,
                        "speaker": "Djordje Cvarkov",
                        "text": "Hvala vam!",
                    },
                    {
                        "start": 95,
                        "end": 96,
                        "speaker": "Djordje Cvarkov",
                        "text": "Hvala vam!",
                    },
                    {
                        "start": 115,
                        "end": 116,
                        "speaker": "Djordje Cvarkov",
                        "text": "Hvala vam!",
                    },
                    {
                        "start": 223,
                        "end": 225,
                        "speaker": "Djordje Cvarkov",
                        "text": "Državni posao vam je omogućio.",
                    },
                ]
            ),
        )

        with self.assertRaises(TranscriptQualityError) as context:
            build_pair(episode, "Predizborna tišina.")

        self.assertIn("short_transcript", context.exception.reasons)
        self.assertIn("fewer_than_two_core_speakers", context.exception.reasons)
        self.assertIn("repeated_4grams_gt_6pct", context.exception.reasons)

    def test_assigns_chronological_interleaved_splits(self):
        episodes = [
            self.episode("3", date(2012, 9, 26)),
            self.episode("1", date(2012, 9, 24)),
            self.episode("2", date(2012, 9, 25)),
        ]

        folds = assign_folds(episodes, 3)

        self.assertEqual(folds, {"1": 0, "2": 1, "3": 2})
        self.assertEqual(
            select_splits(folds, 0, 1),
            {"1": "test", "2": "validation", "3": "train"},
        )

    @staticmethod
    def episode(key, broadcast_date):
        return SimpleNamespace(
            key=key,
            metadata=SimpleNamespace(date=broadcast_date),
        )

    @classmethod
    def quality_episode(
        cls,
        *,
        include_anonymous=False,
        word_score=0.9,
    ):
        segments = [
            {
                "start": 0,
                "end": 150,
                "speaker": "Djordje Cvarkov",
                "text": cls.unique_words("djordje", 210),
                "words": [{"score": word_score}],
            }
        ]
        if include_anonymous:
            segments.append(
                {
                    "start": 150,
                    "end": 151,
                    "speaker": "[anonymous]",
                    "text": "Ovo ne treba da ostane.",
                    "words": [{"score": word_score}],
                }
            )
        segments.append(
            {
                "start": 151,
                "end": 300,
                "speaker": "Boskic",
                "text": cls.unique_words("boskic", 210),
                "words": [{"score": word_score}],
            }
        )
        return SimpleNamespace(key="330", transcript=Transcript(segments))

    @staticmethod
    def unique_words(prefix, count):
        return " ".join(
            f"{prefix}{chr(97 + index % 26)}{chr(97 + (index // 26) % 26)}"
            for index in range(count)
        )


if __name__ == "__main__":
    unittest.main()
