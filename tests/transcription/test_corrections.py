import unittest

from drzavni_posao.transcription.corrections import (
    apply_episode_corrections,
)
from drzavni_posao.transcription.transcript import Transcript


def raw_segment(
    start: float,
    text: str,
    speaker: str,
) -> dict:
    return {
        "start": start,
        "end": start + 1,
        "text": text,
        "speaker": speaker,
        "words": [{"word": text, "start": start, "end": start + 1}],
    }


class TranscriptCorrectionTests(unittest.TestCase):
    def test_applies_selected_edits_to_collapsed_generation_turns(self) -> None:
        transcript = Transcript(
            [
                raw_segment(0, "Prvi pogrešan deo.", "Boskic"),
                raw_segment(1, "Drugi deo.", "Boskic"),
                raw_segment(2, "Anonimno.", "SPEAKER_00"),
                raw_segment(3, "Treći deo.", "Boskic"),
            ],
            file="episode.mp3",
        )
        edits = [
            {
                "turn_id": 0,
                "original": "pogrešan",
                "replacement": "ispravan",
                "confidence": "high",
            },
            {
                "turn_id": 0,
                "original": "Treći",
                "replacement": "3.",
                "confidence": "medium",
            },
        ]

        corrected, stats = apply_episode_corrections(transcript, edits)

        self.assertEqual(corrected.segments[0]["text"], "Prvi ispravan deo.")
        self.assertEqual(corrected.segments[3]["text"], "Treći deo.")
        self.assertNotIn("words", corrected.segments[0])
        self.assertIn("words", corrected.segments[1])
        self.assertEqual(corrected.segments[0]["start"], 0)
        self.assertEqual(stats.applied, 1)
        self.assertEqual(stats.ignored_confidence, 1)
        self.assertEqual(stats.changed_segments, 1)

    def test_skips_edit_that_generation_notebook_would_skip(self) -> None:
        transcript = Transcript(
            [raw_segment(0, "isto isto", "Djordje Cvarkov")],
            file="episode.mp3",
        )

        corrected, stats = apply_episode_corrections(
            transcript,
            [
                {
                    "turn_id": 0,
                    "original": "isto",
                    "replacement": "drugo",
                    "confidence": "high",
                }
            ],
        )

        self.assertEqual(corrected.segments[0]["text"], "isto isto")
        self.assertEqual(stats.applied, 0)
        self.assertEqual(stats.skipped, 1)

    def test_projects_cross_segment_edit_back_to_timestamped_segments(self) -> None:
        transcript = Transcript(
            [
                raw_segment(0, "prvi kraj", "Dragan Torbica"),
                raw_segment(1, "drugi početak", "Dragan Torbica"),
            ],
            file="episode.mp3",
        )

        corrected, stats = apply_episode_corrections(
            transcript,
            [
                {
                    "turn_id": 0,
                    "original": "kraj drugi",
                    "replacement": "sredina",
                    "confidence": "high",
                }
            ],
        )

        self.assertEqual(
            " ".join(segment["text"] for segment in corrected.segments),
            "prvi sredina početak",
        )
        self.assertEqual(corrected.segments[0]["start"], 0)
        self.assertEqual(corrected.segments[1]["start"], 1)
        self.assertEqual(stats.applied, 1)


if __name__ == "__main__":
    unittest.main()
