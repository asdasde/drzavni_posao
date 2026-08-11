import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from drzavni_posao.constants import SR
from drzavni_posao.transcriber import (
    Transcriber,
    TranscriberConfig,
    is_collapsed,
)


DEFAULTS = dict(
    min_duration=8.0,
    max_chars_per_sec=1.5,
)


class CollapsedChunkTests(unittest.TestCase):
    def test_short_segment_is_not_collapsed(self):
        self.assertFalse(
            is_collapsed(
                {"start": 0.0, "end": 3.0, "text": "Pa što je to?"},
                **DEFAULTS,
            )
        )

    def test_dense_long_segment_is_not_collapsed(self):
        text = " ".join(["reci"] * 80)
        self.assertFalse(
            is_collapsed({"start": 0.0, "end": 20.0, "text": text}, **DEFAULTS)
        )

    def test_sparse_long_segment_is_collapsed(self):
        self.assertTrue(
            is_collapsed(
                {"start": 288.838, "end": 311.838, "text": "Pa što je to?"},
                **DEFAULTS,
            )
        )

    def test_thresholds_come_from_arguments(self):
        segment = {"start": 0.0, "end": 20.0, "text": "x" * 40}
        self.assertFalse(is_collapsed(segment, min_duration=8.0, max_chars_per_sec=1.5))
        self.assertTrue(is_collapsed(segment, min_duration=8.0, max_chars_per_sec=3.0))


class TranscriberConfigTrackingTests(unittest.TestCase):
    def test_default_retry_knobs_omitted_when_excluding_defaults(self):
        data = TranscriberConfig(model_name="large-v3").model_dump(
            exclude_defaults=True
        )
        self.assertEqual(data, {"model_name": "large-v3"})

    def test_custom_retry_knobs_kept_when_excluding_defaults(self):
        data = TranscriberConfig(
            model_name="large-v3",
            chunk_size=20,
            retry_collapsed=False,
            retry_chunk_size=5,
            retry_collapsed_min_duration=6.0,
        ).model_dump(exclude_defaults=True)
        self.assertEqual(data["chunk_size"], 20)
        self.assertFalse(data["retry_collapsed"])
        self.assertEqual(data["retry_chunk_size"], 5)
        self.assertEqual(data["retry_collapsed_min_duration"], 6.0)
        self.assertNotIn("retry_collapsed_max_chars_per_sec", data)

    def test_empty_vad_options_match_absent(self):
        self.assertEqual(
            TranscriberConfig(model_name="large-v3").model_dump(exclude_defaults=True),
            TranscriberConfig(model_name="large-v3", vad_options={}).model_dump(
                exclude_defaults=True
            ),
        )

    def test_vad_alignment_repair_is_enabled_by_min_silence(self):
        default = TranscriberConfig(model_name="large-v3")
        enabled = TranscriberConfig(
            model_name="large-v3",
            vad_repair_min_silence=1.0,
        )

        self.assertIsNone(default.vad_repair_min_silence)
        self.assertFalse(hasattr(default, "repair_alignment_with_vad"))
        self.assertEqual(
            enabled.model_dump(exclude_defaults=True)["vad_repair_min_silence"],
            1.0,
        )


class VadRealignmentTests(unittest.TestCase):
    def test_transcriber_runs_second_alignment_on_vad_split_segments(self):
        class FakeVad:
            def merge_chunks(self, segments, chunk_size, onset, offset):
                return [
                    {
                        "start": 10.0,
                        "end": 20.0,
                        "segments": [(10.0, 13.0), (17.0, 20.0)],
                    }
                ]

        transcriber = Transcriber.__new__(Transcriber)
        transcriber.config = TranscriberConfig(
            model_name="large-v3",
            vad_repair_min_silence=1.0,
        )
        transcriber.asr = MagicMock()
        transcriber.asr.vad_model = FakeVad()
        transcriber.asr.transcribe.side_effect = lambda *args, **kwargs: (
            transcriber.asr.vad_model.merge_chunks([], 30, onset=0.5, offset=0.5),
            {
                "language": "sr",
                "segments": [
                    {
                        "start": 10.0,
                        "end": 20.0,
                        "text": "ne može tako. Đe ste ljudi",
                    }
                ],
            },
        )[1]
        transcriber.align_model = object()
        transcriber.align_meta = {}
        episode = MagicMock()
        episode.metadata.episode_number = 72
        episode.full_name = "72.mp3"
        episode.audio.return_value = np.zeros(30 * SR, dtype=np.float32)

        first_alignment = {
            "segments": [
                {
                    "start": 10.2,
                    "end": 20.0,
                    "text": "ne može tako. Đe ste ljudi",
                    "words": [
                        {"word": "ne", "start": 10.2, "end": 10.5},
                        {"word": "može", "start": 10.6, "end": 11.0},
                        {"word": "tako.", "start": 12.0, "end": 16.0},
                        {"word": "Đe", "start": 14.0, "end": 18.0},
                        {"word": "ste", "start": 18.1, "end": 18.5},
                        {"word": "ljudi", "start": 19.0, "end": 20.0},
                    ],
                }
            ]
        }
        second_alignment = {
            "segments": [
                {
                    "start": 10.2,
                    "end": 12.8,
                    "text": "ne može tako.",
                    "words": [],
                },
                {
                    "start": 17.2,
                    "end": 20.0,
                    "text": "Đe ste ljudi",
                    "words": [],
                },
            ]
        }

        with patch(
            "drzavni_posao.transcriber.whisperx.align",
            side_effect=[first_alignment, second_alignment],
        ) as align:
            transcriber.run(episode)

        self.assertEqual(align.call_count, 2)
        self.assertEqual(
            align.call_args_list[1].args[0],
            [
                {"start": 10.1, "end": 13.1, "text": "ne može tako."},
                {"start": 16.9, "end": 20.1, "text": "Đe ste ljudi"},
            ],
        )
        self.assertEqual(episode.transcript.segments, second_alignment["segments"])


class RetryCollapsedTests(unittest.TestCase):
    def setUp(self):
        self.transcriber = Transcriber.__new__(Transcriber)
        self.transcriber.config = TranscriberConfig(model_name="large-v3")
        self.transcriber.asr = MagicMock()

    def test_retries_only_collapsed_spans_and_shifts_timestamps(self):
        audio = np.zeros(40 * SR, dtype=np.float32)
        segments = [
            {"start": 0.0, "end": 5.0, "text": "zdravo svima kako ste"},
            {"start": 10.0, "end": 33.0, "text": "Pa što je to?"},
            {"start": 35.0, "end": 38.0, "text": "ajde"},
        ]
        self.transcriber.asr.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 9.0, "text": "prvi deo"},
                {"start": 9.0, "end": 23.0, "text": "drugi deo monologa"},
            ]
        }

        repaired = self.transcriber._retry_collapsed(audio, segments)

        self.transcriber.asr.transcribe.assert_called_once()
        _, kwargs = self.transcriber.asr.transcribe.call_args
        self.assertEqual(kwargs["chunk_size"], 10)
        self.assertEqual(
            [s["text"] for s in repaired],
            ["zdravo svima kako ste", "prvi deo", "drugi deo monologa", "ajde"],
        )
        self.assertEqual(repaired[1]["start"], 10.0)
        self.assertEqual(repaired[1]["end"], 19.0)
        self.assertEqual(repaired[2]["start"], 19.0)
        self.assertEqual(repaired[2]["end"], 33.0)

    def test_uses_config_retry_chunk_size(self):
        self.transcriber.config = TranscriberConfig(
            model_name="large-v3",
            retry_chunk_size=5,
        )
        audio = np.zeros(30 * SR, dtype=np.float32)
        segments = [{"start": 0.0, "end": 20.0, "text": "..."}]
        self.transcriber.asr.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 20.0, "text": "popravljeno"}]
        }

        self.transcriber._retry_collapsed(audio, segments)

        _, kwargs = self.transcriber.asr.transcribe.call_args
        self.assertEqual(kwargs["chunk_size"], 5)

    def test_keeps_original_when_retry_returns_nothing(self):
        audio = np.zeros(30 * SR, dtype=np.float32)
        segments = [{"start": 0.0, "end": 20.0, "text": "..."}]
        self.transcriber.asr.transcribe.return_value = {"segments": []}

        repaired = self.transcriber._retry_collapsed(audio, segments)

        self.assertEqual(repaired, segments)

    def test_skips_retry_when_disabled(self):
        self.transcriber.config = TranscriberConfig(
            model_name="large-v3",
            retry_collapsed=False,
        )
        audio = np.zeros(30 * SR, dtype=np.float32)
        segments = [{"start": 0.0, "end": 20.0, "text": "..."}]

        repaired = self.transcriber._retry_collapsed(audio, segments)

        self.assertEqual(repaired, segments)
        self.transcriber.asr.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
