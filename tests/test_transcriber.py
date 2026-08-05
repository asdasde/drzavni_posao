import unittest
from unittest.mock import MagicMock

import numpy as np

from drzavni_posao.constants import SR
from drzavni_posao.transcriber import Transcriber, TranscriberConfig, is_collapsed


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
            retry_chunk_size=5,
            retry_collapsed_min_duration=6.0,
        ).model_dump(exclude_defaults=True)
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


if __name__ == "__main__":
    unittest.main()
