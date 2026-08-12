import json
import tempfile
import unittest
from pathlib import Path

from drzavni_posao.comparison import compare, load_transcripts, scribe_to_transcript
from drzavni_posao.transcript import Transcript


class ComparisonTranscriptTests(unittest.TestCase):
    def test_scribe_conversion_and_comparison_use_transcripts(self) -> None:
        transcript = scribe_to_transcript(
            {
                "words": [
                    {
                        "type": "word",
                        "text": "zdravo",
                        "start": 0.0,
                        "end": 1.0,
                        "speaker_id": "A",
                        "logprob": -0.1,
                    }
                ]
            }
        )

        self.assertIsInstance(transcript, Transcript)
        result = compare("1", transcript, transcript)
        self.assertEqual(result.counts, {"equal": 1})

    def test_load_transcripts_returns_transcript_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "1.json").write_text(
                json.dumps(
                    {
                        "file": "episode.mp3",
                        "segments": [{"start": 0, "end": 1, "text": "zdravo"}],
                    }
                ),
                encoding="utf-8",
            )

            transcripts = load_transcripts(directory)

            self.assertIsInstance(transcripts["1"], Transcript)
            self.assertEqual(transcripts["1"].file, "episode.mp3")


if __name__ == "__main__":
    unittest.main()
