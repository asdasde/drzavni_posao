import json
import tempfile
import unittest
from pathlib import Path

from drzavni_posao.search.conform.pipeline import (
    build_corpus,
    chunk_raw_segments,
    parse_episode_filename,
)


def raw_segment(
    start: float, end: float, text: str, speaker: str | None = None
) -> dict:
    value = {"start": start, "end": end, "text": text, "words": []}
    if speaker is not None:
        value["speaker"] = speaker
    return value


class SegmentationTests(unittest.TestCase):
    def test_filename_parser_handles_decomposed_unicode_and_fullwidth_colon(
        self,
    ) -> None:
        parsed = parse_episode_filename(
            "DRŽAVNI POSAO [HQ] - Ep.25： Boško šef (26.10.2012.).mp3"
        )
        self.assertEqual(parsed, ("DP-E0025", 25, "Boško šef", "2012-10-26"))

    def test_filename_parser_preserves_extensionless_transcript_date(self) -> None:
        parsed = parse_episode_filename(
            "DRŽAVNI POSAO [HQ] - Ep.1： Rakija (24.09.2012.)"
        )
        self.assertEqual(parsed, ("DP-E0001", 1, "Rakija", "2012-09-24"))

    def test_windows_overlap_and_do_not_cross_huge_gap(self) -> None:
        segments = [raw_segment(i, i + 0.5, str(i)) for i in range(10)]
        segments.extend(
            [raw_segment(100 + i, 100 + i + 0.5, f"b{i}") for i in range(3)]
        )
        chunks = chunk_raw_segments(
            segments,
            max_utterances=8,
            overlap_utterances=2,
            max_gap_seconds=30,
        )
        indices = [[index for index, _ in chunk] for chunk in chunks]
        self.assertEqual(indices, [list(range(8)), [6, 7, 8, 9], [10, 11, 12]])

    def test_hard_time_limit_shrinks_a_window(self) -> None:
        segments = [
            raw_segment(0, 1, "a"),
            raw_segment(20, 21, "b"),
            raw_segment(40, 41, "c"),
        ]
        chunks = chunk_raw_segments(
            segments,
            max_utterances=3,
            overlap_utterances=1,
            max_seconds=25,
            max_gap_seconds=30,
        )
        self.assertEqual([[i for i, _ in item] for item in chunks], [[0, 1], [1, 2]])

    def test_corpus_build_preserves_original_and_applies_video_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_dir = root / "named"
            transcript_dir.mkdir()
            source_name = "DRŽAVNI POSAO [HQ] - Ep.1： Rakija (24.09.2012.)"
            transcript = {
                "file": source_name,
                "text": "",
                "segments": [
                    raw_segment(1.0, 2.0, "ЉУБАВ и ракија.", "Boskic"),
                    raw_segment(2.1, 3.0, "Čvarkov odgovara.", "Djordje Cvarkov"),
                ],
            }
            (transcript_dir / "episode.json").write_text(
                json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
            )
            episodes, segments = build_corpus(
                transcript_dir,
                root / "episodes.jsonl",
                root / "segments.jsonl",
                source_time_offset_seconds=25,
                max_utterances=8,
            )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(segments), 1)
            self.assertEqual(episodes[0].episode_title, "Rakija")
            self.assertEqual(episodes[0].broadcast_date, "2012-09-24")
            segment = segments[0]
            self.assertIn("ЉУБАВ", segment.text_original)
            self.assertIn("ljubav", segment.text_normalized)
            self.assertEqual(segment.speaker_keys, ["boskic", "cvarkov"])
            self.assertEqual(segment.video_start_seconds, 26.0)
            self.assertEqual(segment.video_start_time, "00:00:26")


if __name__ == "__main__":
    unittest.main()
