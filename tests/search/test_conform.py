import json
import tempfile
import unittest
import unicodedata
from pathlib import Path

from drzavni_posao.transcription.episode import Episode, EpisodeCatalog, EpisodeMetadata
from drzavni_posao.search.conform.pipeline import (
    build_records,
    chunk_raw_segments,
    load_episodes_from_mlflow_export,
    source_time_offset_from_export,
)
from drzavni_posao.transcription.store import MediaStore
from drzavni_posao.transcription.transcript import Transcript


def raw_segment(
    start: float, end: float, text: str, speaker: str | None = None
) -> dict:
    value = {"start": start, "end": end, "text": text, "words": []}
    if speaker is not None:
        value["speaker"] = speaker
    return value


def catalog_record() -> dict:
    return {
        "episode_number": 1,
        "title": "Rakija",
        "date": "2012-09-24",
        "is_special": False,
        "original_url": "https://www.youtube.com/watch?v=McwPB-eQ2BY",
        "youtube_video_id": "McwPB-eQ2BY",
        "youtube_title": "DRŽAVNI POSAO [HQ] - Ep.1: Rakija (24.09.2012.)",
        "duration_seconds": 350,
        "view_count": 4_200_000,
        "season": 1,
        "playlist_id": "playlist-1",
        "playlist_index": None,
    }


class SegmentationTests(unittest.TestCase):
    def test_catalog_matches_normalized_audio_name(self) -> None:
        catalog = EpisodeCatalog.model_validate({"episodes": [catalog_record()]})
        title = catalog_record()["youtube_title"].replace(":", "：")
        source = unicodedata.normalize("NFD", f"{title}.mp3")

        self.assertEqual(
            catalog.metadata_for(Path(source)),
            EpisodeMetadata.model_validate(catalog_record()),
        )

    def test_episode_holds_audio_and_catalog_metadata(self) -> None:
        metadata = EpisodeMetadata.model_validate(catalog_record())
        episode = Episode(Path("anything.mp3"), metadata)
        self.assertEqual(
            (
                episode.metadata.episode_number,
                episode.metadata.title,
                episode.metadata.date.isoformat(),
                episode.metadata.original_url,
            ),
            (
                1,
                "Rakija",
                "2012-09-24",
                "https://www.youtube.com/watch?v=McwPB-eQ2BY",
            ),
        )
        self.assertEqual(episode.metadata, metadata)

    def test_corpus_supports_special_episodes_without_numbers(self) -> None:
        record = {
            **catalog_record(),
            "episode_number": None,
            "title": "Specijal",
            "is_special": True,
            "youtube_video_id": "special-video",
            "youtube_title": "DRŽAVNI POSAO [HQ] - Specijal (01.01.2014.)",
        }
        catalog = EpisodeCatalog.model_validate({"episodes": [record]})
        transcript = {
            "file": record["youtube_title"],
            "segments": [raw_segment(0, 1, "Specijalna epizoda")],
        }
        episode = Episode(
            Path(f"{record['youtube_title']}.mp3"),
            catalog.episodes[0],
            transcript=Transcript.from_dict(transcript),
        )

        segments = build_records([episode])

        self.assertEqual(segments[0].episode_title, "Specijal")
        self.assertEqual(segments[0].youtube_video_id, "special-video")

    def test_media_store_requires_catalog_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "anything unknown.mp3").touch()
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not found in episode catalog"):
                MediaStore(audio_dir, root / "runs", catalog_path=catalog_path)

    def test_media_store_matches_normalized_youtube_title(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            title = catalog_record()["youtube_title"].replace(":", "：")
            audio_name = unicodedata.normalize("NFD", f"{title}.mp3")
            (audio_dir / audio_name).touch()
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )

            media = MediaStore(audio_dir, root / "runs", catalog_path=catalog_path)

            self.assertEqual(
                media.episode("1").metadata,
                EpisodeMetadata.model_validate(catalog_record()),
            )

    def test_media_store_configures_episode_transcript_and_trims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )
            media = MediaStore(
                None,
                root / "runs",
                trim_start=25,
                trim_end=3,
                catalog_path=catalog_path,
            )

            episode = media.episode("1")

            self.assertIsNone(episode.transcript)
            self.assertEqual(episode.trim_start, 25)
            self.assertEqual(episode.trim_end, 3)

    def test_media_store_loads_transcripts_into_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_dir = root / "audio"
            run_dir = root / "runs"
            audio_dir.mkdir()
            run_dir.mkdir()
            title = catalog_record()["youtube_title"]
            (audio_dir / f"{title}.mp3").touch()
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )
            (run_dir / "1.json").write_text(
                json.dumps({"file": title, "segments": []}), encoding="utf-8"
            )
            media = MediaStore(audio_dir, run_dir, catalog_path=catalog_path)

            episodes = media.load_all()

            self.assertEqual(episodes, [media.episode("1")])
            self.assertEqual(episodes[0].transcript.file, title)

    def test_media_store_loads_transcripts_without_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs"
            run_dir.mkdir()
            title = catalog_record()["youtube_title"]
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )
            (run_dir / "1.json").write_text(
                json.dumps({"file": title, "segments": []}), encoding="utf-8"
            )

            media = MediaStore(None, run_dir, catalog_path=catalog_path)
            episodes = media.load_all()

            self.assertEqual(episodes, [media.episode("1")])
            self.assertEqual(episodes[0].transcript.file, title)

    def test_media_store_loads_special_transcript_without_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs"
            run_dir.mkdir()
            title = "Novogodišnji specijal"
            key = f"{title} (2014-12-31)"
            record = {
                **catalog_record(),
                "episode_number": None,
                "title": title,
                "date": "2014-12-31",
                "is_special": True,
                "youtube_title": f"DRŽAVNI POSAO [HQ] - {title} (31.12.2014.)",
            }
            later_record = {
                **record,
                "date": "2015-12-31",
                "youtube_video_id": "later-special",
                "youtube_title": f"DRŽAVNI POSAO [HQ] - {title} (31.12.2015.)",
            }
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [record, later_record]}), encoding="utf-8"
            )
            decomposed_key = unicodedata.normalize("NFD", key)
            (run_dir / f"{decomposed_key}.json").write_text(
                json.dumps({"file": record["youtube_title"], "segments": []}),
                encoding="utf-8",
            )

            media = MediaStore(None, run_dir, catalog_path=catalog_path)
            episodes = media.load_all()

            self.assertEqual(len(list(media.episodes())), 2)
            self.assertEqual(episodes, [media.episode(key)])
            self.assertEqual(episodes[0].metadata.date.isoformat(), "2014-12-31")

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
            audio_dir = root / "audio"
            transcript_dir = root / "transcripts"
            audio_dir.mkdir()
            transcript_dir.mkdir()
            source_name = "DRŽAVNI POSAO [HQ] - Ep.1： Rakija (24.09.2012.)"
            (audio_dir / f"{source_name}.mp3").touch()
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )
            transcript = {
                "file": source_name,
                "text": "",
                "segments": [
                    raw_segment(1.0, 2.0, "ЉУБАВ и ракија.", "Boskic"),
                    raw_segment(2.1, 3.0, "Čvarkov odgovara.", "Djordje Cvarkov"),
                ],
            }
            (transcript_dir / "1.json").write_text(
                json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
            )
            media = MediaStore(audio_dir, transcript_dir, catalog_path=catalog_path)
            segments = build_records(
                media.load_all(),
                source_time_offset_seconds=25,
                max_utterances=8,
            )
            self.assertEqual(len(segments), 1)
            segment = segments[0]
            self.assertEqual(segment.youtube_video_id, "McwPB-eQ2BY")
            self.assertEqual(segment.episode_title, "Rakija")
            self.assertEqual(
                segment.youtube_url,
                "https://www.youtube.com/watch?v=McwPB-eQ2BY",
            )
            self.assertIn("ЉУБАВ", segment.text_original)
            self.assertIn("ljubav", segment.text_normalized)
            self.assertEqual(segment.speaker_keys, ["boskic", "cvarkov"])
            self.assertEqual(segment.video_start_seconds, 26.0)
            self.assertEqual(segment.video_start_time, "00:00:26")

    def test_mlflow_export_prefers_character_assigner_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_dir = root / "export"
            episode_dir = export_dir / "episodes" / "1"
            (episode_dir / "transcription").mkdir(parents=True)
            (episode_dir / "character_assigner").mkdir()
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )
            (export_dir / "config.json").write_text(
                json.dumps({"audio": {"trim_start": 25, "trim_end": 0}}),
                encoding="utf-8",
            )
            (episode_dir / "transcription" / "transcript.json").write_text(
                json.dumps(
                    {
                        "file": "episode.mp3",
                        "segments": [raw_segment(1.0, 2.0, "Anoniman.", "SPEAKER_00")],
                    }
                ),
                encoding="utf-8",
            )
            (episode_dir / "character_assigner" / "transcript.json").write_text(
                json.dumps(
                    {
                        "file": "episode.mp3",
                        "segments": [raw_segment(1.0, 2.0, "Boskic govori.", "Boskic")],
                    }
                ),
                encoding="utf-8",
            )
            media = MediaStore(None, export_dir, catalog_path=catalog_path)

            episodes = load_episodes_from_mlflow_export(media, export_dir)
            segments = build_records(
                episodes,
                source_time_offset_seconds=source_time_offset_from_export(export_dir),
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].transcript.segments[0]["speaker"], "Boskic")
            self.assertEqual(segments[0].speaker_keys, ["boskic"])
            self.assertEqual(segments[0].video_start_seconds, 26.0)

    def test_mlflow_export_rejects_flat_transcript_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "youtube_episodes.json"
            catalog_path.write_text(
                json.dumps({"episodes": [catalog_record()]}), encoding="utf-8"
            )
            (root / "1.json").write_text(
                json.dumps({"file": "episode.mp3", "segments": []}),
                encoding="utf-8",
            )
            media = MediaStore(None, root, catalog_path=catalog_path)

            with self.assertRaisesRegex(ValueError, "not an MLflow export"):
                load_episodes_from_mlflow_export(media, root)


if __name__ == "__main__":
    unittest.main()
