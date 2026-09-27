import tempfile
import unittest
from pathlib import Path

import numpy as np

from drzavni_posao.search.common.models import SegmentRecord
from drzavni_posao.search.topics import (
    build_dialogue_chunk_corpus,
    build_episode_documents,
    build_topic_result,
    load_stop_words,
    mean_pool_dialogue_chunk_embeddings,
)


def segment(
    *,
    video_id: str,
    episode_number: int,
    title: str,
    segment_number: int,
    source_indices: list[int],
    lines: list[str],
    start: float,
) -> SegmentRecord:
    segment_id = f"{video_id}_{segment_number:04d}"
    return SegmentRecord(
        youtube_video_id=video_id,
        episode_number=episode_number,
        episode_title=title,
        segment_id=segment_id,
        start_seconds=start,
        end_seconds=start + 10,
        video_start_seconds=start,
        video_end_seconds=start + 10,
        text_original="\n".join(lines),
        text_semantic=" ".join(lines),
        text_normalized=" ".join(lines),
        source_segment_indices=source_indices,
        youtube_url=f"https://www.youtube.com/watch?v={video_id}",
    )


class EpisodeTopicTests(unittest.TestCase):
    def setUp(self) -> None:
        # Non-overlapping segments: source indices are disjoint across the episode
        # (the topics corpus is built with --overlap-utterances 0).
        self.episode_one = [
            segment(
                video_id="video-a",
                episode_number=1,
                title="Rakija",
                segment_number=0,
                source_indices=[0, 1],
                lines=[
                    "[Boškić] Prva rečenica.",
                    "[Đorđe Čvarkov] Druga rečenica.",
                ],
                start=0,
            ),
            segment(
                video_id="video-a",
                episode_number=1,
                title="Rakija",
                segment_number=1,
                source_indices=[2, 3],
                lines=[
                    "[Dragan Torbica] Treća rečenica.",
                    "[Boškić] Četvrta rečenica.",
                ],
                start=10,
            ),
        ]
        self.episode_two = [
            segment(
                video_id="video-b",
                episode_number=2,
                title="Pilići",
                segment_number=0,
                source_indices=[0],
                lines=["[Boškić] Peta rečenica."],
                start=0,
            )
        ]

    def test_episode_documents_join_dialogue_without_metadata(self) -> None:
        documents = build_episode_documents(
            [self.episode_two[0], self.episode_one[1], self.episode_one[0]]
        )

        self.assertEqual([document.episode_number for document in documents], [1, 2])
        self.assertEqual(
            documents[0].text,
            "prva recenica druga recenica treca recenica cetvrta recenica",
        )
        self.assertNotIn("rakija", documents[0].text)
        self.assertNotIn("boskic", documents[0].text)
        self.assertEqual(
            documents[0].segment_ids,
            ("video-a_0000", "video-a_0001"),
        )

    def test_overlapping_segments_are_rejected(self) -> None:
        overlapping = [
            self.episode_one[0],
            segment(
                video_id="video-a",
                episode_number=1,
                title="Rakija",
                segment_number=1,
                source_indices=[1, 2],
                lines=[
                    "[Đorđe Čvarkov] Druga rečenica.",
                    "[Dragan Torbica] Treća rečenica.",
                ],
                start=8,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "overlapping segments"):
            build_episode_documents(overlapping)

    def test_dialogue_chunks_are_non_overlapping_and_exclude_metadata(self) -> None:
        corpus = build_dialogue_chunk_corpus(
            [*self.episode_one, *self.episode_two],
            utterances_per_chunk=2,
        )

        self.assertEqual(
            corpus.chunk_texts,
            (
                "Prva rečenica. Druga rečenica.",
                "Treća rečenica. Četvrta rečenica.",
                "Peta rečenica.",
            ),
        )
        self.assertEqual(corpus.chunk_episode_rows, (0, 0, 1))
        self.assertEqual(corpus.chunk_utterance_counts, (2, 2, 1))
        self.assertEqual(
            corpus.raw_episode_texts[0],
            "Prva rečenica. Druga rečenica. Treća rečenica. Četvrta rečenica.",
        )
        self.assertEqual(
            corpus.documents[0].text,
            "prva recenica druga recenica treca recenica cetvrta recenica",
        )
        self.assertNotIn("rakija", corpus.documents[0].text)

    def test_dialogue_chunk_embeddings_are_utterance_weighted(self) -> None:
        corpus = build_dialogue_chunk_corpus(
            [*self.episode_one, *self.episode_two],
            utterances_per_chunk=2,
        )

        pooled = mean_pool_dialogue_chunk_embeddings(
            corpus,
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        np.testing.assert_allclose(
            pooled,
            np.asarray(
                [
                    [2**-0.5, 2**-0.5],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_custom_stop_words_are_normalized_and_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stopwords.txt"
            path.write_text("Takođe\n# comment\nNEŠTO_NOVO\n", encoding="utf-8")

            words = load_stop_words(path)

        self.assertIn("takodje", words)
        self.assertIn("nesto", words)
        self.assertIn("novo", words)
        self.assertIn("boskic", words)

    def test_academic_serbian_stop_word_dictionary_is_loaded(self) -> None:
        words = load_stop_words()

        self.assertGreater(len(words), 1_000)
        self.assertIn("cetvoronoske", words)
        self.assertIn("drugde", words)
        self.assertTrue({"aj", "ca", "imas", "ou"}.issubset(words))

    def test_topic_result_contains_labels_and_representative_episodes(self) -> None:
        class FakeTopicModel:
            @staticmethod
            def get_topic(topic_id: int) -> list[tuple[str, float]]:
                if topic_id == 0:
                    return [
                        ("rakija", 0.8),
                        ("slavlje", 0.5),
                        ("", 0.0),
                    ]
                return []

        documents = build_episode_documents([*self.episode_one, *self.episode_two])
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        result = build_topic_result(
            documents=documents,
            embeddings=embeddings,
            topics=[0, -1],
            topic_model=FakeTopicModel(),
            corpus_fingerprint="corpus",
            settings={"min_topic_size": 2},
            representative_episode_count=1,
        )

        self.assertEqual(result["topic_count"], 1)
        self.assertEqual(result["outlier_episode_count"], 1)
        self.assertEqual(result["topics"][1]["label"], "rakija · slavlje")
        self.assertEqual(len(result["topics"][1]["terms"]), 2)
        self.assertEqual(
            result["topics"][1]["representative_episodes"][0]["episode_number"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
