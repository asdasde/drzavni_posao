from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from drzavni_posao.search.common.models import SegmentRecord, format_timestamp
from drzavni_posao.search.indexing.bm25 import BM25Index
from drzavni_posao.search.indexing.vector import VectorIndex
from drzavni_posao.search.retrieval.engine import (
    SearchEngine,
    build_argument_parser,
    timestamped_youtube_url,
)
from drzavni_posao.search.retrieval.fusion import normalize_branch_scores


class StubSemanticEncoder:
    name = "stub-semantic"
    dimension = 2
    uses_e5_prefixes = False

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if any(
            term in lowered
            for term in ("election scene", "democratic choice", "glasanje")
        ):
            return [1.0, 0.0]
        return [0.0, 1.0]

    def encode_passages(self, texts):
        return [self._vector(text) for text in texts]

    def encode_query(self, text):
        return self._vector(text)


class StubCrossEncoder:
    name = "stub-cross-encoder"

    def score(self, query, segments):
        del query
        return {
            segment.segment_id: 10.0 if segment.segment_id == "s2" else -10.0
            for segment in segments
        }


def make_segment(
    segment_id: str,
    episode_id: str,
    text: str,
    *,
    semantic: str | None = None,
    speaker_keys: list[str] | None = None,
    start: float = 0.0,
) -> SegmentRecord:
    end = start + 20
    keys = speaker_keys or []
    return SegmentRecord(
        episode_id=episode_id,
        episode_title=f"Title {episode_id}",
        segment_id=segment_id,
        start_seconds=start,
        end_seconds=end,
        start_time=format_timestamp(start),
        end_time=format_timestamp(end),
        video_start_seconds=start + 25,
        video_end_seconds=end + 25,
        video_start_time=format_timestamp(start + 25),
        video_end_time=format_timestamp(end + 25),
        text_original=text,
        text_semantic=semantic or text,
        text_normalized=text,
        speakers=keys,
        speaker_keys=keys,
        source_segment_indices=[0],
    )


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            make_segment(
                "s1",
                "e1",
                "boskic cvarkov izbori rasprava",
                semantic="election scene with an argument",
                speaker_keys=["boskic", "cvarkov"],
            ),
            make_segment(
                "s2",
                "e2",
                "torbica pokvaren automobil",
                semantic="broken car scene",
                speaker_keys=["torbica"],
            ),
            make_segment(
                "s3",
                "e3",
                "izbori kandidat",
                semantic="election scene",
            ),
        ]
        self.bm25 = BM25Index.build(self.segments)
        self.encoder = StubSemanticEncoder()
        self.vector = VectorIndex.build(self.segments, self.encoder)
        self.engine = SearchEngine(
            self.segments,
            self.bm25,
            vector_index=self.vector,
            text_encoder=self.encoder,
        )

    def test_score_normalization_handles_negative_and_constant_values(self) -> None:
        self.assertEqual(normalize_branch_scores({"a": -2, "b": 0}), {"a": 0, "b": 1})
        self.assertEqual(normalize_branch_scores({"a": 3, "b": 3}), {"a": 1, "b": 1})
        self.assertEqual(normalize_branch_scores({}), {})

    def test_bm25_exact_entity_and_topic(self) -> None:
        results = self.engine.search(
            "Boškić Čvarkov", method="bm25", group_by_episode=False
        )
        self.assertEqual(results[0].segment.segment_id, "s1")

    def test_search_defaults_to_bm25_without_vector_assets(self) -> None:
        engine = SearchEngine(self.segments, self.bm25)
        results = engine.search("Boškić Čvarkov", group_by_episode=False)
        self.assertEqual(results[0].segment.segment_id, "s1")

    def test_cli_defaults_to_bm25(self) -> None:
        args = build_argument_parser().parse_args(["Boškić"])
        self.assertEqual(args.method, "bm25")
        self.assertIsNone(args.vectors)

    def test_dense_cross_wording_match(self) -> None:
        results = self.engine.search(
            "democratic choice",
            method="dense",
            group_by_episode=False,
            deduplicate_overlaps=False,
        )
        self.assertIn(results[0].segment.segment_id, {"s1", "s3"})

    def test_hybrid_character_boost_prefers_named_scene(self) -> None:
        results = self.engine.search(
            "Boškić and Čvarkov democratic choice",
            method="hybrid",
            group_by_episode=False,
            deduplicate_overlaps=False,
        )
        self.assertEqual(results[0].segment.segment_id, "s1")
        self.assertEqual(results[0].score_components["metadata"], 1.0)

    def test_cross_encoder_scores_the_entire_corpus_without_other_retrievers(
        self,
    ) -> None:
        engine = SearchEngine(
            self.segments,
            self.bm25,
            reranker=StubCrossEncoder(),
        )
        results = engine.search(
            "unrelated wording",
            method="cross_encoder",
            candidate_k=1,
            group_by_episode=False,
            deduplicate_overlaps=False,
        )
        self.assertEqual(results[0].segment.segment_id, "s2")
        self.assertEqual(len(results), 3)
        self.assertIn("cross_encoder", results[0].score_components)

    def test_vector_and_bm25_serialization_preserve_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.bm25.save(root / "bm25.json")
            self.vector.save(root / "vectors")
            loaded_bm25 = BM25Index.load(root / "bm25.json", segments=self.segments)
            loaded_vector = VectorIndex.load(root / "vectors", segments=self.segments)
            engine = SearchEngine(
                self.segments,
                loaded_bm25,
                vector_index=loaded_vector,
                text_encoder=self.encoder,
            )
            self.assertEqual(
                engine.search("Boškić Čvarkov democratic choice", method="hybrid")[
                    0
                ].segment.segment_id,
                "s1",
            )

    def test_vector_load_rejects_changed_semantic_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors"
            self.vector.save(path)
            changed = [
                replace(self.segments[0], text_semantic="different embedded passage"),
                *self.segments[1:],
            ]
            with self.assertRaisesRegex(
                ValueError, "semantic segment corpus"
            ):
                VectorIndex.load(path, segments=changed)

    def test_youtube_timestamp_preserves_existing_query(self) -> None:
        self.assertEqual(
            timestamped_youtube_url("https://youtu.be/abc?si=xyz", 65.4),
            "https://youtu.be/abc?si=xyz&t=65s",
        )


if __name__ == "__main__":
    unittest.main()
