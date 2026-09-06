import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from drzavni_posao.search.common.models import SegmentRecord
from drzavni_posao.search.indexing.bm25 import BM25Hit, BM25Index
from drzavni_posao.search.indexing.vector import (
    VECTOR_INDEX_VERSION,
    DenseHit,
    VectorIndex,
    encoder_for_index,
    normalize_vector,
    resolve_model_path,
)
from drzavni_posao.search.retrieval import engine as engine_module
from drzavni_posao.search.retrieval.engine import (
    SearchEngine,
    build_argument_parser,
    timestamped_youtube_url,
)
from drzavni_posao.search.retrieval.fusion import normalize_branch_scores


class StubSemanticEncoder:
    name = "stub-semantic"
    revision = "stub-revision"
    dimension = 2
    uses_e5_prefixes = False

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.casefold()
        if any(
            term in lowered
            for term in ("election scene", "democratic choice", "glasanje")
        ):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    def encode_passages(self, texts):
        return np.stack([self._vector(text) for text in texts])

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
    youtube_video_id: str,
    text: str,
    *,
    semantic: str | None = None,
    speaker_keys: list[str] | None = None,
    start: float = 0.0,
    episode_number: int | None = 1,
) -> SegmentRecord:
    end = start + 20
    keys = speaker_keys or []
    return SegmentRecord(
        youtube_video_id=youtube_video_id,
        episode_number=episode_number,
        episode_title=f"Title {youtube_video_id}",
        segment_id=segment_id,
        start_seconds=start,
        end_seconds=end,
        video_start_seconds=start + 25,
        video_end_seconds=end + 25,
        text_original=text,
        text_semantic=semantic or text,
        text_normalized=text,
        speaker_keys=keys,
        source_segment_indices=[0],
    )


class RetrievalTests(unittest.TestCase):
    def test_segment_record_derives_display_fields(self) -> None:
        segment = SegmentRecord(
            youtube_video_id="e1",
            episode_number=42,
            episode_title="Title",
            segment_id="s1",
            start_seconds=1,
            end_seconds=21,
            video_start_seconds=26,
            video_end_seconds=46,
            text_original="text",
            text_semantic="text",
            text_normalized="text",
            speaker_keys=["boskic"],
            source_segment_indices=[0],
        )

        self.assertEqual(segment.start_time, "00:00:01")
        self.assertEqual(segment.video_end_time, "00:00:46")
        self.assertEqual(segment.speakers, ["Boškić"])
        self.assertEqual(segment.episode_label, "Ep. 42 — Title")
        self.assertEqual(
            segment.model_copy(update={"episode_number": None}).episode_label,
            "Title",
        )
        self.assertNotIn("start_time", segment.model_dump())
        self.assertNotIn("speakers", segment.model_dump())

    def test_resolve_model_path_returns_path_for_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)

            resolved = resolve_model_path(path.as_posix(), local_files_only=True)

            self.assertEqual(resolved, path.resolve())
            self.assertIsInstance(resolved, Path)

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

    def test_vector_normalization_uses_float32_numpy_arrays(self) -> None:
        normalized = normalize_vector([3.0, 4.0])
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(normalized, np.array([0.6, 0.8], dtype=np.float32))
        self.assertIsInstance(self.vector.vectors, np.ndarray)

    def test_vector_normalization_rejects_invalid_vectors(self) -> None:
        for vector in ([], [0.0, 0.0], [1.0, float("nan")]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                normalize_vector(vector)

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

    def test_result_filtering_keeps_ranks_contiguous_and_fills_top_k(self) -> None:
        segments = [
            make_segment("s1", "e1", "shared", start=0),
            make_segment("s2", "e1", "shared", start=5),
            make_segment("s3", "e1", "shared", start=40),
            make_segment("s4", "e2", "shared", start=0),
        ]
        engine = SearchEngine(segments, BM25Index.build(segments), dense_device="cpu")
        for group, deduplicate, expected in (
            (True, True, ["s1", "s4"]),
            (True, False, ["s1", "s4"]),
            (False, True, ["s1", "s3", "s4"]),
            (False, False, ["s1", "s2", "s3"]),
        ):
            with self.subTest(group=group, deduplicate=deduplicate):
                results = engine.search(
                    "shared",
                    top_k=3,
                    group_by_episode=group,
                    deduplicate_overlaps=deduplicate,
                )
                self.assertEqual(
                    [result.segment.segment_id for result in results], expected
                )
                self.assertEqual(
                    [result.rank for result in results], list(range(1, len(expected) + 1))
                )

    def test_cli_defaults_to_bm25(self) -> None:
        args = build_argument_parser().parse_args(["Boškić"])
        self.assertEqual(args.method, "bm25")
        self.assertIsNone(args.vectors)

    def test_cli_converts_path_strings_at_the_boundary(self) -> None:
        args = build_argument_parser().parse_args(
            [
                "Boškić",
                "--segments",
                "custom/segments.jsonl",
                "--bm25",
                "custom/bm25.json",
                "--vectors",
                "custom/vectors",
            ]
        )
        self.assertEqual(args.segments, Path("custom/segments.jsonl"))
        self.assertEqual(args.bm25, Path("custom/bm25.json"))
        self.assertEqual(args.vectors, Path("custom/vectors"))

    def test_human_readable_results_are_logged_as_one_multiline_message(self) -> None:
        result = engine_module.SearchResult(
            rank=1,
            segment=make_segment("s1", "e1", "first line\nsecond line"),
            score=0.75,
            score_components={},
            matched_characters=[],
            matched_terms=[],
            reason="test reason",
            youtube_url_with_timestamp="https://youtu.be/e1?t=25s",
        )

        with self.assertLogs(engine_module.logger, level="INFO") as captured:
            engine_module._log_results([result])

        self.assertEqual(
            captured.records[0].getMessage(),
            "1. Ep. 1 — Title e1\n"
            "   Time: 00:00:25–00:00:45\n"
            "   Characters: unknown\n"
            "   Relative score: 0.750\n"
            "   Reason: test reason\n"
            "   YouTube: https://youtu.be/e1?t=25s\n"
            "   Transcript:\n"
            "     first line\n"
            "     second line\n",
        )
        self.assertEqual(len(captured.records), 1)

    def test_json_cli_output_remains_a_bare_json_payload(self) -> None:
        result = engine_module.SearchResult(
            rank=1,
            segment=make_segment("s1", "e1", "text"),
            score=0.75,
            score_components={},
            matched_characters=[],
            matched_terms=[],
            reason="test reason",
        )
        fake_engine = SimpleNamespace(search=lambda *_args, **_kwargs: [result])
        stdout = StringIO()

        with (
            patch.object(
                engine_module.SearchEngine, "from_paths", return_value=fake_engine
            ),
            patch.object(engine_module.logging, "basicConfig") as configure_logging,
            patch.object(engine_module.logger, "info") as log_info,
            patch("sys.argv", ["dp-search", "Boškić", "--json"]),
            redirect_stdout(stdout),
        ):
            engine_module.main()

        self.assertEqual(json.loads(stdout.getvalue()), [result.to_dict()])
        configure_logging.assert_not_called()
        log_info.assert_not_called()

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

    def test_hybrid_reranking_excludes_unscored_tail_candidates(self) -> None:
        segments = [
            make_segment(f"s{index:02}", f"e{index:02}", "shared")
            for index in range(60)
        ]
        segment_ids = [segment.segment_id for segment in segments]
        bm25 = SimpleNamespace(
            segment_ids=segment_ids,
            search=lambda _tokens, *, top_k: [
                BM25Hit(segment_id, 1.0) for segment_id in segment_ids[:50]
            ][:top_k],
        )
        vector = SimpleNamespace(
            segment_ids=segment_ids,
            dimension=2,
            search_vector=lambda _query, *, top_k: [
                DenseHit(segment_id, 1.0) for segment_id in segment_ids[10:60]
            ][:top_k],
        )
        encoder = SimpleNamespace(
            dimension=2,
            encode_query=lambda _text: np.array([1.0, 0.0], dtype=np.float32),
        )
        reranker = SimpleNamespace(
            name="selective-reranker",
            score=lambda _query, candidates: {
                segment.segment_id: (1.0 if segment.segment_id == "s10" else 0.0)
                for segment in candidates
            },
        )
        engine = SearchEngine(
            segments,
            bm25,
            vector_index=vector,
            text_encoder=encoder,
            reranker=reranker,
        )

        results = engine.search(
            "shared",
            method="hybrid_rerank",
            top_k=10,
            candidate_k=50,
            rerank_k=50,
            group_by_episode=False,
            deduplicate_overlaps=False,
        )

        self.assertEqual(results[0].segment.segment_id, "s10")
        self.assertTrue(
            all("reranker" in result.score_components for result in results)
        )
        self.assertFalse(
            {result.segment.segment_id for result in results} & set(segment_ids[50:60])
        )

    def test_hybrid_reranking_requires_enough_scored_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "rerank_k must be at least top_k"):
            self.engine.search(
                "Boškić",
                method="hybrid_rerank",
                top_k=3,
                rerank_k=2,
            )

    def test_vector_and_bm25_serialization_preserve_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.bm25.save(root / "bm25.json")
            self.vector.save(root / "vectors")
            metadata = json.loads((root / "vectors.json").read_text(encoding="utf-8"))
            self.assertEqual(VECTOR_INDEX_VERSION, 1)
            self.assertEqual(metadata["version"], 1)
            loaded_bm25 = BM25Index.load(root / "bm25.json", segments=self.segments)
            loaded_vector = VectorIndex.load(root / "vectors", segments=self.segments)
            self.assertEqual(loaded_vector.model_revision, "stub-revision")
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

    def test_vector_load_accepts_legacy_version_two_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors"
            self.vector.save(path)
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["version"] = 2
            metadata.pop("model_revision")
            metadata.pop("model_requested_revision")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            loaded = VectorIndex.load(path, segments=self.segments)

            self.assertIsNone(loaded.model_revision)
            self.assertIsNone(loaded.model_requested_revision)

    def test_vector_load_rejects_changed_semantic_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors"
            self.vector.save(path)
            changed = [
                self.segments[0].model_copy(
                    update={"text_semantic": "different embedded passage"}
                ),
                *self.segments[1:],
            ]
            with self.assertRaisesRegex(ValueError, "semantic segment corpus"):
                VectorIndex.load(path, segments=changed)

    def test_index_encoder_reuses_the_recorded_model_revision(self) -> None:
        with patch(
            "drzavni_posao.search.indexing.vector.SentenceTransformerTextEncoder",
            return_value=self.encoder,
        ) as factory:
            encoder_for_index(self.vector, device="cpu", local_files_only=True)

        factory.assert_called_once_with(
            self.vector.model_name,
            device="cpu",
            local_files_only=True,
            uses_e5_prefixes=self.vector.uses_e5_prefixes,
            model_revision="stub-revision",
        )

    def test_youtube_timestamp_preserves_existing_query(self) -> None:
        self.assertEqual(
            timestamped_youtube_url("https://youtu.be/abc?si=xyz", 65.4),
            "https://youtu.be/abc?si=xyz&t=65s",
        )


if __name__ == "__main__":
    unittest.main()
