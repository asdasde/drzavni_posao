import json
import math
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

from drzavni_posao.search.evaluation.runner import (
    EvaluationQuery,
    Judgment,
    build_argument_parser,
    build_evaluation_manifest,
    calculate_query_metrics,
    load_evaluation_queries,
    relevance_grades,
)
from drzavni_posao.search.retrieval.fusion import HybridWeights
from drzavni_posao.search.retrieval.models import SearchResult

from .test_engine import make_segment


def result(
    rank: int, segment_id: str, youtube_video_id: str, start: float
) -> SearchResult:
    return SearchResult(
        rank=rank,
        segment=make_segment(segment_id, youtube_video_id, "text", start=start),
        score=1 / rank,
        score_components={},
        matched_characters=[],
        matched_terms=[],
    )


class EvaluationTests(unittest.TestCase):
    def test_evaluate_cli_defaults_to_bm25(self) -> None:
        args = build_argument_parser().parse_args([])
        self.assertEqual(args.methods, "bm25")
        self.assertIsNone(args.vectors)
        self.assertEqual(args.queries, Path("data/evaluation_queries.jsonl"))
        self.assertEqual(args.relevance_level, "scene")

    def test_evaluate_cli_accepts_episode_relevance(self) -> None:
        args = build_argument_parser().parse_args(["--relevance-level", "episode"])
        self.assertEqual(args.relevance_level, "episode")

    def test_episode_only_judgment_parses_only_in_episode_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "ona epizoda sa jubilejom",
                        "judgments": [{"youtube_video_id": "e1"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            episode_queries = load_evaluation_queries(path, relevance_level="episode")
            self.assertEqual(episode_queries[0].judgments, (Judgment("e1"),))
            with self.assertRaisesRegex(ValueError, "A scene judgment needs"):
                load_evaluation_queries(path)

    def test_evaluation_manifest_captures_configuration_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_path = root / "queries.jsonl"
            segments_path = root / "segments.jsonl"
            bm25_path = root / "bm25.json"
            vector_path = root / "vectors"
            vector_metadata_path = vector_path.with_suffix(".json")
            vector_data_path = vector_path.with_suffix(".f32")
            query_path.write_text('{"query_id":"q1"}\n', encoding="utf-8")
            segments_path.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
            bm25_path.write_text('{"version":1}\n', encoding="utf-8")
            vector_data_path.write_bytes(b"\x00\x00\x80?")
            vector_metadata_path.write_text(
                json.dumps({"vectors_file": vector_data_path.name}),
                encoding="utf-8",
            )
            engine = SimpleNamespace(
                segments=[make_segment("s1", "e1", "text")],
                vector_index=SimpleNamespace(
                    model_name="dense-model",
                    model_revision="dense-commit",
                    semantic_corpus_fingerprint="semantic-fingerprint",
                    dimension=768,
                    uses_e5_prefixes=True,
                ),
                reranker=SimpleNamespace(
                    name="reranker-model",
                    revision="reranker-commit",
                    batch_size=16,
                ),
                dense_device="cpu",
                local_models_only=True,
            )

            manifest = build_evaluation_manifest(
                engine=engine,
                methods=["bm25", "hybrid_rerank"],
                queries=[
                    EvaluationQuery(
                        query_id="q1",
                        query="scene query",
                        judgments=(Judgment("e1", 25, 45),),
                    )
                ],
                query_files=[query_path],
                segments_path=segments_path,
                bm25_path=bm25_path,
                vector_path=vector_path,
                split="test",
                ranking_unit="segment",
                candidate_k=50,
                tolerance_seconds=2.0,
                bootstrap_samples=1000,
                warmup=True,
                weights=HybridWeights(),
            )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["models"]["vector"]["revision"], "dense-commit")
        self.assertEqual(manifest["models"]["reranker"]["revision"], "reranker-commit")
        self.assertEqual(manifest["evaluation"]["top_k"], 10)
        self.assertEqual(manifest["evaluation"]["rerank_k"], 50)
        self.assertEqual(manifest["evaluation"]["relevance_level"], "scene")
        self.assertEqual(manifest["data"]["query_count"], 1)
        self.assertEqual(len(manifest["data"]["query_set_fingerprint"]), 64)
        self.assertEqual(len(manifest["data"]["corpus_fingerprint"]), 64)
        self.assertEqual(
            manifest["data"]["semantic_corpus_fingerprint"],
            "semantic-fingerprint",
        )
        self.assertEqual(
            manifest["evaluation"]["hybrid_weights"],
            {"bm25": 0.45, "dense": 0.45, "metadata": 0.10},
        )
        self.assertEqual(
            manifest["artifacts"]["vector_index"]["vectors"]["size_bytes"], 4
        )
        self.assertEqual(
            manifest["artifacts"]["queries"][0]["sha256"],
            "f2c527db2ab8602e4a44ea2e24db69d9b1848236868c9039a7207850d1166e37",
        )
        self.assertEqual(len(manifest["artifacts"]["segments"]["sha256"]), 64)
        self.assertIn("commit", manifest["software"]["git"])

    def test_single_relevant_at_rank_three(self) -> None:
        results = [
            result(1, "a", "e1", 0),
            result(2, "b", "e2", 0),
            result(3, "c", "e3", 30),
        ]
        judgments = (
            Judgment(
                youtube_video_id="e3",
                start_seconds=55,
                end_seconds=75,
                time_reference="video",
            ),
        )
        metrics = calculate_query_metrics(results, judgments)
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertAlmostEqual(metrics.reciprocal_rank_at_10, 1 / 3)
        self.assertAlmostEqual(metrics.ndcg_at_10, 1 / math.log2(4))

    def test_absent_relevant_scene_scores_zero(self) -> None:
        judgments = (
            Judgment(
                "missing", start_seconds=0, end_seconds=10, time_reference="transcript"
            ),
        )
        metrics = calculate_query_metrics([result(1, "a", "e1", 0)], judgments)
        self.assertEqual(metrics.recall_at_5, 0.0)
        self.assertEqual(metrics.reciprocal_rank_at_10, 0.0)
        self.assertEqual(metrics.ndcg_at_10, 0.0)

    def test_overlapping_duplicate_results_cannot_reuse_one_judgment(self) -> None:
        results = [result(1, "a", "e1", 0), result(2, "b", "e1", 5)]
        judgments = (
            Judgment("e1", start_seconds=25, end_seconds=45, time_reference="video"),
        )
        grades, matched = relevance_grades(results, judgments)
        self.assertEqual(grades, [1, 0])
        self.assertEqual(matched, {0})

    def test_graded_judgments_use_ideal_order(self) -> None:
        results = [result(1, "a", "e1", 0), result(2, "b", "e2", 0)]
        judgments = (
            Judgment("e1", 25, 45, grade=1, time_reference="video"),
            Judgment("e2", 25, 45, grade=2, time_reference="video"),
        )
        metrics = calculate_query_metrics(results, judgments)
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics.ndcg_at_10, expected)

    def test_episode_relevance_matches_video_id_without_scene_overlap(self) -> None:
        judgments = (Judgment("e1"),)
        metrics = calculate_query_metrics(
            [result(1, "far-away-scene", "e1", 5000)],
            judgments,
            relevance_level="episode",
        )
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.reciprocal_rank_at_10, 1.0)
        self.assertEqual(metrics.ndcg_at_10, 1.0)

    def test_episode_metrics_collapse_duplicate_segments_before_ranking(self) -> None:
        results = [
            result(1, "e1-a", "e1", 0),
            result(2, "e1-b", "e1", 100),
            result(3, "e2-a", "e2", 0),
        ]
        metrics = calculate_query_metrics(
            results, (Judgment("e2"),), relevance_level="episode"
        )
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.reciprocal_rank_at_10, 0.5)
        self.assertAlmostEqual(metrics.ndcg_at_10, 1 / math.log2(3))

    def test_episode_metrics_collapse_duplicate_judgments(self) -> None:
        metrics = calculate_query_metrics(
            [result(1, "e1-a", "e1", 0)],
            (Judgment("e1", grade=1), Judgment("e1", grade=2)),
            relevance_level="episode",
        )
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.reciprocal_rank_at_10, 1.0)
        self.assertEqual(metrics.ndcg_at_10, 1.0)


if __name__ == "__main__":
    unittest.main()
