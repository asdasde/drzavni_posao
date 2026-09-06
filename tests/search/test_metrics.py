import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from drzavni_posao.search.evaluation.runner import (
    EvaluationQuery,
    build_argument_parser,
    build_evaluation_manifest,
    calculate_query_metrics,
    evaluate_method,
    load_evaluation_queries,
)
from drzavni_posao.search.retrieval.fusion import HybridWeights
from drzavni_posao.search.retrieval.models import SearchResult

from .test_engine import make_segment


def result(rank: int, segment_id: str, youtube_video_id: str) -> SearchResult:
    return SearchResult(
        rank=rank,
        segment=make_segment(segment_id, youtube_video_id, "text"),
        score=1 / rank,
        score_components={},
        matched_characters=[],
        matched_terms=[],
    )


class EvaluationTests(unittest.TestCase):
    def test_evaluate_cli_defaults_to_bm25(self) -> None:
        args = build_argument_parser().parse_args(["--queries", "queries.jsonl"])
        self.assertEqual(args.methods, "bm25")
        self.assertIsNone(args.vectors)
        self.assertEqual(args.queries, Path("queries.jsonl"))
        self.assertFalse(hasattr(args, "relevance_level"))
        self.assertFalse(hasattr(args, "additional_queries"))

    def test_tiktok_query_loader_keeps_lineage(self) -> None:
        row = {
            "query_id": "tt_1",
            "query": "ona epizoda sa jubilejom",
            "relevant_youtube_video_id": "e1",
            "tiktok_url": "https://www.tiktok.com/@viewer/video/1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            queries = load_evaluation_queries(path)

        self.assertEqual(
            queries,
            [
                EvaluationQuery(
                    query_id="tt_1",
                    query="ona epizoda sa jubilejom",
                    relevant_youtube_video_id="e1",
                    tiktok_url="https://www.tiktok.com/@viewer/video/1",
                )
            ],
        )

    def test_tiktok_query_loader_requires_lineage_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "tt_1",
                        "query": "query",
                        "relevant_youtube_video_id": "e1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "public TikTok URL"):
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
                        "q1",
                        "episode query",
                        "e1",
                        "https://www.tiktok.com/@viewer/video/1",
                    )
                ],
                query_file=query_path,
                segments_path=segments_path,
                bm25_path=bm25_path,
                vector_path=vector_path,
                candidate_k=50,
                bootstrap_samples=1000,
                warmup=True,
                weights=HybridWeights(),
            )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["models"]["vector"]["revision"], "dense-commit")
        self.assertEqual(manifest["models"]["reranker"]["revision"], "reranker-commit")
        self.assertEqual(manifest["evaluation"]["ranking_unit"], "episode")
        self.assertEqual(manifest["evaluation"]["top_k"], 10)
        self.assertEqual(manifest["evaluation"]["rerank_k"], 50)
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
            manifest["artifacts"]["queries"]["sha256"],
            "f2c527db2ab8602e4a44ea2e24db69d9b1848236868c9039a7207850d1166e37",
        )
        self.assertEqual(len(manifest["artifacts"]["segments"]["sha256"]), 64)
        self.assertIn("commit", manifest["software"]["git"])

    def test_single_relevant_episode_at_rank_three(self) -> None:
        metrics = calculate_query_metrics(
            [
                result(1, "a", "e1"),
                result(2, "b", "e2"),
                result(3, "c", "e3"),
            ],
            "e3",
        )
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertAlmostEqual(metrics.reciprocal_rank_at_10, 1 / 3)
        self.assertAlmostEqual(metrics.ndcg_at_10, 1 / math.log2(4))

    def test_absent_relevant_episode_scores_zero(self) -> None:
        metrics = calculate_query_metrics([result(1, "a", "e1")], "missing")
        self.assertEqual(metrics.recall_at_5, 0.0)
        self.assertEqual(metrics.reciprocal_rank_at_10, 0.0)
        self.assertEqual(metrics.ndcg_at_10, 0.0)

    def test_duplicate_segments_do_not_consume_episode_ranks(self) -> None:
        metrics = calculate_query_metrics(
            [
                result(1, "e1-a", "e1"),
                result(2, "e1-b", "e1"),
                result(3, "e2-a", "e2"),
            ],
            "e2",
        )
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.reciprocal_rank_at_10, 0.5)
        self.assertAlmostEqual(metrics.ndcg_at_10, 1 / math.log2(3))

    def test_evaluation_checkpoint_resumes_completed_queries(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.segments = [make_segment("s1", "e1", "text")]
                self.reranker = None
                self.calls = 0

            def search(self, *_args, **_kwargs):
                self.calls += 1
                return [result(1, "s1", "e1")]

        queries = [
            EvaluationQuery(
                f"q{index}",
                f"query {index}",
                "e1",
                f"https://www.tiktok.com/@viewer/video/{index}",
            )
            for index in (1, 2)
        ]
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.jsonl"
            first = evaluate_method(engine, queries, "bm25", checkpoint_path=checkpoint)
            second = evaluate_method(
                engine, queries, "bm25", checkpoint_path=checkpoint
            )

            rows = [
                json.loads(line)
                for line in checkpoint.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0]["tiktok_url"],
                "https://www.tiktok.com/@viewer/video/1",
            )
        self.assertEqual(engine.calls, 2)
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
