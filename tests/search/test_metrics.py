import math
import tempfile
import unittest
from pathlib import Path

from drzavni_posao.search.evaluation.runner import (
    EvaluationQuery,
    Judgment,
    build_argument_parser,
    build_evaluation_provenance,
    calculate_query_metrics,
    evaluation_query_fingerprint,
    relevance_grades,
)
from drzavni_posao.search.retrieval.models import SearchResult

from .test_engine import make_segment


def result(rank: int, segment_id: str, episode_id: str, start: float) -> SearchResult:
    return SearchResult(
        rank=rank,
        segment=make_segment(segment_id, episode_id, "text", start=start),
        score=1 / rank,
        score_components={},
        matched_characters=[],
        matched_terms=[],
        reason="test",
    )


class EvaluationTests(unittest.TestCase):
    def test_evaluate_cli_defaults_to_bm25(self) -> None:
        args = build_argument_parser().parse_args([])
        self.assertEqual(args.methods, "bm25")
        self.assertIsNone(args.vectors)

    def test_evaluation_provenance_binds_queries_and_files(self) -> None:
        query = EvaluationQuery(
            query_id="q1",
            query="scene query",
            judgments=(Judgment("e1", 25, 45),),
            split="test",
        )
        changed = EvaluationQuery(
            query_id="q1",
            query="changed scene query",
            judgments=query.judgments,
            split="test",
        )
        self.assertNotEqual(
            evaluation_query_fingerprint([query]),
            evaluation_query_fingerprint([changed]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            query_path = Path(temporary) / "queries.jsonl"
            query_path.write_text('{"query_id":"q1"}\n', encoding="utf-8")
            provenance = build_evaluation_provenance(
                query_files=[query_path],
                queries=[query],
                segments=[make_segment("s1", "e1", "text")],
            )
        self.assertEqual(provenance["status"], "current")
        self.assertTrue(provenance["usable_for_current_benchmark"])
        self.assertEqual(provenance["query_count"], 1)
        self.assertEqual(
            provenance["query_file_fingerprints"],
            [
                {
                    "path": str(query_path),
                    "sha256": (
                        "f2c527db2ab8602e4a44ea2e24db69d9b1848236868c9039a7207850d1166e37"
                    ),
                }
            ],
        )

    def test_single_relevant_at_rank_three(self) -> None:
        results = [
            result(1, "a", "e1", 0),
            result(2, "b", "e2", 0),
            result(3, "c", "e3", 30),
        ]
        judgments = (
            Judgment(
                episode_id="e3",
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


if __name__ == "__main__":
    unittest.main()
