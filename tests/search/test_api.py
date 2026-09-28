import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from drzavni_posao.search.common.models import SegmentRecord
from drzavni_posao.search.retrieval.models import SearchResult

with patch.dict(os.environ, {"DP_DEVICE": "cpu"}):
    from drzavni_posao.search.api import app as api_app

from fastapi.testclient import TestClient

EXPECTED_RESULT_KEYS = {
    "rank",
    "youtube_video_id",
    "episode_number",
    "episode_title",
    "segment_id",
    "start_seconds",
    "end_seconds",
    "start_time",
    "end_time",
    "transcript_start_time",
    "transcript_end_time",
    "youtube_url",
    "speakers",
    "snippet",
    "score",
    "score_components",
    "matched_characters",
    "matched_terms",
}


def _make_result(rank: int = 1) -> SearchResult:
    segment = SegmentRecord(
        youtube_video_id="abc",
        episode_number=1,
        episode_title="Izbori",
        segment_id="abc-seg-0001",
        start_seconds=12.0,
        end_seconds=34.0,
        video_start_seconds=12.0,
        video_end_seconds=34.0,
        text_original="Boškić i Čvarkov se svađaju oko izbora.",
        text_semantic="Boškić i Čvarkov se svađaju oko izbora.",
        text_normalized="boskic i cvarkov se svadjaju oko izbora",
        speaker_keys=["boskic", "cvarkov"],
        source_segment_indices=[0],
        youtube_url="https://youtube.com/watch?v=abc",
    )
    return SearchResult(
        rank=rank,
        segment=segment,
        score=0.75,
        score_components={"bm25": 0.75},
        matched_characters=["Boškić"],
        matched_terms=["izbora"],
        youtube_url_with_timestamp="https://youtube.com/watch?v=abc&t=12s",
    )


class SearchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_app.app)
        self.engine = Mock()
        self.engine.search.return_value = [_make_result()]
        self.load_engine = Mock(return_value=self.engine)
        patcher = patch.object(api_app, "load_engine", self.load_engine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _patch_vector_path(self, exists: bool) -> None:
        directory = tempfile.mkdtemp()
        vector_path = Path(directory) / "vector_index"
        if exists:
            vector_path.with_suffix(".json").write_text("{}")
        patcher = patch.object(api_app, "VECTOR_PATH", vector_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_bm25_search_returns_result_dicts(self) -> None:
        response = self.client.post(
            "/search", json={"query": "svađa oko izbora", "method": "bm25"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload), {"query", "method", "took_seconds", "results"})
        self.assertEqual(payload["query"], "svađa oko izbora")
        self.assertEqual(payload["method"], "bm25")
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(set(payload["results"][0]), EXPECTED_RESULT_KEYS)
        self.load_engine.assert_called_once_with(False, False)
        self.engine.search.assert_called_once_with(
            "svađa oko izbora",
            method="bm25",
            top_k=5,
            candidate_k=50,
            group_by_episode=True,
        )

    def test_hybrid_rerank_requests_reranker_engine(self) -> None:
        self._patch_vector_path(exists=True)
        response = self.client.post(
            "/search", json={"query": "plata", "method": "hybrid_rerank"}
        )
        self.assertEqual(response.status_code, 200)
        self.load_engine.assert_called_once_with(True, True)

    def test_blank_query_is_rejected(self) -> None:
        for query in ("", "   "):
            response = self.client.post(
                "/search", json={"query": query, "method": "bm25"}
            )
            self.assertEqual(response.status_code, 422)
        self.load_engine.assert_not_called()

    def test_unknown_method_is_rejected(self) -> None:
        response = self.client.post(
            "/search", json={"query": "plata", "method": "cross_encoder"}
        )
        self.assertEqual(response.status_code, 422)

    def test_out_of_range_parameters_are_rejected(self) -> None:
        for overrides in ({"top_k": 0}, {"top_k": 21}, {"candidate_k": 5}):
            response = self.client.post(
                "/search", json={"query": "plata", "method": "bm25", **overrides}
            )
            self.assertEqual(response.status_code, 422)

    def test_dense_without_vector_index_returns_503(self) -> None:
        self._patch_vector_path(exists=False)
        response = self.client.post(
            "/search", json={"query": "plata", "method": "dense"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("vector index", response.json()["detail"])
        self.load_engine.assert_not_called()

    def test_missing_artifacts_return_503(self) -> None:
        self.load_engine.side_effect = FileNotFoundError("artifacts/bm25_index.json")
        response = self.client.post(
            "/search", json={"query": "plata", "method": "bm25"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("bm25_index.json", response.json()["detail"])

    def test_health_reports_bm25_only_without_vectors(self) -> None:
        self._patch_vector_path(exists=False)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["methods"], ["bm25"])
        self.assertIsInstance(payload["episodes"], int)
        self.assertIsInstance(payload["segments"], int)

    def test_health_reports_all_methods_with_vectors(self) -> None:
        self._patch_vector_path(exists=True)
        response = self.client.get("/health")
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(
            payload["methods"],
            ["bm25", "dense", "hybrid", "hybrid_rerank"],
        )
        self.assertIsInstance(payload["episodes"], int)
        self.assertIsInstance(payload["segments"], int)


if __name__ == "__main__":
    unittest.main()
