import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from drzavni_posao.search.retrieval.rerank import CrossEncoderReranker


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_passes_local_only_directly_to_cross_encoder(self) -> None:
        cross_encoder = Mock(return_value=object())
        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        fake_sentence_transformers.CrossEncoder = cross_encoder

        with (
            patch.dict(
                sys.modules,
                {"sentence_transformers": fake_sentence_transformers},
            ),
            patch(
                "drzavni_posao.search.retrieval.rerank.resolve_model_path",
                return_value=Path("/cache/models--reranker/snapshots/abc123"),
            ) as resolver,
        ):
            reranker = CrossEncoderReranker(
                local_files_only=True,
                device="cpu",
                model_revision="release-tag",
            )

        cross_encoder.assert_called_once_with(
            "/cache/models--reranker/snapshots/abc123",
            device="cpu",
            local_files_only=True,
        )
        resolver.assert_called_once_with(
            reranker.name,
            local_files_only=True,
            revision="release-tag",
        )
        self.assertEqual(reranker.revision, "abc123")
        self.assertEqual(reranker.requested_revision, "release-tag")


if __name__ == "__main__":
    unittest.main()
