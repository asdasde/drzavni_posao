import sys
import types
import unittest
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
                return_value="/models/reranker",
            ),
        ):
            CrossEncoderReranker(local_files_only=True)

        cross_encoder.assert_called_once_with(
            "/models/reranker",
            device="cpu",
            local_files_only=True,
        )


if __name__ == "__main__":
    unittest.main()
