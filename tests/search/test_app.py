import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class StreamlitAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fake_streamlit = types.ModuleType("streamlit")
        fake_streamlit.cache_resource = lambda **_kwargs: lambda function: function
        app_path = (
            Path(__file__).resolve().parents[2]
            / "drzavni_posao"
            / "search"
            / "app"
            / "streamlit.py"
        )
        with (
            patch.dict(os.environ, {"DP_DEVICE": "cpu"}),
            patch.dict(sys.modules, {"streamlit": fake_streamlit}),
        ):
            cls.namespace = runpy.run_path(
                str(app_path), run_name="streamlit_file_execution_test"
            )

    def test_documented_file_path_imports_search_package(self) -> None:
        namespace = self.namespace
        methods = namespace["_available_methods"](vector_available=True)
        self.assertEqual(
            methods,
            ["BM25", "Dense", "Hybrid", "Hybrid + reranker"],
        )
        self.assertEqual(
            namespace["_default_method_index"](methods),
            2,
        )
        self.assertEqual(
            namespace["_method_for_label"]("Hybrid + reranker"),
            "hybrid_rerank",
        )

    def test_reranker_is_only_loaded_when_requested(self) -> None:
        namespace = self.namespace
        reranker_factory = Mock(return_value=object())
        engine_factory = Mock(return_value=object())
        fake_engine = types.SimpleNamespace(from_paths=engine_factory)
        load_engine = namespace["load_engine"]
        with patch.dict(
            load_engine.__globals__,
            {
                "CrossEncoderReranker": reranker_factory,
                "SearchEngine": fake_engine,
            },
        ):
            load_engine(use_vectors=True, use_reranker=False)
            reranker_factory.assert_not_called()
            self.assertIsNone(engine_factory.call_args.kwargs["reranker"])

            load_engine(use_vectors=True, use_reranker=True)
            reranker_factory.assert_called_once()
            self.assertIs(
                engine_factory.call_args.kwargs["reranker"],
                reranker_factory.return_value,
            )


if __name__ == "__main__":
    unittest.main()
