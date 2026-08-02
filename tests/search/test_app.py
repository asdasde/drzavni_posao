import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class StreamlitAppTests(unittest.TestCase):
    def test_documented_file_path_imports_search_package(self) -> None:
        fake_streamlit = types.ModuleType("streamlit")
        fake_streamlit.cache_resource = lambda **_kwargs: lambda function: function
        app_path = (
            Path(__file__).resolve().parents[2]
            / "drzavni_posao"
            / "search"
            / "app"
            / "streamlit.py"
        )
        with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            namespace = runpy.run_path(
                str(app_path), run_name="streamlit_file_execution_test"
            )
        self.assertEqual(
            namespace["_default_method_index"](["BM25", "Dense", "Hybrid"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
