import sys
import types
import unittest
from unittest.mock import patch

from drzavni_posao.search.common.device import resolve_torch_device


class ResolveTorchDeviceTests(unittest.TestCase):
    def test_explicit_preference_wins(self) -> None:
        self.assertEqual(resolve_torch_device("cpu"), "cpu")
        self.assertEqual(resolve_torch_device("mps"), "mps")
        self.assertEqual(resolve_torch_device("cuda"), "cuda")

    def test_auto_and_none_use_detection(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_torch_device("auto"), "mps")
            self.assertEqual(resolve_torch_device(None), "mps")

    def test_prefers_cuda_over_mps(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_torch_device("auto"), "cuda")

    def test_falls_back_to_cpu_without_accelerators(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=None)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_torch_device("auto"), "cpu")

    def test_falls_back_to_cpu_without_torch(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            # ImportError path: torch missing from import
            real_import = __import__

            def _import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("no torch")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_import):
                self.assertEqual(resolve_torch_device("auto"), "cpu")


if __name__ == "__main__":
    unittest.main()
