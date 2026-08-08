from __future__ import annotations


def resolve_torch_device(preferred: str | None = None) -> str:
    """Pick a torch device: explicit preference, else cuda → mps → cpu."""
    if preferred is not None and preferred not in {"", "auto"}:
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
