"""Public text-encoder adapters used by dense indexing and retrieval."""

from .vector import (
    DEFAULT_MODEL,
    HashingTextEncoder,
    SentenceTransformerTextEncoder,
    TextEncoder,
    model_revision_from_path,
    normalize_vector,
    resolve_model_path,
)

__all__ = [
    "DEFAULT_MODEL",
    "HashingTextEncoder",
    "SentenceTransformerTextEncoder",
    "TextEncoder",
    "model_revision_from_path",
    "normalize_vector",
    "resolve_model_path",
]
