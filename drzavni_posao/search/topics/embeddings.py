"""Per-episode embeddings: build from dialogue chunks and cache by identity."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..common.io import read_float32_matrix, write_float32_matrix, write_json
from ..indexing.vector import build_text_encoder
from .core import DialogueChunkCorpus, mean_pool_dialogue_chunk_embeddings

logger = logging.getLogger(__name__)
Float32Matrix = NDArray[np.float32]
EMBEDDING_CACHE_VERSION = 1
EPISODE_EMBEDDING_CACHE_DIR = Path("artifacts/episode_embeddings")


def _embedding_cache_path(
    *,
    corpus_fingerprint: str,
    model_name: str,
    model_revision: str | None,
    utterances_per_chunk: int,
) -> Path:
    identity = "\0".join(
        [corpus_fingerprint, model_name, model_revision or "", str(utterances_per_chunk)]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    slug = model_name.rsplit("/", 1)[-1].replace(" ", "-")
    name = f"{slug}_u{utterances_per_chunk:03d}_{digest}.f32"
    return EPISODE_EMBEDDING_CACHE_DIR / name


def _read_cached_embeddings(
    cache_path: Path, expected: dict[str, Any], *, rows: int
) -> tuple[Float32Matrix, dict[str, Any]] | None:
    """Return cached embeddings + manifest when they match ``expected``, else None."""
    metadata_path = cache_path.with_suffix(".json")
    if not (cache_path.exists() and metadata_path.exists()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if any(metadata.get(key) != value for key, value in expected.items()):
        logger.info("Ignoring incompatible embedding cache at %s", cache_path)
        return None
    values = read_float32_matrix(
        cache_path,
        rows=rows,
        cols=int(metadata["dimension"]),
        byte_order=metadata.get("byte_order", sys.byteorder),
    )
    logger.info("Loaded %d cached episode embeddings", len(values))
    return values, metadata


def load_or_build_embeddings(
    *,
    corpus: DialogueChunkCorpus,
    source_fingerprint: str,
    model_name: str,
    model_revision: str | None,
    device: str,
    batch_size: int,
    allow_download: bool,
    utterances_per_chunk: int,
) -> tuple[Float32Matrix, dict[str, Any]]:
    cache_path = _embedding_cache_path(
        corpus_fingerprint=source_fingerprint,
        model_name=model_name,
        model_revision=model_revision,
        utterances_per_chunk=utterances_per_chunk,
    )
    rows = len(corpus.documents)
    expected: dict[str, Any] = {
        "version": EMBEDDING_CACHE_VERSION,
        "source_corpus_fingerprint": source_fingerprint,
        "model_name": model_name,
        "model_requested_revision": model_revision,
        "utterances_per_chunk": utterances_per_chunk,
        "episode_ids": [document.youtube_video_id for document in corpus.documents],
        "chunk_count": len(corpus.chunk_texts),
    }

    cached = _read_cached_embeddings(cache_path, expected, rows=rows)
    if cached is not None:
        return cached

    encoder = build_text_encoder(
        model_name,
        device=device,
        batch_size=batch_size,
        allow_download=allow_download,
        model_revision=model_revision,
    )
    logger.info(
        "Encoding %d non-overlapping dialogue chunks for %d episodes",
        len(corpus.chunk_texts),
        rows,
    )
    chunk_embeddings = encoder.encode_passages(corpus.chunk_texts)
    values = mean_pool_dialogue_chunk_embeddings(corpus, chunk_embeddings)
    manifest = {
        **expected,
        "algorithm": "utterance_weighted_mean_of_dialogue_chunk_embeddings",
        "model_revision": encoder.revision,
        "dimension": encoder.dimension,
        "episode_count": rows,
        "byte_order": sys.byteorder,
    }
    write_float32_matrix(cache_path, values)
    write_json(cache_path.with_suffix(".json"), manifest)
    logger.info("Wrote episode embedding cache to %s", cache_path)
    return values, manifest
