from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..common.device import resolve_torch_device
from ..common.io import write_json
from ..text import normalize_text

logger = logging.getLogger(__name__)

LEMMA_CACHE_VERSION = 1
LEMMA_PROCESSORS = "tokenize,pos,lemma"


def _text_corpus_fingerprint(
    episode_ids: Sequence[str],
    texts: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for episode_id, text in zip(episode_ids, texts):
        digest.update(episode_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalized_lemmas(document: Any, *, fallback: str) -> str:
    lemmas = [
        word.lemma or word.text
        for sentence in document.sentences
        for word in sentence.words
        if word.lemma or word.text
    ]
    value = normalize_text(" ".join(lemmas))
    return value or normalize_text(fallback)


def _load_stanza_pipeline(
    *,
    allow_download: bool,
    device: str,
    model_dir: Path | None,
) -> tuple[Any, str]:
    import stanza

    selected_device = resolve_torch_device(device)
    download_kwargs: dict[str, Any] = {
        "lang": "sr",
        "processors": LEMMA_PROCESSORS,
        "verbose": False,
    }
    pipeline_kwargs: dict[str, Any] = {
        "lang": "sr",
        "processors": LEMMA_PROCESSORS,
        "download_method": None,
        "device": selected_device,
        "verbose": False,
    }
    if model_dir is not None:
        download_kwargs["model_dir"] = str(model_dir)
        pipeline_kwargs["dir"] = str(model_dir)
    if allow_download:
        logger.info("Downloading Stanza tokenization/POS/lemma models")
        stanza.download(**download_kwargs)
    try:
        return stanza.Pipeline(**pipeline_kwargs), selected_device
    except Exception as error:
        raise RuntimeError(
            "Could not load local Stanza models. Rerun with "
            "--allow-lemma-download once."
        ) from error


def _lemma_metadata(cache: dict[str, Any], payload_key: str) -> dict[str, Any]:
    """Cache contents minus the bulky payload and episode-id list."""
    hidden = {payload_key, "episode_ids"}
    return {key: value for key, value in cache.items() if key not in hidden}


def _read_lemma_cache(
    cache_path: Path, expected: dict[str, Any], payload_key: str
) -> tuple[list[Any], dict[str, Any]] | None:
    """Return (payload, metadata) from a matching, non-empty cache, else None."""
    if not cache_path.exists():
        return None
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if any(
        cached.get(key) != value for key, value in expected.items()
    ) or not cached.get(payload_key):
        logger.info("Ignoring incompatible lemma cache at %s", cache_path)
        return None
    return cached[payload_key], _lemma_metadata(cached, payload_key)


def _write_lemma_cache(
    cache_path: Path,
    expected: dict[str, Any],
    payload_key: str,
    payload: list[str],
    device: str,
) -> dict[str, Any]:
    """Persist ``payload`` under the manifest and return its (payload-free) metadata."""
    cache = {**expected, "device": device, payload_key: payload}
    write_json(cache_path, cache)
    logger.info("Wrote %d lemma entries to %s", len(payload), cache_path)
    return _lemma_metadata(cache, payload_key)


def load_or_build_lemma_texts(
    *,
    episode_ids: Sequence[str],
    raw_texts: Sequence[str],
    source_corpus_fingerprint: str,
    cache_path: Path,
    allow_download: bool,
    device: str,
    model_dir: Path | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Lemmatize label text with Stanza and cache by corpus identity."""
    if not episode_ids or len(episode_ids) != len(raw_texts):
        raise ValueError("Episode IDs and raw lemma texts must be non-empty and align")
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Episode IDs for lemmatization must be unique")

    expected = {
        "version": LEMMA_CACHE_VERSION,
        "algorithm": "stanza_pos_aware_lemmatization",
        "processors": LEMMA_PROCESSORS,
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "raw_text_fingerprint": _text_corpus_fingerprint(episode_ids, raw_texts),
        "episode_ids": list(episode_ids),
    }
    cached = _read_lemma_cache(cache_path, expected, "lemma_texts")
    if cached is not None and len(cached[0]) == len(episode_ids):
        logger.info("Loaded %d cached lemma documents", len(cached[0]))
        return tuple(str(text) for text in cached[0]), cached[1]

    pipeline, selected_device = _load_stanza_pipeline(
        allow_download=allow_download, device=device, model_dir=model_dir
    )
    lemma_texts: list[str] = []
    for row, text in enumerate(raw_texts, start=1):
        lemma_texts.append(_normalized_lemmas(pipeline(text), fallback=text))
        if row % 100 == 0 or row == len(raw_texts):
            logger.info("Lemmatized %d/%d episode documents", row, len(raw_texts))

    metadata = _write_lemma_cache(
        cache_path, expected, "lemma_texts", lemma_texts, selected_device
    )
    return tuple(lemma_texts), metadata
