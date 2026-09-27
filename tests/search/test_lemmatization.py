from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from drzavni_posao.search.topics.lemmatization import (
    LEMMA_CACHE_VERSION,
    LEMMA_PROCESSORS,
    _normalized_lemmas,
    _text_corpus_fingerprint,
    load_or_build_lemma_texts,
)


def test_normalized_lemmas_collapse_inflections() -> None:
    document = SimpleNamespace(
        sentences=[
            SimpleNamespace(
                words=[
                    SimpleNamespace(text="Životinje", lemma="životinja"),
                    SimpleNamespace(text="životinja", lemma="životinja"),
                ]
            )
        ]
    )

    assert (
        _normalized_lemmas(document, fallback="Životinje životinja")
        == "zivotinja zivotinja"
    )


def test_valid_lemma_cache_avoids_model_loading(
    tmp_path: Path,
) -> None:
    episode_ids = ["video-a"]
    raw_texts = ["Životinje trče."]
    source_fingerprint = "source"
    cache_path = tmp_path / "lemmas.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": LEMMA_CACHE_VERSION,
                "algorithm": "stanza_pos_aware_lemmatization",
                "processors": LEMMA_PROCESSORS,
                "source_corpus_fingerprint": source_fingerprint,
                "raw_text_fingerprint": _text_corpus_fingerprint(
                    episode_ids, raw_texts
                ),
                "episode_ids": episode_ids,
                "device": "cpu",
                "lemma_texts": ["zivotinja trcati"],
            }
        ),
        encoding="utf-8",
    )

    texts, metadata = load_or_build_lemma_texts(
        episode_ids=episode_ids,
        raw_texts=raw_texts,
        source_corpus_fingerprint=source_fingerprint,
        cache_path=cache_path,
        allow_download=False,
        device="cpu",
    )

    assert texts == ("zivotinja trcati",)
    assert metadata["algorithm"] == "stanza_pos_aware_lemmatization"
    assert "lemma_texts" not in metadata
