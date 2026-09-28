"""Clustering engine: fit one topic model and assemble its run result."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import CountVectorizer

from .core import DialogueChunkCorpus, EpisodeDocument, build_topic_result
from .spherical import SphericalKMeans

logger = logging.getLogger(__name__)
Float32Matrix = NDArray[np.float32]
RESULT_VERSION = 1


def fit_topic_labels(
    *,
    documents: tuple[EpisodeDocument, ...],
    topic_texts: Sequence[str],
    embeddings: Float32Matrix,
    n_topics: int,
    seed: int,
    n_init: int,
    min_df: int,
    top_n_words: int,
    stop_words: list[str],
) -> tuple[Any, list[int]]:
    if len(topic_texts) != len(documents):
        raise ValueError("Topic label texts do not match episode documents")
    # Imported lazily so a non-fitting sweep step avoids BERTopic's slow import.
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer

    model = BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=SphericalKMeans(
            n_clusters=n_topics,
            n_init=n_init,
            random_state=seed,
        ),
        vectorizer_model=CountVectorizer(
            ngram_range=(1, 2),
            min_df=min_df,
            stop_words=stop_words,
        ),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(
        list(topic_texts),
        embeddings,
    )
    return model, [int(topic) for topic in topics]


def build_clustering_result(
    *,
    corpus: DialogueChunkCorpus,
    embeddings: Float32Matrix,
    labels: list[int],
    assigned_cosines: NDArray[np.float32],
    margins: NDArray[np.float32],
    topic_model: Any,
    embedding_metadata: dict[str, Any],
    representation_metadata: dict[str, Any],
    source_fingerprint: str,
    settings: dict[str, Any],
    representative_episode_count: int,
) -> dict[str, Any]:
    result = build_topic_result(
        documents=corpus.documents,
        embeddings=embeddings,
        topics=labels,
        topic_model=topic_model,
        corpus_fingerprint=source_fingerprint,
        settings=settings,
        representative_episode_count=representative_episode_count,
    )
    chunk_counts = np.bincount(
        corpus.chunk_episode_rows, minlength=len(corpus.documents)
    )
    result["embedding"] = {
        "aggregation": (
            "utterance_weighted_mean_then_l2_normalize_non_overlapping_dialogue_chunks"
        ),
        "model_name": embedding_metadata["model_name"],
        "model_revision": embedding_metadata["model_revision"],
        "source_dimension": embedding_metadata["dimension"],
        "dimension": embeddings.shape[1],
        "utterances_per_chunk": embedding_metadata["utterances_per_chunk"],
        "speaker_metadata_included": False,
        "episode_title_included": False,
        "clustering_representation": representation_metadata,
    }
    for row, episode in enumerate(result["episodes"]):
        episode["dialogue_chunk_count"] = int(chunk_counts[row])
        episode["assignment_cosine_similarity"] = round(float(assigned_cosines[row]), 6)
        episode["assignment_margin"] = round(float(margins[row]), 6)
    return result
