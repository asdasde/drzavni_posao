from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from ...transcription.episode import episode_order_key
from ..common.models import SegmentRecord
from ..indexing.vector import normalize_vector
from ..text import normalize_text

logger = logging.getLogger(__name__)

TOPIC_RESULT_VERSION = 1
_RESOURCES = Path(__file__).parent / "resources"
# Show-specific and common-conversation stop words; the SSW dictionary is a
# separate CC BY-NC-SA resource kept verbatim (see SERBIAN_STOPWORDS_NOTICE.md).
DP_STOP_WORDS_PATH = _RESOURCES / "drzavni_posao_stopwords.csv"
SSW_STOP_WORDS_PATH = _RESOURCES / "serbian_stopwords_ssw.csv"
_SPEAKER_PREFIX = re.compile(r"^\[[^\]\n]+\]\s*")

Float32Matrix = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EpisodeDocument:
    youtube_video_id: str
    episode_number: int | None
    episode_title: str
    youtube_url: str | None
    segment_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class DialogueChunkCorpus:
    documents: tuple[EpisodeDocument, ...]
    raw_episode_texts: tuple[str, ...]
    chunk_texts: tuple[str, ...]
    chunk_episode_rows: tuple[int, ...]
    chunk_utterance_counts: tuple[int, ...]


def _episode_dialogue(segments: Sequence[SegmentRecord]) -> list[str]:
    """Ordered, speaker-stripped dialogue lines for one episode.

    Assumes a non-overlapping corpus (build it with ``--overlap-utterances 0``):
    a repeated source index means the segments overlap and would double-count.
    """
    seen: set[int] = set()
    dialogue: list[str] = []
    for segment in sorted(
        segments, key=lambda item: (item.start_seconds, item.segment_id)
    ):
        lines = [
            line.strip() for line in segment.text_original.splitlines() if line.strip()
        ]
        indices = segment.source_segment_indices
        if not indices or len(indices) != len(lines):
            raise ValueError(
                f"{segment.segment_id} needs one source index per transcript line"
            )
        if not seen.isdisjoint(indices):
            raise ValueError(
                f"Episode {segment.youtube_video_id} has overlapping segments; "
                "rebuild the corpus with --overlap-utterances 0"
            )
        seen.update(indices)
        dialogue.extend(
            value for line in lines if (value := _SPEAKER_PREFIX.sub("", line).strip())
        )
    return dialogue


def _episode_documents_with_dialogue(
    segments: Sequence[SegmentRecord],
) -> list[tuple[EpisodeDocument, list[str]]]:
    """Build each episode's document paired with its ordered dialogue lines."""
    if not segments:
        raise ValueError("Cannot build episode documents from an empty corpus")

    grouped: dict[str, list[SegmentRecord]] = defaultdict(list)
    for segment in segments:
        grouped[segment.youtube_video_id].append(segment)

    pairs: list[tuple[EpisodeDocument, list[str]]] = []
    for video_id, episode_segments in grouped.items():
        first = episode_segments[0]
        for segment in episode_segments[1:]:
            if (
                segment.episode_number != first.episode_number
                or segment.episode_title != first.episode_title
                or segment.youtube_url != first.youtube_url
            ):
                raise ValueError(f"Inconsistent metadata for episode {video_id}")
        ordered = sorted(
            episode_segments, key=lambda item: (item.start_seconds, item.segment_id)
        )
        dialogue = _episode_dialogue(ordered)
        if not dialogue:
            raise ValueError(f"Episode {video_id} has no dialogue")
        document = EpisodeDocument(
            youtube_video_id=video_id,
            episode_number=first.episode_number,
            episode_title=first.episode_title,
            youtube_url=first.youtube_url,
            segment_ids=tuple(segment.segment_id for segment in ordered),
            text=" ".join(
                value for line in dialogue if (value := normalize_text(line))
            ),
        )
        pairs.append((document, dialogue))
    return sorted(
        pairs,
        key=lambda pair: episode_order_key(
            pair[0].episode_number, pair[0].youtube_video_id
        ),
    )


def build_episode_documents(
    segments: Sequence[SegmentRecord],
) -> list[EpisodeDocument]:
    return [document for document, _ in _episode_documents_with_dialogue(segments)]


def build_dialogue_chunk_corpus(
    segments: Sequence[SegmentRecord],
    *,
    utterances_per_chunk: int = 16,
) -> DialogueChunkCorpus:
    """Build dialogue-only, non-overlapping chunks for episode embeddings."""
    if utterances_per_chunk < 1:
        raise ValueError("utterances_per_chunk must be positive")

    documents: list[EpisodeDocument] = []
    raw_episode_texts: list[str] = []
    chunk_texts: list[str] = []
    chunk_episode_rows: list[int] = []
    chunk_utterance_counts: list[int] = []
    for episode_row, (document, dialogue) in enumerate(
        _episode_documents_with_dialogue(segments)
    ):
        documents.append(document)
        raw_episode_texts.append(" ".join(dialogue))
        for start in range(0, len(dialogue), utterances_per_chunk):
            chunk = dialogue[start : start + utterances_per_chunk]
            chunk_texts.append(" ".join(chunk))
            chunk_episode_rows.append(episode_row)
            chunk_utterance_counts.append(len(chunk))

    return DialogueChunkCorpus(
        documents=tuple(documents),
        raw_episode_texts=tuple(raw_episode_texts),
        chunk_texts=tuple(chunk_texts),
        chunk_episode_rows=tuple(chunk_episode_rows),
        chunk_utterance_counts=tuple(chunk_utterance_counts),
    )


def mean_pool_dialogue_chunk_embeddings(
    corpus: DialogueChunkCorpus,
    chunk_embeddings: Float32Matrix,
) -> Float32Matrix:
    """Utterance-weight chunk vectors, then L2-normalize each episode vector."""
    values = np.asarray(chunk_embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(corpus.chunk_texts):
        raise ValueError("Chunk vector matrix does not match dialogue chunks")
    if not np.isfinite(values).all():
        raise ValueError("Chunk vector matrix contains non-finite values")

    episode_vectors = np.zeros(
        (len(corpus.documents), values.shape[1]), dtype=np.float32
    )
    episode_weights = np.zeros(len(corpus.documents), dtype=np.float32)
    for chunk_row, (episode_row, weight) in enumerate(
        zip(corpus.chunk_episode_rows, corpus.chunk_utterance_counts)
    ):
        episode_vectors[episode_row] += values[chunk_row] * weight
        episode_weights[episode_row] += weight
    if np.any(episode_weights == 0):
        raise ValueError("At least one episode has no chunk weight")

    episode_vectors /= episode_weights[:, None]
    return np.stack([normalize_vector(row) for row in episode_vectors])


def pca_transform_episode_embeddings(
    embeddings: Float32Matrix,
    *,
    n_components: int | None,
    whiten: bool,
    random_state: int,
) -> tuple[Float32Matrix, dict[str, Any]]:
    """Mean-center, optionally reduce with PCA, then normalize episode vectors."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Expected a non-empty episode embedding matrix")
    source_dimension = values.shape[1]
    if n_components is None:
        return values, {
            "algorithm": "none",
            "source_dimension": source_dimension,
            "dimension": source_dimension,
        }
    if n_components >= min(values.shape):
        raise ValueError(
            "PCA components must be below both episode count and source dimension"
        )

    reducer = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="randomized",
        random_state=random_state,
    )
    transformed = np.asarray(reducer.fit_transform(values), dtype=np.float32)
    normalized = np.stack([normalize_vector(row) for row in transformed])
    metadata = {
        "algorithm": "centered_pca_then_l2_normalize",
        "source_dimension": source_dimension,
        "dimension": n_components,
        "whiten": whiten,
        "explained_variance_ratio": round(
            float(reducer.explained_variance_ratio_.sum()), 6
        ),
        "random_state": random_state,
    }
    logger.info(
        "Reduced episode embeddings %d→%d dimensions with PCA "
        "(whiten=%s, explained variance=%.1f%%)",
        source_dimension,
        n_components,
        whiten,
        100 * metadata["explained_variance_ratio"],
    )
    return normalized, metadata


def _csv_stop_words(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not row[0].strip():
                raise ValueError(f"Invalid stop-word row {row_number} in {path}")
            words.update(normalize_text(row[0]).split())
    return words


def load_stop_words(path: Path | None = None) -> list[str]:
    words = _csv_stop_words(DP_STOP_WORDS_PATH) | _csv_stop_words(SSW_STOP_WORDS_PATH)
    if path is not None:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.split("#", 1)[0].strip()
            if value:
                words.update(normalize_text(value).split())
    return sorted(words)


def build_topic_result(
    *,
    documents: Sequence[EpisodeDocument],
    embeddings: Float32Matrix,
    topics: Sequence[int],
    topic_model: Any,
    corpus_fingerprint: str,
    settings: dict[str, Any],
    representative_episode_count: int,
) -> dict[str, Any]:
    if len(topics) != len(documents):
        raise ValueError("Topic assignments do not match episode documents")

    members_by_topic: dict[int, list[int]] = defaultdict(list)
    for row, topic_id in enumerate(topics):
        members_by_topic[int(topic_id)].append(row)

    topic_rows: list[dict[str, Any]] = []
    labels: dict[int, str] = {}
    for topic_id in sorted(members_by_topic):
        raw_terms = topic_model.get_topic(topic_id) or []
        terms = [
            {"term": str(term), "score": round(float(score), 8)}
            for term, score in raw_terms
            if str(term).strip()
        ]
        label = (
            "outlier / noise"
            if topic_id == -1
            else " · ".join(item["term"] for item in terms[:3])
        )
        labels[topic_id] = label

        member_rows = members_by_topic[topic_id]
        centroid = normalize_vector(embeddings[member_rows].mean(axis=0))
        similarities = embeddings[member_rows] @ centroid
        top_indices = np.argsort(-similarities)[:representative_episode_count]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "label": label,
                "episode_count": len(member_rows),
                "terms": terms,
                "representative_episodes": [
                    {
                        "youtube_video_id": documents[member_rows[index]].youtube_video_id,
                        "episode_number": documents[member_rows[index]].episode_number,
                        "episode_title": documents[member_rows[index]].episode_title,
                        "similarity": round(float(similarities[index]), 6),
                    }
                    for index in top_indices
                ],
            }
        )

    episode_rows = []
    for document, topic_id in zip(documents, topics):
        episode_rows.append(
            {
                "youtube_video_id": document.youtube_video_id,
                "episode_number": document.episode_number,
                "episode_title": document.episode_title,
                "youtube_url": document.youtube_url,
                "topic_id": int(topic_id),
                "topic_label": labels[int(topic_id)],
                "scene_count": len(document.segment_ids),
            }
        )

    return {
        "version": TOPIC_RESULT_VERSION,
        "algorithm": "bertopic_episode_clustering",
        "corpus_fingerprint": corpus_fingerprint,
        "episode_count": len(documents),
        "topic_count": sum(topic_id >= 0 for topic_id in members_by_topic),
        "outlier_episode_count": len(members_by_topic.get(-1, [])),
        "settings": settings,
        "topics": topic_rows,
        "episodes": episode_rows,
    }
