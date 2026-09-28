"""Evaluation metrics for dialogue-episode topic clusterings."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..indexing.vector import normalize_vector
from .spherical import SphericalKMeans

Float32Matrix = NDArray[np.float32]


def cluster_diagnostics(
    embeddings: Float32Matrix,
    labels: list[int],
) -> tuple[dict[str, Any], NDArray[np.float32], NDArray[np.float32]]:
    assignments = np.asarray(labels, dtype=np.int64)
    topic_ids = sorted(set(labels))
    centers = np.stack(
        [
            normalize_vector(embeddings[assignments == topic_id].mean(axis=0))
            for topic_id in topic_ids
        ]
    )
    topic_column = {topic_id: column for column, topic_id in enumerate(topic_ids)}
    assigned_columns = np.asarray([topic_column[value] for value in labels])
    similarities = embeddings @ centers.T
    assigned = similarities[np.arange(len(embeddings)), assigned_columns]
    competing = similarities.copy()
    competing[np.arange(len(embeddings)), assigned_columns] = -np.inf
    margins = assigned - np.max(competing, axis=1)
    sizes = np.bincount(assigned_columns, minlength=len(topic_ids))
    probabilities = sizes / sizes.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / np.log(len(topic_ids))
    centroid_similarities = centers @ centers.T
    off_diagonal = centroid_similarities[~np.eye(len(topic_ids), dtype=bool)]

    metrics = {
        "cosine_silhouette": round(
            float(silhouette_score(embeddings, assignments, metric="cosine")),
            6,
        ),
        "mean_assigned_cosine": round(float(assigned.mean()), 6),
        "median_assigned_cosine": round(float(np.median(assigned)), 6),
        "p10_assigned_cosine": round(float(np.quantile(assigned, 0.1)), 6),
        "mean_assignment_margin": round(float(margins.mean()), 6),
        "median_assignment_margin": round(float(np.median(margins)), 6),
        "p10_assignment_margin": round(float(np.quantile(margins, 0.1)), 6),
        "mean_centroid_cosine": round(float(off_diagonal.mean()), 6),
        "max_centroid_cosine": round(float(off_diagonal.max()), 6),
        "cluster_size": {
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
        },
        "cluster_size_normalized_entropy": round(normalized_entropy, 6),
    }
    return metrics, assigned.astype(np.float32), margins.astype(np.float32)


def topic_diversity(model: Any, labels: list[int], top_n_words: int) -> float:
    terms = [
        term
        for topic_id in sorted(set(labels))
        for term, _ in (model.get_topic(topic_id) or [])[:top_n_words]
    ]
    return len(set(terms)) / len(terms) if terms else 0.0


def stability(
    *,
    embeddings: Float32Matrix,
    canonical_labels: list[int],
    n_topics: int,
    seeds: list[int],
    n_init: int,
) -> dict[str, Any]:
    labels_by_seed: dict[int, NDArray[np.int64]] = {
        seeds[0]: np.asarray(canonical_labels, dtype=np.int64)
    }
    for seed in seeds[1:]:
        labels_by_seed[seed] = SphericalKMeans(
            n_clusters=n_topics,
            n_init=n_init,
            random_state=seed,
        ).fit_predict(embeddings)

    pairwise = [
        {
            "seed_a": seed_a,
            "seed_b": seed_b,
            "adjusted_rand_index": round(
                float(
                    adjusted_rand_score(
                        labels_by_seed[seed_a],
                        labels_by_seed[seed_b],
                    )
                ),
                6,
            ),
        }
        for seed_a, seed_b in combinations(seeds, 2)
    ]
    scores = [row["adjusted_rand_index"] for row in pairwise]
    return {
        "seeds": seeds,
        "pairwise": pairwise,
        "mean_adjusted_rand_index": round(float(np.mean(scores)), 6),
        "min_adjusted_rand_index": round(float(np.min(scores)), 6),
    }


SELECTION_SCORE_WEIGHTS = {
    "cosine_silhouette": 0.25,
    "stability.mean_adjusted_rand_index": 0.35,
    "mean_assignment_margin": 0.15,
    "cluster_size_normalized_entropy": 0.10,
    "topic_diversity": 0.15,
}


def add_selection_scores(results: list[dict[str, Any]]) -> None:
    def value(result: dict[str, Any], path: str) -> float:
        current: Any = result
        for key in path.split("."):
            current = current[key]
        return float(current)

    normalized: dict[str, list[float]] = {}
    for field in SELECTION_SCORE_WEIGHTS:
        values = np.asarray([value(result, field) for result in results])
        span = float(values.max() - values.min())
        normalized[field] = (
            [1.0] * len(results)
            if span == 0
            else ((values - values.min()) / span).tolist()
        )
    for row, result in enumerate(results):
        result["selection_score"] = round(
            sum(
                weight * normalized[field][row]
                for field, weight in SELECTION_SCORE_WEIGHTS.items()
            ),
            6,
        )


def pool_sweep_results(
    summaries: Sequence[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten every run's per-K rows, tag each with its config, and score them
    under one global normalization so whole configs are comparable, not just Ks
    within a run. Returns rows sorted by selection score, best first."""
    rows: list[dict[str, Any]] = []
    for source, summary in summaries:
        embedding = summary.get("embedding", {})
        reduction = summary.get("clustering_representation", {})
        config = {
            "summary": source,
            "model_name": embedding.get("model_name"),
            "utterances_per_chunk": embedding.get("utterances_per_chunk"),
            "reduction": reduction.get("algorithm"),
            "reduction_dimension": reduction.get("dimension"),
        }
        rows.extend({**config, **result} for result in summary["results"])
    if not rows:
        raise ValueError("No sweep results to pool")
    add_selection_scores(rows)
    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    return rows
