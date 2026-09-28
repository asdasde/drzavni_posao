from __future__ import annotations

import numpy as np
import pytest

from drzavni_posao.search.topics import pca_transform_episode_embeddings
from drzavni_posao.search.topics.metrics import pool_sweep_results


def _summary(*, u: int, sil: float, stab: float, margin: float, div: float) -> dict:
    return {
        "embedding": {"model_name": "m", "utterances_per_chunk": u},
        "clustering_representation": {"algorithm": "none", "dimension": 8},
        "results": [
            {
                "n_topics": 25,
                "cosine_silhouette": sil,
                "mean_assignment_margin": margin,
                "cluster_size_normalized_entropy": div,
                "topic_diversity": div,
                "stability": {"mean_adjusted_rand_index": stab},
            }
        ],
    }


def test_pool_sweep_results_normalizes_across_runs() -> None:
    # "good" dominates "bad" on every metric; global min-max must rank it first
    # even though each run alone would self-normalize to selection_score 1.0.
    good = _summary(u=16, sil=0.03, stab=0.9, margin=0.02, div=0.95)
    bad = _summary(u=8, sil=0.001, stab=0.1, margin=0.001, div=0.5)

    ranked = pool_sweep_results([("bad", bad), ("good", good)])

    assert [row["summary"] for row in ranked] == ["good", "bad"]
    assert ranked[0]["selection_score"] == 1.0  # max on every metric
    assert ranked[1]["selection_score"] == 0.0  # min on every metric


def test_pca_transform_centers_reduces_and_normalizes() -> None:
    embeddings = np.random.RandomState(7).normal(size=(20, 8)).astype(np.float32)

    transformed, metadata = pca_transform_episode_embeddings(
        embeddings,
        n_components=4,
        whiten=False,
        random_state=42,
    )

    assert transformed.shape == (20, 4)
    np.testing.assert_allclose(
        np.linalg.norm(transformed, axis=1),
        np.ones(20),
        atol=1e-6,
    )
    assert metadata["algorithm"] == "centered_pca_then_l2_normalize"
    assert metadata["source_dimension"] == 8
    assert metadata["dimension"] == 4
    assert 0 < metadata["explained_variance_ratio"] < 1


def test_pca_transform_rejects_excessive_components() -> None:
    with pytest.raises(ValueError, match="below both"):
        pca_transform_episode_embeddings(
            np.eye(4, dtype=np.float32),
            n_components=4,
            whiten=False,
            random_state=42,
        )
