from __future__ import annotations

import numpy as np
import pytest

from drzavni_posao.search.topics.spherical import SphericalKMeans


def test_spherical_kmeans_finds_cosine_clusters() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, -0.1],
            [-1.0, 0.0],
            [-0.9, 0.1],
            [-0.8, -0.1],
        ],
        dtype=np.float32,
    )

    model = SphericalKMeans(n_clusters=2, random_state=7, n_init=3)
    labels = model.fit_predict(vectors)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    np.testing.assert_allclose(
        np.linalg.norm(model.cluster_centers_, axis=1),
        np.ones(2),
        atol=1e-6,
    )
    assert model.inertia_ >= 0
    assert model.n_iter_ >= 1


def test_spherical_kmeans_is_scale_invariant_and_predicts() -> None:
    vectors = np.array(
        [[10.0, 0.0], [2.0, 0.1], [0.0, 4.0], [0.1, 3.0]],
        dtype=np.float32,
    )
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    scaled_model = SphericalKMeans(n_clusters=2, random_state=4, n_init=2).fit(vectors)
    normalized_model = SphericalKMeans(n_clusters=2, random_state=4, n_init=2).fit(
        normalized
    )

    assert np.array_equal(scaled_model.labels_, normalized_model.labels_)
    predictions = scaled_model.predict(
        np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    )
    assert predictions[0] != predictions[1]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.empty((0, 2), dtype=np.float32), "empty"),
        (np.array([[0.0, 0.0]], dtype=np.float32), "zero"),
        (np.array([1.0, 2.0], dtype=np.float32), "2-dimensional"),
    ],
)
def test_spherical_kmeans_rejects_invalid_vectors(
    vectors: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SphericalKMeans(n_clusters=1).fit(vectors)


def test_spherical_kmeans_rejects_too_many_clusters() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        SphericalKMeans(n_clusters=3).fit(np.eye(2, dtype=np.float32))


def test_spherical_kmeans_requires_fit_before_predict() -> None:
    with pytest.raises(ValueError, match="not been fitted"):
        SphericalKMeans(n_clusters=2).predict(np.eye(2, dtype=np.float32))
