from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import kmeans_plusplus

Float32Matrix = NDArray[np.float32]


def _normalize_rows(values: ArrayLike) -> Float32Matrix:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2-dimensional matrix")
    if not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("Expected a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains non-finite values")
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("Cannot normalize zero vectors")
    return matrix / lengths


class SphericalKMeans:
    """Small scikit-learn-compatible spherical K-means estimator."""

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(
        self,
        values: ArrayLike,
        y: ArrayLike | None = None,
    ) -> SphericalKMeans:
        del y
        matrix = _normalize_rows(values)
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if self.n_clusters > len(matrix):
            raise ValueError("n_clusters cannot exceed the row count")
        if self.n_init < 1:
            raise ValueError("n_init must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol < 0:
            raise ValueError("tol cannot be negative")

        random = np.random.RandomState(self.random_state)
        best: tuple[float, NDArray[np.int64], Float32Matrix, int] | None = None
        for _ in range(self.n_init):
            seed = int(random.randint(np.iinfo(np.int32).max))
            result = self._fit_once(matrix, seed)
            if best is None or result[0] < best[0]:
                best = result

        assert best is not None
        self.inertia_, self.labels_, self.cluster_centers_, self.n_iter_ = best
        return self

    def _fit_once(
        self,
        matrix: Float32Matrix,
        random_state: int,
    ) -> tuple[float, NDArray[np.int64], Float32Matrix, int]:
        centers, _ = kmeans_plusplus(
            matrix,
            n_clusters=self.n_clusters,
            random_state=random_state,
        )
        centers = _normalize_rows(centers)
        labels: NDArray[np.int64] | None = None
        for iteration in range(1, self.max_iter + 1):
            similarities = matrix @ centers.T
            new_labels = np.argmax(similarities, axis=1).astype(np.int64)
            counts = np.bincount(new_labels, minlength=self.n_clusters)
            weakest_rows = iter(np.argsort(np.max(similarities, axis=1)))
            for cluster_id in np.flatnonzero(counts == 0):
                row = next(row for row in weakest_rows if counts[new_labels[row]] > 1)
                source_cluster = new_labels[row]
                counts[source_cluster] -= 1
                new_labels[row] = cluster_id
                counts[cluster_id] = 1

            new_centers = np.stack(
                [
                    matrix[new_labels == cluster_id].mean(axis=0)
                    for cluster_id in range(self.n_clusters)
                ]
            )
            new_centers = _normalize_rows(new_centers)
            center_shift = float(np.max(1.0 - np.sum(centers * new_centers, axis=1)))
            converged = labels is not None and np.array_equal(labels, new_labels)
            centers = new_centers
            labels = new_labels
            if converged or center_shift <= self.tol:
                break

        assert labels is not None
        assigned_similarities = np.sum(matrix * centers[labels], axis=1)
        inertia = float(np.sum(np.maximum(0.0, 1.0 - assigned_similarities)))
        return inertia, labels, centers, iteration

    def fit_predict(
        self,
        values: ArrayLike,
        y: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        return self.fit(values, y).labels_

    def predict(self, values: ArrayLike) -> NDArray[np.int64]:
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("SphericalKMeans has not been fitted")
        matrix = _normalize_rows(values)
        return np.argmax(matrix @ self.cluster_centers_.T, axis=1).astype(np.int64)
