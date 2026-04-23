"""Clustering protocol + default implementations (PRD supplement §2.2).

The clustering stage produces per-user persona assignments. The rest of
the persona pipeline (rate aggregation, TF-IDF, matching) is
algorithm-agnostic: personas are stored as item-space vectors built
from member aggregates, not from cluster centroids. This decoupling
lets us ship HDBSCAN (which doesn't support online ``predict()``) as
the default while keeping K-means as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    """Output of a clustering run.

    assignments:
        Shape (n_users,). Entity-to-persona mapping. -1 indicates a
        noise point (no persona) and is handled as "zero persona
        signal" downstream.
    probabilities:
        Shape (n_users,). Membership probability in [0, 1]. For K-means
        this is 1.0 for all assigned users; for HDBSCAN it comes from
        the cluster's density.
    n_personas:
        Number of distinct non-noise persona labels in the output.
    """

    assignments: np.ndarray
    probabilities: np.ndarray
    n_personas: int


@runtime_checkable
class ClusteringProtocol(Protocol):
    name: str

    def fit(self, user_vectors: np.ndarray) -> ClusterResult:
        """Cluster ``user_vectors`` (shape (n_users, n_features))."""
        ...


@dataclass
class HDBSCANClustering:
    """Density-based clustering over UMAP-reduced vectors.

    Default for the persona signal because cluster size naturally
    varies with the density of taste structure and we keep noise
    points honest (users whose behavior doesn't fit any cluster
    contribute zero persona signal rather than being force-fit).
    """

    name: str = "hdbscan"
    min_cluster_size_pct: float = 0.005  # 0.5% of user base
    absolute_min_cluster_size: int = 5
    reduction_method: str = "umap"  # "umap", "als", "none"
    reduction_dims: int = 20
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    random_state: int = 0

    def fit(self, user_vectors: np.ndarray) -> ClusterResult:
        try:
            import hdbscan
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HDBSCANClustering requires the 'hdbscan' package. "
                "Install with `pip install 'kindling[personas]'`."
            ) from exc

        reduced = self._reduce(user_vectors)
        n_users = reduced.shape[0]
        min_cluster_size = max(
            self.absolute_min_cluster_size,
            int(round(n_users * self.min_cluster_size_pct)),
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,  # defaults to min_cluster_size
            prediction_data=False,
            core_dist_n_jobs=1,
        )
        labels = clusterer.fit_predict(reduced)
        probabilities = getattr(clusterer, "probabilities_", np.ones(n_users, dtype=np.float64))
        n_personas = int(labels.max()) + 1 if (labels >= 0).any() else 0
        return ClusterResult(
            assignments=labels.astype(np.int64),
            probabilities=probabilities.astype(np.float64),
            n_personas=n_personas,
        )

    def _reduce(self, user_vectors: np.ndarray) -> np.ndarray:
        if user_vectors.shape[1] <= self.reduction_dims:
            return user_vectors
        if self.reduction_method == "none":
            return user_vectors
        if self.reduction_method == "als":
            # Caller is expected to have passed already-reduced factors.
            return user_vectors
        if self.reduction_method == "umap":
            try:
                import umap  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "HDBSCANClustering(reduction_method='umap') requires "
                    "the 'umap-learn' package. Install with "
                    "`pip install 'kindling[personas]'`."
                ) from exc
            reducer = umap.UMAP(
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                n_components=self.reduction_dims,
                random_state=self.random_state,
                verbose=False,
            )
            return np.asarray(reducer.fit_transform(user_vectors), dtype=np.float64)
        raise ValueError(f"Unknown reduction_method: {self.reduction_method!r}")


@dataclass
class KMeansClustering:
    """Simple K-means fallback.

    Fixed K (with a sane default of 30). No dimensionality reduction -
    K-means handles high-dim reasonably well when the underlying
    factors are already low-dim (e.g., ALS factors). Deterministic
    given ``random_state``.
    """

    name: str = "kmeans"
    n_clusters: int = 30
    random_state: int = 0
    n_init: int = 10

    def fit(self, user_vectors: np.ndarray) -> ClusterResult:
        from sklearn.cluster import KMeans

        n_users = user_vectors.shape[0]
        k = min(self.n_clusters, max(2, n_users // 5))
        km = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        assignments = km.fit_predict(user_vectors).astype(np.int64)
        probabilities = np.ones(n_users, dtype=np.float64)
        return ClusterResult(
            assignments=assignments,
            probabilities=probabilities,
            n_personas=int(assignments.max()) + 1,
        )
