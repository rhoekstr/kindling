"""Persona signal configuration surface (PRD supplement §4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kindling.personas.clustering import ClusteringProtocol


@dataclass
class PersonaConfig:
    """Configuration for the persona signal pipeline.

    Attributes
    ----------
    enabled:
        Master switch. When False, the engine skips persona construction
        entirely even if the config object is passed. Default False so
        the signal is opt-in.
    clustering:
        Pluggable clustering strategy. Default HDBSCAN over UMAP-reduced
        ALS factors; K-means ships as a fallback for environments where
        HDBSCAN + UMAP aren't available.
    z_threshold:
        Items with per-persona rate more than this many std-devs BELOW
        the persona's mean rate are dropped from the persona vector
        (PRD supplement §2.3 step 2). Default 1.5 is single-tailed noise
        filtering without discarding legitimately-popular items.
    min_cluster_membership:
        Users with max-persona membership probability below this value
        produce zero persona signal (graceful degradation for users the
        clustering can't place).
    min_activation_users:
        Persona signal skips entirely when fewer than this many users
        are in the training set. HDBSCAN + UMAP behave poorly below
        a few thousand users; we default to 1000 (per supplement §4.3)
        but emit a warning when the resulting noise-fraction exceeds
        50% (clustering not finding meaningful structure).
    cold_start_overperformance_threshold:
        For new items, persona-affinity must exceed baseline by this
        multiplier to be considered "fitted" to a persona. Wired in
        commit 3.
    cold_start_min_interactions:
        Minimum interaction count before cold-start inference activates
        for a new item. Wired in commit 3.
    """

    enabled: bool = False
    clustering: "ClusteringProtocol | None" = None
    z_threshold: float = 1.5
    min_cluster_membership: float = 0.5
    min_activation_users: int = 1000
    cold_start_overperformance_threshold: float = 1.0
    cold_start_min_interactions: int = 1
    # Matches the L2-normalized main persona_vectors scale after the
    # cold_start_weights log1p+L2 normalization fix. 0.5 keeps main
    # persona signal dominant while letting cold-start contribute.
    cold_start_weight: float = 0.5

    def resolved_clustering(self) -> "ClusteringProtocol":
        """Return a concrete clustering instance, defaulting to HDBSCAN."""
        if self.clustering is not None:
            return self.clustering
        # Lazy import avoids a hard dep on hdbscan/umap when the user is
        # happy with the K-means fallback.
        from kindling.personas.clustering import HDBSCANClustering

        return HDBSCANClustering()
