"""Persona signal (PRD supplement: persona_signal).

Group-level taste matching: users are clustered into personas during
fit, each persona becomes a TF-IDF-weighted vector in item space, and at
query time the user's interaction vector is matched against personas
via cosine similarity. Candidate items are scored by persona-weighted
importance.

This module is opt-in (``Engine(persona_config=PersonaConfig(...))``).
It requires the ``umap-learn`` and ``hdbscan`` extras for the default
HDBSCAN clustering path; the K-means fallback works with just
scikit-learn.
"""

from kindling.personas.clustering import (
    ClusteringProtocol,
    ClusterResult,
    HDBSCANClustering,
    KMeansClustering,
    KMeansWithNoiseClustering,
)
from kindling.personas.config import PersonaConfig
from kindling.personas.index import PersonaIndex

__all__ = [
    "ClusterResult",
    "ClusteringProtocol",
    "HDBSCANClustering",
    "KMeansClustering",
    "KMeansWithNoiseClustering",
    "PersonaConfig",
    "PersonaIndex",
]
