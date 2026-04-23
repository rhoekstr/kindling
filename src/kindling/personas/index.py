"""PersonaIndex: fitted persona vectors + IDF + user assignments.

The runtime state produced by ``build.build_persona_index`` and consumed
by ``matching.match_user`` + ``matching.score_candidates``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class PersonaIndex:
    """Compact persistable snapshot of a fitted persona signal.

    Attributes
    ----------
    persona_vectors:
        (n_personas, n_items) sparse CSR matrix. Each row is a persona's
        TF-IDF-weighted, L2-normalized item vector. Only items that
        passed the z-score filter (§2.3 step 2) have non-zero entries.
    idf:
        (n_items,) IDF vector, rate-weighted per supplement §2.3 step 3.
    persona_sizes:
        (n_personas,) number of unique users per persona.
    item_id_to_idx:
        Mapping from original item id to the internal column index in
        ``persona_vectors``. Built from the engine's item graph.
    user_to_persona:
        (n_users,) int array of persona assignments, -1 for noise.
    user_membership:
        (n_users,) float array of persona-membership probability.
    entity_id_to_user_idx:
        Mapping from entity id to the row index used in ``user_to_persona``.
    """

    persona_vectors: sp.csr_matrix
    idf: np.ndarray
    persona_sizes: np.ndarray
    item_id_to_idx: dict[object, int] = field(default_factory=dict)
    user_to_persona: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    user_membership: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    entity_id_to_user_idx: dict[object, int] = field(default_factory=dict)

    @property
    def n_personas(self) -> int:
        return int(self.persona_vectors.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.persona_vectors.shape[1])

    def persona_of_entity(self, entity_id: object) -> int:
        """Return the persona index for an entity, or -1 if unknown/noise."""
        user_idx = self.entity_id_to_user_idx.get(entity_id, -1)
        if user_idx < 0 or user_idx >= len(self.user_to_persona):
            return -1
        return int(self.user_to_persona[user_idx])

    def membership_of_entity(self, entity_id: object) -> float:
        """Return the membership probability for an entity, or 0 if unknown."""
        user_idx = self.entity_id_to_user_idx.get(entity_id, -1)
        if user_idx < 0 or user_idx >= len(self.user_membership):
            return 0.0
        return float(self.user_membership[user_idx])
