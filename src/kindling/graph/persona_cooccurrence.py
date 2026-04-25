"""Per-persona item cooccurrence.

A different cooc structure: rather than `U.T @ U` over all users
(``ItemGraph``) or sessions (``SessionCooccurrenceGraph``), we
build a **separate cooc graph per persona**:

    adjacency[p][i, j] = #users in persona p who interacted with both i and j

At recommend time, the query user's score for candidate ``c`` against
their owned items ``O`` is computed by **soft-weighted** sum across
the personas the user matches:

    score(c, u) = sum_p match(u, p) * sum_o adjacency[p][c, o]

where ``match(u, p)`` is the cosine similarity from ``persona.matching.
match_user`` (the same soft-match used elsewhere). This gives:

- Cold-start users (sparse history): even with weak match across
  multiple personas, the persona-cooc has thousands of users worth
  of evidence per persona. Much fatter signal than global cooc on
  their 1-3 owned items.
- Warm/hot users: match concentrates on one persona; that persona's
  cooc is strong on items the user's taste cluster collectively
  touches.

Compare to the alternatives:

- ``ItemGraph`` (global cooc): ``U.T @ U`` over all users. Ignores
  taste structure. Cold-start users get tiny scores because few
  users share their specific items.
- Persona-as-boost-layer: would add a constant boost when persona
  match is confident, but the boost magnitude is decoupled from
  per-(user, candidate) co-touch evidence. Doesn't address the
  zero-cooc-baseline problem on cold-start.

Build cost: same as global cooc, distributed across personas. For
30 personas of 1000 users each, build runs 30 sparse matmuls each
on a smaller submatrix - net total O(n_users^2) work, about same
as one global cooc.

Storage: 30 sparse (n_items, n_items) matrices. On amazon-beauty
(12k items, 30 personas) ≈ 30 × few MB ≈ 100MB. Manageable. On
gowalla-scale (38k items × 30 personas) may need item-pruning per
persona.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class PersonaCooccurrenceGraph:
    """Per-persona item-item cooccurrence with soft-weighted scoring.

    Attributes
    ----------
    per_persona_adjacency:
        Length-n_personas list. Each element is a CSR
        (n_items, n_items) where adjacency[p][i, j] = #users in
        persona p who interacted with both items i and j.
        Diagonal is zeroed.
    item_ids:
        Internal-index -> external item_id mapping (mirrors ItemGraph).
    item_index:
        item_id -> internal-index mapping.
    n_personas:
        Number of personas. Personas with empty member sets contribute
        zero-matrices.
    persona_sizes:
        (n_personas,) array of distinct users per persona.
    """

    per_persona_adjacency: list[sparse.csr_matrix]
    item_ids: np.ndarray
    item_index: dict[object, int]
    n_personas: int
    persona_sizes: np.ndarray

    @property
    def n_items(self) -> int:
        if not self.per_persona_adjacency:
            return 0
        return int(self.per_persona_adjacency[0].shape[0])

    @property
    def n_edges(self) -> int:
        """Total stored non-zero entries across all persona graphs."""
        return sum(int(a.nnz) for a in self.per_persona_adjacency)

    def score_against_owned_soft(
        self,
        owned_indices: np.ndarray,
        match_weights: np.ndarray,
        exclude_indices: set[int] | None = None,
        match_threshold: float = 1e-6,
    ) -> np.ndarray:
        """Soft-weighted persona-cooc score per item.

        ``score[c] = sum_p match[p] * sum_o adjacency[p][c, o]``

        Parameters
        ----------
        owned_indices:
            Internal indices of the user's owned items.
        match_weights:
            (n_personas,) per-persona match scores from
            ``persona.matching.match_user``. Negative or near-zero
            entries (below ``match_threshold``) are skipped.
        exclude_indices:
            Items to zero in the output (typically the owned items
            themselves).
        match_threshold:
            Skip personas whose match weight falls below this.
            Default 1e-6.

        Returns
        -------
        (n_items,) array max-normalized to [0, 1].
        """
        n = self.n_items
        if n == 0 or owned_indices.size == 0 or match_weights.size == 0:
            return np.zeros(n, dtype=np.float64)
        owned_indices = owned_indices[(owned_indices >= 0) & (owned_indices < n)]
        if owned_indices.size == 0:
            return np.zeros(n, dtype=np.float64)

        scores = np.zeros(n, dtype=np.float64)
        for p, weight in enumerate(match_weights):
            if weight < match_threshold:
                continue
            if p >= len(self.per_persona_adjacency):
                continue
            adj = self.per_persona_adjacency[p]
            if adj.nnz == 0:
                continue
            # Sum the rows of owned items - per-item score under persona p.
            rows = adj[owned_indices]
            persona_scores = np.asarray(rows.sum(axis=0)).ravel()
            scores += float(weight) * persona_scores

        if exclude_indices:
            for idx in exclude_indices:
                if 0 <= idx < n:
                    scores[idx] = 0.0
        max_s = float(scores.max())
        if max_s > 0:
            scores = scores / max_s
        return scores


def build_persona_cooccurrence_graph(
    interactions: pd.DataFrame,
    item_index: dict[object, int],
    persona_index,
    min_persona_users: int = 5,
) -> PersonaCooccurrenceGraph | None:
    """Build per-persona item-cooc graphs from interactions + persona assignments.

    Parameters
    ----------
    interactions:
        Validated interactions with entity_id + item_id columns.
        ``_interaction_weight`` is used when present.
    item_index:
        Engine ItemGraph's item_id -> internal-index mapping.
    persona_index:
        Fitted ``PersonaIndex`` with ``user_to_persona`` (hard
        assignments). Soft scoring at query time uses
        ``persona.matching.match_user`` separately.
    min_persona_users:
        Personas with fewer than this many users get an empty
        (zero) cooc matrix. Avoids degenerate cooc on tiny
        clusters.

    Returns
    -------
    PersonaCooccurrenceGraph or None when the persona index is empty
    or the catalog has no items.
    """
    from kindling.preprocess import weights_of

    if persona_index is None or persona_index.n_personas == 0:
        return None
    n_items = max(item_index.values()) + 1 if item_index else 0
    if n_items == 0 or len(interactions) == 0:
        return None

    n_personas = persona_index.n_personas

    # Map interactions to (entity_idx, item_idx, weight).
    weights = weights_of(interactions)
    item_idx_array = np.asarray(
        [item_index.get(x, -1) for x in interactions["item_id"].to_numpy()],
        dtype=np.int64,
    )
    entity_array = interactions["entity_id"].to_numpy()
    keep = (item_idx_array >= 0) & (weights > 0)

    # Map entity -> persona via persona_index.
    user_to_persona = persona_index.user_to_persona
    entity_to_user_idx = persona_index.entity_id_to_user_idx
    persona_of_entity = np.asarray(
        [user_to_persona[entity_to_user_idx.get(e, -1)] if entity_to_user_idx.get(e, -1) >= 0 else -1
         for e in entity_array],
        dtype=np.int64,
    )
    keep = keep & (persona_of_entity >= 0)
    if not keep.any():
        # No interactions have valid persona assignments.
        return PersonaCooccurrenceGraph(
            per_persona_adjacency=[
                sparse.csr_matrix((n_items, n_items), dtype=np.float32)
                for _ in range(n_personas)
            ],
            item_ids=_item_ids_from_index(item_index, n_items),
            item_index=dict(item_index),
            n_personas=n_personas,
            persona_sizes=np.zeros(n_personas, dtype=np.int64),
        )

    item_idx_array = item_idx_array[keep]
    persona_of_entity = persona_of_entity[keep]
    weights = weights[keep].astype(np.float32)
    entity_array_kept = entity_array[keep]

    # Build per-persona adjacencies.
    per_persona_adjacency: list[sparse.csr_matrix] = []
    persona_sizes: list[int] = []

    # Map entities to dense indices within their persona for U_p construction.
    entity_to_user_idx_arr = np.asarray(
        [entity_to_user_idx.get(e, -1) for e in entity_array_kept],
        dtype=np.int64,
    )

    for p in range(n_personas):
        in_persona = persona_of_entity == p
        n_p_users = int(np.unique(entity_to_user_idx_arr[in_persona]).size)
        persona_sizes.append(n_p_users)
        if n_p_users < min_persona_users or not in_persona.any():
            per_persona_adjacency.append(
                sparse.csr_matrix((n_items, n_items), dtype=np.float32)
            )
            continue

        # Build U_p (n_p_users, n_items) and compute U_p.T @ U_p.
        sub_user_indices = entity_to_user_idx_arr[in_persona]
        sub_item_indices = item_idx_array[in_persona]
        sub_weights = weights[in_persona]
        # Re-index sub_user_indices to dense [0..n_p_users).
        unique_users, inverse = np.unique(sub_user_indices, return_inverse=True)
        # Max-aggregate per (user, item) pair.
        df_sub = pd.DataFrame({
            "u": inverse,
            "i": sub_item_indices,
            "w": sub_weights,
        })
        agg = df_sub.groupby(["u", "i"], sort=False, as_index=False)["w"].max()
        bipartite = sparse.csr_matrix(
            (agg["w"].to_numpy(dtype=np.float32),
             (agg["u"].to_numpy(dtype=np.int64), agg["i"].to_numpy(dtype=np.int64))),
            shape=(unique_users.size, n_items),
            dtype=np.float32,
        )
        adj = (bipartite.T @ bipartite).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        per_persona_adjacency.append(adj)

    return PersonaCooccurrenceGraph(
        per_persona_adjacency=per_persona_adjacency,
        item_ids=_item_ids_from_index(item_index, n_items),
        item_index=dict(item_index),
        n_personas=n_personas,
        persona_sizes=np.asarray(persona_sizes, dtype=np.int64),
    )


def _item_ids_from_index(item_index: dict[object, int], n_items: int) -> np.ndarray:
    item_ids = np.empty(n_items, dtype=object)
    for item_id, idx in item_index.items():
        item_ids[idx] = item_id
    return item_ids
