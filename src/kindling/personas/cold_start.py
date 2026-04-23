"""Cold-start affinity for new items (PRD supplement §2.7).

For every (item, persona) pair, compute the overperformance ratio:

    persona_affinity(c, P) = Σ_{u ∈ U_c} match(u, P)
    expected_affinity(c, P) = |U_c| × base_rate(P)
    overperformance(c, P) = persona_affinity / expected_affinity

Base rate is the persona's share of **interactions** (not users) —
active personas over-interact and the base-rate normalization corrects
for that, per the design decision to use interaction-volume-weighted
denominators.

Per the decision to skip the significance gate for v1, we record
raw overperformance for every (item, persona) pair above 1.0 (elevated)
rather than thresholding on confidence intervals. The measurement in
commit 4 tells us whether this helps or hurts; the threshold is
configurable if we need to raise it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kindling.personas.index import PersonaIndex


def compute_cold_start_weights(
    interactions: pd.DataFrame,
    index: PersonaIndex,
    overperformance_threshold: float = 1.0,
) -> sp.csr_matrix:
    """Build the (n_personas, n_items) cold-start weight matrix.

    Entry ``(P, c)`` is the overperformance ratio when it exceeds
    ``overperformance_threshold``, else zero. The matrix is
    intentionally sparse so storage + score-time gather stays cheap.

    Parameters
    ----------
    interactions:
        DataFrame with ``entity_id`` and ``item_id`` columns (the same
        frame passed to ``build_persona_index``).
    index:
        The fitted PersonaIndex. We read ``user_to_persona``,
        ``user_membership``, ``item_id_to_idx``, and ``entity_id_to_user_idx``.
    overperformance_threshold:
        Ratios at or below this value produce zero entries. Default 1.0
        records any elevation above base rate; raise to 1.5 or 2.0 for a
        stricter signal.
    """
    n_personas = index.n_personas
    n_items = index.n_items
    if n_personas == 0 or n_items == 0:
        return sp.csr_matrix((n_personas, n_items), dtype=np.float64)

    # Map interactions to (user_idx, item_idx).
    ent_col = interactions["entity_id"]
    item_col = interactions["item_id"]
    user_idx = ent_col.map(index.entity_id_to_user_idx).to_numpy()
    item_idx = item_col.map(index.item_id_to_idx).to_numpy()
    mask = ~(pd.isna(user_idx) | pd.isna(item_idx))
    user_idx = user_idx[mask].astype(np.int64)
    item_idx = item_idx[mask].astype(np.int64)

    assignments = np.asarray(index.user_to_persona, dtype=np.int64)
    memberships = np.asarray(index.user_membership, dtype=np.float64)

    # Filter to interactions whose user is assigned to a valid persona.
    valid_persona = (assignments[user_idx] >= 0) & (assignments[user_idx] < n_persona_max(n_personas))
    user_idx = user_idx[valid_persona]
    item_idx = item_idx[valid_persona]
    n_obs = len(user_idx)
    if n_obs == 0:
        return sp.csr_matrix((n_personas, n_items), dtype=np.float64)

    # Per-interaction persona match weight: for hard cluster assignment
    # this is just the membership probability of the user's persona.
    persona_of_int = assignments[user_idx]
    match_weight = memberships[user_idx]

    # persona_affinity[P, c] = sum_{(u, i) : P_u = P, i_u = c} match(u, P)
    affinity = sp.csr_matrix(
        (match_weight, (persona_of_int, item_idx)), shape=(n_personas, n_items)
    )
    affinity.sum_duplicates()

    # |U_c| per item: number of distinct users who interacted with c
    # (with valid persona). Use drop_duplicates on (user, item) pairs.
    user_item_unique = pd.DataFrame({"u": user_idx, "i": item_idx}).drop_duplicates()
    items_col = user_item_unique["i"].to_numpy()
    n_users_per_item = np.zeros(n_items, dtype=np.float64)
    np.add.at(n_users_per_item, items_col, 1.0)

    # Base rate per persona: persona's share of total interaction weight.
    total_weight_per_persona = np.zeros(n_personas, dtype=np.float64)
    np.add.at(total_weight_per_persona, persona_of_int, match_weight)
    total_weight = total_weight_per_persona.sum()
    if total_weight <= 0.0:
        return sp.csr_matrix((n_personas, n_items), dtype=np.float64)
    base_rate = total_weight_per_persona / total_weight  # (n_personas,)

    # expected_affinity[P, c] = |U_c| * base_rate[P]
    # Compute as outer product (small: n_personas x n_items).
    # Only use sparse dense product where n_users_per_item > 0.
    affinity_dense = affinity.toarray()
    expected = np.outer(base_rate, n_users_per_item)  # (n_personas, n_items)

    with np.errstate(divide="ignore", invalid="ignore"):
        overperf = np.where(expected > 0.0, affinity_dense / expected, 0.0)
    overperf[overperf <= overperformance_threshold] = 0.0

    return sp.csr_matrix(overperf, dtype=np.float64)


def n_persona_max(n_personas: int) -> int:
    """Tiny helper used above for readability."""
    return int(n_personas)
