"""Persona vs cooc differentiation analysis.

Answers: "When personas are active, are we surfacing meaningfully
different recommendations than cooc, or just a smaller subset of the
same cooc top-K?"

For each non-empty persona, sample users in that persona and for each:

  1. Compute cooc-base score vector (sum of global cooc rows for the
     user's owned items, with owned items masked out).
  2. Compute persona-cooc score vector (same construction but using
     persona p's per-persona cooc CSR).
  3. Take top-K of each.
  4. Measure differentiation:
     - **jaccard@K** = |C ∩ P| / |C ∪ P|  — set overlap
     - **kendall_tau** on the items in the intersection (rank
       agreement on shared items)
     - **mean_rank_shift** for items unique to persona top-K = the
       cooc rank those items would have had. Large = persona is
       genuinely surfacing items cooc would have ranked far down.
       Small (≈ K) = persona is just shuffling near-misses.

Aggregates → per-persona stats → global mean across personas.

This is purely diagnostic; nothing is wired into recommend(). Output is
a dict suitable for merging into a benchmark report.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from kindling.engine_v2 import EngineV2


def _csr_row(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, row: int):
    s = int(indptr[row])
    e = int(indptr[row + 1])
    return indices[s:e], data[s:e]


def _user_score_vec(
    items: np.ndarray,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    n_items: int,
) -> np.ndarray:
    """Sum of cooc rows for `items` → length-n_items score vector."""
    out = np.zeros(n_items, dtype=np.float64)
    for i in items:
        if 0 <= int(i) < len(indptr) - 1:
            cols, vals = _csr_row(data, indices, indptr, int(i))
            out[cols] += vals
    return out


def _topk_indices(scores: np.ndarray, k: int, mask_out: np.ndarray) -> np.ndarray:
    """Top-k item indices excluding `mask_out`. Stable order by score desc."""
    s = scores.copy()
    s[mask_out] = -np.inf
    if k >= s.size:
        return np.argsort(-s, kind="stable")
    # argpartition then sort the top-k slice for stable order
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx], kind="stable")]


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Pairwise concordance ratio between two ranked sequences (same length).

    Returns -1..1. Used on the **intersection** of two top-K sets where
    we re-rank each by its position in the source list.
    """
    n = a.size
    if n < 2:
        return 0.0
    # Map each item to its rank in a and b respectively.
    rank_a = {item: r for r, item in enumerate(a.tolist())}
    rank_b = {item: r for r, item in enumerate(b.tolist())}
    common = [it for it in a.tolist() if it in rank_b]
    m = len(common)
    if m < 2:
        return 0.0
    pairs = m * (m - 1) // 2
    concord = 0
    discord = 0
    for i in range(m):
        for j in range(i + 1, m):
            xi, xj = rank_a[common[i]], rank_a[common[j]]
            yi, yj = rank_b[common[i]], rank_b[common[j]]
            if (xi - xj) * (yi - yj) > 0:
                concord += 1
            elif (xi - xj) * (yi - yj) < 0:
                discord += 1
    if pairs == 0:
        return 0.0
    return (concord - discord) / pairs


def compute_persona_diff(
    engine: EngineV2,
    sample_users_per_persona: int = 30,
    k: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute per-persona and aggregate persona-vs-cooc differentiation.

    Returns:
      ```
      {
        "global": {
            "mean_jaccard_at_k": float,
            "mean_kendall_tau": float,
            "mean_rank_shift_unique": float,  # for items uniquely in persona top-K
            "fraction_identical": float,      # users where persona top-K == cooc top-K
            "n_users_sampled": int,
            "n_personas_sampled": int,
        },
        "per_persona": {
            persona_id: {
                "n_users": int,
                "jaccard_at_k": float,
                "kendall_tau": float,
                "rank_shift_unique": float,
            }
            for persona_id in non_empty
        },
      }
      ```
    """
    if engine._state is None:
        return {}
    st = engine._state
    if not st.personas_enabled or st.n_personas == 0:
        return {"global": {}, "per_persona": {}}

    rng = np.random.RandomState(seed)
    n_items = st.n_items
    user_to_persona = np.asarray(st.user_to_persona, dtype=np.int64)
    n_users = len(user_to_persona)
    if n_users == 0:
        return {"global": {}, "per_persona": {}}

    # entity_id list in user_idx order (insertion order = user_idx order).
    entity_ids = list(st.owned_by_entity.keys())
    if len(entity_ids) != n_users:
        # Defensive: if the dict order disagrees with the assignment array
        # length, abort cleanly.
        return {"global": {}, "per_persona": {}}

    # Group user indices by persona (skip -1 noise).
    by_persona: dict[int, list[int]] = {}
    for u_idx, p in enumerate(user_to_persona):
        p_int = int(p)
        if p_int < 0:
            continue
        by_persona.setdefault(p_int, []).append(u_idx)

    # Iterate personas with non-empty membership AND a non-empty
    # persona_cooc CSR (some persona slots can be empty after filtering).
    per_persona: dict[int, dict[str, float | int]] = {}
    aggregate_jacc: list[float] = []
    aggregate_tau: list[float] = []
    aggregate_shift: list[float] = []
    aggregate_identical: list[bool] = []

    for p, users in by_persona.items():
        if p >= len(st.persona_cooc_data):
            continue
        pc_data = st.persona_cooc_data[p]
        pc_indices = st.persona_cooc_indices[p]
        pc_indptr = st.persona_cooc_indptr[p]
        if pc_data.size == 0:
            continue

        if len(users) > sample_users_per_persona:
            sampled_idx = rng.choice(len(users), sample_users_per_persona, replace=False)
            sample = [users[i] for i in sampled_idx]
        else:
            sample = users

        jacc_list: list[float] = []
        tau_list: list[float] = []
        shift_list: list[float] = []
        identical_list: list[bool] = []

        for u_idx in sample:
            entity = entity_ids[u_idx]
            owned = st.owned_by_entity.get(entity)
            if owned is None or owned.size == 0:
                continue
            mask = np.zeros(n_items, dtype=bool)
            mask[owned] = True

            # cooc base score vector
            cooc_scores = _user_score_vec(
                owned, st.cooc_data, st.cooc_indices, st.cooc_indptr, n_items
            )
            persona_scores = _user_score_vec(
                owned, pc_data, pc_indices, pc_indptr, n_items
            )

            cooc_topk = _topk_indices(cooc_scores, k, mask)
            persona_topk = _topk_indices(persona_scores, k, mask)
            if cooc_topk.size == 0 or persona_topk.size == 0:
                continue

            cooc_set = set(int(x) for x in cooc_topk.tolist())
            persona_set = set(int(x) for x in persona_topk.tolist())
            inter = cooc_set & persona_set
            union = cooc_set | persona_set
            jacc = len(inter) / len(union) if union else 0.0
            jacc_list.append(jacc)
            identical_list.append(cooc_set == persona_set)

            # Kendall tau on intersection items (using their order within
            # each top-K list).
            tau = _kendall_tau(cooc_topk, persona_topk)
            tau_list.append(tau)

            # For items uniquely in persona top-K: their rank in the
            # full cooc ranking. Mean rank shift = how far down would
            # they have been? Use ascending rank from full cooc sorted
            # desc; lower is better for cooc, higher is "different from cooc".
            unique_persona = persona_set - cooc_set
            if unique_persona:
                cooc_full_order = np.argsort(-cooc_scores, kind="stable")
                rank_lookup = np.empty(n_items, dtype=np.int64)
                rank_lookup[cooc_full_order] = np.arange(n_items)
                shifts = [int(rank_lookup[i]) for i in unique_persona]
                shift_list.append(float(np.mean(shifts)))

        per_persona[p] = {
            "n_users_sampled": len(sample),
            "jaccard_at_k": float(np.mean(jacc_list)) if jacc_list else 0.0,
            "kendall_tau": float(np.mean(tau_list)) if tau_list else 0.0,
            "rank_shift_unique": float(np.mean(shift_list)) if shift_list else 0.0,
            "fraction_identical": float(np.mean(identical_list)) if identical_list else 0.0,
        }
        aggregate_jacc.extend(jacc_list)
        aggregate_tau.extend(tau_list)
        aggregate_shift.extend(shift_list)
        aggregate_identical.extend(identical_list)

    glob = {
        "mean_jaccard_at_k": (
            float(np.mean(aggregate_jacc)) if aggregate_jacc else 0.0
        ),
        "mean_kendall_tau": (
            float(np.mean(aggregate_tau)) if aggregate_tau else 0.0
        ),
        "mean_rank_shift_unique": (
            float(np.mean(aggregate_shift)) if aggregate_shift else 0.0
        ),
        "fraction_identical": (
            float(np.mean(aggregate_identical)) if aggregate_identical else 0.0
        ),
        "n_users_sampled": len(aggregate_jacc),
        "n_personas_sampled": len(per_persona),
    }
    return {"global": glob, "per_persona": per_persona}
