"""Temporal interaction graph - shared substrate for two new signals.

A weighted undirected item-item graph where edge weights reflect
**temporal adjacency** rather than aggregate co-occurrence. For each
user, every pair of items they interacted with contributes an edge
weight modulated by the time gap between interactions:

    weight(item_a, item_b) = sum_{users with both items} kernel(|t_a - t_b|)

The kernel is a logistic decay:

    kernel(dt) = 1 / (1 + exp((dt - midpoint) / steepness))

It stays near 1 for ``dt`` well below the midpoint, drops sharply
around the midpoint, and decays toward 0 well above. The shape matches
"items rated within the same session-like window are roughly
equivalent; items across distinct sessions are sharply less related."

Auto-calibration of (midpoint, steepness) reuses the existing
``ingest.sessions._fit_gmm_threshold`` machinery — the GMM's bimodal
threshold on log-inter-event-deltas is the natural midpoint, and the
within-component sigma is the natural steepness scale.

Pure-count fallback when timestamps are absent or the GMM doesn't
detect bimodality. In that regime ``kernel = 1`` for every pair, and
the temporal graph collapses to a re-weighted cooccurrence graph.

Two derived signals (built in separate modules) consume this graph:

1. ``interaction_network`` - PPR walks for breadth.
2. ``interaction_neighborhood`` - Louvain communities + per-query
   centrality on the union of the user's top-N communities.

Storage: scipy CSR adjacency, mirrors ``ItemGraph``. Caps per-user
history at ``max_history_per_user`` (default 200) to keep build cost
bounded - O(sum_users * history_len * window_radius) dominated by
the within-window walk, not O(history_len^2).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

if TYPE_CHECKING:
    from kindling.graph.item_graph import ItemGraph


# Kernel cutoff: stop generating pairs once kernel(dt) drops below this.
# 1e-3 is two orders of magnitude below the midpoint, ~5x past the steepness.
_KERNEL_CUTOFF = 1e-3

# Per-user history cap. Above this we keep the most-recent N events.
_DEFAULT_MAX_HISTORY = 200


@dataclass(frozen=True)
class KernelParams:
    """Logistic-decay kernel parameters.

    Attributes
    ----------
    midpoint_seconds:
        Time gap (seconds) where the kernel transitions through 0.5.
        For timestamped data this is the GMM's bimodal threshold (the
        natural session-vs-cross-session boundary).
    steepness_seconds:
        Width of the transition region (seconds). Larger = smoother
        transition. Defaults to the within-component sigma from the GMM.
    pure_count:
        When True, kernel(dt) = 1 for every pair (fallback when no
        usable timestamps or no bimodal session structure).
    strategy:
        How the kernel was calibrated. ``"gmm"`` | ``"manual_fallback"``
        | ``"pure_count"``.
    """

    midpoint_seconds: float
    steepness_seconds: float
    pure_count: bool
    strategy: str

    def kernel(self, dt: np.ndarray | float) -> np.ndarray | float:
        if self.pure_count:
            if isinstance(dt, np.ndarray):
                return np.ones_like(dt, dtype=np.float64)
            return 1.0
        # Numerically stable logistic, branched on sign to avoid overflow
        # for large positive z.
        z = (np.asarray(dt, dtype=np.float64) - self.midpoint_seconds) / max(
            self.steepness_seconds, 1e-6
        )
        with np.errstate(over="ignore"):
            pos = z >= 0
            out = np.empty_like(z, dtype=np.float64)
            # For z >= 0: 1/(1 + e^z) — exp doesn't overflow because we
            # mask the negative branch out, but the e^z for very large z
            # rounds to inf and 1/inf = 0, which is the correct limit.
            out[pos] = 1.0 / (1.0 + np.exp(z[pos]))
            out[~pos] = 1.0 - 1.0 / (1.0 + np.exp(-z[~pos]))
        return float(out) if np.isscalar(dt) or np.ndim(dt) == 0 else out

    def cutoff_seconds(self) -> float:
        """Time gap above which kernel(dt) < ``_KERNEL_CUTOFF``."""
        if self.pure_count:
            return float("inf")
        # Solve 1/(1+exp(z)) = cutoff → z = log(1/cutoff - 1)
        z = float(np.log(1.0 / _KERNEL_CUTOFF - 1.0))
        return self.midpoint_seconds + z * self.steepness_seconds


@dataclass(frozen=True)
class TemporalInteractionGraph:
    """Sparse item-item adjacency under the temporal-decay kernel.

    Same structural shape as ``ItemGraph`` so retrievers and signals
    can interoperate (item_index alignment, sparse access, prune/decay).

    Attributes
    ----------
    adjacency:
        Symmetric CSR (n_items, n_items). adjacency[i, j] = sum over
        users of kernel(|t_a - t_b|) for that user's interactions on
        items (i, j). Diagonal zeroed.
    item_ids:
        Aligned with the engine's ItemGraph item ordering.
    item_index:
        Mapping item_id -> internal index.
    kernel_params:
        How the kernel was calibrated for this build (for diagnostics
        and reproducibility).
    n_users_contributed:
        Distinct users whose data fed at least one edge weight.
    n_pairs_generated:
        Total (item_a, item_b, weight) tuples accumulated before the
        sparse aggregation. Diagnostic of build cost.
    """

    adjacency: sparse.csr_matrix
    item_ids: np.ndarray
    item_index: dict[object, int]
    kernel_params: KernelParams
    n_users_contributed: int
    n_pairs_generated: int

    @property
    def n_items(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_edges(self) -> int:
        """Stored non-zero entries (undirected edge count is n_edges/2)."""
        return int(self.adjacency.nnz)

    def neighbors(self, item_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(neighbor_indices, weights)`` for ``item_idx``."""
        row_start = self.adjacency.indptr[item_idx]
        row_end = self.adjacency.indptr[item_idx + 1]
        return (
            self.adjacency.indices[row_start:row_end],
            self.adjacency.data[row_start:row_end],
        )


def calibrate_kernel(
    interactions: pd.DataFrame,
    manual_fallback_seconds: float = 1800.0,
    min_samples: int = 50,
) -> KernelParams:
    """Auto-calibrate the logistic-decay kernel from inter-event deltas.

    Reuses the same GMM-on-log-inter-event-deltas approach the session
    inference module uses. The GMM's bimodal threshold is the kernel
    midpoint; the within-component sigma (in original seconds, after
    inverse log) is the steepness scale.

    Falls back to (manual_fallback_seconds, manual_fallback_seconds/4)
    when timestamps are absent, sample size is too small, or the GMM
    doesn't detect bimodality.
    """
    if "timestamp" not in interactions.columns or len(interactions) == 0:
        return KernelParams(
            midpoint_seconds=manual_fallback_seconds,
            steepness_seconds=manual_fallback_seconds / 4.0,
            pure_count=True,
            strategy="pure_count",
        )

    sorted_df = interactions.sort_values(["entity_id", "timestamp"], kind="mergesort")
    ts_seconds = sorted_df["timestamp"].astype("int64").to_numpy() // 10**9
    entity_ids = sorted_df["entity_id"].to_numpy()
    same_entity = np.zeros(len(sorted_df), dtype=bool)
    same_entity[1:] = entity_ids[1:] == entity_ids[:-1]
    deltas = np.zeros(len(sorted_df), dtype=np.float64)
    deltas[1:] = np.where(same_entity[1:], ts_seconds[1:] - ts_seconds[:-1], 0.0)
    deltas = deltas[same_entity & (deltas > 0)]

    if len(deltas) < min_samples:
        return KernelParams(
            midpoint_seconds=manual_fallback_seconds,
            steepness_seconds=manual_fallback_seconds / 4.0,
            pure_count=False,
            strategy="manual_fallback",
        )

    log_deltas = np.log(deltas)
    threshold_log, llr, within_sigma_log = _fit_gmm_kernel_params(log_deltas)
    if threshold_log is None or llr is None or llr < 10.0:
        # Bimodality not detected - same threshold as the session module's
        # _GOF_MIN_LLR_GAIN. Pure-count fallback because using a manual
        # threshold here would invent structure that isn't there.
        return KernelParams(
            midpoint_seconds=manual_fallback_seconds,
            steepness_seconds=manual_fallback_seconds / 4.0,
            pure_count=True,
            strategy="pure_count",
        )

    # Convert from log-seconds back to seconds, expand sigma to a
    # geometric-mean steepness in original-time units.
    midpoint = float(np.exp(threshold_log))
    # within_sigma_log is in log-seconds; the steepness in seconds at
    # the midpoint is approximately midpoint * within_sigma_log.
    steepness = float(midpoint * max(within_sigma_log, 0.1))
    return KernelParams(
        midpoint_seconds=midpoint,
        steepness_seconds=steepness,
        pure_count=False,
        strategy="gmm",
    )


def _fit_gmm_kernel_params(
    log_deltas: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Fit a 2-component GMM on log-deltas; return (threshold,
    log_likelihood_ratio_2_vs_1, within_component_sigma).

    Mirrors ``ingest.sessions._fit_gmm_threshold`` but additionally
    returns the within-component (smaller-mean) sigma, which the
    sessions module discards.
    """
    rng = np.random.default_rng(seed=0)
    n = len(log_deltas)
    if n < 50:
        return None, None, None

    mu_1 = float(log_deltas.mean())
    sigma_1 = float(log_deltas.std(ddof=0)) or 1e-6
    ll_1 = float(norm.logpdf(log_deltas, loc=mu_1, scale=sigma_1).sum())

    p25, p75 = np.percentile(log_deltas, [25, 75])
    mu = np.array([p25, p75], dtype=np.float64)
    if mu[0] >= mu[1]:
        mu = np.array([mu_1 - 1.0, mu_1 + 1.0])
    sigma = np.array([sigma_1, sigma_1], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    for _ in range(50):
        log_resp = np.stack(
            [
                np.log(pi[0] + 1e-12) + norm.logpdf(log_deltas, mu[0], sigma[0]),
                np.log(pi[1] + 1e-12) + norm.logpdf(log_deltas, mu[1], sigma[1]),
            ],
            axis=1,
        )
        log_norm = np.logaddexp(log_resp[:, 0], log_resp[:, 1])
        resp = np.exp(log_resp - log_norm[:, None])
        nk = resp.sum(axis=0)
        if (nk < 1.0).any():
            return None, None, None
        pi = nk / n
        mu = (resp * log_deltas[:, None]).sum(axis=0) / nk
        var = (resp * (log_deltas[:, None] - mu) ** 2).sum(axis=0) / nk
        sigma = np.sqrt(np.maximum(var, 1e-8))

    ll_2 = float(np.logaddexp(
        np.log(pi[0] + 1e-12) + norm.logpdf(log_deltas, mu[0], sigma[0]),
        np.log(pi[1] + 1e-12) + norm.logpdf(log_deltas, mu[1], sigma[1]),
    ).sum())
    llr = ll_2 - ll_1

    # Threshold = midpoint between component means weighted by sigma.
    order = np.argsort(mu)
    mu_lo, mu_hi = mu[order[0]], mu[order[1]]
    sigma_lo = sigma[order[0]]
    threshold = (mu_lo + mu_hi) / 2.0
    return float(threshold), float(llr), float(sigma_lo)


def build_temporal_interaction_graph(
    interactions: pd.DataFrame,
    item_index: dict[object, int],
    kernel_params: KernelParams | None = None,
    max_history_per_user: int = _DEFAULT_MAX_HISTORY,
) -> TemporalInteractionGraph:
    """Build the temporal interaction graph.

    Parameters
    ----------
    interactions:
        Validated interaction DataFrame. ``timestamp`` optional but
        kernel collapses to pure-count without it.
    item_index:
        Engine ItemGraph's item_id -> internal-index mapping. Items
        not in this index are dropped (they're not in the served
        catalog).
    kernel_params:
        If None, calls ``calibrate_kernel`` to derive from data.
    max_history_per_user:
        Cap on per-user interactions used. Most-recent ``max_history_per_user``
        events are kept (oldest dropped) when a user exceeds the cap.

    Returns
    -------
    TemporalInteractionGraph with symmetric CSR adjacency on the same
    item ordering as the input ``item_index``.
    """
    if kernel_params is None:
        kernel_params = calibrate_kernel(interactions)

    n_items = max(item_index.values()) + 1 if item_index else 0
    item_ids = np.empty(n_items, dtype=object)
    for item_id, idx in item_index.items():
        item_ids[idx] = item_id

    if n_items == 0 or len(interactions) == 0:
        return TemporalInteractionGraph(
            adjacency=sparse.csr_matrix((n_items, n_items), dtype=np.float32),
            item_ids=item_ids,
            item_index=dict(item_index),
            kernel_params=kernel_params,
            n_users_contributed=0,
            n_pairs_generated=0,
        )

    has_timestamp = "timestamp" in interactions.columns and not kernel_params.pure_count

    # Map item_ids → indices; drop interactions whose item isn't in the catalog.
    item_idx_array = np.asarray(
        [item_index.get(x, -1) for x in interactions["item_id"].to_numpy()],
        dtype=np.int64,
    )
    keep = item_idx_array >= 0
    if not keep.any():
        return TemporalInteractionGraph(
            adjacency=sparse.csr_matrix((n_items, n_items), dtype=np.float32),
            item_ids=item_ids,
            item_index=dict(item_index),
            kernel_params=kernel_params,
            n_users_contributed=0,
            n_pairs_generated=0,
        )

    if has_timestamp:
        ts_seconds = (
            interactions["timestamp"].astype("int64").to_numpy() // 10**9
        )
    else:
        # Pure-count fallback: synthesize sequence positions per user as
        # pseudo-timestamps so the per-user sort is deterministic.
        ts_seconds = np.arange(len(interactions), dtype=np.int64)

    sub = pd.DataFrame({
        "entity_id": interactions["entity_id"].to_numpy()[keep],
        "item_idx": item_idx_array[keep],
        "ts": ts_seconds[keep],
    })

    cutoff = kernel_params.cutoff_seconds()
    pair_rows: list[np.ndarray] = []
    pair_cols: list[np.ndarray] = []
    pair_weights: list[np.ndarray] = []
    n_users = 0

    # Group by entity, sort by ts, then per-user pair enumeration.
    for _entity, group in sub.groupby("entity_id", sort=False):
        n_users += 1
        if len(group) > max_history_per_user:
            group = group.nlargest(max_history_per_user, "ts")
        group = group.sort_values("ts")
        items = group["item_idx"].to_numpy(dtype=np.int64)
        ts = group["ts"].to_numpy(dtype=np.float64)
        if items.size < 2:
            continue
        rows, cols, weights = _user_pairs(items, ts, kernel_params, cutoff)
        if rows.size > 0:
            pair_rows.append(rows)
            pair_cols.append(cols)
            pair_weights.append(weights)

    if not pair_rows:
        return TemporalInteractionGraph(
            adjacency=sparse.csr_matrix((n_items, n_items), dtype=np.float32),
            item_ids=item_ids,
            item_index=dict(item_index),
            kernel_params=kernel_params,
            n_users_contributed=n_users,
            n_pairs_generated=0,
        )

    rows = np.concatenate(pair_rows)
    cols = np.concatenate(pair_cols)
    weights = np.concatenate(pair_weights).astype(np.float32)
    n_pairs = rows.size

    # Symmetric: stack with swapped indices.
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_weights = np.concatenate([weights, weights])

    adjacency = sparse.csr_matrix(
        (sym_weights, (sym_rows, sym_cols)),
        shape=(n_items, n_items),
    )
    adjacency.sum_duplicates()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()

    return TemporalInteractionGraph(
        adjacency=adjacency,
        item_ids=item_ids,
        item_index=dict(item_index),
        kernel_params=kernel_params,
        n_users_contributed=n_users,
        n_pairs_generated=n_pairs,
    )


def _user_pairs(
    items: np.ndarray,
    ts: np.ndarray,
    kernel: KernelParams,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate within-window pair contributions for one sorted user history.

    For each event i, walk forward through events j > i while
    ts[j] - ts[i] <= cutoff and emit (i, j, kernel(dt)) tuples. This
    avoids the O(N^2) all-pairs enumeration when most pairs would
    contribute kernel weight below the cutoff anyway.

    For pure-count kernels (cutoff=inf), every pair contributes 1.0;
    we still walk forward to avoid duplicating pairs.
    """
    n = items.size
    if n < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64)

    # Two-pointer walk: for each i, find largest j_max with ts[j_max] - ts[i] <= cutoff.
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    j = 1
    for i in range(n - 1):
        while j < n and ts[j] - ts[i] <= cutoff:
            j += 1
        # j is now first index past the cutoff. Pairs (i, i+1)..(i, j-1) are valid.
        for k in range(i + 1, j):
            a, b = int(items[i]), int(items[k])
            if a == b:
                continue
            w = float(kernel.kernel(np.float64(ts[k] - ts[i])))
            if w >= _KERNEL_CUTOFF:
                rows.append(a)
                cols.append(b)
                weights.append(w)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )
