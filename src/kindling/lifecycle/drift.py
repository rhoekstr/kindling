"""Structural drift detection (PRD §3.5, plan Phase 6).

Drift measures how much the derived structures change across time
windows. Stable domains show near-zero drift; fast-changing domains show
drift that grows with lag. The drift report feeds into retention
strategy selection.

Two default metrics ship in v1:

1. **Item-graph edge correlation (Spearman)**: rank correlation of
   co-occurrence weights on the shared edge set between two time
   windows. Stable = high correlation (near 1); drifting = low or
   negative.
2. **Community stability (Adjusted Rand Index)**: ARI of Louvain
   community assignments across windows. Stable communities = ARI near
   1; shifting communities = ARI near 0.

Plan-closed PRD gap: the drift threshold bootstrap. The PRD says
"calibrated against held-out data during the first stable retrain" but
doesn't specify how. Phase 6 rule:

- At first retrain, compute ``lag_30d_drift`` as a baseline and record.
- Subsequent retrains flag drift as "concerning" when it exceeds
  ``3x baseline``. The baseline is revisable per retrain - if the
  environment genuinely becomes more dynamic, the threshold widens.

Two optional metrics (path KL, basket JS) live in extended metrics and
are opt-in via the ``extended_metrics`` config field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse, stats

from kindling.graph.item_graph import ItemGraph, build_item_graph

DEFAULT_LAG_BUCKETS_DAYS = (30, 90, 365)
DRIFT_CONCERN_MULTIPLIER = 3.0


@dataclass(frozen=True)
class DriftMetrics:
    """Per-lag drift measurements.

    Attributes
    ----------
    item_graph_drift:
        1 - Spearman correlation of co-occurrence weights across windows.
        0 = no drift, 1 = anti-correlated.
    community_stability:
        Louvain ARI across windows. Phase 6 uses the proxy described in
        ``_community_stability_proxy`` because a full Louvain partition
        requires networkx/igraph; the proxy measures stability of the
        item-neighbor ranking distribution, which correlates with
        community assignments without the extra dependency.
    """

    item_graph_drift: float
    community_stability: float


@dataclass(frozen=True)
class DriftReport:
    """Full drift report (PRD §3.5 surface).

    Attributes
    ----------
    metrics_by_lag:
        ``{lag_days: DriftMetrics}``.
    estimated_retention_horizon_days:
        Longest lag at which drift stays below the concern threshold.
        Approximation - the adaptive retention strategy uses this to
        decide how far back to keep data.
    recommendation_at_risk:
        True when drift at the smallest lag exceeds the concern
        threshold. Signals that the most recent structural updates are
        themselves untrusted.
    interpretation:
        One-line plain-language summary for the power-user surface.
    baseline_lag_30d_drift:
        First-retrain baseline; ``None`` on the first retrain itself.
    """

    metrics_by_lag: dict[int, DriftMetrics]
    estimated_retention_horizon_days: int
    recommendation_at_risk: bool
    interpretation: str
    baseline_lag_30d_drift: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "item_graph_drift": {
                f"lag_{lag}d": m.item_graph_drift
                for lag, m in self.metrics_by_lag.items()
            },
            "community_stability": {
                f"lag_{lag}d": m.community_stability
                for lag, m in self.metrics_by_lag.items()
            },
            "estimated_retention_horizon_days": self.estimated_retention_horizon_days,
            "recommendation_at_risk": self.recommendation_at_risk,
            "interpretation": self.interpretation,
            "baseline_lag_30d_drift": self.baseline_lag_30d_drift,
        }


@dataclass
class DriftTracker:
    """Per-engine drift bookkeeping. Maintains the baseline across
    retrains so the concern threshold can anchor to historical behavior."""

    baseline_lag_30d_drift: float | None = None
    last_report: DriftReport | None = None
    concern_multiplier: float = DRIFT_CONCERN_MULTIPLIER
    lag_buckets_days: tuple[int, ...] = DEFAULT_LAG_BUCKETS_DAYS

    def compute(
        self,
        interactions: pd.DataFrame,
        reference_timestamp: pd.Timestamp | None = None,
    ) -> DriftReport:
        if "timestamp" not in interactions.columns:
            return self._empty_report(
                reason="no timestamp column - cannot compute drift",
            )
        if len(interactions) == 0:
            return self._empty_report(reason="no interactions")

        if reference_timestamp is None:
            reference_timestamp = interactions["timestamp"].max()

        # Build the reference (most-recent 30-day) item graph to compare
        # against each lag window.
        recent_window = _window_of(interactions, reference_timestamp, days_back=30)
        if len(recent_window) == 0:
            return self._empty_report(reason="most-recent 30-day window is empty")
        recent_graph = build_item_graph(recent_window)

        metrics: dict[int, DriftMetrics] = {}
        for lag in self.lag_buckets_days:
            window = _window_of(
                interactions, reference_timestamp, days_back=lag, skip_recent_days=30
            )
            if len(window) == 0:
                metrics[lag] = DriftMetrics(
                    item_graph_drift=0.0, community_stability=1.0
                )
                continue
            prior_graph = build_item_graph(window)
            metrics[lag] = DriftMetrics(
                item_graph_drift=_item_graph_drift(recent_graph, prior_graph),
                community_stability=_community_stability_proxy(
                    recent_graph, prior_graph
                ),
            )

        return self._finalize(metrics)

    def _finalize(self, metrics: dict[int, DriftMetrics]) -> DriftReport:
        lag_30d = metrics.get(30)
        drift_30d = lag_30d.item_graph_drift if lag_30d is not None else 0.0

        # First-retrain baseline is the current measurement; subsequent
        # retrains compare against it.
        is_first_retrain = self.baseline_lag_30d_drift is None
        if is_first_retrain:
            self.baseline_lag_30d_drift = float(drift_30d)
            concerning_threshold = 0.3  # initial-run conservative default
        else:
            baseline = self.baseline_lag_30d_drift or 0.0
            concerning_threshold = max(
                0.05, self.concern_multiplier * float(baseline)
            )

        # Retention horizon: the longest lag at which drift is still
        # under the threshold. Fall back to the smallest lag when all
        # exceed it.
        sorted_lags = sorted(metrics)
        horizon = sorted_lags[0] if sorted_lags else 0
        for lag in sorted_lags:
            if metrics[lag].item_graph_drift <= concerning_threshold:
                horizon = lag
            else:
                break

        at_risk = drift_30d > concerning_threshold
        interpretation = _interpret(drift_30d, horizon, at_risk, is_first_retrain)

        report = DriftReport(
            metrics_by_lag=metrics,
            estimated_retention_horizon_days=horizon,
            recommendation_at_risk=at_risk,
            interpretation=interpretation,
            baseline_lag_30d_drift=self.baseline_lag_30d_drift,
        )
        self.last_report = report
        return report

    def _empty_report(self, reason: str) -> DriftReport:
        return DriftReport(
            metrics_by_lag={},
            estimated_retention_horizon_days=0,
            recommendation_at_risk=False,
            interpretation=f"Drift not computed: {reason}.",
            baseline_lag_30d_drift=self.baseline_lag_30d_drift,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window_of(
    interactions: pd.DataFrame,
    reference: pd.Timestamp,
    days_back: int,
    skip_recent_days: int = 0,
) -> pd.DataFrame:
    """Return interactions in ``[reference - days_back, reference - skip_recent_days]``.

    The recent window (``skip_recent_days=0``) is inclusive of the
    reference timestamp so events landing exactly at the max observed
    time aren't dropped. Older windows (``skip_recent_days > 0``) exclude
    the upper bound so the recent and older windows don't overlap.
    """
    if skip_recent_days == 0:
        upper = reference
        upper_op = interactions["timestamp"] <= upper
    else:
        upper = reference - pd.Timedelta(days=skip_recent_days)
        upper_op = interactions["timestamp"] < upper
    lower = reference - pd.Timedelta(days=days_back)
    mask = (interactions["timestamp"] >= lower) & upper_op
    return interactions[mask]


def _item_graph_drift(a: ItemGraph, b: ItemGraph) -> float:
    """``1 - Spearman correlation`` of edge weights on the shared edge set.
    0 = stable, 1 = maximally different. Returns 0 when there are no
    shared edges or when either input is constant (no correlation
    defined)."""
    shared_indices = _shared_edge_indices(a, b)
    if shared_indices is None:
        return 0.0
    weights_a, weights_b = shared_indices
    if len(weights_a) < 3:
        return 0.0
    # Guard against constant-input warnings from scipy.
    if weights_a.std() == 0 or weights_b.std() == 0:
        return 0.0
    with np.errstate(invalid="ignore"):
        corr, _ = stats.spearmanr(weights_a, weights_b)
    if np.isnan(corr):
        return 0.0
    return float(max(0.0, 1.0 - corr))


def _shared_edge_indices(
    a: ItemGraph, b: ItemGraph
) -> tuple[np.ndarray, np.ndarray] | None:
    """For items appearing in both graphs, return their co-occurrence
    row vectors flattened into comparable float arrays."""
    shared_items = set(a.item_index).intersection(b.item_index)
    if len(shared_items) < 3:
        return None
    shared_sorted = sorted(shared_items, key=str)[:500]
    a_indices = [a.item_index[i] for i in shared_sorted]
    b_indices = [b.item_index[i] for i in shared_sorted]
    # Build the shared-subgraph weights by selecting rows + columns.
    a_sub = a.adjacency[a_indices][:, a_indices]
    b_sub = b.adjacency[b_indices][:, b_indices]
    # Flatten (using COO to compare on shared entries).
    return np.asarray(a_sub.toarray()).ravel(), np.asarray(b_sub.toarray()).ravel()


def _community_stability_proxy(a: ItemGraph, b: ItemGraph) -> float:
    """Proxy for Louvain ARI without the extra dependency. Measures
    stability of each shared item's top-3 neighbors across the two
    graphs. Returns a value in ``[0, 1]`` - 1 = all items keep the same
    top neighbors, 0 = fully disjoint neighbor sets."""
    shared = sorted(set(a.item_index).intersection(b.item_index), key=str)[:200]
    if len(shared) < 3:
        return 1.0
    overlap_sum = 0.0
    valid = 0
    for item in shared:
        a_top = _top_k_neighbors(a.adjacency, a.item_index[item], k=3)
        b_top = _top_k_neighbors(b.adjacency, b.item_index[item], k=3)
        if not a_top or not b_top:
            continue
        # Overlap fraction over top-3 positions.
        # Compare by position index back to the original item id.
        a_items = {a.item_ids[idx] for idx in a_top}
        b_items = {b.item_ids[idx] for idx in b_top}
        if a_items or b_items:
            overlap = len(a_items & b_items) / max(len(a_items | b_items), 1)
            overlap_sum += overlap
            valid += 1
    return float(overlap_sum / valid) if valid else 1.0


def _top_k_neighbors(adj: sparse.csr_matrix, row_idx: int, k: int) -> list[int]:
    row = adj.getrow(row_idx).toarray().ravel()
    if not row.any():
        return []
    top = np.argpartition(-row, min(k, len(row) - 1))[:k]
    top = top[np.argsort(-row[top])]
    mask = row[top] > 0
    return list(top[mask])


def _interpret(
    drift_30d: float,
    horizon_days: int,
    at_risk: bool,
    is_first_retrain: bool,
) -> str:
    if is_first_retrain:
        return (
            f"First retrain - baseline drift recorded at {drift_30d:.3f}. "
            "Subsequent retrains will flag concerning drift at 3x this value."
        )
    if at_risk:
        return (
            f"Recent drift ({drift_30d:.3f}) exceeds the concern threshold. "
            "Data structures may not reflect current domain state; consider "
            "a shorter retention or an immediate retrain."
        )
    return (
        f"Stable structure; 30-day drift {drift_30d:.3f}. Estimated "
        f"retention horizon: {horizon_days} days."
    )
