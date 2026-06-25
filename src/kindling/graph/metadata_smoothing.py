"""Metadata smoothing of the base item-item matrix.

Augments the active base scorer (wilson-cooc CSR or the EASE matrix) with a
metadata-kNN graph whose edge weights are the *fitted prediction* of the base's
own item-item weight from metadata similarity (``base_weight ~ metadata_sim``).

Two properties make this safe to leave on:

* **Self-scaling** — the dose is fit against the active base's own weights, so
  it lands on that base's scale automatically (no cooc-vs-EASE rescaling knob).
* **Self-gating** — when metadata doesn't predict the base, the fitted slope
  collapses to ~0 and the imputed weights vanish, so a useless-metadata catalog
  is a near no-op without an explicit alive/dead gate.

Validated at the cooc-base level in ``bench/run_graft_revisit.py``: all-items
smoothing lifts cold-tier recall and overall NDCG on rich-metadata catalogs
(H&M +11%, steam +3%) and self-disables on shuffled (dead) metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp


def _knn_edges(
    features: sp.csr_matrix, topk: int, block: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed top-k metadata neighbours per item, by sparse feature dot product.

    Returns ``(ei, ej, esim)``: for each item with metadata, its ``topk`` most
    metadata-similar items (self excluded, positive similarity only).

    Uses the Rust core (``kindling_core.metadata_knn`` — inverted-index, rayon
    parallel) when available so full catalogs run without subsampling; falls
    back to the pure-NumPy block matmul otherwise.
    """
    from kindling._native import CORE_AVAILABLE, kindling_core

    if CORE_AVAILABLE and hasattr(kindling_core, "metadata_knn"):
        ei_r, ej_r, es_r = kindling_core.metadata_knn(
            np.ascontiguousarray(features.data, dtype=np.float32),
            np.ascontiguousarray(features.indices, dtype=np.int32),
            np.ascontiguousarray(features.indptr, dtype=np.int32),
            int(features.shape[1]),
            int(topk),
            0,
        )
        return (
            np.asarray(ei_r, dtype=np.int64),
            np.asarray(ej_r, dtype=np.int64),
            np.asarray(es_r, dtype=np.float64),
        )

    has = np.diff(features.indptr) > 0
    items = np.where(has)[0]
    if items.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)
    fa = features[items]
    ei: list[np.ndarray] = []
    ej: list[np.ndarray] = []
    es: list[np.ndarray] = []
    for s in range(0, len(items), block):
        pos = np.arange(s, min(s + block, len(items)))
        sims = np.asarray((features[items[pos]] @ fa.T).todense())
        for bi, gpos in enumerate(pos):
            row = sims[bi]
            row[gpos] = -1.0  # exclude self
            k = min(topk, len(row) - 1)
            if k <= 0:
                continue
            top = np.argpartition(-row, k)[:k]
            top = top[row[top] > 0]
            if top.size == 0:
                continue
            ei.append(np.full(top.size, int(items[gpos]), dtype=np.int64))
            ej.append(items[top].astype(np.int64))
            es.append(row[top])
    if not ei:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)
    return np.concatenate(ei), np.concatenate(ej), np.concatenate(es).astype(np.float64)


def _r2(target: np.ndarray, pred: np.ndarray) -> float:
    ss = float(((target - target.mean()) ** 2).sum())
    return 1.0 - float(((target - pred) ** 2).sum()) / ss if ss > 0 else 0.0


def _fit_dose(
    sim: np.ndarray, target: np.ndarray, family: str
) -> tuple[Callable[[np.ndarray], np.ndarray], float, float]:
    """Fit ``base_weight ~ metadata_sim`` and return ``(predict_fn, slope, r2)``.

    ``family`` selects the link, matched to the data-generating process:
      * ``ols``      — linear (additive), clipped >= 0.
      * ``poisson``  — exp link; for repeat/count co-occurrence (unbounded).
      * ``logistic`` — sigmoid saturating at the observed max; for no-repeat
        binary implicit data where co-occurrence is a bounded proportion.

    The slope is the self-gating signal: ~0 ⇒ metadata doesn't predict the base
    ⇒ the dose vanishes.
    """

    def _zero(s: np.ndarray) -> np.ndarray:
        return np.zeros_like(s)

    if sim.size < 2 or float(np.ptp(sim)) == 0.0 or float(np.ptp(target)) == 0.0:
        return _zero, 0.0, 0.0

    if family == "poisson":
        from sklearn.linear_model import PoissonRegressor

        # Poisson requires a non-negative target; the EASE base has signed
        # entries, so clip (negatives are not 'counts' anyway).
        y = np.clip(target, 0.0, None)
        if float(np.ptp(y)) == 0.0:
            return _zero, 0.0, 0.0
        pr = PoissonRegressor(alpha=1e-6, max_iter=500).fit(sim.reshape(-1, 1), y)

        def _pois(s: np.ndarray) -> np.ndarray:
            return np.asarray(pr.predict(s.reshape(-1, 1)), dtype=np.float64)

        return _pois, float(pr.coef_[0]), _r2(target, _pois(sim))

    if family == "logistic":
        # Sigmoid saturating at the observed ceiling: model the cooc weight as a
        # bounded proportion. Fit logit(p) ~ sim by OLS on the logit-transformed
        # (normalized) target — cheap, captures saturation, can't over-predict.
        mx = float(target.max())
        if mx <= 0.0:
            return _zero, 0.0, 0.0
        p = np.clip(target / mx, 1e-4, 1.0 - 1e-4)
        b1, b0 = np.polyfit(sim, np.log(p / (1.0 - p)), 1)

        def _logit(s: np.ndarray) -> np.ndarray:
            return np.asarray(mx / (1.0 + np.exp(-(b0 + b1 * s))), dtype=np.float64)

        return _logit, float(b1), _r2(target, _logit(sim))

    # ols (default)
    b1, b0 = np.polyfit(sim, target, 1)

    def _ols(s: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip(b0 + b1 * s, 0.0, None), dtype=np.float64)

    return _ols, float(b1), _r2(target, _ols(sim))


def resolve_family(family: str, *, is_repeat: bool) -> str:
    """``auto`` ⇒ logistic always.

    The fit target is a bounded quantity (the wilson-normalized cooc weight, or
    the signed EASE entry), not a raw count — so the saturating sigmoid is the
    universal link. It is also the cheapest (a polyfit on the logit, no sklearn)
    and the only family valid on the signed EASE base. We collapse repeat/count
    co-occurrence to its binary presence rather than switching to Poisson.
    """
    return family if family != "auto" else "logistic"


def smoothing_graph(
    features: sp.csr_matrix,
    base_value_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_items: int,
    *,
    topk: int = 20,
    family: str = "auto",
    is_repeat: bool = False,
    cap: float | None = None,
    base_max: float | None = None,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Build the metadata-smoothing addend ``M`` for the active base.

    ``base_value_fn(ei, ej)`` returns the active base matrix's values for the
    candidate edges (0 where the pair is absent). The fit of
    ``base_value ~ metadata_sim`` always supplies the **gate** (its slope sign).

    Two dose modes:
      * ``cap`` set → edge weight = ``sim · cap · base_max`` (a *fixed* fraction
        of the base's max edge). This is the effective dose — the grounded
        prediction below systematically *under-doses* because E[cooc|sim] is
        tiny, so a fixed cap (~0.05–0.1) is the default.
      * ``cap`` None → edge weight = the fitted prediction (``family`` link).
        Self-scaling but under-dosing; kept for diagnostics.
    """
    fam = resolve_family(family, is_repeat=is_repeat)
    ei, ej, esim = _knn_edges(features, topk)
    if ei.size == 0:
        return sp.csr_matrix((n_items, n_items)), {"applied": False, "reason": "no_metadata_edges"}
    base_vals = np.asarray(base_value_fn(ei, ej), dtype=np.float64).ravel()
    predict, slope, fit_r2 = _fit_dose(esim, base_vals, fam)
    if cap is not None and cap > 0.0:
        bmax = base_max if base_max is not None else float(np.max(base_vals))
        w = esim * cap * bmax
        dose = f"cap={cap}"
    else:
        w = predict(esim)
        dose = f"predicted/{fam}"
    # The gate IS the fit: only smooth when metadata *positively* predicts the
    # base (slope > 0). On dead/anti-correlated metadata the slope collapses or
    # flips, and this is a clean no-op — no separate alive/dead threshold.
    applied = bool(slope > 0.0 and w.sum() > 0.0)
    info: dict[str, Any] = {
        "applied": applied,
        "dose": dose,
        "family": fam,
        "edges": int(ei.size),
        "slope": round(slope, 6),
        "fit_r2": round(fit_r2, 4),
        "mean_weight": round(float(w.mean()), 6),
        "frac_base_zero": round(float((base_vals == 0).mean()), 3),
    }
    if not info["applied"]:
        return sp.csr_matrix((n_items, n_items)), info
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    data = np.concatenate([w, w]).astype(np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items)), info
