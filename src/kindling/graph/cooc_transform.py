"""Popularity-normalizing transforms for the cooccurrence base scorer.

The raw cooc base (row-sum of co-counts) degenerates toward a popularity ranker
on large catalogs (>20k items), which is the *only* regime that reaches this
path — at <=20k the engine uses EASE, whose O(n^3) inversion subtracts the same
popularity/redundancy structure. Empirically on amazon-book-academic (91k, the
canonical large sparse catalog), replacing raw co-counts with a popularity-
normalized weight lifts NDCG@20 by +68% (0.0286 -> 0.0482, beating published
LightGCN / Mult-VAE), reproducing Dacrema 2019.

These operate on the symmetric cooc CSR (data/indices/indptr) plus per-item
marginals (distinct-user counts; == np.bincount(item_idx) for a deduplicated
log). They are deliberately NOT applied on the EASE path: cosine-as-base was
measured to over-suppress popular items and crash hit/recall/NDCG on
amazon-beauty (see the KindlingV2State base-layer comment). `wilson` is the
default because it was the book winner AND is the safest against that failure
mode — it shrinks only low-confidence rare edges while leaving high-count
(typically popular) edges near their raw conditional probability.
"""

from __future__ import annotations

import numpy as np

TRANSFORMS = ("raw", "cosine", "jaccard", "wilson")
_AUTO = "wilson"


def resolve_cooc_transform(name: str) -> str:
    """Map ``"auto"`` to the evidence-backed default; pass others through."""
    resolved = _AUTO if name == "auto" else name
    if resolved not in TRANSFORMS:
        raise ValueError(f"unknown cooc_base_transform: {name!r}; expected auto/{TRANSFORMS}")
    return resolved


def apply_cooc_transform(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    item_counts: np.ndarray,
    n_users: int,
    transform: str,
    wilson_z: float = 1.96,
) -> np.ndarray:
    """Return a new ``data`` array for the cooc CSR under ``transform``.

    ``item_counts[i]`` is item i's marginal (distinct users). Pure function;
    shape and sparsity pattern are preserved (transforms only rescale weights).
    """
    transform = resolve_cooc_transform(transform)
    if transform == "raw":
        return data
    c = data.astype(np.float64)
    rows = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    # Floor marginals at 1: every item in the cooc appeared in item_idx, but a
    # weighted/temporal kernel can still leave a marginal below 1.
    di = np.maximum(item_counts.astype(np.float64)[rows], 1.0)
    dj = np.maximum(item_counts.astype(np.float64)[indices], 1.0)

    if transform == "cosine":
        out = c / np.sqrt(di * dj)
    elif transform == "jaccard":
        out = c / np.maximum(di + dj - c, 1.0)
    elif transform == "wilson":
        z2 = wilson_z * wilson_z

        def lb(phat: np.ndarray, n: np.ndarray) -> np.ndarray:
            # phat is a conditional probability; weighted/multi-session co-counts
            # can push c/d past 1, so clip before the variance sqrt.
            phat = np.clip(phat, 0.0, 1.0)
            bound = (
                phat + z2 / (2 * n) - wilson_z * np.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
            ) / (1 + z2 / n)
            return np.asarray(bound, dtype=np.float64)

        out = np.minimum(lb(c / di, di), lb(c / dj, dj))
    else:  # pragma: no cover - resolve_cooc_transform guards this
        raise ValueError(f"unknown cooc transform: {transform!r}")

    return np.asarray(out, dtype=data.dtype)
