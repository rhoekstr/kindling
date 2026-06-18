"""Embedding imputation for cold-start item placement.

Cold items carry no co-occurrence evidence, so every interaction channel scores
them zero — structurally unrecommendable. This module predicts a cold item's
*position in cooc-embedding space* from its content features (one vector, not k
grafted edges — flood-free), so the item can be ranked against the user's taste
centroid in the SAME space the warm items already live in.

Mechanism (validated on MovieLens-25M, bench/run_ml25m_lift.py — the first
cost-free cold lift of the enrichment investigation: cold-tier recall 0 ->
~half the warm tier, aggregate NDCG held flat, warm ranking untouched):

    cooc_emb = PPMI-SVD of the warm item-item cooc       (the strong signal)
    W        = ridge map  content_emb -> cooc_emb         (fit on warm items)
    cold     -> predicted position  content_emb(cold) @ W (one vector)
    score    = user_profile (mean owned warm cooc_emb) . item_position  (cosine)

Why this and not edge-grafting: grafting injected k synthetic edges per cold
item into the candidate pool, which floods warm items out of the top-K on
cold-dominated catalogs (amazon-book cratered ~10x). Imputation places *one*
vector and is consumed only by the reserved cold slots — it cannot displace warm
ranking, so the gate is the metadata->cooc mapping floor, not catalog warmth.

The mapping quality (transfer R^2 + cooc-neighbor recovery on a held-out warm
split, bench/run_meta_cooc_map.py) is the validated predictor of whether this
pays: it ranks steam>book matching grafting outcomes, zeros content-orthogonal
catalogs (beauty/book), and tops out ~0.10 even on the richest metadata that
exists (ml-25m tag-genome). Content reconstructs only ~10% of cooc structure at
the ceiling — a weak-but-real cold supplement, gated to where it clears a floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

DIM = 64
WARM_MIN = 5          # an item is "warm" (gets a true cooc embedding) at >= this
N_WARM_MAX = 20_000   # cap the cooc SVD; warmest items by degree win the budget
RIDGE_LAM = 10.0
COLDNESS_GATE = 0.75  # mirrors the cold-slot eligibility gate in recommend()


@dataclass
class ImputeModel:
    """Cold-start placement model, indexed by engine catalog index.

    ``positions`` are L2-comparable cooc-space coordinates: warm rows hold the
    standardized true cooc embedding, cold rows (with content) hold the
    metadata-predicted position, content-less rows are zero (score 0). ``warm``
    marks which rows may contribute to a user's taste centroid.
    """

    positions: np.ndarray   # (n_items, DIM) f32
    warm: np.ndarray        # (n_items,) bool — eligible for the user profile
    r2: float               # held-out transfer R^2 (the gate metric)
    neighbor_recovery: float  # held-out cooc-neighbor recovery@10
    n_warm: int
    dim: int


def ppmi(co: sp.coo_matrix, item_counts: np.ndarray, n_users: int) -> sp.csr_matrix:
    """Positive PMI of a cooc COO; preserves shape, drops non-positive cells."""
    c = co.data.astype(np.float64)
    di, dj = item_counts[co.row], item_counts[co.col]
    pmi = np.log(np.maximum(c * n_users / np.maximum(di * dj, 1.0), 1e-12))
    keep = pmi > 0
    return sp.csr_matrix(
        (pmi[keep], (co.row[keep], co.col[keep])), shape=co.shape
    )


def _truncated_svd(m: sp.spmatrix, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (row embeddings U·S, right factors Vt).

    ``M @ Vt.T == U·S`` because Vt has orthonormal rows, so Vt projects *any*
    row (warm or cold) into the same space the warm rows define.
    """
    dim = min(dim, min(m.shape) - 1)
    u, s, vt = svds(m.asfptype(), k=dim)
    return u * s, vt


def _neighbor_recovery(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Fraction of an item's true top-k cooc neighbors its predicted position
    retrieves — the sharp sub-metric (does the map place items where their real
    neighbors are)."""
    yn = y_true / np.maximum(np.linalg.norm(y_true, axis=1, keepdims=True), 1e-9)
    hn = y_pred / np.maximum(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-9)
    rec = []
    for i in range(min(400, len(y_true))):
        true_nn = np.argpartition(-(yn @ yn[i]), k + 1)[: k + 1]
        pred_nn = np.argpartition(-(yn @ hn[i]), k)[:k]
        rec.append(len(set(true_nn.tolist()) & set(pred_nn.tolist())) / k)
    return float(np.mean(rec)) if rec else 0.0


def fit_impute(
    cooc_data: np.ndarray,
    cooc_indices: np.ndarray,
    cooc_indptr: np.ndarray,
    content_csr: sp.csr_matrix,
    item_counts: np.ndarray,
    n_users: int,
    *,
    n_items: int,
    dim: int = DIM,
    warm_min: int = WARM_MIN,
    n_warm_max: int = N_WARM_MAX,
    ridge_lam: float = RIDGE_LAM,
    seed: int = 0,
) -> ImputeModel:
    """Fit the metadata->cooc-position map and impute cold positions.

    ``cooc_*`` is the RAW (pre-transform) symmetric cooc CSR over the
    ``n_items`` train items. ``content_csr`` is the IDF/L2 item-feature matrix
    over all ``n_items_ext`` catalog rows (extension items included). The
    returned model's ``r2`` is the gate the caller checks before activating.
    """
    n_ext = content_csr.shape[0]
    warm = np.where(item_counts[:n_items] >= warm_min)[0]
    warm = warm[np.argsort(-item_counts[warm])][:n_warm_max]
    warm_mask = np.zeros(n_ext, dtype=bool)
    warm_mask[warm] = True

    # Too few warm items, or no warm content → no usable map. Return an inert
    # model: r2 0 (so "auto" declines) and zero positions (so a forced
    # "impute" scores 0 and falls through to the recency prior).
    if len(warm) < 10 or content_csr[warm].nnz == 0:
        return ImputeModel(
            positions=np.zeros((n_ext, 1), dtype=np.float32),
            warm=warm_mask, r2=0.0, neighbor_recovery=0.0,
            n_warm=len(warm), dim=1,
        )

    # cooc embedding over the warm submatrix (diagonal dropped, like the bench).
    cf = sp.csr_matrix(
        (cooc_data, cooc_indices, cooc_indptr), shape=(n_items, n_items)
    )
    cw = cf[warm][:, warm].tocoo()
    off = cw.row != cw.col
    cw = sp.coo_matrix(
        (cw.data[off], (cw.row[off], cw.col[off])), shape=(len(warm), len(warm))
    )
    cooc_emb, _ = _truncated_svd(ppmi(cw, item_counts[warm], n_users), dim)

    # content embedding in a shared space (warm-derived factors project all
    # rows). Its rank is independent of the cooc rank — thin metadata yields
    # fewer content dims, which is fine: the ridge maps dim_x -> dim_y.
    _, vt = _truncated_svd(content_csr[warm], dim)
    content_emb = np.asarray(content_csr @ vt.T)  # (n_ext, dim_x)
    dim_x = content_emb.shape[1]
    dim_y = cooc_emb.shape[1]

    # standardize both spaces on warm, fit ridge content_emb -> cooc_emb.
    xw = content_emb[warm]
    mx, sx = xw.mean(0), np.maximum(xw.std(0), 1e-9)
    my, sy = cooc_emb.mean(0), np.maximum(cooc_emb.std(0), 1e-9)
    xw_s = (xw - mx) / sx
    yw_s = (cooc_emb - my) / sy

    # gate: 70/30 warm split, fit on train, measure transfer on held-out warm.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(warm))
    cut = int(0.7 * len(warm))
    tr, te = perm[:cut], perm[cut:]
    eye = ridge_lam * np.eye(dim_x)
    w_tr = np.linalg.solve(xw_s[tr].T @ xw_s[tr] + eye, xw_s[tr].T @ yw_s[tr])
    yhat = xw_s[te] @ w_tr
    yte = yw_s[te]
    ss_res = float(((yte - yhat) ** 2).sum())
    ss_tot = float(((yte - yte.mean(0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rec = _neighbor_recovery(yte, yhat)

    # deploy: refit on all warm, impute positions for cold items with content.
    w = np.linalg.solve(xw_s.T @ xw_s + eye, xw_s.T @ yw_s)  # (dim_x, dim_y)
    positions = np.zeros((n_ext, dim_y), dtype=np.float32)
    positions[warm] = yw_s.astype(np.float32)
    has_content = np.diff(content_csr.indptr) > 0
    cold = ~warm_mask & has_content
    if cold.any():
        positions[cold] = (((content_emb[cold] - mx) / sx) @ w).astype(np.float32)

    return ImputeModel(
        positions=positions,
        warm=warm_mask,
        r2=round(r2, 4),
        neighbor_recovery=round(rec, 4),
        n_warm=len(warm),
        dim=dim_y,
    )


def cold_scores(model: ImputeModel, owned: np.ndarray) -> np.ndarray:
    """Full-catalog cooc-space scores for one user (cosine to taste centroid).

    The profile is the mean of the user's owned *warm* item positions; cold
    candidates score by how close their imputed position sits to it.
    """
    n = model.positions.shape[0]
    if owned.size == 0:
        return np.zeros(n, dtype=np.float64)
    warm_owned = owned[model.warm[owned]]
    if warm_owned.size == 0:
        return np.zeros(n, dtype=np.float64)
    profile = model.positions[warm_owned].mean(0)
    return np.asarray(model.positions @ profile, dtype=np.float64)
