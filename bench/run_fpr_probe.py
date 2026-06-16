"""Phase-0 viability probe for the Force Projection Recommender (FPR).

Single question, cheapest decisive form:

    Does kNN in a force-directed layout of the item-item co-occurrence graph
    beat kNN on the sparse graph *directly* (the B1 bar) on ml1m top-K accuracy?

Design discipline — isolate exactly one decision (§6.2 projection vs none):
every arm shares the SAME item-item cosine similarity graph. We fit the library
``ItemItemKNN`` baseline (that IS the B1 "operate on the graph directly" arm) and
then lay out *its* similarity matrix with a force-directed (Fruchterman-Reingold
spring-electrical) algorithm. The only thing that varies between B1 and the FPR
arms is whether scoring happens in graph-space (sum of edge similarity to seen
items) or in the force-layout coordinate space (distance to seen items). Same
split, same eval users, same metric aggregation as the rest of the kindling
scoreboard, so the result is directly comparable to the 0.284 ml1m bests.

Pre-registered decision rule (fixed before the first run):
    FPR "shows life" iff  fpr_32d_nearest NDCG@10  >=  graph_knn (B1) NDCG@10.
    If FPR is materially below B1 (< 0.95x), the top-K-accuracy thesis for force
    projection is dead and the full PRD benchmark is not worth standing up.

Run:
    .venv/bin/python bench/run_fpr_probe.py
Env overrides: FPR_ITERS (layout iterations), FPR_DIMS (csv), FPR_EVAL (max eval
entities), FPR_KNEIGH (graph top-k), FPR_SEED.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from kindling.benchmarks.baselines import ItemItemKNN, PopularityBaseline
from kindling.benchmarks.metrics import aggregate
from kindling.loaders import movielens

REPORT = Path(__file__).parent / "reports" / "fpr_probe_ml1m.json"


# --------------------------------------------------------------------------- #
# Force-directed layout (Fruchterman-Reingold / spring-electrical, §6.2).
# Memory-safe at high dim: the per-step displacement is computed in the factored
# form  disp_i = x_i * sum_j coef_ij - (coef @ x)_i,  so we never materialize the
# (n, n, dim) delta tensor (that would be ~3.8 GB at n=3883, dim=32 and risk an
# OOM kill). Working set is two dense (n, n) matrices ~120 MB each.
# --------------------------------------------------------------------------- #
def fr_layout(
    adj: sp.csr_matrix,
    dim: int,
    seed: int,
    iters: int = 80,
    t0: float = 0.1,
) -> np.ndarray:
    n = adj.shape[0]
    A = np.asarray(adj.todense(), dtype=np.float64)
    A = np.maximum(A, A.T)  # symmetrize the top-k graph
    np.fill_diagonal(A, 0.0)

    rng = np.random.default_rng(seed)
    x = rng.random((n, dim))
    k = np.sqrt(1.0 / n)  # ideal edge length
    eps = 1e-9

    for it in range(iters):
        gram = x @ x.T
        sq = np.diag(gram)
        d2 = sq[:, None] + sq[None, :] - 2.0 * gram
        np.maximum(d2, eps, out=d2)
        d = np.sqrt(d2)
        # coef_ij = repulsion (k^2/d^2) - attraction (A_ij * d / k)
        coef = (k * k) / d2 - A * d / k
        np.fill_diagonal(coef, 0.0)
        disp = x * coef.sum(axis=1)[:, None] - coef @ x
        length = np.sqrt((disp * disp).sum(axis=1))
        np.maximum(length, 1e-3, out=length)
        t = t0 * (1.0 - it / iters)  # linear cooling
        x += (disp / length[:, None]) * t
    return x


# --------------------------------------------------------------------------- #
# FPR recommender: score unseen items by proximity in the layout to the user's
# seen items. Shares the index maps / user-item matrix of the fitted B1 graph.
# --------------------------------------------------------------------------- #
class ForceProjection:
    def __init__(
        self,
        name: str,
        coords: np.ndarray,
        ix_item: np.ndarray,
        user_items: sp.csr_matrix,
        entity_ix: dict[object, int],
        mode: str = "nearest",  # 'nearest' (nearest_seen) | 'centroid'
    ) -> None:
        self.name = name
        self.coords = coords
        self.ix_item = ix_item
        self.user_items = user_items
        self.entity_ix = entity_ix
        self.mode = mode
        self._cc = (coords * coords).sum(axis=1)  # ||item||^2, precomputed

    def recommend(self, entity_id: object, n: int = 10) -> list[object]:
        uid = self.entity_ix.get(entity_id)
        if uid is None:
            return []
        owned = self.user_items.getrow(uid).indices
        if len(owned) == 0:
            return []
        O = self.coords[owned]
        if self.mode == "nearest":
            cross = self.coords @ O.T  # (n_items, n_owned)
            d2 = self._cc[:, None] + (O * O).sum(axis=1)[None, :] - 2.0 * cross
            score = -d2.min(axis=1)  # closeness to nearest seen item
        else:  # centroid
            cen = O.mean(axis=0)
            score = -((self.coords - cen) ** 2).sum(axis=1)
        score[owned] = -np.inf
        if n >= len(score):
            order = np.argsort(-score)
        else:
            part = np.argpartition(-score, n)[:n]
            order = part[np.argsort(-score[part])]
        return [self.ix_item[i] for i in order[:n] if np.isfinite(score[i])]


# --------------------------------------------------------------------------- #
def evaluate(rec, eval_entities, train_items, test_items, catalog_size, k=10):
    per_entity = []
    t0 = time.perf_counter()
    for e in eval_entities:
        relevant = test_items.get(e, set()) - train_items.get(e, set())
        recs = rec.recommend(e, n=k)
        per_entity.append((recs, relevant))
    secs = time.perf_counter() - t0
    return aggregate(per_entity, catalog_size=catalog_size, k=k), secs


def main() -> int:
    iters = int(os.environ.get("FPR_ITERS", "80"))
    dims = [int(d) for d in os.environ.get("FPR_DIMS", "2,32").split(",")]
    max_eval = int(os.environ.get("FPR_EVAL", "2000"))
    k_neigh = int(os.environ.get("FPR_KNEIGH", "200"))
    seed = int(os.environ.get("FPR_SEED", "0"))
    k = 10

    print(f"[load] ml1m chronological split (test_fraction=0.1)")
    split = movielens.load_1m(test_fraction=0.1)
    train, test = split.train, split.test

    train_items = {
        e: set(g["item_id"].tolist())
        for e, g in train.groupby("entity_id", sort=False)
    }
    test_items = {
        e: set(g["item_id"].tolist())
        for e, g in test.groupby("entity_id", sort=False)
    }
    eval_entities = sorted(set(train_items) & set(test_items))
    if len(eval_entities) > max_eval:
        step = len(eval_entities) // max_eval
        eval_entities = eval_entities[::step][:max_eval]
    print(f"[eval] {len(eval_entities)} entities, k={k}")

    # B1 == operate on the graph directly. Its fitted similarity matrix is the
    # shared graph every FPR arm is laid out from.
    print(f"[fit ] ItemItemKNN (B1), k_neighbors={k_neigh}")
    t0 = time.perf_counter()
    b1 = ItemItemKNN(k_neighbors=k_neigh).fit(train)
    b1_fit = time.perf_counter() - t0
    catalog_size = len(b1._ix_item)

    sim = b1._item_sim  # (n_items, n_items) top-k cosine graph
    n_comp, _ = connected_components(sp.csr_matrix(sim) + sp.csr_matrix(sim).T, directed=False)
    print(f"[graph] n_items={catalog_size}  nnz={sim.nnz}  connected_components={n_comp}")

    pop = PopularityBaseline().fit(train)

    arms: list[tuple[str, object, float]] = []
    arms.append(("popularity", pop, 0.0))
    arms.append(("graph_knn_B1", b1, b1_fit))

    for d in dims:
        t0 = time.perf_counter()
        coords = fr_layout(sim, dim=d, seed=seed, iters=iters)
        lay_secs = time.perf_counter() - t0
        print(f"[layout] dim={d} iters={iters} -> {lay_secs:.1f}s")
        arms.append((
            f"fpr_{d}d_nearest",
            ForceProjection(f"fpr_{d}d_nearest", coords, b1._ix_item,
                            b1._user_items, b1._entity_ix, mode="nearest"),
            lay_secs,
        ))
        if d == max(dims):
            arms.append((
                f"fpr_{d}d_centroid",
                ForceProjection(f"fpr_{d}d_centroid", coords, b1._ix_item,
                                b1._user_items, b1._entity_ix, mode="centroid"),
                lay_secs,
            ))

    rows = []
    for name, rec, build_secs in arms:
        report, eval_secs = evaluate(
            rec, eval_entities, train_items, test_items, catalog_size, k=k
        )
        m = report.as_dict()
        rows.append({
            "arm": name,
            "ndcg@10": round(m["ndcg_at_k"], 4),
            "recall@10": round(m["recall_at_k"], 4),
            "mrr": round(m["mrr"], 4),
            "hit@10": round(m["hit_rate"], 4),
            "coverage": round(m["coverage"], 4),
            "build_s": round(build_secs, 1),
            "eval_s": round(eval_secs, 1),
        })
        print(f"  {name:20s} ndcg={m['ndcg_at_k']:.4f} recall={m['recall_at_k']:.4f} "
              f"mrr={m['mrr']:.4f} cov={m['coverage']:.3f}")

    # Pre-registered verdict.
    b1_ndcg = next(r["ndcg@10"] for r in rows if r["arm"] == "graph_knn_B1")
    fpr_best_d = max(dims)
    fpr_ndcg = next(r["ndcg@10"] for r in rows if r["arm"] == f"fpr_{fpr_best_d}d_nearest")
    ratio = fpr_ndcg / b1_ndcg if b1_ndcg else 0.0
    if fpr_ndcg >= b1_ndcg:
        verdict = "ALIVE: FPR >= B1 — escalate to Phase 1 (FPR as a kindling channel)."
    elif ratio >= 0.95:
        verdict = f"MARGINAL: FPR at {ratio:.2f}x B1 — close but not winning; weigh cost."
    else:
        verdict = f"DEAD: FPR at {ratio:.2f}x B1 — force projection loses on top-K accuracy."

    print("\n=== VERDICT (pre-registered) ===")
    print(f"  B1 NDCG@10={b1_ndcg:.4f}  FPR_{fpr_best_d}d NDCG@10={fpr_ndcg:.4f}  ratio={ratio:.3f}")
    print(f"  {verdict}")

    out = {
        "dataset": "movielens-1m",
        "protocol": "global-chronological test_fraction=0.1, nearest_seen scoring",
        "k": k,
        "n_eval_entities": len(eval_entities),
        "graph": {"n_items": int(catalog_size), "nnz": int(sim.nnz),
                  "connected_components": int(n_comp), "k_neighbors": k_neigh},
        "layout": {"method": "fruchterman_reingold", "iters": iters, "seed": seed},
        "arms": rows,
        "verdict": {"b1_ndcg": b1_ndcg, "fpr_ndcg": fpr_ndcg,
                    "ratio": round(ratio, 3), "decision": verdict},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[wrote] {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
