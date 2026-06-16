"""Diagnostic: WHY does the force-directed projection kill retrieval?

B1 ranks by exact graph-neighbor weights. FPR ranks by euclidean proximity in
the layout. So the accuracy gap should equal the *neighbor-fidelity* gap: how
well does the layout preserve (a) who each item's true neighbors are and (b)
their order. We measure that directly, plus the two structural causes:
the spring's ideal-edge-length dynamic-range compression, and the all-pairs
repulsion's anti-popularity geometry.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from kindling.benchmarks.baselines import ItemItemKNN
from kindling.loaders import movielens
from run_fpr_probe import fr_layout

split = movielens.load_1m(test_fraction=0.1)
train = split.train
knn = ItemItemKNN(k_neighbors=200).fit(train)
sim = knn._item_sim  # csr item x item, top-200 cosine
n = sim.shape[0]
print(f"n_items={n} nnz={sim.nnz}")

coords = fr_layout(sim, dim=32, seed=0, iters=300)
cc = (coords * coords).sum(1)
D2 = cc[:, None] + cc[None, :] - 2.0 * (coords @ coords.T)
np.maximum(D2, 0.0, out=D2)
Dlay = np.sqrt(D2)
np.fill_diagonal(Dlay, np.inf)

# item popularity by matrix index
pop_by_item = train["item_id"].value_counts()
idx_pop = np.zeros(n)
for it, ix in knn._item_ix.items():
    idx_pop[ix] = pop_by_item.get(it, 0)

rng = np.random.default_rng(0)
sample = rng.choice(n, size=min(800, n), replace=False)

rec_at_10, rec_at_50, spear_edge = [], [], []
for i in sample:
    row = sim.getrow(i)
    if row.nnz < 10:
        continue
    cols, vals = row.indices, row.data
    order = np.argsort(-vals)
    g10 = set(cols[order[:10]].tolist())
    g50 = set(cols[order[: min(50, len(order))]].tolist())
    l10 = set(np.argpartition(Dlay[i], 10)[:10].tolist())
    l50 = set(np.argpartition(Dlay[i], 50)[:50].tolist())
    rec_at_10.append(len(g10 & l10) / 10)
    rec_at_50.append(len(g50 & l50) / min(50, len(order)))
    if row.nnz >= 5:
        rho, _ = spearmanr(vals, -Dlay[i, cols])
        if not np.isnan(rho):
            spear_edge.append(rho)

# dynamic-range compression: layout distance of strong vs weak edges
coo = sp.triu(sim).tocoo()
w = coo.data
elay = Dlay[coo.row, coo.col]
q = np.quantile(w, [0.1, 0.9])
weak = elay[w <= q[0]]
strong = elay[w >= q[1]]

# anti-popularity geometry: do popular items get pushed to the periphery?
cen = coords.mean(0)
dist_cen = np.sqrt(((coords - cen) ** 2).sum(1))
rho_pop, _ = spearmanr(idx_pop, dist_cen)

rand_recovery = 10 / n
print("\n--- neighbor fidelity (B1 uses these at 1.0 by construction) ---")
print(f"recovery@10 graph->layout : {np.mean(rec_at_10):.3f}  (random baseline {rand_recovery:.4f})")
print(f"recovery@50 graph->layout : {np.mean(rec_at_50):.3f}")
print(f"edge-order Spearman(sim, -dist) over true neighbors: {np.mean(spear_edge):.3f}")
print("\n--- dynamic-range compression (ideal-edge-length K flattens weights) ---")
print(f"layout dist  weak edges (bottom 10% weight): mean {weak.mean():.3f}  median {np.median(weak):.3f}")
print(f"layout dist strong edges (top 10% weight)  : mean {strong.mean():.3f}  median {np.median(strong):.3f}")
print(f"strong/weak distance ratio: {strong.mean() / weak.mean():.3f}  (1.0 = no separation)")
print("\n--- anti-popularity geometry (all-pairs repulsion) ---")
print(f"Spearman(popularity, distance-to-centroid): {rho_pop:.3f}  (>0 = popular pushed outward)")
