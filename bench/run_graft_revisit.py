"""Re-run metadata->cooc edge grafting on the CLEAN engine, on production
datasets with *rich* cold metadata (H&M) — the regime grafting never got tested in.

Grafting (Part III fence post, originally DEAD on thin book categories,
marginal-alive on steam tags) injects, for each cold item (train count <= T,
has metadata), symmetric cooc edges to its top-k *metadata-similar warm* items
on the observed wilson-cooc scale, so cold items are reachable through the same
closed-form base. Two questions, faithful to the 2026-06 probe:

  1. GATE — does metadata similarity predict cooc structure on held-out warm
     pairs?  ALIVE iff Spearman(meta_sim, cooc) >= 0.20 or R2 >= 0.10.
     (book 0.011 DEAD; steam ~marginal). H&M's rich readable metadata is the
     new test the gate was built to screen for.
  2. GRAFT — does grafting lift cold-tier recall, and at what cost to overall
     NDCG?  Arms: no_graft / naive (uniform wt) / calibrated (wt = meta_sim).

Run:  GRAFT_DATASETS=steam,h-and-m .venv/bin/python bench/run_graft_revisit.py
Env:  GRAFT_DATASETS, GRAFT_GATE_ONLY=1, GRAFT_SUBSAMPLE, GRAFT_TOPK, GRAFT_CAP,
      GRAFT_PCTILE (csv), HM_START.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from kindling.graph.cooc_transform import apply_cooc_transform
from kindling.item_features import ItemFeatureExtractor

REPORT = Path(__file__).parent / "reports" / "graft_revisit.json"
K = 20
GATE_R2, GATE_SPEARMAN = 0.10, 0.20
TIERS = {"1-2": (1, 3), "3-5": (3, 6), "6-10": (6, 11), "11-20": (11, 21), "21+": (21, 10**9)}

# H&M readable metadata: categoricals + a free-text description. Rich cold signal.
_HM_META = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]


def load(name):
    """Return (train, test, items) — items has item_id + metadata columns."""
    if name == "steam":
        from kindling.loaders.steam import load_steam

        return load_steam(test_fraction=0.1)
    if name in ("h-and-m", "hm"):
        hm = Path(
            "~/.cache/kagglehub/competitions/h-and-m-personalized-fashion-recommendations"
        ).expanduser()
        start = os.environ.get("HM_START", "2020-06-01")
        tx = pd.read_csv(
            hm / "transactions_train.csv",
            usecols=["t_dat", "customer_id", "article_id"],
            parse_dates=["t_dat"],
        )
        tx = tx[tx["t_dat"] >= pd.Timestamp(start)].rename(
            columns={"customer_id": "entity_id", "article_id": "item_id", "t_dat": "timestamp"}
        )
        cut = tx["timestamp"].quantile(0.9)
        train, test = tx[tx["timestamp"] <= cut].copy(), tx[tx["timestamp"] > cut].copy()
        items = pd.read_csv(hm / "articles.csv", usecols=["article_id", *_HM_META]).rename(
            columns={"article_id": "item_id"}
        )
        return train, test, items
    if name == "retailrocket":
        from kindling.benchmarks.comparison import _load_dataset

        split = _load_dataset("retailrocket", 0.1)
        return split.train, split.test, split.items
    if name == "amazon-book-chrono":
        # The original DEAD case (thin categories, R2~0) — the self-gating control.
        from kindling.loaders.amazon_chrono import load_amazon_chrono, load_amazon_meta

        book = Path("~/.cache/kindling/amazon-book").expanduser()
        train, test = load_amazon_chrono(
            book / "reviews_Books_5.json.gz", cache_dir=book, test_fraction=0.1
        )
        items = load_amazon_meta(
            book / "meta_Books.json.gz",
            cache_dir=book,
            catalog=set(train["item_id"].unique()),
            extension_top_n=0,
        )
        return train, test, items
    raise ValueError(name)


def _features(items, item_to_idx, n_items):
    feat = ItemFeatureExtractor().fit_transform(items, item_to_idx, n_items)
    if feat.n_features == 0:
        return None
    F = sp.csr_matrix((feat.data, feat.indices, feat.indptr), shape=(n_items, feat.n_features))
    if os.environ.get("GRAFT_SHUFFLE_META", "") == "1":
        # Destroy the item<->metadata link (keep the feature distribution) — a
        # synthetic DEAD dataset to test whether grounded smoothing self-gates.
        perm = np.random.default_rng(0).permutation(n_items)
        F = F[perm]
    return F


def _indptr_from_rows(rows, n):
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, rows + 1, 1)
    return np.cumsum(indptr).astype(np.int32)


def _r2(y, yhat):
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss if ss > 0 else 0.0


def effect(xtr, ytr, xte, yte):
    sp_ = float(spearmanr(xte, yte).statistic)
    b1, b0 = np.polyfit(xtr, ytr, 1)
    r2_lin = _r2(yte, b0 + b1 * xte)
    qs = np.quantile(xtr, np.linspace(0, 1, 11))
    edges = qs[1:-1]
    btr = np.digitize(xtr, edges)
    means = np.array([ytr[btr == b].mean() if (btr == b).any() else ytr.mean() for b in range(10)])
    r2_np = _r2(yte, means[np.digitize(xte, edges)])
    return {"spearman": round(sp_, 3), "r2_linear": round(r2_lin, 3), "r2_binned": round(r2_np, 3)}


def gate(name, train, items, n_warm=4000, max_pairs=80000, seed=0):
    """Does metadata cosine predict cooc weight on held-out warm pairs?"""
    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    n_items = len(item_ids)
    iidx = train["item_id"].map(item_to_idx).to_numpy()
    uidx = pd.factorize(train["entity_id"])[0]
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    F = _features(items, item_to_idx, n_items)
    if F is None:
        return {"dataset": name, "alive": False, "note": "no metadata features"}
    has = np.diff(F.indptr) > 0
    coverage = float(has.mean())
    order = np.argsort(-d)
    warm = order[has[order]][:n_warm]

    S = sp.csr_matrix(
        (np.ones(len(iidx), np.float32), (uidx, iidx)), shape=(int(uidx.max()) + 1, n_items)
    )
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    Ssub = S[:, warm]
    C = (Ssub.T @ Ssub).tocoo()
    m = C.row < C.col
    r, c, co = C.row[m], C.col[m], C.data[m].astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(co) > max_pairs:
        sel = rng.choice(len(co), max_pairs, replace=False)
        r, c, co = r[sel], c[sel], co[sel]
    gi, gj = warm[r], warm[c]
    sim = np.asarray(F[gi].multiply(F[gj]).sum(axis=1)).ravel()
    di, dj = d[gi], d[gj]
    cos_w = co / np.sqrt(di * dj)

    ntr = int(0.7 * len(sim))
    perm = rng.permutation(len(sim))
    tr, te = perm[:ntr], perm[ntr:]
    targets = {
        "cosine_weight": effect(sim[tr], cos_w[tr], sim[te], cos_w[te]),
        "raw_cocount": effect(sim[tr], co[tr], sim[te], co[te]),
    }
    aff = targets["cosine_weight"]
    alive = max(aff["r2_linear"], aff["r2_binned"]) >= GATE_R2 or aff["spearman"] >= GATE_SPEARMAN
    out = {
        "dataset": name,
        "n_items": int(n_items),
        "n_warm_with_meta": int(len(warm)),
        "metadata_coverage": round(coverage, 3),
        "n_features": int(F.shape[1]),
        "n_pairs": int(len(sim)),
        "frac_pairs_sim_gt0": round(float((sim > 0).mean()), 3),
        "targets": targets,
        "alive": bool(alive),
    }
    print(f"\n##### GATE {name} #####")
    print(
        f"  n_items={n_items} warm_w/meta={len(warm)} coverage={coverage:.2f} "
        f"feats={F.shape[1]} pairs={len(sim)} sim>0={out['frac_pairs_sim_gt0']:.2f}"
    )
    for t, v in targets.items():
        print(
            f"  {t:14s} spearman={v['spearman']:+.3f}  "
            f"R2_lin={v['r2_linear']:+.3f}  R2_binned={v['r2_binned']:+.3f}"
        )
    print(f"  GATE (rho>={GATE_SPEARMAN} or R2>={GATE_R2}): {'ALIVE' if alive else 'DEAD'}")
    return out


def build(train, test, items, subsample=None):
    if subsample and train["entity_id"].nunique() > subsample:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(train["entity_id"].unique(), subsample, replace=False).tolist())
        train = train[train["entity_id"].isin(keep)]
        test = test[test["entity_id"].isin(keep)]
        print(f"  subsampled to {subsample} users -> {train['item_id'].nunique()} items")
    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    n_items = len(item_ids)
    iidx = train["item_id"].map(item_to_idx).to_numpy()
    uidx = pd.factorize(train["entity_id"])[0]
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    F = _features(items, item_to_idx, n_items)
    has = np.diff(F.indptr) > 0

    S = sp.csr_matrix(
        (np.ones(len(iidx), np.float32), (uidx, iidx)), shape=(int(uidx.max()) + 1, n_items)
    )
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    C = (S.T @ S).tocoo()
    m = C.row != C.col
    rows, cols = C.row[m], C.col[m]
    co = C.data[m].astype(np.float64)
    obs = apply_cooc_transform(
        co.astype(np.float32),
        cols.astype(np.int32),
        _indptr_from_rows(rows, n_items),
        d,
        int(S.shape[0]),
        "wilson",
    )
    Cobs = sp.csr_matrix((obs, (rows, cols)), shape=(n_items, n_items))
    tb = train.groupby("entity_id", sort=False)["item_id"].apply(set)
    te = test.groupby("entity_id", sort=False)["item_id"].apply(set)
    cand = sorted(set(tb.index) & set(te.index))
    if len(cand) > 2000:
        cand = cand[:: len(cand) // 2000][:2000]
    return dict(
        n_items=n_items,
        d=d,
        F=F,
        has=has,
        Cobs=Cobs,
        max_obs=float(obs.max()),
        item_to_idx=item_to_idx,
        tb=tb,
        te=te,
        cand=cand,
    )


def graft(g, threshold, topk, scale):
    """Calibrated grafting: synthetic edge weight = sim * scale * max_obs.

    ``scale`` is the dose dial. With ``scale`` set to the gate's measured
    metadata->cooc R2, the dose is self-calibrating: R2~0 (book) grafts ~nothing
    and cannot flood; R2=0.13 (H&M) grafts gently.
    """
    n_items, d, F, has = g["n_items"], g["d"], g["F"], g["has"]
    if scale <= 0:
        return g["Cobs"]
    eligible = np.where((d >= 1) & (d <= threshold) & has)[0]
    warm = np.where((d > threshold) & has)[0]
    if len(eligible) == 0 or len(warm) == 0:
        return g["Cobs"]
    Fw = F[warm]
    er, ec, ew = [], [], []
    for s in range(0, len(eligible), 512):
        blk = eligible[s : s + 512]
        sims = np.asarray((F[blk] @ Fw.T).todense())
        for bi, c in enumerate(blk):
            row = sims[bi]
            top = np.argpartition(-row, topk)[:topk] if topk < len(row) else np.arange(len(row))
            top = top[row[top] > 0]
            for w in top:
                er.append(c)
                ec.append(int(warm[w]))
                ew.append(row[w] * scale * g["max_obs"])
    syn = sp.csr_matrix(
        (np.array(ew + ew), (np.array(er + ec), np.array(ec + er))), shape=(n_items, n_items)
    )
    return (g["Cobs"] + syn).tocsr()


def metadata_knn(g, topk):
    """Symmetric sim-weighted top-k metadata graph over ALL items with metadata.

    Built once; C_aug = Cobs + scale * max_obs * M lets us sweep the dose
    cheaply. Unlike cold-only grafting, every item (warm included) gains its
    metadata edges, so warm items keep their real-cooc lead instead of being
    leapfrogged by lifted cold items.
    """
    F, has, n_items = g["F"], g["has"], g["n_items"]
    items = np.where(has)[0]
    Fa = F[items]
    er, ec, ew = [], [], []
    for s in range(0, len(items), 512):
        pos = np.arange(s, min(s + 512, len(items)))
        sims = np.asarray((F[items[pos]] @ Fa.T).todense())
        for bi, gpos in enumerate(pos):
            row = sims[bi].copy()
            row[gpos] = -1.0  # exclude self
            k = min(topk, len(row) - 1)
            top = np.argpartition(-row, k)[:k]
            top = top[row[top] > 0]
            c = int(items[gpos])
            for w in top:
                er.append(c)
                ec.append(int(items[w]))
                ew.append(float(row[w]))
    return sp.csr_matrix(
        (np.array(ew + ew), (np.array(er + ec), np.array(ec + er))), shape=(n_items, n_items)
    )


def knn_edges_with_obs(g, topk):
    """Directed top-k metadata edges (i->j) with sim and the ACTUAL wilson cooc
    weight of (i,j) from Cobs (0 where the pair never co-occurs).

    This is the population we impute over: fit cooc~sim on it, predict on it.
    Most edges are obs=0 (metadata-similar pairs that don't co-occur), so the
    fitted E[cooc|sim] is naturally small — that is the grounded 'cap'.
    """
    F, has, n_items = g["F"], g["has"], g["n_items"]
    Cobs = g["Cobs"].tocsr()
    items = np.where(has)[0]
    Fa = F[items]
    ei, ej, es = [], [], []
    for s in range(0, len(items), 512):
        pos = np.arange(s, min(s + 512, len(items)))
        sims = np.asarray((F[items[pos]] @ Fa.T).todense())
        for bi, gpos in enumerate(pos):
            row = sims[bi].copy()
            row[gpos] = -1.0
            k = min(topk, len(row) - 1)
            top = np.argpartition(-row, k)[:k]
            top = top[row[top] > 0]
            ei.append(np.full(len(top), int(items[gpos])))
            ej.append(items[top])
            es.append(row[top])
    ei, ej, esim = np.concatenate(ei), np.concatenate(ej), np.concatenate(es)
    # Vectorized cooc lookup for ALL edges at once (0 where absent).
    eobs = np.asarray(Cobs[ei, ej]).ravel()
    return ei, ej, esim.astype(np.float64), eobs.astype(np.float64)


def _poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * float(np.sum(term - (y - mu)))


def _fit_predictors(sim, obs):
    """Fit cooc-weight ~ metadata-sim. Returns ({name: predict_fn}, gate_info).

    OLS (additive, clipped >=0) and Poisson (exp link, count-aware). gate_info
    carries OLS R2 and the Poisson *deviance-explained* (pseudo-R2) — the
    candidate gate metric that R2_lin gets wrong on sparse binary metadata.
    """
    from sklearn.linear_model import PoissonRegressor

    x = sim.reshape(-1, 1)
    b1, b0 = np.polyfit(sim, obs, 1)
    pr = PoissonRegressor(alpha=1e-6, max_iter=500).fit(x, obs)
    preds = {
        "ols": lambda s: np.clip(b0 + b1 * s, 0.0, None),
        "poisson": lambda s: pr.predict(s.reshape(-1, 1)),
    }
    ss = float(((obs - obs.mean()) ** 2).sum())
    null_dev = _poisson_deviance(obs, np.full_like(obs, obs.mean()))
    pois_dev = _poisson_deviance(obs, preds["poisson"](sim))
    info = {
        "ols_r2": round(1.0 - float(((obs - preds["ols"](sim)) ** 2).sum()) / ss, 4)
        if ss > 0
        else 0.0,
        "pois_dev_expl": round(1.0 - pois_dev / null_dev, 4) if null_dev > 0 else 0.0,
        "slope": round(float(b1), 5),
    }
    return preds, info


def evaluate(g, Cg):
    d, item_to_idx, n_items = g["d"], g["item_to_idx"], g["n_items"]
    idisc = 1.0 / np.log2(np.arange(2, K + 2))
    tier_hit = {t: 0 for t in TIERS}
    tier_tot = {t: 0 for t in TIERS}
    ndcgs, rec_items = [], set()
    for e in g["cand"]:
        owned = np.array([item_to_idx[i] for i in g["tb"][e] if i in item_to_idx])
        rel = [item_to_idx[i] for i in (g["te"][e] - g["tb"][e]) if i in item_to_idx]
        if len(owned) == 0:
            continue
        scores = np.asarray(Cg[:, owned].sum(axis=1)).ravel()
        scores[owned] = -np.inf
        top = np.argpartition(-scores, K)[:K]
        top = top[np.argsort(-scores[top])]
        rec_items.update(top.tolist())
        if not rel:
            continue
        topset, relset = set(top.tolist()), set(rel)
        g_top = np.array([1.0 if int(t) in relset else 0.0 for t in top])
        if g_top.sum() > 0:
            dcg = float((g_top * idisc).sum())
            ideal = float(idisc[: min(len(rel), K)].sum())
            ndcgs.append(dcg / ideal)
        else:
            ndcgs.append(0.0)
        for r in rel:
            for t, (lo, hi) in TIERS.items():
                if lo <= d[r] < hi:
                    tier_tot[t] += 1
                    if r in topset:
                        tier_hit[t] += 1
                    break
    return {
        "ndcg": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "coverage": round(len(rec_items) / n_items, 4),
        **{
            f"rec_{t}": (round(tier_hit[t] / tier_tot[t], 4) if tier_tot[t] else None)
            for t in TIERS
        },
        **{f"n_{t}": tier_tot[t] for t in TIERS},
    }


def run_graft(name, train, test, items, subsample, topk, caps, r2, pctiles):
    print(f"\n########## GRAFT {name}  (gate R2={r2:.3f}) ##########")
    g = build(train, test, items, subsample)
    base = evaluate(g, g["Cobs"])
    print(
        f"  {'no_graft':28s} ndcg={base['ndcg']:.4f} cov={base['coverage']:.3f} | "
        + " ".join(f"{t}={base['rec_' + t]}" for t in TIERS)
    )
    rows = [{"arm": "no_graft", **base}]

    def arm(label, T, scale):
        r = evaluate(g, graft(g, T, topk, scale))
        rows.append({"arm": label, "threshold": T, "scale": round(scale, 4), **r})
        delta = (r["ndcg"] - base["ndcg"]) / base["ndcg"] * 100 if base["ndcg"] else 0.0
        print(
            f"  {label:28s} ndcg={r['ndcg']:.4f} ({delta:+.0f}%) cov={r['coverage']:.3f} | "
            + " ".join(f"{t}={r['rec_' + t]}" for t in TIERS)
        )

    # The headline: self-calibrating dose = the gate's measured R2 (book -> ~0).
    for T in (10, 20):
        arm(f"r2weighted_T{T}", T, r2)
    # Reference fixed caps, for the dose-response curve.
    for cap in caps:
        arm(f"cap{cap}_T10", 10, cap)
    return {
        "dataset": name,
        "k": K,
        "topk": topk,
        "r2": r2,
        "caps": caps,
        "subsample": subsample,
        "rows": rows,
    }


def run_graft_all(name, train, test, items, subsample, topk, caps, r2):
    """All-items metadata smoothing: C_aug = Cobs + scale*max_obs*M (every item)."""
    print(f"\n########## GRAFT-ALL {name}  (gate R2={r2:.3f}) ##########")
    g = build(train, test, items, subsample)
    base = evaluate(g, g["Cobs"])
    print(
        f"  {'no_graft':22s} ndcg={base['ndcg']:.4f} cov={base['coverage']:.3f} | "
        + " ".join(f"{t}={base['rec_' + t]}" for t in TIERS)
    )
    rows = [{"arm": "no_graft", **base}]
    M = metadata_knn(g, topk)  # built ONCE, swept over scales
    mx = g["max_obs"]
    for scale, label in [(r2, "r2"), *[(c, f"cap{c}") for c in caps]]:
        Cg = (g["Cobs"] + (scale * mx) * M).tocsr()
        r = evaluate(g, Cg)
        delta = (r["ndcg"] - base["ndcg"]) / base["ndcg"] * 100 if base["ndcg"] else 0.0
        rows.append({"arm": f"all_{label}", "scale": round(scale, 4), **r})
        print(
            f"  {'all_' + label:22s} ndcg={r['ndcg']:.4f} ({delta:+.0f}%) cov={r['coverage']:.3f} | "
            + " ".join(f"{t}={r['rec_' + t]}" for t in TIERS)
        )
    return {"dataset": name, "mode": "all", "k": K, "topk": topk, "r2": r2, "rows": rows}


def run_graft_ground(name, train, test, items, subsample, topk, r2):
    """Grounded smoothing: impute the FITTED cooc~sim prediction as the edge
    weight (no cap). The mean predicted weight / max_obs reveals the implied
    'effective cap' — grounding the previously hand-set value."""
    print(f"\n########## GRAFT-GROUND {name}  (gate R2={r2:.3f}) ##########")
    g = build(train, test, items, subsample)
    mx = g["max_obs"]
    base = evaluate(g, g["Cobs"])
    print(
        f"  {'no_graft':16s} ndcg={base['ndcg']:.4f} cov={base['coverage']:.3f} | "
        + " ".join(f"{t}={base['rec_' + t]}" for t in TIERS)
    )
    rows = [{"arm": "no_graft", **base}]
    M = metadata_knn(g, topk)  # sim-weighted, all-items (reuses working build)
    Mco = M.tocoo()
    # Fit cooc~sim on a sample of the metadata edges (most have obs=0).
    rng = np.random.default_rng(0)
    n = Mco.data.size
    idx = rng.choice(n, min(60000, n), replace=False)
    s_rows, s_cols, s_sim = Mco.row[idx], Mco.col[idx], Mco.data[idx]
    s_obs = np.asarray(g["Cobs"][s_rows, s_cols]).ravel()
    preds, info = _fit_predictors(s_sim, s_obs)
    print(
        f"  fit: sample={len(s_sim):,} frac_obs_zero={float((s_obs == 0).mean()):.3f} "
        f"max_obs={mx:.4f} | GATE ols_r2={info['ols_r2']} "
        f"pois_dev_expl={info['pois_dev_expl']} slope={info['slope']}"
    )
    for label, fn in preds.items():
        Mp = M.copy()
        Mp.data = np.clip(fn(M.data), 0.0, None)
        eff_cap = float(Mp.data.mean()) / mx if mx else 0.0
        r = evaluate(g, (g["Cobs"] + Mp).tocsr())
        delta = (r["ndcg"] - base["ndcg"]) / base["ndcg"] * 100 if base["ndcg"] else 0.0
        rows.append({"arm": f"predicted_{label}", "eff_cap": round(eff_cap, 4), **r})
        print(
            f"  predicted_{label:8s} ndcg={r['ndcg']:.4f} ({delta:+.0f}%) cov={r['coverage']:.3f} "
            f"eff_cap={eff_cap:.4f} | " + " ".join(f"{t}={r['rec_' + t]}" for t in TIERS)
        )
    return {"dataset": name, "mode": "ground", "k": K, "topk": topk, "r2": r2, "rows": rows}


def main():
    datasets = os.environ.get("GRAFT_DATASETS", "steam,h-and-m").split(",")
    gate_only = os.environ.get("GRAFT_GATE_ONLY", "") == "1"
    all_items = os.environ.get("GRAFT_ALL", "") == "1"
    ground = os.environ.get("GRAFT_GROUND", "") == "1"
    subsample = int(os.environ.get("GRAFT_SUBSAMPLE", "0")) or None
    topk = int(os.environ.get("GRAFT_TOPK", "20"))
    caps = [float(x) for x in os.environ.get("GRAFT_CAPS", "0.05,0.1,0.2").split(",")]
    pctiles = os.environ.get("GRAFT_PCTILE", "")
    report = {"gates": [], "graft": []}
    for name in datasets:
        print(f"\n=========== {name} ===========", flush=True)
        train, test, items = load(name)
        print(
            f"  loaded train={len(train):,} users={train.entity_id.nunique():,} "
            f"items={train.item_id.nunique():,}",
            flush=True,
        )
        gres = gate(name, train, items)
        report["gates"].append(gres)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        if not gate_only:
            # Dose = the gate's own metadata->cooc fidelity (clamped >= 0).
            r2 = max(
                0.0, float(gres.get("targets", {}).get("cosine_weight", {}).get("r2_linear", 0.0))
            )
            if ground:
                res = run_graft_ground(name, train, test, items, subsample, topk, r2)
            elif all_items:
                res = run_graft_all(name, train, test, items, subsample, topk, caps, r2)
            else:
                res = run_graft(name, train, test, items, subsample, topk, caps, r2, pctiles)
            report["graft"].append(res)
            REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[wrote] {REPORT}")


if __name__ == "__main__":
    main()
