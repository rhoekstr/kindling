"""Does store-aisle/section metadata help actual RECOMMENDATIONS (not just the
mapping-R² proxy)? Clean EASE + content-blend A/B on the labeled book subcatalog.

Restrict to the items we have aisle labels for (the 6000 warmest — and 6000 <
the 20k EASE gate, so EASE is the base). For each eval user, blend the EASE
score with a content-similarity channel built from:
  - aisle   : the LLM shelf labels (aisle + section, one-hot)
  - native  : Amazon's own categories + brand (the baseline metadata)
and sweep the blend weight. NDCG@10 / recall@10.

  ease(α=0)               interaction-only baseline
  + aisle@{.25,.5,1}      does aisle/section content add over EASE?
  + native@{.5}           does it beat the native metadata?
  aisle-only / native-only content as a standalone ranker

If "+ aisle" > baseline AND > "+ native", the labels help recommendations.

Run: .venv/bin/python bench/run_aisle_recs.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine_v2 import EngineV2
from kindling.item_features import ItemFeatureExtractor, content_scores
from run_meta_cooc_map import load as load_ds

K = 10
DATASET = "amazon-book-chrono"
REPORT = Path(__file__).parent / "reports" / "aisle_recs.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def zc(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def features(df, item_to_idx, n):
    f = ItemFeatureExtractor().fit_transform(df, item_to_idx, n)
    return f


def main() -> None:
    train, test, items = load_ds(DATASET)
    aisle_rows = [json.loads(line) for line in
                  (Path(__file__).parent / "cache" / f"{DATASET}_aisle.jsonl")
                  .read_text().splitlines() if line.strip()]
    aisle_df = pd.DataFrame([r for r in aisle_rows if r.get("aisle")])
    labeled = set(aisle_df["item_id"])
    log(f"labeled items: {len(labeled)}")

    # Restrict catalog to labeled items.
    train = train[train["item_id"].isin(labeled)].reset_index(drop=True)
    test = test[test["item_id"].isin(labeled)].reset_index(drop=True)
    log(f"restricted: train {len(train):,}  test {len(test):,}")

    eng = EngineV2(persona_min_users=10**9, base_scorer="ease",
                   retrieval_budget=500, random_state=0).fit(train)
    st = eng._state
    n = st.n_items
    if st.ease_b is None:
        raise SystemExit("EASE did not run (catalog too large?)")
    ease = st.ease_b.astype(np.float64)

    # content features aligned to engine indices
    aisle_feat = features(aisle_df[["item_id", "aisle", "section"]], st.item_to_idx, n)
    native_cols = [c for c in ("categories", "brand") if c in items.columns]
    native_feat = features(items[["item_id", *native_cols]], st.item_to_idx, n)
    log(f"aisle feats: {aisle_feat.n_features} (cov {aisle_feat.coverage:.2f})  "
        f"native feats: {native_feat.n_features} (cov {native_feat.coverage:.2f})")

    eval_set = _build_eval_set(train, test, max_users=3000, seed=0)
    log(f"eval users: {len(eval_set)}")

    # precompute per-user channel vectors
    users = []
    for ent, rel in eval_set.items():
        owned = st.owned_by_entity.get(ent)
        if owned is None or owned.size == 0:
            continue
        e = zc(ease[owned].sum(0))
        ca = zc(content_scores(aisle_feat, owned))
        cn = zc(content_scores(native_feat, owned))
        users.append((owned, rel, e, ca, cn))
    log(f"scored users: {len(users)}")

    def ndcg(blend):
        per = []
        for owned, rel, e, ca, cn in users:
            s = blend(e, ca, cn).copy()
            s[owned] = -np.inf
            order = np.argsort(-s)[:K]
            per.append(([st.item_ids[int(i)] for i in order], rel))
        m = aggregate(per, catalog_size=n, k=K)
        return round(m.ndcg_at_k, 4), round(m.recall_at_k, 4)

    arms = {
        "ease (baseline)":   lambda e, ca, cn: e,
        "+ aisle@0.25":      lambda e, ca, cn: e + 0.25 * ca,
        "+ aisle@0.5":       lambda e, ca, cn: e + 0.5 * ca,
        "+ aisle@1.0":       lambda e, ca, cn: e + 1.0 * ca,
        "+ native@0.5":      lambda e, ca, cn: e + 0.5 * cn,
        "aisle-only":        lambda e, ca, cn: ca,
        "native-only":       lambda e, ca, cn: cn,
    }
    out = {"dataset": DATASET, "n_labeled": len(labeled), "n_users": len(users),
           "aisle_features": aisle_feat.n_features, "results": {}}
    for name, fn in arms.items():
        nd, rc = ndcg(fn)
        out["results"][name] = {"ndcg": nd, "recall": rc}
        log(f"  {name:18s} NDCG@10={nd:.4f}  recall@10={rc:.4f}")
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    log(f"[wrote] {REPORT}")


if __name__ == "__main__":
    main()
