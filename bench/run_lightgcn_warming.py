"""Token LightGCN points for the growth grid — comparable to the other warming
rows. Reproduces run_warming_curve's exact random subsample + eval population,
writes RecBole benchmark-split .inter files, trains LightGCN in .venv-recbole
(bench/_recbole_lightgcn.py), then scores the predictions with kindling's own
aggregate() and MERGEs `lightgcn` rows into warming_<dataset>.json.

LightGCN is ~30 min/fit, so this runs only a few points (default 10/50/100%).
A small slice of each subsample is held out as RecBole's validation set (early
stopping), so LightGCN trains on marginally less than the other models — fine for
a positioning line.

Run: DATASET=movielens-1m FRACTIONS=0.1,0.5,1.0 \
     PYTHONPATH=src .venv/bin/python bench/run_lightgcn_warming.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_warming_curve import load_split

from kindling.benchmarks.metrics import aggregate

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = Path(__file__).resolve().parent / "reports"
RECBOLE_PY = ROOT / ".venv-recbole/bin/python"
DSNAME = "lgcn"  # RecBole benchmark dataset name (reused per fraction)


def write_inter(path: Path, df: pd.DataFrame) -> None:
    # timestamp is unused by LightGCN on a pre-split (benchmark) dataset — write
    # a constant so heterogeneous timestamp dtypes (datetime / int / float) never
    # matter.
    with open(path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        u = df["entity_id"].astype(str).to_numpy()
        it = df["item_id"].astype(str).to_numpy()
        for a, b in zip(u, it):
            f.write(f"{a}\t{b}\t1.0\t0.0\n")


def main() -> int:
    dataset = os.environ.get("DATASET", "movielens-1m")
    fractions = [float(x) for x in os.environ.get("FRACTIONS", "0.1,0.5,1.0").split(",")]
    max_eval = int(os.environ.get("MAX_EVAL", "1000"))
    k = 10

    split = load_split(dataset)
    train, test = split.train, split.test
    train_by = train.groupby("entity_id", sort=False)["item_id"].apply(set)
    test_by = test.groupby("entity_id", sort=False)["item_id"].apply(set)
    eval_all = sorted(set(train_by.index) & set(test_by.index))
    step = max(1, len(eval_all) // max_eval)
    eval_entities = eval_all[::step][:max_eval]
    catalog = int(train["item_id"].nunique())
    order = np.random.default_rng(0).permutation(len(train))  # MUST match run_warming_curve
    print(f"{dataset}: train {len(train):,} eval {len(eval_entities)} catalog {catalog:,} "
          f"fractions {fractions}", flush=True)

    d = ROOT / "recbole_data" / DSNAME
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for frac in fractions:
        n = round(len(train) * frac)
        sub = train.iloc[order[:n]].reset_index(drop=True)
        owned_sub = sub.groupby("entity_id", sort=False)["item_id"].apply(set)
        sub_items = set(sub["item_id"].unique())

        # train / valid split (5% valid for early stopping).
        vmask = np.random.default_rng(1).random(len(sub)) < 0.05
        write_inter(d / f"{DSNAME}.train.inter", sub[~vmask])
        write_inter(d / f"{DSNAME}.valid.inter", sub[vmask] if vmask.any() else sub.iloc[:10])
        # test = eval users' future items that exist in the subsample vocab.
        tr = [(e, it) for e in eval_entities for it in (test_by.get(e, set()) & sub_items)]
        write_inter(d / f"{DSNAME}.test.inter", pd.DataFrame(tr, columns=["entity_id", "item_id"]))

        out = ROOT / "recbole_data" / f"lgcn_preds_{dataset}_{frac}.json"
        t0 = time.perf_counter()
        subprocess.run(
            [str(RECBOLE_PY), "bench/_recbole_lightgcn.py", "recbole_data", DSNAME, str(out)],
            check=True, cwd=ROOT,
        )
        wall = time.perf_counter() - t0
        preds = json.loads(out.read_text())
        topk = preds["topk"]

        per = []
        for ent in eval_entities:
            # RecBole id2token yields string item ids; compare as strings.
            rel = {str(x) for x in (test_by.get(ent, set()) - owned_sub.get(ent, set()))}
            per.append((list(topk.get(str(ent), []))[:k], rel))
        m = aggregate(per, catalog_size=catalog, k=k)
        rows.append({
            "fraction": frac, "n_train": len(sub), "n_train_items": len(sub_items),
            "n_eval_nonempty": sum(1 for e in eval_entities if test_by.get(e, set()) - owned_sub.get(e, set())),
            "model": "lightgcn", "fit_seconds": round(preds["fit_seconds"], 3), "p50_ms": 0.0,
            "recall@k": round(m.recall_at_k, 4), "ndcg@k": round(m.ndcg_at_k, 4),
            "mrr": round(m.mrr, 4), "hit_rate": round(m.hit_rate, 4),
        })
        print(f"  frac={frac:<5} lightgcn ndcg={m.ndcg_at_k:.4f} recall={m.recall_at_k:.4f} "
              f"fit={preds['fit_seconds']:.0f}s (wall {wall:.0f}s)", flush=True)

    # Merge into the dataset's warming file.
    path = REPORT_DIR / f"warming_{dataset}.json"
    data = json.loads(path.read_text())
    fresh = {(r["fraction"], r["model"]) for r in rows}
    data["rows"] = sorted(
        [r for r in data["rows"] if (r["fraction"], r["model"]) not in fresh] + rows,
        key=lambda r: (r["fraction"], r["model"]),
    )
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[merged] {len(rows)} lightgcn rows into {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
