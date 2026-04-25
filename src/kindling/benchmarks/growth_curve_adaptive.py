"""Growth-curve benchmark with adaptive layering.

Sweeps training-data fractions {5%, 10%, 25%, 50%, 100%} and at each
fraction:

1. Profile the (subset) data shape.
2. Plan layer activation.
3. Fit the engine + run probe_engine_layered (Bayesian vs Layered).
4. Record: profile.summary, plan.summary, NDCG/Recall/MRR for both
   architectures, and per-subsystem fit timings.

Surfaces three things the user explicitly asked about:

- **When do layers reach meaningful depth/density?** The plan
  rationale per fraction shows when session_cooc / path_basket /
  repeat_module activate as data accumulates.
- **Does layered's lift over Bayesian grow or shrink with data?**
  Per-fraction NDCG comparison answers this.
- **What dataset-shape transitions matter?** The profile evolution
  from sparse → dense surfaces dataset-specific tipping points.

CLI:
    python -m kindling.benchmarks.growth_curve_adaptive \\
        --dataset synthetic-grocery-deep \\
        --fractions 0.05,0.1,0.25,0.5,1.0 \\
        --output bench/reports/growth_curve_adaptive_grocery.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from kindling import Engine, __version__
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate


def _eval_engine(engine, eval_entities, train_items, test_items, k):
    per_entity = []
    latencies = []
    n_with_relevant = 0
    recall_topk_hits = 0
    for entity in eval_entities:
        t0 = time.perf_counter()
        recs = engine.recommend(entity_id=entity, n=k)
        rec_items = [r.item_id for r in recs]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        relevant = test_owned - train_owned
        per_entity.append((rec_items, relevant))
        if relevant:
            n_with_relevant += 1
            if set(rec_items) & relevant:
                recall_topk_hits += 1
    m = aggregate(per_entity, catalog_size=engine._item_graph.n_items, k=k)
    return {
        "ndcg_at_k": m.ndcg_at_k,
        "mrr": m.mrr,
        "recall_topk": recall_topk_hits / max(n_with_relevant, 1),
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def run(
    dataset: str,
    fractions: list[float],
    max_eval_entities: int = 500,
    k: int = 10,
    test_fraction: float = 0.1,
) -> dict:
    """Run the growth-curve sweep on one dataset."""
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== growth_curve_adaptive: {dataset} ({len(split.train):,} full train) ===", flush=True)

    full_train = split.train
    test_items_full = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )

    rows: list[dict] = []
    for fraction in sorted(fractions):
        n = max(1, int(round(len(full_train) * fraction)))
        sub = full_train.iloc[:n].reset_index(drop=True)
        print(f"\n  -- fraction {fraction:.2f} ({n:,} interactions) --", flush=True)

        train_items_sub = cast(
            pd.Series,
            sub.groupby("entity_id", sort=False)["item_id"].apply(
                lambda s: set(s.tolist())
            ),
        )
        eval_entities_all = sorted(
            set(train_items_sub.index).intersection(test_items_full.index)
        )
        if not eval_entities_all:
            print("    no eligible eval entities at this fraction; skipping", flush=True)
            continue
        step = max(1, len(eval_entities_all) // max_eval_entities)
        eval_entities = eval_entities_all[::step][:max_eval_entities]

        # Bayesian baseline.
        t0 = time.perf_counter()
        engine_b = Engine().fit(sub)
        fit_b = time.perf_counter() - t0
        results_b = _eval_engine(engine_b, eval_entities, train_items_sub, test_items_full, k)
        print(
            f"    bayesian: NDCG={results_b['ndcg_at_k']:.4f} "
            f"R@K={results_b['recall_topk']:.3f} fit={fit_b:.1f}s",
            flush=True,
        )

        # Layered with auto-calibration.
        t0 = time.perf_counter()
        engine_l = Engine(layered_scoring=True).fit(sub)
        fit_l = time.perf_counter() - t0
        results_l = _eval_engine(engine_l, eval_entities, train_items_sub, test_items_full, k)
        cfg = engine_l.layered_config
        print(
            f"    layered:  NDCG={results_l['ndcg_at_k']:.4f} "
            f"R@K={results_l['recall_topk']:.3f} fit={fit_l:.1f}s "
            f"(z={cfg.z_threshold} b={cfg.boost_multiplier})",
            flush=True,
        )

        # Profile + plan from the Bayesian engine (same data shape).
        profile = engine_b._dataset_profile
        plan = engine_b._layer_plan
        print(f"    PROFILE: {profile.summary().splitlines()[0]}", flush=True)
        print(f"    PLAN: {plan.summary().splitlines()[1]}", flush=True)

        rows.append({
            "fraction": fraction,
            "n_interactions": int(n),
            "n_eval_entities": len(eval_entities),
            "profile": {
                "n_users": profile.n_users,
                "n_items": profile.n_items,
                "user_density": profile.user_density,
                "item_density": profile.item_density,
                "avg_events_per_user": profile.avg_events_per_user,
                "time_use": profile.time_use,
                "session_depth": profile.session_depth,
                "deep_session_fraction": profile.deep_session_fraction,
                "repeat_dataset": profile.repeat_dataset,
                "repeat_user_fraction": profile.repeat_user_fraction,
                "notes": profile.notes,
            },
            "plan": {
                "subsystems": list(plan.enabled_subsystems),
                "boost_layers": list(plan.enabled_boost_layers),
                "temporal_kernel_active": plan.temporal_kernel_active,
                "repeat_module_active": plan.repeat_module_active,
                "rationale": plan.rationale,
            },
            "bayesian": {**results_b, "fit_seconds": fit_b},
            "layered": {
                **results_l,
                "fit_seconds": fit_l,
                "calibrated_z": cfg.z_threshold,
                "calibrated_boost": cfg.boost_multiplier,
            },
            "delta_ndcg": results_l["ndcg_at_k"] - results_b["ndcg_at_k"],
            "delta_rel_pct": (
                (results_l["ndcg_at_k"] - results_b["ndcg_at_k"])
                / max(results_b["ndcg_at_k"], 1e-9) * 100.0
            ),
        })

    return {
        "dataset": dataset,
        "fractions": fractions,
        "kindling_version": __version__,
        "growth": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="synthetic-grocery-deep",
        choices=[
            "movielens-1m", "synthetic-grocery", "synthetic-grocery-deep",
            "retailrocket", "instacart", "gowalla", "yelp2018",
            "tafeng", "dunnhumby", "amazon-beauty", "amazon-book",
        ],
    )
    parser.add_argument(
        "--fractions", default="0.05,0.1,0.25,0.5,1.0",
        help="Comma-separated training-data fractions to sweep.",
    )
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    report = run(
        args.dataset, fractions=fractions,
        max_eval_entities=args.max_eval_entities, k=args.k,
    )
    pretty = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"\nWrote {args.output}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
