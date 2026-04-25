"""Stratified probe: persona-cooc as primary scorer vs global cooc.

Tests the user's hypothesis: persona-cooc (soft-weighted per-persona
cooc) is a better primary signal than global cooc on cold-start
users where global cooc has thin signal. May tie or lose on warm/hot
users where global cooc has plenty of evidence.

Methodology:

1. Fit engine WITH persona on a sparse-leaning dataset.
2. Stratify users by training-history size (very_cold / cold /
   warm / hot).
3. For each stratum, run Engine.recommend() under three configs:
   - bayesian_blend (current default)
   - layered_cooc_primary (cooc + adaptive boost layers)
   - layered_persona_primary (persona_cooc + adaptive boost layers)
4. Compare NDCG@10 / Recall@10 / R@budget per stratum.

Expected: persona-primary wins on very_cold / cold; ties on warm;
ties or loses on hot (where global cooc has enough signal).

CLI:
    python -m kindling.benchmarks.probe_persona_cooc_stratified \\
        --dataset amazon-beauty \\
        --output bench/reports/probe_persona_cooc_stratified_amazon_beauty.json
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
from kindling.blend.layered import LayeredConfig
from kindling.personas import KMeansClustering, PersonaConfig


STRATUM_BOUNDARIES = [
    ("very_cold", 0, 3),
    ("cold", 4, 10),
    ("warm", 11, 30),
    ("hot", 31, 1_000_000),
]


def _classify_stratum(history_size: int) -> str:
    for name, lo, hi in STRATUM_BOUNDARIES:
        if lo <= history_size <= hi:
            return name
    return "hot"


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
    max_per_stratum: int = 500,
    k: int = 10,
    test_fraction: float = 0.1,
    n_personas: int = 30,
) -> dict:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== probe_persona_cooc_stratified: {dataset} ({len(split.train):,} train) ===", flush=True)

    test_items = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    train_items = cast(
        pd.Series,
        split.train.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    eligible_users = sorted(
        set(train_items.index).intersection(test_items.index)
    )

    persona_config = PersonaConfig(
        enabled=True,
        clustering=KMeansClustering(n_clusters=n_personas, random_state=0),
        min_activation_users=50,
    )

    # Fit each variant once.
    print(f"  fitting Engine() (Bayesian blend, persona ON) ...", flush=True)
    t0 = time.perf_counter()
    engine_blend = Engine(persona_config=persona_config).fit(split.train)
    fit_blend = time.perf_counter() - t0
    print(
        f"    fit {fit_blend:.1f}s, persona n={engine_blend._persona_index.n_personas if engine_blend._persona_index else 0}, "
        f"persona_cooc nnz={engine_blend._persona_cooc_graph.n_edges if engine_blend._persona_cooc_graph else 0}",
        flush=True,
    )

    print(f"  fitting Engine(layered_scoring=True, primary=cooc) ...", flush=True)
    t0 = time.perf_counter()
    engine_cooc = Engine(
        persona_config=persona_config,
        layered_scoring=True,
        layered_config=LayeredConfig(primary_signal="cooccurrence"),
    ).fit(split.train)
    fit_cooc = time.perf_counter() - t0
    print(f"    fit {fit_cooc:.1f}s", flush=True)

    print(f"  fitting Engine(layered_scoring=True, primary=persona_cooc) ...", flush=True)
    t0 = time.perf_counter()
    engine_pc = Engine(
        persona_config=persona_config,
        layered_scoring=True,
        layered_config=LayeredConfig(primary_signal="persona_cooccurrence"),
    ).fit(split.train)
    fit_pc = time.perf_counter() - t0
    print(f"    fit {fit_pc:.1f}s", flush=True)

    # Classify users.
    user_history_size = {ent: len(train_items.get(ent, set())) for ent in eligible_users}
    by_stratum: dict[str, list[object]] = {s[0]: [] for s in STRATUM_BOUNDARIES}
    for ent, sz in user_history_size.items():
        by_stratum[_classify_stratum(sz)].append(ent)

    print(f"  user counts by stratum:", flush=True)
    for sname in [s[0] for s in STRATUM_BOUNDARIES]:
        print(f"    {sname:<12} {len(by_stratum[sname]):,}", flush=True)

    rows: list[dict] = []
    for stratum_name, _, _ in STRATUM_BOUNDARIES:
        users = by_stratum[stratum_name]
        if not users:
            continue
        if len(users) > max_per_stratum:
            step = max(1, len(users) // max_per_stratum)
            users = users[::step][:max_per_stratum]

        print(f"\n  -- stratum={stratum_name} (n={len(users)}) --", flush=True)
        results_blend = _eval_engine(engine_blend, users, train_items, test_items, k)
        print(f"    bayesian_blend         NDCG={results_blend['ndcg_at_k']:.4f} R@K={results_blend['recall_topk']:.3f}", flush=True)
        results_cooc = _eval_engine(engine_cooc, users, train_items, test_items, k)
        print(f"    layered/cooc-primary   NDCG={results_cooc['ndcg_at_k']:.4f} R@K={results_cooc['recall_topk']:.3f}", flush=True)
        results_pc = _eval_engine(engine_pc, users, train_items, test_items, k)
        print(f"    layered/pcooc-primary  NDCG={results_pc['ndcg_at_k']:.4f} R@K={results_pc['recall_topk']:.3f}", flush=True)

        delta_pc_vs_blend = results_pc["ndcg_at_k"] - results_blend["ndcg_at_k"]
        delta_pc_vs_cooc = results_pc["ndcg_at_k"] - results_cooc["ndcg_at_k"]
        rows.append({
            "stratum": stratum_name,
            "n_users": len(users),
            "bayesian_blend": results_blend,
            "layered_cooc_primary": results_cooc,
            "layered_persona_cooc_primary": results_pc,
            "delta_pc_vs_blend_ndcg": delta_pc_vs_blend,
            "delta_pc_vs_cooc_ndcg": delta_pc_vs_cooc,
        })

    return {
        "dataset": dataset,
        "kindling_version": __version__,
        "n_personas": n_personas,
        "fit_seconds": {
            "blend": fit_blend,
            "cooc_primary": fit_cooc,
            "persona_cooc_primary": fit_pc,
        },
        "strata": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="amazon-beauty",
        choices=[
            "movielens-1m", "synthetic-grocery", "synthetic-grocery-deep",
            "retailrocket", "instacart", "gowalla", "yelp2018",
            "tafeng", "dunnhumby", "amazon-beauty", "amazon-book",
        ],
    )
    parser.add_argument("--max-per-stratum", type=int, default=500)
    parser.add_argument("--n-personas", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(
        args.dataset,
        max_per_stratum=args.max_per_stratum,
        n_personas=args.n_personas,
        k=args.k,
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
