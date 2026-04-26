"""Full sweep: hard/soft × n_personas × primary_routing.

Drops Bayesian baseline (per user direction - layered architecture
is the established winner). Tests:

  match_mode      ∈ {soft, hard}
  n_personas      ∈ {5, 10, 30}
  primary_routing ∈ {cooc-fixed, persona_cooc-fixed, adaptive}

Stratified by user training-history density so we see where each
configuration wins. Also computes per-stratum persona-concentration
distribution to inform the adaptive-threshold default.

CLI:
    python -m kindling.benchmarks.probe_persona_cooc_sweep \\
        --dataset amazon-beauty \\
        --output bench/reports/probe_persona_cooc_sweep_amazon_beauty.json
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
from kindling.personas import KMeansWithNoiseClustering, PersonaConfig


STRATUM_BOUNDARIES = [
    ("very_cold", 0, 3),
    ("cold", 4, 10),
    ("warm", 11, 30),
    ("hot", 31, 1_000_000),
]


def _classify(history_size: int) -> str:
    for name, lo, hi in STRATUM_BOUNDARIES:
        if lo <= history_size <= hi:
            return name
    return "hot"


def _eval(engine, eval_entities, train_items, test_items, k):
    per_entity = []
    n_with_relevant = 0
    recall_topk_hits = 0
    for entity in eval_entities:
        recs = engine.recommend(entity_id=entity, n=k)
        rec_items = [r.item_id for r in recs]
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
    }


def _overlap_distribution(engine, eval_entities) -> dict[str, float]:
    """Compute persona_overlap distribution stats over eval users.

    persona_overlap = fraction of user's owned items that have non-zero
    weight in their top persona's TF-IDF/z-filtered vector. This is the
    behavior-based metric driving adaptive primary routing - bounded
    [0, 1] and independent of user-vector scale.
    """
    if engine._persona_index is None or engine._persona_index.n_personas == 0:
        return {}
    from kindling.personas.matching import build_user_query_vector, match_user

    overlaps = []
    for entity in eval_entities:
        owned = engine._owned_by_entity.get(entity, np.array([]))
        history = engine._history_by_entity.get(entity, ())
        user_vec = build_user_query_vector(
            owned_items=owned, history_items=history, index=engine._persona_index
        )
        m = match_user(user_vec, engine._persona_index)
        if not m.any() or owned.size == 0:
            overlaps.append(0.0)
            continue
        top = int(np.asarray(m).argmax())
        owned_idx = np.fromiter(
            (
                engine._persona_index.item_id_to_idx.get(o, -1)
                for o in owned.tolist()
            ),
            dtype=np.int64,
            count=owned.size,
        )
        owned_idx = owned_idx[owned_idx >= 0]
        if owned_idx.size == 0:
            overlaps.append(0.0)
            continue
        row = engine._persona_index.persona_vectors[top]
        weights = np.asarray(row[:, owned_idx].todense()).ravel()
        overlaps.append(float((weights > 0).mean()))
    if not overlaps:
        return {}
    arr = np.asarray(overlaps)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "frac_above_30pct": float((arr >= 0.30).mean()),
        "frac_above_50pct": float((arr >= 0.50).mean()),
    }


def run(
    dataset: str,
    n_personas_list: list[int],
    max_per_stratum: int = 500,
    k: int = 10,
    test_fraction: float = 0.1,
) -> dict:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== sweep: {dataset} ({len(split.train):,} train) ===", flush=True)

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
    eligible = sorted(set(train_items.index).intersection(test_items.index))
    by_stratum: dict[str, list[object]] = {s[0]: [] for s in STRATUM_BOUNDARIES}
    for ent in eligible:
        by_stratum[_classify(len(train_items.get(ent, set())))].append(ent)

    print(f"  user counts by stratum:", flush=True)
    for sname in [s[0] for s in STRATUM_BOUNDARIES]:
        n = len(by_stratum[sname])
        print(f"    {sname:<12} {n:,}", flush=True)

    sweep_rows: list[dict] = []
    for n_personas in n_personas_list:
        print(f"\n  fitting engines with n_personas={n_personas} ...", flush=True)
        pcfg = PersonaConfig(
            enabled=True,
            clustering=KMeansWithNoiseClustering(
                n_clusters=n_personas,
                noise_fraction=0.15,
                random_state=0,
            ),
            min_activation_users=50,
        )

        # Fit once per (n_personas) - the layered_config controls
        # only the recommend-time scoring, so we don't need separate
        # fits per match_mode / routing.
        t0 = time.perf_counter()
        engine = Engine(
            persona_config=pcfg,
            layered_scoring=True,
            layered_config=LayeredConfig(),  # placeholder
        ).fit(split.train)
        fit_seconds = time.perf_counter() - t0
        actual_n_personas = (
            engine._persona_index.n_personas if engine._persona_index else 0
        )
        print(
            f"    fit {fit_seconds:.1f}s, actual n_personas={actual_n_personas}",
            flush=True,
        )

        # Precompute concentration distribution per stratum.
        for stratum_name, _, _ in STRATUM_BOUNDARIES:
            users = by_stratum[stratum_name]
            if not users:
                continue
            if len(users) > max_per_stratum:
                step = max(1, len(users) // max_per_stratum)
                users = users[::step][:max_per_stratum]

            conc = _overlap_distribution(engine, users)
            print(
                f"  -- stratum={stratum_name} (n={len(users)}) "
                f"overlap mean={conc.get('mean', 0):.3f} "
                f"median={conc.get('median', 0):.3f} "
                f"p75={conc.get('p75', 0):.3f}  "
                f"frac>=0.30={conc.get('frac_above_30pct', 0):.2f} "
                f"frac>=0.50={conc.get('frac_above_50pct', 0):.2f}",
                flush=True,
            )

            for primary_signal, routing in [
                ("cooccurrence", "fixed"),
                ("persona_cooccurrence", "fixed"),
                ("cooccurrence", "adaptive"),
            ]:
                for match_mode in ["soft", "hard"]:
                    if routing == "adaptive" and primary_signal != "cooccurrence":
                        continue  # only one adaptive variant
                    cfg = LayeredConfig(
                        primary_signal=primary_signal,
                        persona_match_mode=match_mode,
                        primary_routing=routing,
                    )
                    engine.layered_config = cfg
                    res = _eval(engine, users, train_items, test_items, k)
                    label = (
                        f"{primary_signal}/{routing}/{match_mode}"
                        if routing == "fixed"
                        else f"adaptive/{match_mode}"
                    )
                    print(
                        f"    {label:<48} NDCG={res['ndcg_at_k']:.4f} "
                        f"R@K={res['recall_topk']:.3f}",
                        flush=True,
                    )
                    sweep_rows.append({
                        "n_personas": actual_n_personas,
                        "stratum": stratum_name,
                        "n_users": len(users),
                        "primary_signal": primary_signal,
                        "routing": routing,
                        "match_mode": match_mode,
                        "ndcg_at_k": res["ndcg_at_k"],
                        "recall_topk": res["recall_topk"],
                        "mrr": res["mrr"],
                        "overlap_stats": conc,
                    })

    return {
        "dataset": dataset,
        "kindling_version": __version__,
        "n_personas_list": n_personas_list,
        "rows": sweep_rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon-beauty")
    parser.add_argument("--n-personas-list", default="5,10,30")
    parser.add_argument("--max-per-stratum", type=int, default=400)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    n_list = [int(x) for x in args.n_personas_list.split(",") if x.strip()]
    report = run(
        args.dataset,
        n_personas_list=n_list,
        max_per_stratum=args.max_per_stratum,
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
