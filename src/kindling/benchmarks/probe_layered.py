"""Probe the layered scoring architecture vs cooc-baseline + Bayesian blend.

For one dataset, fits an engine, then evaluates these scoring variants
on the same 500-eval-entity split:

1. **cooc_alone**: pure cooccurrence score (the baseline).
2. **bayesian_blend**: Engine.recommend() default - the current
   architecture with all signals weighted by the variational posterior.
3. **layered_cooc+pb**: cooc primary + path_basket as a one-tailed
   z-gated boost layer.
4. **layered_cooc+sc**: cooc primary + session_cooccurrence boost.
5. **layered_cooc+tc**: cooc primary + temporal_cooccurrence boost.
6. **layered_cooc+stack**: cooc primary + cumulative stack of all
   three above (each layer fires independently).

All variants share the same retrieved candidate pool (cooc top-budget)
so we're isolating the **scoring** decision, not the retrieval one.

CLI:
    python -m kindling.benchmarks.probe_layered \\
        --dataset synthetic-grocery-deep \\
        --output bench/reports/probe_layered_grocery.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from kindling import Engine, __version__
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.blend.layer_scoring import (
    _cooc_scores,
    _path_basket_scores,
    _session_cooc_scores,
    _temporal_cooc_scores,
)
from kindling.blend.layered import LayeredConfig, diagnostic_report, layered_score
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.protocol import Candidate


@dataclass(frozen=True)
class LayeredCell:
    dataset: str
    variant: str
    ndcg_at_k: float
    recall_topk: float
    recall_budget: float
    mrr: float
    p95_ms: float
    n_eval_entities: int
    diagnostic: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        d = {
            "dataset": self.dataset,
            "variant": self.variant,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_topk": self.recall_topk,
            "recall_budget": self.recall_budget,
            "mrr": self.mrr,
            "p95_ms": self.p95_ms,
            "n_eval_entities": self.n_eval_entities,
        }
        if self.diagnostic is not None:
            d["diagnostic"] = self.diagnostic
        return d


def _evaluate_scoring(
    variant: str,
    score_fn,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    engine: Engine,
    catalog_size: int,
    retrieval_budget: int,
    k: int,
    dataset: str,
    diagnostic_collector: dict | None = None,
) -> LayeredCell:
    """Run one scoring variant against the cooc-retrieved pool."""
    cooc_retriever = CoOccurrenceRetriever(engine._item_graph)
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    recall_topk_hits = 0
    recall_budget_hits = 0
    n_with_relevant = 0

    diag_accum: dict[str, list[float]] = {}

    for entity in eval_entities:
        owned = engine._owned_by_entity.get(entity, np.array([]))
        history = engine._history_by_entity.get(entity, ())
        t0 = time.perf_counter()
        candidates = cooc_retriever.retrieve(owned, retrieval_budget)
        cand_ids = [c.item_id for c in candidates]
        if not cand_ids:
            top: list[object] = []
            latencies.append((time.perf_counter() - t0) * 1000.0)
        else:
            scores = score_fn(engine, cand_ids, owned, history, candidates, diag_accum)
            order = np.argsort(-scores)
            top = [cand_ids[int(i)] for i in order[:k] if scores[int(i)] > 0.0]
            latencies.append((time.perf_counter() - t0) * 1000.0)

        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        relevant = test_owned - train_owned
        per_entity.append((top, relevant))
        if relevant:
            n_with_relevant += 1
            all_retrieved = {c.item_id for c in candidates}
            if all_retrieved & relevant:
                recall_budget_hits += 1
            if set(top) & relevant:
                recall_topk_hits += 1

    m = aggregate(per_entity, catalog_size=catalog_size, k=k)
    diag_summary = (
        {k: float(np.mean(v)) for k, v in diag_accum.items() if v}
        if diag_accum else None
    )
    return LayeredCell(
        dataset=dataset,
        variant=variant,
        ndcg_at_k=m.ndcg_at_k,
        recall_topk=recall_topk_hits / max(n_with_relevant, 1),
        recall_budget=recall_budget_hits / max(n_with_relevant, 1),
        mrr=m.mrr,
        p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        n_eval_entities=len(eval_entities),
        diagnostic=diag_summary,
    )


def _make_score_fns(config: LayeredConfig):
    """Closures matching the score_fn signature for _evaluate_scoring."""

    def score_cooc(engine, cand_ids, owned, history, candidates, diag):
        return _cooc_scores(engine, cand_ids, owned)

    def score_layered_pb(engine, cand_ids, owned, history, candidates, diag):
        primary = _cooc_scores(engine, cand_ids, owned)
        pb = _path_basket_scores(engine, cand_ids, history)
        rep = diagnostic_report(primary, {"path_basket": pb}, config=config)
        if "boost_magnitude" in rep:
            diag.setdefault("boost", []).append(rep["boost_magnitude"])
        if "layers" in rep and "path_basket" in rep["layers"]:
            diag.setdefault("pb_fire_rate", []).append(rep["layers"]["path_basket"]["fire_rate"])
        return layered_score(primary, [pb], config=config)

    def score_layered_sc(engine, cand_ids, owned, history, candidates, diag):
        primary = _cooc_scores(engine, cand_ids, owned)
        sc = _session_cooc_scores(engine, cand_ids, owned)
        rep = diagnostic_report(primary, {"session_cooc": sc}, config=config)
        diag.setdefault("boost", []).append(rep["boost_magnitude"])
        if "session_cooc" in rep["layers"]:
            diag.setdefault("sc_fire_rate", []).append(rep["layers"]["session_cooc"]["fire_rate"])
        return layered_score(primary, [sc], config=config)

    def score_layered_tc(engine, cand_ids, owned, history, candidates, diag):
        primary = _cooc_scores(engine, cand_ids, owned)
        tc = _temporal_cooc_scores(engine, cand_ids, owned)
        rep = diagnostic_report(primary, {"temporal_cooc": tc}, config=config)
        diag.setdefault("boost", []).append(rep["boost_magnitude"])
        if "temporal_cooc" in rep["layers"]:
            diag.setdefault("tc_fire_rate", []).append(rep["layers"]["temporal_cooc"]["fire_rate"])
        return layered_score(primary, [tc], config=config)

    def score_layered_stack(engine, cand_ids, owned, history, candidates, diag):
        primary = _cooc_scores(engine, cand_ids, owned)
        pb = _path_basket_scores(engine, cand_ids, history)
        sc = _session_cooc_scores(engine, cand_ids, owned)
        tc = _temporal_cooc_scores(engine, cand_ids, owned)
        rep = diagnostic_report(
            primary, {"path_basket": pb, "session_cooc": sc, "temporal_cooc": tc},
            config=config,
        )
        diag.setdefault("boost", []).append(rep["boost_magnitude"])
        for layer_name in ("path_basket", "session_cooc", "temporal_cooc"):
            if layer_name in rep["layers"]:
                key = f"{layer_name}_fire_rate"
                diag.setdefault(key, []).append(rep["layers"][layer_name]["fire_rate"])
        return layered_score(primary, [pb, sc, tc], config=config)

    return {
        "cooc_alone": score_cooc,
        "layered_cooc+pb": score_layered_pb,
        "layered_cooc+sc": score_layered_sc,
        "layered_cooc+tc": score_layered_tc,
        "layered_cooc+stack": score_layered_stack,
    }


def _evaluate_blend_baseline(
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    engine: Engine,
    catalog_size: int,
    k: int,
    dataset: str,
) -> LayeredCell:
    """The current Bayesian-blend default for comparison. Uses the
    full Engine.recommend() path."""
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    recall_topk_hits = 0
    recall_budget_hits = 0
    n_with_relevant = 0
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
                recall_budget_hits += 1  # blend doesn't expose budget pool here

    m = aggregate(per_entity, catalog_size=catalog_size, k=k)
    return LayeredCell(
        dataset=dataset,
        variant="bayesian_blend",
        ndcg_at_k=m.ndcg_at_k,
        recall_topk=recall_topk_hits / max(n_with_relevant, 1),
        recall_budget=recall_budget_hits / max(n_with_relevant, 1),
        mrr=m.mrr,
        p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        n_eval_entities=len(eval_entities),
    )


def run_probe(
    dataset: str,
    max_eval_entities: int = 500,
    retrieval_budget: int = 500,
    k: int = 10,
    test_fraction: float = 0.1,
    z_threshold: float = 2.0,
    boost_multiplier: float = 3.0,
    skip_heavy_signals: bool = False,
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== probe_layered: {dataset} ({len(split.train):,} train) ===", flush=True)

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
    eval_entities_all = sorted(
        set(train_items.index).intersection(test_items.index)
    )
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities = eval_entities_all[::step][:max_eval_entities]
    catalog_size = int(split.train["item_id"].nunique())

    print(f"  fitting engine ...", flush=True)
    t0 = time.perf_counter()
    engine_kwargs: dict[str, object] = {}
    if not skip_heavy_signals:
        from kindling.personas import KMeansClustering, PersonaConfig

        engine_kwargs["persona_config"] = PersonaConfig(
            enabled=True,
            clustering=KMeansClustering(n_clusters=30, random_state=0),
            min_activation_users=100,
        )
    engine = Engine(**engine_kwargs).fit(split.train)
    print(f"  engine fit: {time.perf_counter() - t0:.1f}s", flush=True)

    config = LayeredConfig(
        z_threshold=z_threshold,
        boost_multiplier=boost_multiplier,
    )
    score_fns = _make_score_fns(config)
    cells: list[LayeredCell] = []

    for variant, fn in score_fns.items():
        cell = _evaluate_scoring(
            variant=variant,
            score_fn=fn,
            eval_entities=eval_entities,
            train_items=train_items,
            test_items=test_items,
            engine=engine,
            catalog_size=catalog_size,
            retrieval_budget=retrieval_budget,
            k=k,
            dataset=dataset,
        )
        cells.append(cell)
        diag = f" diag={cell.diagnostic}" if cell.diagnostic else ""
        print(
            f"  {variant:<22}  NDCG={cell.ndcg_at_k:.4f}  "
            f"R@K={cell.recall_topk:.3f}  R@B={cell.recall_budget:.3f}  "
            f"p95={cell.p95_ms:.1f}ms{diag}",
            flush=True,
        )

    blend_cell = _evaluate_blend_baseline(
        eval_entities, train_items, test_items, engine, catalog_size, k, dataset,
    )
    cells.append(blend_cell)
    print(
        f"  {blend_cell.variant:<22}  NDCG={blend_cell.ndcg_at_k:.4f}  "
        f"R@K={blend_cell.recall_topk:.3f}  R@B={blend_cell.recall_budget:.3f}  "
        f"p95={blend_cell.p95_ms:.1f}ms",
        flush=True,
    )

    return {
        "dataset": dataset,
        "k": k,
        "n_eval_entities": len(eval_entities),
        "z_threshold": z_threshold,
        "boost_multiplier": boost_multiplier,
        "kindling_version": __version__,
        "cells": [c.as_dict() for c in cells],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe layered scoring (cooc + z-gated boost layers) vs Bayesian blend."
    )
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=[
            "movielens-1m", "synthetic-grocery", "synthetic-grocery-deep",
            "retailrocket", "instacart", "gowalla", "yelp2018",
            "tafeng", "dunnhumby", "amazon-beauty", "amazon-book",
        ],
    )
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--retrieval-budget", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--boost-multiplier", type=float, default=3.0)
    parser.add_argument("--skip-heavy-signals", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_probe(
        dataset=args.dataset,
        max_eval_entities=args.max_eval_entities,
        retrieval_budget=args.retrieval_budget,
        k=args.k,
        z_threshold=args.z_threshold,
        boost_multiplier=args.boost_multiplier,
        skip_heavy_signals=args.skip_heavy_signals,
    )
    pretty = json.dumps(report, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"\nWrote {args.output}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
