"""Focused probe for interaction_network and interaction_neighborhood.

Builds the temporal graph + signal models on top of an existing fitted
Engine (so we get cooc + path + cosine + ALS for free as baselines)
and evaluates each NEW signal as a standalone recommender on the
500-eval-entity split. Apples-to-apples diagonal-cell comparison.

For interaction_neighborhood, runs the centrality fallback sweep
(betweenness, pagerank, eigenvector, degree, closeness) to surface
which centrality measure delivers — without committing to an
architecture choice in the engine.

CLI:
    python -m kindling.benchmarks.probe_temporal_signals \\
        --dataset synthetic-grocery-deep \\
        --output bench/reports/probe_temporal_grocery.json
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
from kindling.graph.temporal_interaction import build_temporal_interaction_graph
from kindling.retrieve.interaction_network import (
    InteractionNetworkConfig,
    build_interaction_network,
)
from kindling.retrieve.interaction_neighborhood import (
    ALL_CENTRALITIES,
    InteractionNeighborhoodConfig,
    build_interaction_neighborhood,
)
from kindling.retrieve.protocol import Candidate

MAX_QUERY_BASKET_SIZE = 50


@dataclass(frozen=True)
class ProbeCell:
    dataset: str
    signal: str
    variant: str  # for centrality variants on interaction_neighborhood
    ndcg_at_k: float
    recall_topk: float
    recall_budget: float
    mrr: float
    p95_ms: float
    n_eval_entities: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "signal": self.signal,
            "variant": self.variant,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_topk": self.recall_topk,
            "recall_budget": self.recall_budget,
            "mrr": self.mrr,
            "p95_ms": self.p95_ms,
            "n_eval_entities": self.n_eval_entities,
        }


def _evaluate_retriever(
    name: str,
    variant: str,
    retrieve_fn,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    owned_by_entity: dict,
    history_by_entity: dict,
    catalog_size: int,
    retrieval_budget: int,
    k: int,
    dataset: str,
) -> ProbeCell:
    """Run a retriever standalone and compute the diagonal-cell metrics.

    ``retrieve_fn(entity_id, owned, history, exclude, budget) -> list[Candidate]``
    is the only thing the probe needs from each retriever.
    """
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    recall_topk_hits = 0
    recall_budget_hits = 0
    n_with_relevant = 0

    for entity in eval_entities:
        owned = owned_by_entity.get(entity, np.array([]))
        history = history_by_entity.get(entity, ())
        exclude = set(owned.tolist()) if owned.size else set()
        t0 = time.perf_counter()
        candidates = retrieve_fn(entity, owned, history, exclude, retrieval_budget)
        cand_ids = [c.item_id for c in candidates]
        scored = sorted([(c.item_id, c.score) for c in candidates], key=lambda kv: -kv[1])
        top = [item for item, score in scored[:k] if score > 0.0]
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
    return ProbeCell(
        dataset=dataset,
        signal=name,
        variant=variant,
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
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== probe: {dataset} ({len(split.train):,} train) ===", flush=True)

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
    eval_entities_all: list[object] = sorted(
        set(train_items.index).intersection(test_items.index)
    )
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities = eval_entities_all[::step][:max_eval_entities]
    catalog_size = int(split.train["item_id"].nunique())

    print(f"  fitting engine (cooc baseline + item ordering) ...", flush=True)
    t0 = time.perf_counter()
    engine = Engine().fit(split.train)
    print(f"  engine fit: {time.perf_counter() - t0:.1f}s", flush=True)

    item_index = engine._item_graph.item_index
    item_ids = engine._item_graph.item_ids

    # Build temporal substrate.
    print(f"  building temporal interaction graph ...", flush=True)
    t0 = time.perf_counter()
    tgraph = build_temporal_interaction_graph(split.train, dict(item_index))
    print(
        f"  temporal graph: {time.perf_counter() - t0:.1f}s, "
        f"n_edges={tgraph.n_edges:,}, kernel={tgraph.kernel_params.strategy}",
        flush=True,
    )

    cells: list[ProbeCell] = []

    # --- baseline: cooccurrence (already in engine; mirror the matrix harness) ---
    from kindling.engine import _cooccurrence_signal
    from kindling.retrieve.cooccurrence import CoOccurrenceRetriever

    cooc_retriever = CoOccurrenceRetriever(engine._item_graph)

    def retrieve_cooc(entity, owned, history, exclude, budget):
        return cooc_retriever.retrieve(owned, budget)

    cell = _evaluate_retriever(
        "cooccurrence", "default", retrieve_cooc,
        eval_entities, train_items, test_items,
        engine._owned_by_entity, engine._history_by_entity,
        catalog_size, retrieval_budget, k, dataset,
    )
    cells.append(cell)
    print(
        f"  cooccurrence            NDCG={cell.ndcg_at_k:.4f}  "
        f"R@K={cell.recall_topk:.3f}  R@B={cell.recall_budget:.3f}  "
        f"p95={cell.p95_ms:.1f}ms",
        flush=True,
    )

    # --- interaction_network (PPR on temporal graph) ---
    print(f"  building interaction_network model ...", flush=True)
    t0 = time.perf_counter()
    network_model = build_interaction_network(tgraph)
    print(f"  interaction_network model: {time.perf_counter() - t0:.2f}s", flush=True)

    if network_model is None:
        print(f"  interaction_network: SKIPPED (no edges in temporal graph)", flush=True)
    else:
        def retrieve_network(entity, owned, history, exclude, budget):
            return network_model.retrieve(
                entity_id=entity,
                owned_items=owned,
                history=history,
                budget=budget,
                exclude=exclude,
            )

        variant = (
            f"alpha=0.15;kernel={tgraph.kernel_params.strategy}"
        )
        cell = _evaluate_retriever(
            "interaction_network", variant, retrieve_network,
            eval_entities, train_items, test_items,
            engine._owned_by_entity, engine._history_by_entity,
            catalog_size, retrieval_budget, k, dataset,
        )
        cells.append(cell)
        print(
            f"  interaction_network     NDCG={cell.ndcg_at_k:.4f}  "
            f"R@K={cell.recall_topk:.3f}  R@B={cell.recall_budget:.3f}  "
            f"p95={cell.p95_ms:.1f}ms",
            flush=True,
        )

    # --- interaction_network with kernel forced to pure-count ---
    # On timestamped datasets, this isolates "what does the temporal
    # weighting add over the same walk on a binary-edge graph?"
    if not tgraph.kernel_params.pure_count:
        from kindling.graph.temporal_interaction import KernelParams

        pc_params = KernelParams(
            midpoint_seconds=tgraph.kernel_params.midpoint_seconds,
            steepness_seconds=tgraph.kernel_params.steepness_seconds,
            pure_count=True,
            strategy="pure_count_forced",
        )
        print(f"  building pure-count temporal graph (no kernel) ...", flush=True)
        t0 = time.perf_counter()
        tgraph_pc = build_temporal_interaction_graph(
            split.train, dict(item_index), kernel_params=pc_params
        )
        network_model_pc = build_interaction_network(tgraph_pc)
        print(f"  pure-count graph: {time.perf_counter() - t0:.1f}s, n_edges={tgraph_pc.n_edges:,}", flush=True)

        if network_model_pc is not None:
            def retrieve_network_pc(entity, owned, history, exclude, budget):
                return network_model_pc.retrieve(
                    entity_id=entity,
                    owned_items=owned,
                    history=history,
                    budget=budget,
                    exclude=exclude,
                )

            cell = _evaluate_retriever(
                "interaction_network", "alpha=0.15;kernel=pure_count_forced", retrieve_network_pc,
                eval_entities, train_items, test_items,
                engine._owned_by_entity, engine._history_by_entity,
                catalog_size, retrieval_budget, k, dataset,
            )
            cells.append(cell)
            print(
                f"  interaction_network[pc] NDCG={cell.ndcg_at_k:.4f}  "
                f"R@K={cell.recall_topk:.3f}  R@B={cell.recall_budget:.3f}  "
                f"p95={cell.p95_ms:.1f}ms",
                flush=True,
            )

    # --- interaction_neighborhood with all 5 centrality variants ---
    print(f"  building interaction_neighborhood (Louvain + caches) ...", flush=True)
    t0 = time.perf_counter()
    nbhd_model = build_interaction_neighborhood(tgraph)
    nbhd_build_s = time.perf_counter() - t0
    if nbhd_model is None:
        print(f"  interaction_neighborhood: SKIPPED (no edges or no communities)", flush=True)
    else:
        sizes = sorted(
            (m.size for m in nbhd_model.community_members.values()),
            reverse=True,
        )
        print(
            f"  louvain: {nbhd_build_s:.1f}s, n_communities={nbhd_model.n_communities}, "
            f"size dist (top): {sizes[:5]}",
            flush=True,
        )
        for cent in ALL_CENTRALITIES:
            def retrieve_nbhd(entity, owned, history, exclude, budget, _cent=cent):
                return nbhd_model.retrieve(
                    entity_id=entity,
                    owned_items=owned,
                    history=history,
                    budget=budget,
                    exclude=exclude,
                    centrality_override=_cent,
                )

            cell = _evaluate_retriever(
                "interaction_neighborhood", cent, retrieve_nbhd,
                eval_entities, train_items, test_items,
                engine._owned_by_entity, engine._history_by_entity,
                catalog_size, retrieval_budget, k, dataset,
            )
            cells.append(cell)
            print(
                f"  nbhd[{cent:<11}]    NDCG={cell.ndcg_at_k:.4f}  "
                f"R@K={cell.recall_topk:.3f}  R@B={cell.recall_budget:.3f}  "
                f"p95={cell.p95_ms:.1f}ms",
                flush=True,
            )

    return {
        "dataset": dataset,
        "k": k,
        "n_eval_entities": len(eval_entities),
        "kindling_version": __version__,
        "temporal_graph": {
            "n_edges": tgraph.n_edges,
            "kernel_strategy": tgraph.kernel_params.strategy,
            "kernel_midpoint_seconds": tgraph.kernel_params.midpoint_seconds,
            "kernel_steepness_seconds": tgraph.kernel_params.steepness_seconds,
            "kernel_pure_count": tgraph.kernel_params.pure_count,
        },
        "neighborhood": {
            "n_communities": (nbhd_model.n_communities if nbhd_model else 0),
            "louvain_build_seconds": nbhd_build_s,
        } if nbhd_model else None,
        "cells": [c.as_dict() for c in cells],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Focused probe for interaction_network + interaction_neighborhood signals."
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_probe(
        dataset=args.dataset,
        max_eval_entities=args.max_eval_entities,
        retrieval_budget=args.retrieval_budget,
        k=args.k,
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
