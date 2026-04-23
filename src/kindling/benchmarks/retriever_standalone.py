"""Per-signal retriever standalone-evaluation diagnostic.

For each signal that makes sense as a retriever, treat it as the complete
recommender: it retrieves its own candidates, sorts by its own score,
takes top-N. No Bayesian blend, no LambdaRank, no re-rank. Just the
signal's raw ranking quality.

This isolates each signal's contribution from the blend's ability (or
failure) to combine them. Answers:

1. What NDCG does each signal actually know, standalone?
2. Which signals see candidates the others miss (complementarity)?
3. Where does each signal win; where does it lose?

CLI:
    python -m kindling.benchmarks.retriever_standalone \
        --dataset synthetic-grocery-deep \
        --fraction 1.0 \
        --max-eval-entities 500 \
        --output bench/reports/retriever_standalone_grocery.json
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
from kindling.benchmarks.metrics import MetricReport, aggregate
from kindling.personas import KMeansClustering, PersonaConfig
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.path_endpoint import PathEndpointRetriever
from kindling.retrieve.protocol import Candidate
from kindling.retrieve.signal_retrievers import (
    ALSRetriever,
    CosineRetriever,
    PathBasketRetriever,
    PathFullRetriever,
    PathTailRetriever,
    PersonaRetriever,
)

MAX_QUERY_BASKET_SIZE = 50


@dataclass(frozen=True)
class RetrieverResult:
    name: str
    ndcg_at_k: float
    recall_at_k: float
    mrr: float
    coverage: float
    recall_budget: float  # fraction of eval entities whose positive was in the top-budget retrieval
    recall_topk: float    # fraction whose positive landed in the top-K (same K as ndcg)
    p95_ms: float
    fit_overhead_s: float
    n_eval_entities: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "coverage": self.coverage,
            "recall_budget": self.recall_budget,
            "recall_topk": self.recall_topk,
            "p95_ms": self.p95_ms,
            "fit_overhead_s": self.fit_overhead_s,
            "n_eval_entities": self.n_eval_entities,
        }


def _evaluate_retriever(
    name: str,
    retrieve_fn,
    eval_entities: list[object],
    test_items_by_entity: pd.Series,
    train_items_by_entity: pd.Series,
    catalog_size: int,
    retrieval_budget: int,
    k: int,
) -> RetrieverResult:
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    recall_budget_hits = 0
    recall_topk_hits = 0
    n_with_relevant = 0
    for entity in eval_entities:
        train_owned = train_items_by_entity.get(entity, set())
        test_owned = test_items_by_entity.get(entity, set())
        relevant = test_owned - train_owned
        t0 = time.perf_counter()
        candidates = retrieve_fn(entity)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        rec_items = [c.item_id for c in candidates[:k]]
        per_entity.append((rec_items, relevant))
        if relevant:
            n_with_relevant += 1
            all_retrieved_ids = {c.item_id for c in candidates[:retrieval_budget]}
            top_k_ids = set(rec_items)
            if all_retrieved_ids & relevant:
                recall_budget_hits += 1
            if top_k_ids & relevant:
                recall_topk_hits += 1

    metrics = aggregate(per_entity, catalog_size=catalog_size, k=k)
    return RetrieverResult(
        name=name,
        ndcg_at_k=metrics.ndcg_at_k,
        recall_at_k=metrics.recall_at_k,
        mrr=metrics.mrr,
        coverage=metrics.coverage,
        recall_budget=recall_budget_hits / max(n_with_relevant, 1),
        recall_topk=recall_topk_hits / max(n_with_relevant, 1),
        p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        fit_overhead_s=0.0,
        n_eval_entities=metrics.n_entities_evaluated,
    )


def run_standalone(
    dataset: str,
    fraction: float = 1.0,
    k: int = 10,
    retrieval_budget: int = 500,
    max_eval_entities: int = 500,
    test_fraction: float = 0.1,
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    n_take = int(round(len(split.train) * fraction))
    train_sub = split.train.iloc[:n_take].reset_index(drop=True)

    train_items = cast(
        pd.Series,
        train_sub.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    test_items = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    eval_entities_all: list[object] = sorted(
        set(train_items.index).intersection(test_items.index)
    )
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities: list[object] = eval_entities_all[::step][:max_eval_entities]
    catalog_size = int(train_sub["item_id"].nunique())

    # Fit an engine ONCE so all retrievers share the same fitted state.
    persona_cfg = PersonaConfig(
        enabled=True,
        clustering=KMeansClustering(n_clusters=30, random_state=0),
        min_activation_users=100,
    )
    print(f"Fitting engine on {dataset} frac={fraction} ({len(train_sub):,} interactions)...", flush=True)
    fit_start = time.perf_counter()
    engine = Engine(persona_config=persona_cfg).fit(train_sub)
    fit_s = time.perf_counter() - fit_start
    print(f"  fit: {fit_s:.1f}s", flush=True)

    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)

    # Construct per-signal retrievers from the engine's fitted state.
    retrievers: dict[str, object] = {
        "path_tail": PathTailRetriever(engine._tail_index, item_ids),
        "path_full": PathFullRetriever(engine._path_tree, item_ids),
        "path_basket": PathBasketRetriever(engine._basket_index, item_ids),
        "cooccurrence": CoOccurrenceRetriever(engine._item_graph),
        "path_endpoint_combined": PathEndpointRetriever(engine._path_tree, engine._tail_index),
    }
    if engine._item_cosine is not None:
        retrievers["item_item_cosine"] = CosineRetriever(
            engine._item_cosine, engine._item_graph, item_ids
        )
    if engine._als_factors is not None:
        retrievers["als_factor"] = ALSRetriever(engine._als_factors, engine._item_graph, item_ids)
    if engine._persona_index is not None and engine._persona_index.n_personas > 0:
        retrievers["persona"] = PersonaRetriever(engine._persona_index, item_ids)

    # Entity state needed at query time.
    owned_by_entity = engine._owned_by_entity
    history_by_entity = engine._history_by_entity

    def retrieve_for(retriever_name: str):
        r = retrievers[retriever_name]

        def inner(entity_id: object) -> list[Candidate]:
            owned = owned_by_entity.get(entity_id, np.array([]))
            history = history_by_entity.get(entity_id, ())
            exclude: set[object] = set(owned.tolist()) if owned.size else set()
            query_basket = frozenset(history[-MAX_QUERY_BASKET_SIZE:])
            if retriever_name == "cooccurrence":
                return r.retrieve(owned, retrieval_budget)  # type: ignore[attr-defined]
            if retriever_name == "path_endpoint_combined":
                return r.retrieve(  # type: ignore[attr-defined]
                    recent_history=history, budget=retrieval_budget, exclude=exclude
                )
            if retriever_name in ("path_tail", "path_full"):
                return r.retrieve(  # type: ignore[attr-defined]
                    recent_history=history, budget=retrieval_budget, exclude=exclude
                )
            if retriever_name == "path_basket":
                return r.retrieve(  # type: ignore[attr-defined]
                    query_basket=query_basket, budget=retrieval_budget, exclude=exclude
                )
            if retriever_name == "item_item_cosine":
                return r.retrieve(  # type: ignore[attr-defined]
                    owned_items=owned, budget=retrieval_budget, exclude=exclude
                )
            if retriever_name == "als_factor":
                return r.retrieve(  # type: ignore[attr-defined]
                    entity_id=entity_id, budget=retrieval_budget, exclude=exclude
                )
            if retriever_name == "persona":
                return r.retrieve(  # type: ignore[attr-defined]
                    entity_id=entity_id,
                    owned_items=owned,
                    history=history,
                    budget=retrieval_budget,
                    exclude=exclude,
                )
            raise ValueError(f"unknown retriever: {retriever_name}")

        return inner

    results: list[RetrieverResult] = []
    for name in retrievers:
        print(f"  evaluating {name}...", flush=True)
        res = _evaluate_retriever(
            name=name,
            retrieve_fn=retrieve_for(name),
            eval_entities=eval_entities,
            test_items_by_entity=test_items,
            train_items_by_entity=train_items,
            catalog_size=catalog_size,
            retrieval_budget=retrieval_budget,
            k=k,
        )
        results.append(res)
        print(
            f"    NDCG={res.ndcg_at_k:.4f} MRR={res.mrr:.4f} "
            f"recall@budget={res.recall_budget:.2f} recall@{k}={res.recall_topk:.2f} "
            f"cov={res.coverage:.3f} p95={res.p95_ms:.1f}ms",
            flush=True,
        )

    return {
        "dataset": dataset,
        "fraction": fraction,
        "n_train_interactions": len(train_sub),
        "n_eval_entities": len(eval_entities),
        "retrieval_budget": retrieval_budget,
        "k": k,
        "kindling_version": __version__,
        "results": [r.as_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate each signal as a standalone retriever/ranker."
    )
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=["movielens-1m", "synthetic-grocery", "synthetic-grocery-deep"],
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--retrieval-budget", type=int, default=500)
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_standalone(
        dataset=args.dataset,
        fraction=args.fraction,
        k=args.k,
        retrieval_budget=args.retrieval_budget,
        max_eval_entities=args.max_eval_entities,
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
