"""Union-retrieval measurement: do combinations of retrievers beat the
best individual retriever?

For each dataset, fit the engine once, then run multiple union
configurations. Each config specifies a set of retrievers and (optional)
per-retriever budgets. Candidates from all retrievers are concatenated
and deduplicated by max score; top-K is taken from the combined list.

Measures whether the retrievers surface complementary candidates (so
their union expands the reachable NDCG ceiling) or just overlap (so the
union wastes budget without gaining recall).

CLI:
    python -m kindling.benchmarks.retriever_union \
        --dataset synthetic-grocery-deep \
        --fraction 1.0 \
        --output bench/reports/retriever_union_grocery.json
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
class UnionResult:
    name: str
    retrievers: list[str]
    per_retriever_budget: int
    total_budget: int
    ndcg_at_k: float
    mrr: float
    coverage: float
    recall_budget: float
    recall_topk: float
    p95_ms: float
    n_eval_entities: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "retrievers": self.retrievers,
            "per_retriever_budget": self.per_retriever_budget,
            "total_budget": self.total_budget,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "coverage": self.coverage,
            "recall_budget": self.recall_budget,
            "recall_topk": self.recall_topk,
            "p95_ms": self.p95_ms,
            "n_eval_entities": self.n_eval_entities,
        }


def _dedup_union(candidates: list[Candidate], budget: int) -> list[Candidate]:
    """Keep the max-score Candidate per item_id, sort desc by score.

    NOTE: produces biased results when retrievers have different score
    scales. Cooc scores are raw edge weights (100-10000); cosine and
    persona are [0,1]; ALS is ~[0, 5]. Max-score sort always picks
    cooc's top regardless of the others' relevance. Use ``_rrf_union``
    for score-scale-independent fusion.
    """
    best: dict[object, Candidate] = {}
    for c in candidates:
        existing = best.get(c.item_id)
        if existing is None or c.score > existing.score:
            best[c.item_id] = c
    out = sorted(best.values(), key=lambda c: -c.score)
    return out[:budget]


def _rrf_union(
    candidates_by_retriever: dict[str, list[Candidate]],
    budget: int,
    k_constant: float = 60.0,
) -> list[Candidate]:
    """Reciprocal Rank Fusion: score(item) = Σ_r 1/(k + rank_r(item)).

    Each retriever's candidates are already sorted by their own score
    (so position 0 in the list = rank 1). Items are scored by their
    RRF contribution across retrievers; items not surfaced by a
    retriever contribute 0 from that retriever. Item provenance is
    merged into a single Candidate with the aggregated score and
    source = "rrf".
    """
    rrf_scores: dict[object, float] = {}
    for retriever_name, cands in candidates_by_retriever.items():
        for rank_zero_based, c in enumerate(cands):
            rank = rank_zero_based + 1
            delta = 1.0 / (k_constant + rank)
            rrf_scores[c.item_id] = rrf_scores.get(c.item_id, 0.0) + delta
    merged = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:budget]
    return [Candidate(item_id=item, score=score, source="rrf") for item, score in merged]


def _fit_engine(dataset: str, fraction: float) -> tuple[Engine, pd.DataFrame]:
    split = _load_dataset(dataset, test_fraction=0.1)
    n_take = int(round(len(split.train) * fraction))
    train_sub = split.train.iloc[:n_take].reset_index(drop=True)
    cfg = PersonaConfig(
        enabled=True,
        clustering=KMeansClustering(n_clusters=30, random_state=0),
        min_activation_users=100,
        cold_start_weight=0.0,  # off by default - scale bug lives here, see ADR
    )
    engine = Engine(persona_config=cfg).fit(train_sub)
    return engine, split.test


def _build_retriever_dict(engine: Engine) -> dict[str, object]:
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    retrievers: dict[str, object] = {
        "cooccurrence": CoOccurrenceRetriever(engine._item_graph),
        "path_tail": PathTailRetriever(engine._tail_index, item_ids),
        "path_full": PathFullRetriever(engine._path_tree, item_ids),
        "path_basket": PathBasketRetriever(engine._basket_index, item_ids),
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
    return retrievers


def _retrieve_union(
    retrievers: dict[str, object],
    names: list[str],
    per_budget: int,
    entity_id: object,
    engine: Engine,
) -> dict[str, list[Candidate]]:
    """Run each retriever, return per-retriever candidate lists.

    Keeps the retriever name so the caller can choose max-score merge
    or rank-based RRF fusion.
    """
    owned = engine._owned_by_entity.get(entity_id, np.array([]))
    history = engine._history_by_entity.get(entity_id, ())
    exclude: set[object] = set(owned.tolist()) if owned.size else set()
    query_basket = frozenset(history[-MAX_QUERY_BASKET_SIZE:])
    per_retriever: dict[str, list[Candidate]] = {}
    for name in names:
        r = retrievers.get(name)
        if r is None:
            continue
        if name == "cooccurrence":
            out = r.retrieve(owned, per_budget)  # type: ignore[attr-defined]
        elif name == "path_endpoint_combined":
            out = r.retrieve(  # type: ignore[attr-defined]
                recent_history=history, budget=per_budget, exclude=exclude
            )
        elif name in ("path_tail", "path_full"):
            out = r.retrieve(  # type: ignore[attr-defined]
                recent_history=history, budget=per_budget, exclude=exclude
            )
        elif name == "path_basket":
            out = r.retrieve(  # type: ignore[attr-defined]
                query_basket=query_basket, budget=per_budget, exclude=exclude
            )
        elif name == "item_item_cosine":
            out = r.retrieve(  # type: ignore[attr-defined]
                owned_items=owned, budget=per_budget, exclude=exclude
            )
        elif name == "als_factor":
            out = r.retrieve(  # type: ignore[attr-defined]
                entity_id=entity_id, budget=per_budget, exclude=exclude
            )
        elif name == "persona":
            out = r.retrieve(  # type: ignore[attr-defined]
                entity_id=entity_id,
                owned_items=owned,
                history=history,
                budget=per_budget,
                exclude=exclude,
            )
        else:
            continue
        per_retriever[name] = out
    return per_retriever


def _evaluate_union(
    config_name: str,
    retrievers: dict[str, object],
    names: list[str],
    per_budget: int,
    engine: Engine,
    eval_entities: list[object],
    test_items: pd.Series,
    train_items: pd.Series,
    catalog_size: int,
    k: int,
    fusion: str = "rrf",
) -> UnionResult:
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    recall_budget_hits = 0
    recall_topk_hits = 0
    n_with_relevant = 0
    total_budget = per_budget * len(names)

    for entity in eval_entities:
        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        relevant = test_owned - train_owned
        t0 = time.perf_counter()
        per_retriever = _retrieve_union(retrievers, names, per_budget, entity, engine)
        if fusion == "rrf":
            candidates = _rrf_union(per_retriever, total_budget)
        elif fusion == "max_score":
            flat = [c for lst in per_retriever.values() for c in lst]
            candidates = _dedup_union(flat, total_budget)
        else:
            raise ValueError(f"unknown fusion: {fusion}")
        latencies.append((time.perf_counter() - t0) * 1000.0)
        rec_items = [c.item_id for c in candidates[:k]]
        per_entity.append((rec_items, relevant))
        if relevant:
            n_with_relevant += 1
            union_ids = {c.item_id for c in candidates}
            if union_ids & relevant:
                recall_budget_hits += 1
            if set(rec_items) & relevant:
                recall_topk_hits += 1

    metrics = aggregate(per_entity, catalog_size=catalog_size, k=k)
    return UnionResult(
        name=config_name,
        retrievers=names,
        per_retriever_budget=per_budget,
        total_budget=total_budget,
        ndcg_at_k=metrics.ndcg_at_k,
        mrr=metrics.mrr,
        coverage=metrics.coverage,
        recall_budget=recall_budget_hits / max(n_with_relevant, 1),
        recall_topk=recall_topk_hits / max(n_with_relevant, 1),
        p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        n_eval_entities=metrics.n_entities_evaluated,
    )


def run_union(
    dataset: str,
    fraction: float = 1.0,
    k: int = 10,
    per_retriever_budget: int = 100,
    max_eval_entities: int = 500,
    fusion: str = "rrf",
) -> dict[str, object]:
    engine, test_df = _fit_engine(dataset, fraction)
    train_items = cast(
        pd.Series,
        engine._interactions.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    test_items = cast(
        pd.Series,
        test_df.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    eval_entities_all: list[object] = sorted(
        set(train_items.index).intersection(test_items.index)
    )
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities: list[object] = eval_entities_all[::step][:max_eval_entities]
    catalog_size = int(engine._interactions["item_id"].nunique())

    retrievers = _build_retriever_dict(engine)

    # Define a ladder of union configs. Each one builds on the previous.
    configs: list[tuple[str, list[str]]] = [
        ("cooc_only", ["cooccurrence"]),
        ("cooc+als", ["cooccurrence", "als_factor"]),
        ("cooc+als+cosine", ["cooccurrence", "als_factor", "item_item_cosine"]),
        ("cooc+als+path_basket", ["cooccurrence", "als_factor", "path_basket"]),
        ("cooc+als+cosine+path_basket", ["cooccurrence", "als_factor", "item_item_cosine", "path_basket"]),
        ("cooc+als+path_basket+persona", ["cooccurrence", "als_factor", "path_basket", "persona"]),
        ("all_useful",
            ["cooccurrence", "item_item_cosine", "als_factor", "path_basket", "path_tail", "persona"]),
        ("engine_current_stage1",
            ["cooccurrence", "path_endpoint_combined"]),
    ]
    # Only include configs whose retrievers actually exist (ALS needs
    # `implicit`; persona needs the index).
    configs = [
        (name, names)
        for name, names in configs
        if all(n in retrievers for n in names)
    ]

    results: list[UnionResult] = []
    for name, names in configs:
        print(f"  evaluating {name} ({len(names)} retrievers, {per_retriever_budget}/each)...",
              flush=True)
        res = _evaluate_union(
            config_name=name,
            retrievers=retrievers,
            names=names,
            per_budget=per_retriever_budget,
            engine=engine,
            eval_entities=eval_entities,
            test_items=test_items,
            train_items=train_items,
            catalog_size=catalog_size,
            k=k,
            fusion=fusion,
        )
        results.append(res)
        print(
            f"    NDCG={res.ndcg_at_k:.4f} MRR={res.mrr:.4f} "
            f"recall@union={res.recall_budget:.3f} recall@{k}={res.recall_topk:.3f} "
            f"p95={res.p95_ms:.1f}ms",
            flush=True,
        )

    return {
        "dataset": dataset,
        "fraction": fraction,
        "n_eval_entities": len(eval_entities),
        "per_retriever_budget": per_retriever_budget,
        "fusion": fusion,
        "k": k,
        "kindling_version": __version__,
        "results": [r.as_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure union-of-retrievers NDCG vs best individual retriever."
    )
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=["movielens-1m", "synthetic-grocery", "synthetic-grocery-deep"],
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--per-retriever-budget", type=int, default=100)
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--fusion", default="rrf", choices=["rrf", "max_score"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_union(
        dataset=args.dataset,
        fraction=args.fraction,
        k=args.k,
        per_retriever_budget=args.per_retriever_budget,
        max_eval_entities=args.max_eval_entities,
        fusion=args.fusion,
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
