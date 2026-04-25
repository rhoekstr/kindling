"""Probe persona-as-candidate-expansion-retriever on cold-start users.

Tests the user's hypothesis: persona's value isn't as a top-of-blend
ranker (NDCG ~0.20s) but as a candidate-expansion retriever - it
takes a sparse user (few training interactions) and uses cluster-
level taste to surface items cooc would miss.

Methodology:

1. Fit engine WITH persona on the dataset.
2. Stratify test users by training-history size:
     very_cold:  <= 3 interactions
     cold:       4-10
     warm:       11-30
     hot:        > 30
3. For each stratum, evaluate three retrieval strategies (all using
   cooc as the FINAL ranker over the candidate pool, so we isolate
   the candidate-expansion question):

   - cooc_only:    top-budget candidates from cooc retrieval
   - cooc + persona union:  cooc top-budget UNION persona top-N (deduplicated;
                             max-score by cooc)
   - persona_only: top-budget from persona retrieval (sanity baseline)

4. Compare NDCG@10, recall@K, recall@budget per stratum.

Expected pattern if persona-as-retriever is valuable:

   stratum         cooc_only   cooc+persona   delta
   very_cold       low         lifted         large
   cold            mid         lifted         moderate
   warm            high        ~tied          small
   hot             high        ~tied          ~zero

CLI:
    python -m kindling.benchmarks.probe_persona_coldstart \\
        --dataset synthetic-grocery-deep \\
        --output bench/reports/probe_persona_coldstart_grocery.json
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
from kindling.engine import _cooccurrence_signal
from kindling.personas import KMeansClustering, PersonaConfig
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.protocol import Candidate
from kindling.retrieve.signal_retrievers import PersonaRetriever


STRATUM_BOUNDARIES = [
    ("very_cold", 0, 3),     # <= 3
    ("cold", 4, 10),          # 4-10
    ("warm", 11, 30),         # 11-30
    ("hot", 31, 1_000_000),    # > 30
]


@dataclass
class StratumResult:
    stratum: str
    n_users: int
    history_range: tuple[int, int]
    cooc_only: dict
    cooc_union_persona: dict
    persona_only: dict


def _classify_stratum(history_size: int) -> str:
    for name, lo, hi in STRATUM_BOUNDARIES:
        if lo <= history_size <= hi:
            return name
    return "hot"


def _evaluate_strategy(
    label: str,
    retrieve_fn,
    rank_with_cooc: bool,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    engine: Engine,
    catalog_size: int,
    k: int,
) -> dict:
    """Score one retrieval strategy on a set of entities. Optionally
    re-rank the retrieved candidates by cooc (so we test
    candidate-expansion-then-cooc-rank vs candidate-expansion-only)."""
    per_entity = []
    n_with_relevant = 0
    recall_topk_hits = 0
    recall_budget_hits = 0
    for entity in eval_entities:
        owned = engine._owned_by_entity.get(entity, np.array([]))
        history = engine._history_by_entity.get(entity, ())
        candidates = retrieve_fn(entity, owned, history)
        if not candidates:
            top: list[object] = []
        else:
            cand_ids = [c.item_id for c in candidates]
            if rank_with_cooc:
                # Rank the retrieved candidates by cooc score against owned.
                cooc_scores = _cooccurrence_signal(cand_ids, owned, engine._item_graph)
                # Tie-break with the retrieval score so persona-only items
                # whose cooc is 0 still get an order.
                retrieval_scores = np.asarray([c.score for c in candidates], dtype=np.float64)
                composite = cooc_scores + 1e-9 * retrieval_scores
                order = np.argsort(-composite)
                top = [cand_ids[int(i)] for i in order[:k]]
            else:
                top = cand_ids[:k]
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
    return {
        "label": label,
        "ndcg_at_k": m.ndcg_at_k,
        "mrr": m.mrr,
        "recall_topk": recall_topk_hits / max(n_with_relevant, 1),
        "recall_budget": recall_budget_hits / max(n_with_relevant, 1),
    }


def run(
    dataset: str,
    max_eval_entities: int = 1000,
    retrieval_budget: int = 200,
    persona_share: int = 100,  # how many candidates from persona side
    k: int = 10,
    test_fraction: float = 0.1,
) -> dict:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== probe_persona_coldstart: {dataset} ({len(split.train):,} train) ===", flush=True)

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

    print(f"  fitting engine WITH persona ...", flush=True)
    t0 = time.perf_counter()
    engine = Engine(
        persona_config=PersonaConfig(
            enabled=True,
            clustering=KMeansClustering(n_clusters=30, random_state=0),
            min_activation_users=50,
        )
    ).fit(split.train)
    fit_seconds = time.perf_counter() - t0
    print(f"    fit {fit_seconds:.1f}s", flush=True)

    if engine._persona_index is None or engine._persona_index.n_personas == 0:
        print("  persona index not fitted; aborting.", flush=True)
        return {"dataset": dataset, "error": "persona not fitted"}

    print(f"    persona n={engine._persona_index.n_personas} personas", flush=True)

    # Classify users into strata.
    user_history_size = {
        ent: len(train_items.get(ent, set())) for ent in eligible_users
    }
    by_stratum: dict[str, list[object]] = {s[0]: [] for s in STRATUM_BOUNDARIES}
    for ent, sz in user_history_size.items():
        by_stratum[_classify_stratum(sz)].append(ent)

    print(f"  user counts by stratum:", flush=True)
    for stratum_name in [s[0] for s in STRATUM_BOUNDARIES]:
        n = len(by_stratum[stratum_name])
        print(f"    {stratum_name:<12} {n:,} users", flush=True)

    # Build retrievers.
    item_ids = np.asarray(engine._item_graph.item_ids, dtype=object)
    cooc_retriever = CoOccurrenceRetriever(engine._item_graph)
    persona_retriever = PersonaRetriever(
        persona_index=engine._persona_index, item_ids=item_ids,
    )

    def retrieve_cooc(entity, owned, history):
        return cooc_retriever.retrieve(owned, retrieval_budget)

    def retrieve_persona(entity, owned, history):
        return persona_retriever.retrieve(
            entity_id=entity, owned_items=owned, history=history,
            budget=retrieval_budget, exclude=set(owned.tolist()) if owned.size else None,
        )

    def retrieve_union(entity, owned, history):
        # cooc top-budget + persona top-persona_share, deduplicated by item_id
        # with max(cooc_score, persona_score). This is the candidate-
        # expansion strategy the user proposed.
        cooc_cands = cooc_retriever.retrieve(owned, retrieval_budget)
        persona_cands = persona_retriever.retrieve(
            entity_id=entity, owned_items=owned, history=history,
            budget=persona_share, exclude=set(owned.tolist()) if owned.size else None,
        )
        seen = {c.item_id: c for c in cooc_cands}
        for p in persona_cands:
            if p.item_id in seen:
                # Keep the existing (cooc) candidate; persona doesn't override.
                continue
            seen[p.item_id] = p
        return list(seen.values())

    rows: list[dict] = []
    for stratum_name, _, _ in STRATUM_BOUNDARIES:
        users = by_stratum[stratum_name]
        if not users:
            continue
        # Subsample for speed.
        if len(users) > max_eval_entities:
            step = max(1, len(users) // max_eval_entities)
            users = users[::step][:max_eval_entities]
        catalog_size = engine._item_graph.n_items
        print(f"\n  -- stratum={stratum_name} (n={len(users)}) --", flush=True)
        cooc_only_results = _evaluate_strategy(
            "cooc_only", retrieve_cooc, rank_with_cooc=True,
            eval_entities=users, train_items=train_items, test_items=test_items,
            engine=engine, catalog_size=catalog_size, k=k,
        )
        union_results = _evaluate_strategy(
            "cooc_union_persona", retrieve_union, rank_with_cooc=True,
            eval_entities=users, train_items=train_items, test_items=test_items,
            engine=engine, catalog_size=catalog_size, k=k,
        )
        persona_only_results = _evaluate_strategy(
            "persona_only", retrieve_persona, rank_with_cooc=False,
            eval_entities=users, train_items=train_items, test_items=test_items,
            engine=engine, catalog_size=catalog_size, k=k,
        )
        delta = union_results["ndcg_at_k"] - cooc_only_results["ndcg_at_k"]
        rel = (delta / max(cooc_only_results["ndcg_at_k"], 1e-9)) * 100
        rb_delta = union_results["recall_budget"] - cooc_only_results["recall_budget"]
        print(
            f"    cooc_only          NDCG={cooc_only_results['ndcg_at_k']:.4f} "
            f"R@K={cooc_only_results['recall_topk']:.3f} "
            f"R@B={cooc_only_results['recall_budget']:.3f}",
            flush=True,
        )
        print(
            f"    cooc+persona       NDCG={union_results['ndcg_at_k']:.4f} "
            f"R@K={union_results['recall_topk']:.3f} "
            f"R@B={union_results['recall_budget']:.3f}  "
            f"(delta NDCG {rel:+.2f}%, R@B {rb_delta:+.3f})",
            flush=True,
        )
        print(
            f"    persona_only       NDCG={persona_only_results['ndcg_at_k']:.4f} "
            f"R@K={persona_only_results['recall_topk']:.3f} "
            f"R@B={persona_only_results['recall_budget']:.3f}",
            flush=True,
        )
        rows.append({
            "stratum": stratum_name,
            "n_users": len(users),
            "cooc_only": cooc_only_results,
            "cooc_union_persona": union_results,
            "persona_only": persona_only_results,
            "delta_ndcg_abs": delta,
            "delta_ndcg_rel_pct": rel,
            "delta_recall_budget": rb_delta,
        })

    return {
        "dataset": dataset,
        "kindling_version": __version__,
        "fit_seconds": fit_seconds,
        "persona_n_personas": engine._persona_index.n_personas,
        "retrieval_budget": retrieval_budget,
        "persona_share": persona_share,
        "k": k,
        "strata": rows,
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
    parser.add_argument("--max-eval-entities", type=int, default=1000)
    parser.add_argument("--retrieval-budget", type=int, default=200)
    parser.add_argument("--persona-share", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(
        args.dataset,
        max_eval_entities=args.max_eval_entities,
        retrieval_budget=args.retrieval_budget,
        persona_share=args.persona_share,
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
