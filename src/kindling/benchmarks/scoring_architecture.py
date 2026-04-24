"""Architecture comparison: Bayesian blend vs gating network vs RRF-of-signals.

For each (dataset, method), fit the engine then measure NDCG@10,
Recall@10, MRR on the standard 500-entity split. Three methods:

1. **Bayesian blend** (``method="bayesian"``): current default.
   Posterior-mean over signals with data-characteristic priors.
2. **Gating network** (``method="gating"``): per-entity learned
   softmax weights over signals.
3. **RRF-of-signals** (``method="rrf"``): each signal ranks the
   candidate pool independently, reciprocal-rank fusion combines
   the rankings. Score-scale independent.

Reports per-method fit time, recommend p95 latency, NDCG/Recall/MRR.

CLI:
    python -m kindling.benchmarks.scoring_architecture \\
        --dataset movielens-1m \\
        --output bench/reports/scoring_architecture_ml1m.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from kindling import __version__
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import MetricReport, aggregate
from kindling.engine import MAX_QUERY_BASKET_SIZE, Engine, _compute_signal_features
from kindling.gate import GatingConfig
from kindling.retrieve.protocol import Candidate


Method = Literal["bayesian", "gating", "rrf"]


@dataclass(frozen=True)
class ArchResult:
    method: str
    fit_seconds: float
    recommend_p95_ms: float
    ndcg_at_k: float
    recall_at_k: float
    mrr: float
    n_eval_entities: int

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "fit_seconds": self.fit_seconds,
            "recommend_p95_ms": self.recommend_p95_ms,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "n_eval_entities": self.n_eval_entities,
        }


def _rrf_score_candidates(
    engine: Engine,
    candidates: list[Candidate],
    entity_id: object,
    owned_items: np.ndarray,
    history: tuple,
    k_rrf: float = 60.0,
) -> np.ndarray:
    """Score each candidate by its rank under each signal, then RRF-sum.

    Each of the K signal columns is treated as a separate ranker: we
    rank the candidate pool by that column, assign rank-based RRF
    contribution 1/(k_rrf + rank), and sum across signals. Signals with
    identical (zero) output contribute nothing (they get equal rank).
    """
    features = _compute_signal_features(
        candidates=candidates,
        owned_items=owned_items,
        query_basket=frozenset(history[-MAX_QUERY_BASKET_SIZE:]),
        history=history[-engine.max_history_for_recommend :],
        item_graph=engine._item_graph,
        tail_index=engine._tail_index,
        path_tree=engine._path_tree,
        basket_index=engine._basket_index,
        basket_similarity=engine.basket_similarity,
        cost_graph=engine._cost_graph,
        entity_id=entity_id,
        item_cosine=engine._item_cosine,
        als_factors=engine._als_factors,
        persona_index=engine._persona_index,
        lightgcn=engine._lightgcn,
    )
    n_cands, n_signals = features.matrix.shape
    rrf = np.zeros(n_cands, dtype=np.float64)
    for s in range(n_signals):
        col = features.matrix[:, s]
        # Skip dead columns (all zero / constant).
        if col.max() - col.min() < 1e-12:
            continue
        # Rank 1 = largest. Ties share rank (descending argsort on -col).
        order = np.argsort(-col)
        ranks = np.empty(n_cands, dtype=np.int64)
        ranks[order] = np.arange(1, n_cands + 1)
        rrf += 1.0 / (k_rrf + ranks)
    return rrf


def _evaluate(
    engine: Engine,
    method: Method,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    catalog_size: int,
    k: int,
) -> tuple[MetricReport, float]:
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    for entity in eval_entities:
        t0 = time.perf_counter()
        if method == "rrf":
            # RRF scoring: retrieve via the engine's stack, then re-rank
            # via RRF over signal columns instead of the blend.
            owned = engine._owned_by_entity.get(entity, np.array([]))
            history = engine._history_by_entity.get(entity, ())
            exclude = set(owned.tolist()) if owned.size else set()
            query_basket = frozenset(history[-MAX_QUERY_BASKET_SIZE:])
            candidates = engine._retrieve_rrf(
                entity_id=entity,
                owned_items=owned,
                owned_set=exclude,
                history=history,
                query_basket=query_basket,
            )
            if candidates:
                scores = _rrf_score_candidates(
                    engine=engine,
                    candidates=candidates,
                    entity_id=entity,
                    owned_items=owned,
                    history=history,
                )
                order = np.argsort(-scores)
                rec_items = [candidates[int(i)].item_id for i in order[:k]]
            else:
                rec_items = []
        else:
            # Bayesian / gating are both driven by Engine.recommend;
            # the engine state already selects the right path because
            # `engine._gate` is either None (bayesian) or fitted (gating).
            recs = engine.recommend(entity_id=entity, n=k)
            rec_items = [r.item_id for r in recs]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        relevant = test_owned - train_owned
        per_entity.append((rec_items, relevant))

    metrics = aggregate(per_entity, catalog_size=catalog_size, k=k)
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    return metrics, p95


def run_comparison(
    dataset: str,
    max_eval_entities: int = 500,
    test_fraction: float = 0.1,
    k: int = 10,
    gating_epochs: int = 10,
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    train_items = cast(
        pd.Series,
        split.train.groupby("entity_id", sort=False)["item_id"].apply(
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
    catalog_size = int(split.train["item_id"].nunique())

    results: list[ArchResult] = []

    # Method 1: Bayesian (default Engine config).
    print(f"[{dataset}] fitting Bayesian baseline...", flush=True)
    t0 = time.perf_counter()
    engine_b = Engine().fit(split.train)
    fit_b = time.perf_counter() - t0
    metrics, p95 = _evaluate(engine_b, "bayesian", eval_entities, train_items, test_items, catalog_size, k)
    results.append(ArchResult(
        method="bayesian", fit_seconds=fit_b, recommend_p95_ms=p95,
        ndcg_at_k=metrics.ndcg_at_k, recall_at_k=metrics.recall_at_k,
        mrr=metrics.mrr, n_eval_entities=metrics.n_entities_evaluated,
    ))

    # Method 2: Gating.
    print(f"[{dataset}] fitting gating network...", flush=True)
    t0 = time.perf_counter()
    gate_cfg = GatingConfig(
        enabled=True,
        n_epochs=gating_epochs,
        batch_size=512,
        min_users=100,
        seed=0,
    )
    engine_g = Engine(gating_config=gate_cfg).fit(split.train)
    fit_g = time.perf_counter() - t0
    metrics, p95 = _evaluate(engine_g, "gating", eval_entities, train_items, test_items, catalog_size, k)
    results.append(ArchResult(
        method="gating", fit_seconds=fit_g, recommend_p95_ms=p95,
        ndcg_at_k=metrics.ndcg_at_k, recall_at_k=metrics.recall_at_k,
        mrr=metrics.mrr, n_eval_entities=metrics.n_entities_evaluated,
    ))

    # Method 3: RRF-of-signals (uses the Bayesian engine's fitted state).
    print(f"[{dataset}] evaluating RRF-of-signals...", flush=True)
    metrics, p95 = _evaluate(engine_b, "rrf", eval_entities, train_items, test_items, catalog_size, k)
    results.append(ArchResult(
        method="rrf", fit_seconds=fit_b, recommend_p95_ms=p95,
        ndcg_at_k=metrics.ndcg_at_k, recall_at_k=metrics.recall_at_k,
        mrr=metrics.mrr, n_eval_entities=metrics.n_entities_evaluated,
    ))

    return {
        "dataset": dataset,
        "k": k,
        "n_eval_entities": len(eval_entities),
        "kindling_version": __version__,
        "results": [r.as_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare scoring architectures.")
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=["movielens-1m", "synthetic-grocery", "synthetic-grocery-deep"],
    )
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--gating-epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_comparison(
        dataset=args.dataset,
        max_eval_entities=args.max_eval_entities,
        gating_epochs=args.gating_epochs,
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
