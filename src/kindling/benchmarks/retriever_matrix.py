"""Per-fraction, per-signal retriever/ranker matrix.

Two measurement surfaces on top of the standalone-retriever harness:

1. **Fraction sweep.** For each signal acting as a complete recommender
   (its own retriever + its own scoring), measure NDCG@10 and recall@10
   at 20%, 40%, 60%, 80%, 100% of the training data. Decomposes
   "what each signal knows" into (a) what candidates it surfaces
   (recall@10) and (b) how well it ranks among those candidates (NDCG).

2. **Retriever × Ranker cross.** For each pair of signals (R, S), use R
   as the retriever and S as the ranker. Scores each R's candidates
   with S's scoring function. Quantifies the user's insight that
   "persona retriever + cooc ranker" might out-perform either alone
   because persona has high recall and cooc has precise ranking.

CLI:
    python -m kindling.benchmarks.retriever_matrix \
        --dataset synthetic-grocery-deep \
        --fractions 0.2,0.4,0.6,0.8,1.0 \
        --cross
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd

from kindling import Engine, __version__
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.path.basket_index import BasketSimilarity
from kindling.personas import KMeansClustering, PersonaConfig
from kindling.personas.matching import (
    build_user_query_vector,
    match_user,
    score_candidates as score_candidates_persona,
)
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.protocol import Candidate
from kindling.retrieve.signal_retrievers import (
    ALSRetriever,
    CosineRetriever,
    LightGCNRetriever,
    PathBasketRetriever,
    PathFullRetriever,
    PathTailRetriever,
    PersonaRetriever,
)

MAX_QUERY_BASKET_SIZE = 50


@dataclass(frozen=True)
class MatrixCell:
    dataset: str
    fraction: float
    retriever: str
    ranker: str
    ndcg_at_k: float
    recall_topk: float
    recall_budget: float
    mrr: float
    p95_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "fraction": self.fraction,
            "retriever": self.retriever,
            "ranker": self.ranker,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_topk": self.recall_topk,
            "recall_budget": self.recall_budget,
            "mrr": self.mrr,
            "p95_ms": self.p95_ms,
        }


# Each ranker takes (candidate_ids, entity_context) and returns per-candidate scores.
RankerFn = Callable[[list[object], dict[str, object]], np.ndarray]


def _build_rankers(engine: Engine) -> dict[str, RankerFn]:
    item_graph = engine._item_graph
    assert item_graph is not None
    tail_index = engine._tail_index
    path_tree = engine._path_tree
    basket_index = engine._basket_index
    als_factors = engine._als_factors
    item_cosine = engine._item_cosine
    persona_index = engine._persona_index

    def cooc_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        from kindling.engine import _cooccurrence_signal

        owned = cast(np.ndarray, ctx["owned"])
        return _cooccurrence_signal(cands, owned, item_graph)

    def cosine_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        owned = cast(np.ndarray, ctx["owned"])
        if item_cosine is None or owned.size == 0:
            return np.zeros(len(cands), dtype=np.float64)
        cand_idx = np.fromiter(
            (item_graph.item_index.get(c, -1) for c in cands), dtype=np.int64, count=len(cands)
        )
        owned_idx = np.asarray(
            [item_graph.item_index[o] for o in owned.tolist() if o in item_graph.item_index],
            dtype=np.int64,
        )
        if owned_idx.size == 0:
            return np.zeros(len(cands), dtype=np.float64)
        valid = cand_idx >= 0
        scores = np.zeros(len(cands), dtype=np.float64)
        if valid.any():
            s = item_cosine.score_many(cand_idx[valid], owned_idx)
            scores[valid] = s
        return scores

    def als_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        entity_id = ctx["entity_id"]
        if als_factors is None:
            return np.zeros(len(cands), dtype=np.float64)
        cand_idx = np.fromiter(
            (item_graph.item_index.get(c, -1) for c in cands), dtype=np.int64, count=len(cands)
        )
        valid = cand_idx >= 0
        scores = np.zeros(len(cands), dtype=np.float64)
        if valid.any():
            s = als_factors.score_many(entity_id, cand_idx[valid])
            scores[valid] = s
        return scores

    def path_tail_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        history = cast(tuple, ctx["history"])
        if not history:
            return np.zeros(len(cands), dtype=np.float64)
        return np.asarray(tail_index.score_many(cands, history[-1]), dtype=np.float64)

    def path_full_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        history = cast(tuple, ctx["history"])
        if not history:
            return np.zeros(len(cands), dtype=np.float64)
        return np.asarray(path_tree.score_many(cands, history), dtype=np.float64)

    def path_basket_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        q = cast(frozenset, ctx["query_basket"])
        if not q:
            return np.zeros(len(cands), dtype=np.float64)
        return np.asarray(
            basket_index.score_many(cands, q, BasketSimilarity.COVERAGE), dtype=np.float64
        )

    def persona_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        if persona_index is None or persona_index.n_personas == 0:
            return np.zeros(len(cands), dtype=np.float64)
        owned = cast(np.ndarray, ctx["owned"])
        history = cast(tuple, ctx["history"])
        user_vec = build_user_query_vector(
            owned_items=owned, history_items=history, index=persona_index
        )
        matches = match_user(user_vec, persona_index)
        if not matches.any():
            return np.zeros(len(cands), dtype=np.float64)
        return score_candidates_persona(matches, persona_index, cands)

    lightgcn = engine._lightgcn

    def lightgcn_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        if lightgcn is None:
            return np.zeros(len(cands), dtype=np.float64)
        entity_id = ctx["entity_id"]
        cand_idx = np.fromiter(
            (item_graph.item_index.get(c, -1) for c in cands),
            dtype=np.int64,
            count=len(cands),
        )
        valid = cand_idx >= 0
        scores = np.zeros(len(cands), dtype=np.float64)
        if valid.any():
            s = lightgcn.score_many(entity_id, cand_idx[valid])
            scores[valid] = s
        return scores

    temporal_graph = engine._temporal_graph

    def temporal_cooc_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        if temporal_graph is None or temporal_graph.n_edges == 0:
            return np.zeros(len(cands), dtype=np.float64)
        owned = cast(np.ndarray, ctx["owned"])
        if owned.size == 0:
            return np.zeros(len(cands), dtype=np.float64)
        owned_idx = np.fromiter(
            (temporal_graph.item_index.get(o, -1) for o in owned.tolist()),
            dtype=np.int64, count=owned.size,
        )
        owned_idx = owned_idx[owned_idx >= 0]
        if owned_idx.size == 0:
            return np.zeros(len(cands), dtype=np.float64)
        scores_full = temporal_graph.score_against_owned(
            owned_idx, exclude_indices={int(i) for i in owned_idx.tolist()}
        )
        cand_idx = np.fromiter(
            (temporal_graph.item_index.get(c, -1) for c in cands),
            dtype=np.int64, count=len(cands),
        )
        valid = cand_idx >= 0
        out = np.zeros(len(cands), dtype=np.float64)
        if valid.any():
            out[valid] = scores_full[cand_idx[valid]]
        return out

    session_cooc_graph = engine._session_cooc_graph

    def session_cooc_ranker(cands: list[object], ctx: dict[str, object]) -> np.ndarray:
        if session_cooc_graph is None or session_cooc_graph.n_edges == 0:
            return np.zeros(len(cands), dtype=np.float64)
        owned = cast(np.ndarray, ctx["owned"])
        if owned.size == 0:
            return np.zeros(len(cands), dtype=np.float64)
        owned_idx = np.fromiter(
            (session_cooc_graph.item_index.get(o, -1) for o in owned.tolist()),
            dtype=np.int64, count=owned.size,
        )
        owned_idx = owned_idx[owned_idx >= 0]
        if owned_idx.size == 0:
            return np.zeros(len(cands), dtype=np.float64)
        scores_full = session_cooc_graph.score_against_owned(
            owned_idx, exclude_indices={int(i) for i in owned_idx.tolist()}
        )
        cand_idx = np.fromiter(
            (session_cooc_graph.item_index.get(c, -1) for c in cands),
            dtype=np.int64, count=len(cands),
        )
        valid = cand_idx >= 0
        out = np.zeros(len(cands), dtype=np.float64)
        if valid.any():
            out[valid] = scores_full[cand_idx[valid]]
        return out

    rankers: dict[str, RankerFn] = {
        "cooccurrence": cooc_ranker,
        "path_tail": path_tail_ranker,
        "path_full": path_full_ranker,
        "path_basket": path_basket_ranker,
    }
    if item_cosine is not None:
        rankers["item_item_cosine"] = cosine_ranker
    if als_factors is not None:
        rankers["als_factor"] = als_ranker
    if persona_index is not None and persona_index.n_personas > 0:
        rankers["persona"] = persona_ranker
    if lightgcn is not None:
        rankers["lightgcn"] = lightgcn_ranker
    if temporal_graph is not None and temporal_graph.n_edges > 0:
        rankers["temporal_cooccurrence"] = temporal_cooc_ranker
    if session_cooc_graph is not None and session_cooc_graph.n_edges > 0:
        rankers["session_cooccurrence"] = session_cooc_ranker
    return rankers


def _retrieve(
    name: str,
    retriever: object,
    budget: int,
    ctx: dict[str, object],
) -> list[Candidate]:
    owned = cast(np.ndarray, ctx["owned"])
    history = cast(tuple, ctx["history"])
    exclude = cast(set, ctx["exclude"])
    query_basket = cast(frozenset, ctx["query_basket"])
    entity_id = ctx["entity_id"]
    if name == "cooccurrence":
        return retriever.retrieve(owned, budget)  # type: ignore[attr-defined]
    if name in ("path_tail", "path_full"):
        return retriever.retrieve(history, budget, exclude)  # type: ignore[attr-defined]
    if name == "path_basket":
        return retriever.retrieve(query_basket, budget, exclude)  # type: ignore[attr-defined]
    if name == "item_item_cosine":
        return retriever.retrieve(owned_items=owned, budget=budget, exclude=exclude)  # type: ignore[attr-defined]
    if name == "als_factor":
        return retriever.retrieve(entity_id=entity_id, budget=budget, exclude=exclude)  # type: ignore[attr-defined]
    if name == "persona":
        return retriever.retrieve(  # type: ignore[attr-defined]
            entity_id=entity_id, owned_items=owned, history=history,
            budget=budget, exclude=exclude,
        )
    if name == "lightgcn":
        return retriever.retrieve(entity_id=entity_id, budget=budget, exclude=exclude)  # type: ignore[attr-defined]
    if name == "temporal_cooccurrence":
        return retriever.retrieve(  # type: ignore[attr-defined]
            owned_items=owned, history=history, budget=budget, exclude=exclude,
        )
    if name == "session_cooccurrence":
        return retriever.retrieve(  # type: ignore[attr-defined]
            owned_items=owned, budget=budget, exclude=exclude,
        )
    raise ValueError(name)


class _TemporalCoocRetriever:
    """Thin adapter that scores all items by direct kernel-weighted
    cooccurrence lookup on the engine's temporal graph, then returns
    the top-budget."""

    name = "temporal_cooccurrence"
    budget_fraction = 1.0

    def __init__(self, temporal_graph, item_ids: np.ndarray) -> None:
        self.graph = temporal_graph
        self.item_ids = item_ids

    def retrieve(
        self, owned_items: np.ndarray, history: tuple, budget: int,
        exclude: set[object] | None = None,
    ) -> list[Candidate]:
        if owned_items.size == 0 or self.graph is None or self.graph.n_edges == 0:
            return []
        owned_idx = np.fromiter(
            (self.graph.item_index.get(o, -1) for o in owned_items.tolist()),
            dtype=np.int64, count=owned_items.size,
        )
        owned_idx = owned_idx[owned_idx >= 0]
        if owned_idx.size == 0:
            return []
        exclude_idx = {int(i) for i in owned_idx.tolist()}
        if exclude:
            for it in exclude:
                idx = self.graph.item_index.get(it, -1)
                if idx >= 0:
                    exclude_idx.add(int(idx))
        scores = self.graph.score_against_owned(owned_idx, exclude_indices=exclude_idx)
        if scores.max() <= 0:
            return []
        if budget < scores.size:
            top_idx = np.argpartition(-scores, budget)[:budget]
            top_idx = top_idx[scores[top_idx] > 0]
            order = np.argsort(-scores[top_idx])
            top_idx = top_idx[order]
        else:
            top_idx = np.argsort(-scores)
            top_idx = top_idx[scores[top_idx] > 0]
        return [
            Candidate(
                item_id=self.graph.item_ids[i],
                score=float(scores[i]),
                source="temporal_cooccurrence",
            )
            for i in top_idx
        ]


class _SessionCoocRetriever:
    """Direct lookup on the session-cooccurrence graph (S.T @ S where
    S is session-row indexed). Mirrors the cooccurrence retriever shape."""

    name = "session_cooccurrence"
    budget_fraction = 1.0

    def __init__(self, session_cooc_graph, item_ids: np.ndarray) -> None:
        self.graph = session_cooc_graph
        self.item_ids = item_ids

    def retrieve(
        self, owned_items: np.ndarray, budget: int,
        exclude: set[object] | None = None,
    ) -> list[Candidate]:
        if owned_items.size == 0 or self.graph is None or self.graph.n_edges == 0:
            return []
        owned_idx = np.fromiter(
            (self.graph.item_index.get(o, -1) for o in owned_items.tolist()),
            dtype=np.int64, count=owned_items.size,
        )
        owned_idx = owned_idx[owned_idx >= 0]
        if owned_idx.size == 0:
            return []
        exclude_idx = {int(i) for i in owned_idx.tolist()}
        if exclude:
            for it in exclude:
                idx = self.graph.item_index.get(it, -1)
                if idx >= 0:
                    exclude_idx.add(int(idx))
        scores = self.graph.score_against_owned(owned_idx, exclude_indices=exclude_idx)
        if scores.max() <= 0:
            return []
        if budget < scores.size:
            top_idx = np.argpartition(-scores, budget)[:budget]
            top_idx = top_idx[scores[top_idx] > 0]
            order = np.argsort(-scores[top_idx])
            top_idx = top_idx[order]
        else:
            top_idx = np.argsort(-scores)
            top_idx = top_idx[scores[top_idx] > 0]
        return [
            Candidate(
                item_id=self.graph.item_ids[i],
                score=float(scores[i]),
                source="session_cooccurrence",
            )
            for i in top_idx
        ]


def _build_retrievers(engine: Engine) -> dict[str, object]:
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    retrievers: dict[str, object] = {
        "cooccurrence": CoOccurrenceRetriever(engine._item_graph),
        "path_tail": PathTailRetriever(engine._tail_index, item_ids),
        "path_full": PathFullRetriever(engine._path_tree, item_ids),
        "path_basket": PathBasketRetriever(engine._basket_index, item_ids),
    }
    if engine._item_cosine is not None:
        retrievers["item_item_cosine"] = CosineRetriever(
            engine._item_cosine, engine._item_graph, item_ids
        )
    if engine._als_factors is not None:
        retrievers["als_factor"] = ALSRetriever(engine._als_factors, engine._item_graph, item_ids)
    if engine._persona_index is not None and engine._persona_index.n_personas > 0:
        retrievers["persona"] = PersonaRetriever(engine._persona_index, item_ids)
    if engine._lightgcn is not None:
        retrievers["lightgcn"] = LightGCNRetriever(engine._lightgcn, engine._item_graph, item_ids)
    if engine._temporal_graph is not None and engine._temporal_graph.n_edges > 0:
        retrievers["temporal_cooccurrence"] = _TemporalCoocRetriever(
            engine._temporal_graph, item_ids
        )
    if engine._session_cooc_graph is not None and engine._session_cooc_graph.n_edges > 0:
        retrievers["session_cooccurrence"] = _SessionCoocRetriever(
            engine._session_cooc_graph, item_ids
        )
    return retrievers


def _eval(
    dataset: str,
    fraction: float,
    retriever_name: str,
    retriever: object,
    ranker_name: str,
    ranker: RankerFn | None,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    owned_by_entity: dict,
    history_by_entity: dict,
    catalog_size: int,
    retrieval_budget: int,
    k: int,
) -> MatrixCell:
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    recall_budget_hits = 0
    recall_topk_hits = 0
    n_with_relevant = 0
    for entity in eval_entities:
        owned = owned_by_entity.get(entity, np.array([]))
        history = history_by_entity.get(entity, ())
        exclude = set(owned.tolist()) if owned.size else set()
        query_basket = frozenset(history[-MAX_QUERY_BASKET_SIZE:])
        ctx = {
            "entity_id": entity,
            "owned": owned,
            "history": history,
            "exclude": exclude,
            "query_basket": query_basket,
        }
        t0 = time.perf_counter()
        candidates = _retrieve(retriever_name, retriever, retrieval_budget, ctx)
        cand_ids = [c.item_id for c in candidates]
        if ranker is None:
            # Use retriever's own score.
            scored = [(c.item_id, c.score) for c in candidates]
        else:
            scores = ranker(cand_ids, ctx)
            scored = list(zip(cand_ids, scores, strict=True))
        # Sort by score desc, drop zeros.
        scored.sort(key=lambda kv: -kv[1])
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
    return MatrixCell(
        dataset=dataset,
        fraction=fraction,
        retriever=retriever_name,
        ranker=ranker_name,
        ndcg_at_k=m.ndcg_at_k,
        recall_topk=recall_topk_hits / max(n_with_relevant, 1),
        recall_budget=recall_budget_hits / max(n_with_relevant, 1),
        mrr=m.mrr,
        p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
    )


def run_matrix(
    dataset: str,
    fractions: list[float],
    cross: bool,
    k: int = 10,
    retrieval_budget: int = 500,
    max_eval_entities: int = 500,
    skip_heavy_signals: bool = False,
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction=0.1)
    test_items = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )

    all_cells: list[MatrixCell] = []
    per_fraction_fit_timings: dict[float, dict[str, float]] = {}
    for fraction in fractions:
        n_take = int(round(len(split.train) * fraction))
        sub = split.train.iloc[:n_take].reset_index(drop=True)
        print(f"\n=== {dataset} @ frac={fraction:.2f} ({len(sub):,} interactions) ===", flush=True)

        if skip_heavy_signals:
            # Persona (HDBSCAN+UMAP) and LightGCN scale poorly on multi-million-
            # interaction datasets. Skip them when we already have their results
            # on smaller datasets and only need cooc / cosine / path / ALS /
            # temporal_cooccurrence cells.
            engine_kwargs: dict[str, object] = {}
        else:
            cfg = PersonaConfig(
                enabled=True,
                clustering=KMeansClustering(n_clusters=30, random_state=0),
                min_activation_users=100,
            )
            from kindling.graph.lightgcn import LightGCNConfig

            # Use the LightGCNConfig defaults (n_epochs=30, batch_size=8192) which
            # are tuned for the end-to-end-gradient training architecture. The
            # earlier 10-epoch config was calibrated for the abandoned two-stage
            # shortcut and undertrains the real model.
            lgcn_cfg = LightGCNConfig(
                dim=64, min_users=50, min_items=50, seed=0
            )
            engine_kwargs = {"persona_config": cfg, "lightgcn_config": lgcn_cfg}

        t0 = time.perf_counter()
        engine = Engine(**engine_kwargs).fit(sub)
        print(f"  fit {time.perf_counter() - t0:.1f}s  per-subsystem={engine._fit_timings}", flush=True)
        per_fraction_fit_timings.setdefault(fraction, dict(engine._fit_timings))

        train_items = cast(
            pd.Series,
            sub.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s.tolist())),
        )
        eval_entities_all: list[object] = sorted(
            set(train_items.index).intersection(test_items.index)
        )
        step = max(1, len(eval_entities_all) // max_eval_entities)
        eval_entities: list[object] = eval_entities_all[::step][:max_eval_entities]
        catalog_size = int(sub["item_id"].nunique())

        retrievers = _build_retrievers(engine)
        rankers = _build_rankers(engine)

        # (1) Standalone: retriever = ranker (use retriever's own score).
        for name, r in retrievers.items():
            cell = _eval(
                dataset=dataset,
                fraction=fraction,
                retriever_name=name,
                retriever=r,
                ranker_name=name,  # same as retriever = standalone
                ranker=None,
                eval_entities=eval_entities,
                train_items=train_items,
                test_items=test_items,
                owned_by_entity=engine._owned_by_entity,
                history_by_entity=engine._history_by_entity,
                catalog_size=catalog_size,
                retrieval_budget=retrieval_budget,
                k=k,
            )
            all_cells.append(cell)
            print(
                f"  {name:<20} (standalone)  NDCG={cell.ndcg_at_k:.3f} "
                f"rec@K={cell.recall_topk:.3f} rec@B={cell.recall_budget:.3f}",
                flush=True,
            )

        # (2) Cross: every retriever x every ranker (expensive; only
        # run on the largest fraction when requested).
        if cross and fraction == fractions[-1]:
            print(f"\n  === cross combinations (retriever x ranker) ===", flush=True)
            for r_name, r in retrievers.items():
                for rk_name, rk_fn in rankers.items():
                    if rk_name == r_name:
                        continue  # already done as standalone
                    cell = _eval(
                        dataset=dataset,
                        fraction=fraction,
                        retriever_name=r_name,
                        retriever=r,
                        ranker_name=rk_name,
                        ranker=rk_fn,
                        eval_entities=eval_entities,
                        train_items=train_items,
                        test_items=test_items,
                        owned_by_entity=engine._owned_by_entity,
                        history_by_entity=engine._history_by_entity,
                        catalog_size=catalog_size,
                        retrieval_budget=retrieval_budget,
                        k=k,
                    )
                    all_cells.append(cell)
                    print(
                        f"  R={r_name:<18} K={rk_name:<18} NDCG={cell.ndcg_at_k:.3f} "
                        f"rec@K={cell.recall_topk:.3f}",
                        flush=True,
                    )

    return {
        "dataset": dataset,
        "fractions": fractions,
        "k": k,
        "retrieval_budget": retrieval_budget,
        "kindling_version": __version__,
        "cells": [c.as_dict() for c in all_cells],
        "fit_timings_per_fraction": per_fraction_fit_timings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-fraction per-signal retriever/ranker matrix.")
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=[
            "movielens-1m",
            "synthetic-grocery",
            "synthetic-grocery-deep",
            "retailrocket",
            "instacart",
            "gowalla",
            "yelp2018",
            "tafeng",
            "dunnhumby",
            "amazon-beauty",
            "amazon-book",
        ],
    )
    parser.add_argument("--fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--cross", action="store_true",
                        help="Run every retriever x ranker combination at the final fraction.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--retrieval-budget", type=int, default=500)
    parser.add_argument(
        "--skip-heavy-signals", action="store_true",
        help="Skip persona + LightGCN configs at fit time (cuts hours off "
             "very large datasets when those signals are already validated "
             "on smaller datasets).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    report = run_matrix(
        dataset=args.dataset,
        fractions=fractions,
        cross=args.cross,
        k=args.k,
        retrieval_budget=args.retrieval_budget,
        max_eval_entities=args.max_eval_entities,
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
