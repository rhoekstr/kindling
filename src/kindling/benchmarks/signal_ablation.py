"""Signal ablation audit.

For each (dataset, fraction), fit the engine once and evaluate NDCG
under a sweep of signal masks:

- ``full``: all 9 signals active (baseline).
- ``-<signal>``: leave-one-out -- mask a single signal, keep the rest.
  Delta vs. full measures that signal's marginal contribution.
- ``only_<family>``: mask every signal except one family (paths, cooc,
  cosine, als, cost). Measures the family's standalone accuracy floor.

Also records the Bayesian posterior weights per fraction so we can see
how the blend adapts across the growth curve -- the user's ask:
"I want to see how our blends are adapting over time and whether they
are adding value at each stage."

CLI:
    python -m kindling.benchmarks.signal_ablation \
        --dataset synthetic-grocery-deep \
        --fractions 0.3,0.6,1.0 \
        --max-eval-entities 500 \
        --output bench/reports/signal_ablation_grocery.json
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
from kindling.engine import SIGNAL_ORDER

PATH_SIGNALS = ["path_full", "path_tail", "path_basket"]
COOC_SIGNALS = ["cooccurrence"]
COSINE_SIGNALS = ["item_item_cosine"]
ALS_SIGNALS = ["als_factor"]
COST_SIGNALS = ["cost_population", "cost_entity", "cost_context"]


@dataclass(frozen=True)
class AblationPoint:
    fraction: float
    n_interactions: int
    config: str
    ndcg_at_k: float
    recall_at_k: float
    mrr: float
    coverage: float
    n_eval_entities: int

    def as_dict(self) -> dict[str, object]:
        return {
            "fraction": self.fraction,
            "n_interactions": self.n_interactions,
            "config": self.config,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "coverage": self.coverage,
            "n_eval_entities": self.n_eval_entities,
        }


def _mask_posterior(engine: Engine, names_to_mask: list[str]) -> np.ndarray | None:
    """Zero the posterior_beta entry for each named signal.

    Returns the original posterior_beta so the caller can restore.
    """
    blend = engine._bayesian_blend
    if blend is None:
        return None
    orig = blend.posterior_beta.copy()
    new = orig.copy()
    for name in names_to_mask:
        new[SIGNAL_ORDER.index(name)] = 1e-9
    blend.posterior_beta = new
    return orig


def _restore_posterior(engine: Engine, orig: np.ndarray | None) -> None:
    if orig is not None and engine._bayesian_blend is not None:
        engine._bayesian_blend.posterior_beta = orig


def _evaluate(
    engine: Engine,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    catalog_size: int,
    k: int,
) -> MetricReport:
    per: list[tuple[list[object], set[object]]] = []
    for entity in eval_entities:
        rel = test_items.get(entity, set()) - train_items.get(entity, set())
        recs = engine.recommend(entity_id=entity, n=k)
        per.append(([r.item_id for r in recs], rel))
    return aggregate(per, catalog_size=catalog_size, k=k)


def run_ablation(
    dataset: str,
    fractions: list[float],
    k: int = 10,
    max_eval_entities: int = 500,
    test_fraction: float = 0.1,
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

    points: list[AblationPoint] = []
    posterior_trace: list[dict[str, object]] = []

    family_configs: list[tuple[str, list[str]]] = [
        ("only_paths", PATH_SIGNALS),
        ("only_cooc", COOC_SIGNALS),
        ("only_cosine", COSINE_SIGNALS),
        ("only_als", ALS_SIGNALS),
        ("only_cost", COST_SIGNALS),
    ]

    for frac in fractions:
        n_take = int(round(len(split.train) * frac))
        sub = split.train.iloc[:n_take].reset_index(drop=True)
        catalog_size = int(sub["item_id"].nunique())
        print(f"\n=== {dataset} @ frac={frac:.2f} ({len(sub):,} interactions) ===", flush=True)
        t0 = time.perf_counter()
        engine = Engine().fit(sub)
        fit_s = time.perf_counter() - t0
        print(f"  fit: {fit_s:.1f}s", flush=True)

        # Record posterior weights at this fraction.
        blend = engine._bayesian_blend
        if blend is not None:
            posterior_trace.append({
                "fraction": frac,
                "n_interactions": len(sub),
                "signal_weights": {
                    name: float(w)
                    for name, w in zip(blend.signal_names, blend.posterior_mean, strict=True)
                },
                "prior_alpha": {
                    name: float(a)
                    for name, a in zip(blend.signal_names, blend.prior_alpha, strict=True)
                },
            })

        def _point(config: str) -> AblationPoint:
            m = _evaluate(engine, eval_entities, train_items, test_items, catalog_size, k)
            return AblationPoint(
                fraction=frac,
                n_interactions=len(sub),
                config=config,
                ndcg_at_k=m.ndcg_at_k,
                recall_at_k=m.recall_at_k,
                mrr=m.mrr,
                coverage=m.coverage,
                n_eval_entities=m.n_entities_evaluated,
            )

        # Full baseline.
        points.append(_point("full"))
        print(f"  full         NDCG={points[-1].ndcg_at_k:.4f}", flush=True)

        # LOO per signal.
        for sig in SIGNAL_ORDER:
            orig = _mask_posterior(engine, [sig])
            pt = _point(f"-{sig}")
            _restore_posterior(engine, orig)
            points.append(pt)
            print(f"  -{sig:<20} NDCG={pt.ndcg_at_k:.4f}  delta={pt.ndcg_at_k - points[-2].ndcg_at_k:+.4f}", flush=True)

        # Family-only.
        for family_name, family in family_configs:
            mask = [s for s in SIGNAL_ORDER if s not in family]
            orig = _mask_posterior(engine, mask)
            pt = _point(family_name)
            _restore_posterior(engine, orig)
            points.append(pt)
            print(f"  {family_name:<12} NDCG={pt.ndcg_at_k:.4f}", flush=True)

    return {
        "dataset": dataset,
        "k": k,
        "kindling_version": __version__,
        "posterior_trace": posterior_trace,
        "ablation_points": [p.as_dict() for p in points],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-signal ablation of the Bayesian blend across data sizes."
    )
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=["movielens-1m", "synthetic-grocery", "synthetic-grocery-deep"],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fractions", default="0.3,0.6,1.0")
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    report = run_ablation(
        dataset=args.dataset,
        fractions=fractions,
        k=args.k,
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
