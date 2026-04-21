"""Critical-path likelihood benchmark suite (PRD §12.4, plan Phase 3).

For each of the four shipped likelihoods, fit the Bayesian blend on the
chronological tail outcome batch and report:

  - NDCG@10 predictive accuracy on held-out test
  - Brier score of predicted-vs-observed selection probability
  - Expected Calibration Error (ECE)
  - Posterior predictive coverage (does the 90% CI cover observed rates)
  - Weight-stability under bootstrap of the outcome batch
  - Wall-time per VI fit

Decision rule for the v1 default (plan Phase 3 exit):
listwise calibration retains its default status if it wins calibration
metrics (Brier, ECE) and is within 5% on predictive accuracy (NDCG@10) on
at least three of four datasets. If another likelihood dominates clearly,
the default changes. MovieLens-1M is the only dataset in Phase 3; the
final decision lands after Phase 7 extends to Instacart, Amazon, and
RetailRocket.

This module is the scaffolding. The CLI entry point
``python -m kindling.benchmarks.likelihood_suite`` runs the suite on
ML-1M and writes JSON reports to bench/reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

from kindling import __version__
from kindling.benchmarks.metrics import aggregate
from kindling.blend.likelihoods import (
    BinaryIndependent,
    LikelihoodProtocol,
    ListwiseCalibration,
    MultinomialSoftmax,
    PairwiseBradleyTerry,
)
from kindling.engine import Engine
from kindling.loaders import movielens


@dataclass(frozen=True)
class LikelihoodResult:
    """One row of the likelihood comparison table."""

    likelihood: str
    ndcg_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    brier: float
    ece: float
    posterior_mean: list[float]
    credible_width_mean: float
    diagnostic_all_pass: bool
    diagnostic_warnings: list[str]
    fit_seconds: float
    recommend_seconds: float


_ALL_LIKELIHOODS: dict[str, LikelihoodProtocol] = {
    "listwise_calibration": cast(LikelihoodProtocol, ListwiseCalibration()),
    "binary_independent": cast(LikelihoodProtocol, BinaryIndependent()),
    "pairwise_bradley_terry": cast(LikelihoodProtocol, PairwiseBradleyTerry()),
    "multinomial_softmax": cast(LikelihoodProtocol, MultinomialSoftmax()),
}


def _expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE."""
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_mean_prob = float(probs[mask].mean())
        bin_mean_outcome = float(outcomes[mask].mean())
        ece += (mask.sum() / len(probs)) * abs(bin_mean_prob - bin_mean_outcome)
    return float(ece)


def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - outcomes.astype(np.float64)) ** 2))


def run_likelihood_suite_movielens(
    k: int = 10,
    max_eval_entities: int = 500,
    vi_max_iter: int = 200,
    seed: int = 0,
) -> list[LikelihoodResult]:
    """Run the four-likelihood comparison on ML-1M."""
    split = movielens.load_1m(test_fraction=0.1)
    train_items = split.train.groupby("entity_id", sort=False)["item_id"].apply(
        lambda s: set(s.tolist())
    )
    test_items = split.test.groupby("entity_id", sort=False)["item_id"].apply(
        lambda s: set(s.tolist())
    )
    eval_entities = sorted(set(train_items.index).intersection(test_items.index))
    step = max(1, len(eval_entities) // max_eval_entities)
    eval_entities = eval_entities[::step][:max_eval_entities]

    results: list[LikelihoodResult] = []
    for name, likelihood in _ALL_LIKELIHOODS.items():
        engine = Engine(
            use_bayesian_blend=True,
            likelihood=likelihood,
            seed=seed,
            vi_max_iter=vi_max_iter,
        )
        fit_start = time.perf_counter()
        engine.fit(split.train)
        fit_seconds = time.perf_counter() - fit_start

        per_entity: list[tuple[list[object], set[object]]] = []
        probs_actual: list[float] = []
        outcomes_actual: list[float] = []
        rec_start = time.perf_counter()
        for entity in eval_entities:
            relevant = test_items.get(entity, set()) - train_items.get(entity, set())
            recs = engine.recommend(entity_id=entity, n=k)
            rec_items = [r.item_id for r in recs]
            per_entity.append((rec_items, relevant))
            # For Brier/ECE we use the credible-mean score squashed to [0, 1].
            for r in recs:
                probs_actual.append(1.0 / (1.0 + np.exp(-np.clip(r.score, -30, 30))))
                outcomes_actual.append(1.0 if r.item_id in relevant else 0.0)
        recommend_seconds = time.perf_counter() - rec_start
        metrics = aggregate(per_entity, catalog_size=engine.item_graph.n_items, k=k)

        probs_arr = np.asarray(probs_actual)
        outcomes_arr = np.asarray(outcomes_actual)
        brier = _brier_score(probs_arr, outcomes_arr)
        ece = _expected_calibration_error(probs_arr, outcomes_arr)
        summary = engine.posterior_summary()
        ci_raw = summary.get("credible_interval", [[0.0, 0.0]])
        ci = np.asarray(ci_raw)
        width_mean = float((ci[:, 1] - ci[:, 0]).mean()) if ci.size else 0.0
        diagnostics = cast(dict[str, object], summary.get("diagnostics", {}) or {})
        posterior_mean_raw = cast(list[float], summary.get("posterior_mean", []))

        results.append(
            LikelihoodResult(
                likelihood=name,
                ndcg_at_k=metrics.ndcg_at_k,
                precision_at_k=metrics.precision_at_k,
                recall_at_k=metrics.recall_at_k,
                mrr=metrics.mrr,
                hit_rate=metrics.hit_rate,
                brier=brier,
                ece=ece,
                posterior_mean=list(posterior_mean_raw),
                credible_width_mean=width_mean,
                diagnostic_all_pass=bool(diagnostics.get("all_pass", False)),
                diagnostic_warnings=cast(list[str], diagnostics.get("warnings", []) or []),
                fit_seconds=fit_seconds,
                recommend_seconds=recommend_seconds,
            )
        )
    return results


def default_decision(results: list[LikelihoodResult]) -> str:
    """Apply the plan's decision rule and return a short rationale."""
    listwise = next((r for r in results if r.likelihood == "listwise_calibration"), None)
    if listwise is None:
        return "listwise_calibration missing from results - cannot decide"

    max_ndcg = max(r.ndcg_at_k for r in results)
    listwise_ndcg_ok = listwise.ndcg_at_k >= 0.95 * max_ndcg
    best_brier = min(r.brier for r in results)
    listwise_brier_best = listwise.brier <= best_brier + 1e-9
    best_ece = min(r.ece for r in results)
    listwise_ece_best = listwise.ece <= best_ece + 1e-9

    if listwise_brier_best and listwise_ece_best and listwise_ndcg_ok:
        return "keep: listwise_calibration wins on calibration (Brier, ECE) and matches on NDCG"
    if listwise_ndcg_ok:
        return (
            "review: listwise_calibration matches on NDCG but not calibration. "
            "Consider keeping on domain-knowledge grounds or switching."
        )
    return (
        "switch: another likelihood dominates. Compare table and update the default with rationale."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 3 critical-path likelihood comparison."
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--vi-max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_likelihood_suite_movielens(
        k=args.k,
        max_eval_entities=args.max_eval_entities,
        vi_max_iter=args.vi_max_iter,
        seed=args.seed,
    )
    decision = default_decision(results)
    report = {
        "engine_version": __version__,
        "dataset": "movielens-1m",
        "decision": decision,
        "results": [asdict(r) for r in results],
    }
    pretty = json.dumps(report, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"Wrote {args.output}")
        print(f"Decision: {decision}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
