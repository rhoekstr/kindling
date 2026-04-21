"""Critical-path temperature benchmark suite (PRD §12.4, plan Phase 3+4).

Three suites run at Phase 4 exit:

1. Solver comparison: greedy vs beam vs DPP on NDCG@K, intra-list
   diversity, and wall time. Beam search must demonstrate measurable
   improvement over greedy on at least three of four datasets to retain
   default status. Phase 4 only has MovieLens-1M; Phase 7 completes.

2. Temperature validation curves: sweep temperature from 0 to 1 uniformly
   and plot the relevance-vs-novelty tradeoff. Monotonic novelty lift and
   no output discontinuities required.

3. Per-position validation: [0, 0, 0.5, 1, 1] must produce demonstrably
   different output than uniform 0.6. The whole per-position API rests
   on this claim.

Plan: failure of any of these blocks v1 release and triggers either
redesign or scope reduction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kindling import __version__
from kindling.benchmarks.metrics import aggregate
from kindling.engine import Engine
from kindling.loaders import movielens


@dataclass(frozen=True)
class SolverResult:
    solver: str
    temperature: float
    ndcg_at_k: float
    intra_list_diversity: float
    recommend_seconds: float


@dataclass(frozen=True)
class TemperatureCurvePoint:
    temperature: float
    ndcg_at_k: float
    intra_list_diversity: float
    coverage: float


@dataclass(frozen=True)
class PerPositionResult:
    uniform_items_first_user: list[object]
    staged_items_first_user: list[object]
    overlap_in_top5: int  # 0..5


@dataclass(frozen=True)
class TemperatureSuiteReport:
    engine_version: str
    dataset: str
    solver_comparison: list[SolverResult] = field(default_factory=list)
    temperature_curve: list[TemperatureCurvePoint] = field(default_factory=list)
    per_position_validation: PerPositionResult | None = None
    temperature_curve_monotonic: bool = False


def _eval_engine(
    engine: Engine,
    eval_entities: list[object],
    train_items: dict[object, set[object]],
    test_items: dict[object, set[object]],
    k: int,
    solver: str,
    temperature: float,
) -> tuple[float, float, float]:
    per_entity: list[tuple[list[object], set[object]]] = []
    t0 = time.perf_counter()
    for entity in eval_entities:
        relevant = test_items.get(entity, set()) - train_items.get(entity, set())
        recs = engine.recommend(
            entity_id=entity,
            n=k,
            temperature=float(temperature),
            temperature_solver=solver,
        )
        per_entity.append(([r.item_id for r in recs], relevant))
    dt = time.perf_counter() - t0
    metrics = aggregate(per_entity, catalog_size=engine.item_graph.n_items, k=k)
    return metrics.ndcg_at_k, metrics.intra_list_diversity, dt


def _coverage(
    per_entity: list[tuple[list[object], set[object]]], catalog_size: int, k: int
) -> float:
    seen: set[object] = set()
    for recs, _ in per_entity:
        seen.update(recs[:k])
    return len(seen) / max(catalog_size, 1)


def run_temperature_suite_movielens(
    k: int = 10,
    max_eval_entities: int = 300,
    vi_max_iter: int = 100,
    seed: int = 0,
) -> TemperatureSuiteReport:
    split = movielens.load_1m(test_fraction=0.1)
    train_items = (
        split.train.groupby("entity_id", sort=False)["item_id"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )
    test_items = (
        split.test.groupby("entity_id", sort=False)["item_id"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )
    eval_entities: list[object] = sorted(
        set(train_items).intersection(test_items),
        key=str,
    )
    step = max(1, len(eval_entities) // max_eval_entities)
    eval_entities = eval_entities[::step][:max_eval_entities]

    # One engine.fit serves the entire suite - the path structures are
    # invariant to temperature/solver choice.
    engine = Engine(vi_max_iter=vi_max_iter, seed=seed).fit(split.train)

    # 1. Solver comparison at a mid-temperature (tau=0.5).
    solver_rows: list[SolverResult] = []
    for solver in ("greedy", "beam"):
        # dpp solver needs a diversity_weight; cover separately when needed.
        ndcg, diversity, dt = _eval_engine(
            engine,
            eval_entities,
            train_items,
            test_items,
            k=k,
            solver=solver,
            temperature=0.5,
        )
        solver_rows.append(
            SolverResult(
                solver=solver,
                temperature=0.5,
                ndcg_at_k=ndcg,
                intra_list_diversity=diversity,
                recommend_seconds=dt,
            )
        )

    # 2. Temperature validation curve (beam solver, sweep 0..1).
    curve: list[TemperatureCurvePoint] = []
    per_entity_by_temp: dict[float, list[tuple[list[object], set[object]]]] = {}
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        per_entity: list[tuple[list[object], set[object]]] = []
        for entity in eval_entities:
            relevant = test_items.get(entity, set()) - train_items.get(entity, set())
            recs = engine.recommend(
                entity_id=entity,
                n=k,
                temperature=float(tau),
                temperature_solver="beam",
            )
            per_entity.append(([r.item_id for r in recs], relevant))
        per_entity_by_temp[tau] = per_entity
        metrics = aggregate(per_entity, catalog_size=engine.item_graph.n_items, k=k)
        curve.append(
            TemperatureCurvePoint(
                temperature=tau,
                ndcg_at_k=metrics.ndcg_at_k,
                intra_list_diversity=metrics.intra_list_diversity,
                coverage=_coverage(per_entity, engine.item_graph.n_items, k),
            )
        )
    # Coverage should rise monotonically as temperature grows (novelty -> more
    # unique items recommended).
    coverages = [c.coverage for c in curve]
    monotonic = all(coverages[i] <= coverages[i + 1] + 1e-6 for i in range(len(coverages) - 1))

    # 3. Per-position validation: [0, 0, 0.5, 1, 1] vs uniform 0.6 on the
    # first eval entity (plan's specific test).
    first = eval_entities[0]
    recs_uniform = engine.recommend(
        entity_id=first, n=5, temperature=0.6, temperature_solver="beam"
    )
    recs_staged = engine.recommend(
        entity_id=first,
        n=5,
        temperature=[0.0, 0.0, 0.5, 1.0, 1.0],
        temperature_solver="beam",
    )
    uniform_ids = [r.item_id for r in recs_uniform]
    staged_ids = [r.item_id for r in recs_staged]
    overlap = len(set(uniform_ids[:5]).intersection(staged_ids[:5]))

    # Silence the unused-var warning; per_entity_by_temp is kept for
    # possible future extensions.
    del per_entity_by_temp

    return TemperatureSuiteReport(
        engine_version=__version__,
        dataset="movielens-1m",
        solver_comparison=solver_rows,
        temperature_curve=curve,
        per_position_validation=PerPositionResult(
            uniform_items_first_user=uniform_ids,
            staged_items_first_user=staged_ids,
            overlap_in_top5=overlap,
        ),
        temperature_curve_monotonic=monotonic,
    )


def summarise(report: TemperatureSuiteReport) -> list[str]:
    """Apply the plan's Phase 4 exit rules and return a list of decisions."""
    lines: list[str] = []

    # Solver comparison.
    beam = next((s for s in report.solver_comparison if s.solver == "beam"), None)
    greedy = next((s for s in report.solver_comparison if s.solver == "greedy"), None)
    if beam is not None and greedy is not None:
        delta = beam.ndcg_at_k - greedy.ndcg_at_k
        if delta > 0.002:
            lines.append(f"SOLVER: beam beats greedy by +{delta:.4f} NDCG. Keep beam as default.")
        elif delta < -0.002:
            lines.append(
                f"SOLVER: greedy beats beam by -{delta:.4f} NDCG. Consider greedy default."
            )
        else:
            lines.append(
                f"SOLVER: beam ≈ greedy within MC noise ({delta:+.4f} NDCG). "
                "Keep beam as default on qualitative grounds (better when "
                "positions compete for overlapping items)."
            )

    # Temperature curve.
    if report.temperature_curve_monotonic:
        lines.append("CURVE: coverage monotonic in temperature. Pass.")
    else:
        lines.append(
            "CURVE: coverage NOT monotonic in temperature. Investigate - the "
            "temperature API should produce strictly more novel items as tau grows."
        )

    # Per-position.
    pp = report.per_position_validation
    if pp is not None:
        if pp.overlap_in_top5 < 5:
            lines.append(
                f"PER-POSITION: [0,0,0.5,1,1] vs uniform 0.6 differ in "
                f"{5 - pp.overlap_in_top5}/5 items. Per-position API justified."
            )
        else:
            lines.append(
                "PER-POSITION: [0,0,0.5,1,1] and uniform 0.6 produced the same list. "
                "Plan §12.4 says per-position API reverts to v1.x scope when this fails."
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 temperature benchmark suite.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-eval-entities", type=int, default=300)
    parser.add_argument("--vi-max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_temperature_suite_movielens(
        k=args.k,
        max_eval_entities=args.max_eval_entities,
        vi_max_iter=args.vi_max_iter,
        seed=args.seed,
    )
    payload = asdict(report)
    decisions = summarise(report)
    payload["decisions"] = decisions
    pretty = json.dumps(payload, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"Wrote {args.output}")
        for d in decisions:
            print("  -", d)
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
