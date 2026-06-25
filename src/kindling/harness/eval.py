"""Realistic-tier evaluation, packaged.

The same protocol the project validates itself with (chronological split,
full-catalog ranking, sliced by user history length), exposed as a reusable
function so you can point it at your own data and a fitted ``Engine`` and get
back per-warmth-bucket NDCG / Recall / MRR / HR — alongside the baselines.

The headline question this answers is the one that matters for adoption:
*does the model beat popularity, and the trained baselines, on MY data, and
in which warmth regime?*
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from kindling.benchmarks.parity import _build_eval_set
from kindling.blend.layer_scoring import aggregate
from kindling.engine import Engine
from kindling.harness.baselines import build_baselines

# (label, lo, hi) inclusive history-length bands; "all" is added implicitly.
DEFAULT_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1-4", 1, 4),
    ("5-19", 5, 19),
    ("20+", 20, 1_000_000_000),
)


def _bucket_of(history_len: int, buckets: tuple[tuple[str, int, int], ...]) -> str | None:
    for label, lo, hi in buckets:
        if lo <= history_len <= hi:
            return label
    return None


@dataclass
class BucketResult:
    """Per-warmth-bucket metrics for every model, plus the user count."""

    bucket: str
    n_users: int
    metrics: dict[str, dict[str, float]]  # model -> {ndcg@k, recall@k, mrr, hr@k}


@dataclass
class EvalReport:
    """Structured result of one :func:`evaluate` run (JSON-serializable)."""

    dataset: str
    k: int
    n_items: int
    n_train_interactions: int
    n_eval_users: int
    base_scorer: str
    active_channels: list[str]
    fit_seconds: float
    models: list[str]
    buckets: list[BucketResult]
    skipped_baselines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def metric(self, model: str, bucket: str = "all", name: str = "ndcg@k") -> float:
        """Convenience accessor used by tests and quick scripts."""
        for b in self.buckets:
            if b.bucket == bucket:
                return b.metrics.get(model, {}).get(name, 0.0)
        raise KeyError(f"no bucket {bucket!r} in report")


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    items: pd.DataFrame | None = None,
    *,
    dataset: str = "custom",
    k: int = 10,
    engine_kwargs: dict[str, Any] | None = None,
    baselines: list[str] | None = None,
    max_users: int = 2000,
    buckets: tuple[tuple[str, int, int], ...] = DEFAULT_BUCKETS,
    seed: int = 0,
    log: Any = None,
) -> EvalReport:
    """Fit ``Engine(**engine_kwargs)`` on ``train`` and score it against ``test``.

    Returns an :class:`EvalReport` with NDCG/Recall/MRR/HR at ``k`` for
    kindling and each baseline, sliced by user history length. ``baselines``
    defaults to ``["popularity"]``; pass names from
    :func:`kindling.harness.baselines.available_baselines` (``item-knn`` /
    ``als`` / ``bpr`` need the optional ``implicit`` library). ``log`` is an
    optional ``callable(str)`` for progress lines.
    """
    engine_kwargs = dict(engine_kwargs or {})
    engine_kwargs.setdefault("random_state", seed)
    baselines = baselines if baselines is not None else ["popularity"]
    emit = log if callable(log) else (lambda _m: None)

    t0 = time.perf_counter()
    engine = Engine(**engine_kwargs)
    engine.fit(train, item_metadata=items)
    fit_s = time.perf_counter() - t0
    state = engine._state
    assert state is not None
    plan = engine.activation_plan
    emit(
        f"fit {fit_s:.1f}s  base={plan.base_scorer}  "
        f"channels={plan.active_channels}  n_items={state.n_items:,}"
    )

    built, skipped = build_baselines(train, baselines, seed=seed)
    for s in skipped:
        emit(f"skip baseline {s}")

    eval_set = _build_eval_set(train, test, max_users=max_users, seed=seed)
    owned_by_entity: dict[object, set[object]] = {
        u: set(g["item_id"].tolist()) for u, g in train.groupby("entity_id")
    }

    model_names = ["kindling", *[b.name for b in built]]
    bucket_labels = [b[0] for b in buckets] + ["all"]
    # model -> bucket -> list[(recs, relevant)]
    acc: dict[str, dict[str, list[tuple[list[object], set[object]]]]] = {
        m: {b: [] for b in bucket_labels} for m in model_names
    }

    for entity, relevant in eval_set.items():
        owned = owned_by_entity.get(entity, set())
        label = _bucket_of(len(owned), buckets)
        target_buckets = [b for b in (label, "all") if b is not None]
        recs_by_model: dict[str, list[object]] = {
            "kindling": [r.item_id for r in engine.recommend(entity_id=entity, n=k)],
        }
        for b in built:
            recs_by_model[b.name] = b.recommend(entity, owned, k)
        for model, recs in recs_by_model.items():
            for tb in target_buckets:
                acc[model][tb].append((recs, relevant))

    n_items = state.n_items
    bucket_results: list[BucketResult] = []
    for label in bucket_labels:
        n_users = len(acc["kindling"][label])
        if n_users == 0:
            continue
        metrics: dict[str, dict[str, float]] = {}
        for model in model_names:
            rep = aggregate(acc[model][label], catalog_size=max(n_items, 1), k=k)
            metrics[model] = {
                f"ndcg@{k}": round(float(rep.ndcg_at_k), 4),
                f"recall@{k}": round(float(rep.recall_at_k), 4),
                "mrr": round(float(rep.mrr), 4),
                f"hr@{k}": round(float(rep.hit_rate), 4),
            }
        bucket_results.append(BucketResult(bucket=label, n_users=n_users, metrics=metrics))

    return EvalReport(
        dataset=dataset,
        k=k,
        n_items=n_items,
        n_train_interactions=len(train),
        n_eval_users=len(eval_set),
        base_scorer=plan.base_scorer,
        active_channels=list(plan.active_channels),
        fit_seconds=round(fit_s, 1),
        models=model_names,
        buckets=bucket_results,
        skipped_baselines=skipped,
    )


def format_report(report: EvalReport) -> str:
    """Render an :class:`EvalReport` as a fixed-width NDCG@k comparison table."""
    k = report.k
    lines: list[str] = []
    lines.append(
        f"{report.dataset} — NDCG@{k} by user history "
        f"(base={report.base_scorer}, n_items={report.n_items:,}, "
        f"fit={report.fit_seconds:.1f}s)"
    )
    if report.active_channels:
        lines.append(f"  active channels: {', '.join(report.active_channels)}")
    if report.skipped_baselines:
        lines.append(f"  skipped: {'; '.join(report.skipped_baselines)}")
    header = f"{'bucket':<8}{'n':>7}" + "".join(f"{m:>12}" for m in report.models)
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    metric_key = f"ndcg@{k}"
    for b in report.buckets:
        row = f"{b.bucket:<8}{b.n_users:>7}"
        best = max((b.metrics[m].get(metric_key, 0.0) for m in report.models), default=0.0)
        for m in report.models:
            val = b.metrics[m].get(metric_key, 0.0)
            mark = "*" if val == best and best > 0 else " "
            row += f"{val:>11.4f}{mark}"
        lines.append(row)
    lines.append("")
    lines.append("  * = best in row")
    return "\n".join(lines)
