"""Canonical default-config verification harness — the regression gate.

One entry point that reproduces the REFERENCE §3.3 engine-default numbers
for the four reference datasets, so any phase of the production transition
can be checked against a stable baseline. Mirrors the canonical eval
(``parity._build_eval_set`` + ``metrics.aggregate``) used by the frozen
runners; encodes the documented per-dataset config.

Run:  DATASET=movielens-1m .venv/bin/python bench/verify.py
      (movielens-1m | amazon-beauty | steam | amazon-book-chrono)

Reference NDCG@10: ml1m 0.2931 · beauty 0.0343 · steam 0.0660 · book 0.0318
"""

from __future__ import annotations

import json
import os
import time

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set

try:
    # Pre-rename: EngineV2. Post-rename (Phase 3): Engine forwards here.
    from kindling.engine_v2 import EngineV2 as _Engine
except ImportError:  # pragma: no cover - after the v2→Engine promotion
    from kindling import Engine as _Engine

# Documented per-dataset config (REFERENCE §3.3 / §5). persona_min_users is
# pinned high to disable the (dead, to-be-deleted) persona path.
_BASE = dict(persona_min_users=10**9, retrieval_budget=500, random_state=0)
_CONFIG = {
    "movielens-1m": {},
    "amazon-beauty": {"ease_lambda": 250.0},
    "steam": {"cold_slots": 1},
    "amazon-book-chrono": {"cold_slots": 1},
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate(dataset: str, *, quiet: bool = False) -> dict:
    """Fit the documented default config for ``dataset`` and return metrics.

    Importable by the CI regression gate (``bench/check_gate.py``).
    """
    extra = _CONFIG.get(dataset, {})

    t0 = time.perf_counter()
    split = _load_dataset(dataset, test_fraction=0.1)
    train, test = split.train, split.test
    if not quiet:
        _log(
            f"{dataset}: loaded {time.perf_counter() - t0:.0f}s  "
            f"train {len(train):,}  users {train.entity_id.nunique():,}  "
            f"items {train.item_id.nunique():,}"
        )
    eval_set = _build_eval_set(train, test, max_users=500, seed=0)
    has_meta = getattr(split, "items", None) is not None
    cfg = {**_BASE, **extra}
    if dataset == "amazon-book-chrono" and not has_meta:
        cfg["cold_slots"] = 0

    t0 = time.perf_counter()
    engine = _Engine(**cfg)
    engine.fit(train, item_metadata=split.items if has_meta else None)
    fit_s = time.perf_counter() - t0
    st = engine._state
    base = st.profile.get("base_scorer_used", "?")
    if not quiet:
        _log(f"fit {fit_s:.0f}s  base={base}  n_items={st.n_items:,}")

    per = []
    for entity, rel in eval_set.items():
        recs = engine.recommend(entity_id=entity, n=10)
        per.append(([r.item_id for r in recs], rel))
    rep = aggregate(per, catalog_size=max(st.n_items, 1), k=10)

    return {
        "dataset": dataset,
        "config": {k: v for k, v in cfg.items() if k != "random_state"},
        "base_scorer": base,
        "n_eval_users": len(eval_set),
        "fit_seconds": round(fit_s, 1),
        "ndcg@10": round(float(rep.ndcg_at_k), 4),
        "recall@10": round(float(rep.recall_at_k), 4),
        "mrr": round(float(rep.mrr), 4),
        "hr@10": round(float(rep.hit_rate), 4),
    }


def main() -> None:
    out = evaluate(os.environ.get("DATASET", "movielens-1m"))
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
