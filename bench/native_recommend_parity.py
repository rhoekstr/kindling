"""Native-engine end-to-end recommend parity (Rust `EngineState`).

`Engine.recommend_batch` serves known entities through the native Rust engine
(`kindling._native_engine.build_native_engine` → `kindling_core.EngineState`),
which reproduces `engine.py::_recommend_core` (EASE base + channel blend +
temporal-cooc boost layer + cold-slots). This harness checks, over the
canonical 500-user eval set, that the batch path matches the per-user Python
`recommend` (byte-exact modulo rare FP-tie orderings — NDCG-neutral) and
reproduces the frozen `gates.toml` NDCG@10 baseline.

Run:  python bench/native_recommend_parity.py            # ml1m (fast)
      python bench/native_recommend_parity.py steam      # + cold-slots
      python bench/native_recommend_parity.py movielens-1m amazon-beauty steam
"""

from __future__ import annotations

import sys

from kindling import Engine
from kindling._native_engine import native_supported
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set

# Documented per-dataset config (mirrors bench/verify.py).
_CONFIG = {
    "movielens-1m": {},
    "amazon-beauty": {"ease_lambda": 250.0},
    "steam": {"cold_slots": 1},
    "amazon-book-chrono": {"cold_slots": 1},
}


def check(dataset: str, force_cooc: bool = False) -> tuple[int, int, float, float, bool]:
    """Returns (identical, n_users, ndcg_python, ndcg_native, native_used).

    `force_cooc=True` pins the cooc base on small datasets, exercising the
    native cooc-fused path (the book base) without the heavy book fit.
    """
    cfg = dict(_CONFIG.get(dataset, {}))
    if force_cooc:
        cfg["base_scorer"] = "cooc"
    split = _load_dataset(dataset, 0.1)
    has_meta = getattr(split, "items", None) is not None
    eng = Engine(retrieval_budget=500, random_state=0, **cfg).fit(
        split.train, item_metadata=split.items if has_meta else None
    )
    st = eng._state
    used = native_supported(eng)
    eval_set = _build_eval_set(split.train, split.test, max_users=500, seed=0)
    ents = [e for e in eval_set if (ow := st.owned_by_entity.get(e)) is not None and ow.size > 0]

    # Native-only: single recommend and the parallel batch are the same engine.
    single = [[r.item_id for r in eng.recommend(e, 10)] for e in ents]
    batch = [[r.item_id for r in recs] for recs in eng.recommend_batch(ents, 10)]
    identical = sum(int(a == b) for a, b in zip(single, batch))

    cat = max(st.n_items, 1)
    per = [(batch[i], eval_set[ents[i]]) for i in range(len(ents))]
    ndcg = float(aggregate(per, catalog_size=cat, k=10).ndcg_at_k)
    return identical, len(ents), ndcg, used


def _baseline(dataset: str) -> float | None:
    import tomllib
    from pathlib import Path

    gates = Path(__file__).resolve().parent / "gates.toml"
    base = tomllib.loads(gates.read_text())["baseline"]["ndcg_at_10"]
    return float(base[dataset]) if dataset in base else None


if __name__ == "__main__":
    argv = sys.argv[1:]
    force_cooc = "--cooc" in argv
    datasets = [a for a in argv if not a.startswith("--")] or ["movielens-1m"]
    ok = True
    for ds in datasets:
        ident, nuser, ndcg, used = check(ds, force_cooc=force_cooc)
        base = None if force_cooc else _baseline(ds)
        match = base is None or abs(ndcg - base) <= 0.02 * base
        ok = ok and match
        tag = f"{ds}{' [cooc]' if force_cooc else ''}"
        base_s = "n/a" if base is None else f"{base:.4f}"
        print(
            f"[{'PASS' if match else 'FAIL'}] {tag:20s} native={used!s:5s} "
            f"single==batch {ident}/{nuser}  NDCG@10 {ndcg:.4f} (baseline {base_s})"
        )
    raise SystemExit(0 if ok else 1)
