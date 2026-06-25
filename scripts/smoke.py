"""Post-install smoke test — run against a freshly installed wheel.

Exercises the whole user-facing path with only the *core* dependencies
(numpy / pandas / scipy + the Rust core), so a green run proves
``pip install kindling`` yields a working engine end to end:

    fit → recommend → recommend_for_items (+ popularity fallback) →
    save/load round-trip → the ``kindling`` console command.

Used by the ``wheels`` CI job and runnable locally:  ``python scripts/smoke.py``.
Exits non-zero on any failure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import pandas as pd

    import kindling
    from kindling import Engine
    from kindling._native import CORE_AVAILABLE

    assert CORE_AVAILABLE, "Rust core (kindling._core) not importable from the wheel"
    print(f"kindling {kindling.__version__}  (core available)")

    # A tiny but non-degenerate log: shared co-purchases so EASE has signal.
    rows = []
    for user in range(40):
        base = (user % 5) * 3
        for item in (base, base + 1, base + 2, (base + 3) % 15):
            rows.append({"entity_id": user, "item_id": int(item)})
    df = pd.DataFrame(rows)

    engine = Engine(random_state=0).fit(df)
    plan = engine.activation_plan
    print(f"fit ok  base={plan.base_scorer}  channels={plan.active_channels}")

    recs = engine.recommend(entity_id=0, n=5)
    assert recs, "recommend returned nothing for a known user"
    assert all(isinstance(r.score, float) for r in recs), "scores must be floats"
    print(f"recommend(0) → {[r.item_id for r in recs]}")

    anon = engine.recommend_for_items(seed_item_ids=[0, 1], n=5)
    assert anon, "recommend_for_items returned nothing for valid seeds"
    fallback = engine.recommend_for_items(seed_item_ids=[], n=5)
    assert fallback, "empty seeds should still return popularity recs"
    assert fallback[0].base_kind.startswith("cold"), "empty seeds should fall back to cold path"
    print(f"recommend_for_items ok  (fallback base_kind={fallback[0].base_kind})")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "engine.kindling"
        engine.save(path)
        reloaded = Engine.load(path)
        assert [r.item_id for r in reloaded.recommend(entity_id=0, n=5)] == [
            r.item_id for r in recs
        ], "save/load changed recommendations"
    print("save/load round-trip ok")

    # The console entry point resolves and runs.
    out = subprocess.run(
        [sys.executable, "-m", "kindling.cli", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == kindling.__version__, "CLI version mismatch"
    print(f"CLI ok  (kindling version → {out.stdout.strip()})")

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
