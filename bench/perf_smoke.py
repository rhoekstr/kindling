"""Perf smoke: fit time + serve latency on a fixed synthetic-large fixture.

Catches gross performance regressions (a 2-3x slowdown), not noise. The
envelope is deliberately generous because CI runners vary; this is a
manual / local guard rather than a blocking CI gate (perf gating on shared
runners flakes). Memory profiling is intentionally out of scope here — the
large-catalog memory model is covered by tests/unit/test_extension_cap.py.

Run:  python bench/perf_smoke.py
Exit non-zero if fit or serve latency blows the envelope.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from kindling import Engine
from kindling.loaders import synthetic

# Fixed fixture: ~3k users x 2k items, dense enough to exercise EASE.
N_ENTITIES, N_ITEMS = 3_000, 2_000
# Generous envelope (a healthy laptop fits in ~2-4s and serves in ~1-3ms).
FIT_SECONDS_MAX = 30.0
SERVE_P95_MS_MAX = 50.0


def main() -> int:
    split = synthetic.make_ratings(
        n_entities=N_ENTITIES, n_items=N_ITEMS, ratings_per_entity=20, seed=0
    )
    t0 = time.perf_counter()
    engine = Engine(persona_min_users=10**9, random_state=0)
    engine.fit(split.train)
    fit_s = time.perf_counter() - t0

    ents = list(engine._state.owned_by_entity.keys())
    for x in ents[:20]:
        engine.recommend(entity_id=x, n=10)  # warm up
    lat = []
    for x in ents[20:120]:
        t = time.perf_counter()
        engine.recommend(entity_id=x, n=10)
        lat.append((time.perf_counter() - t) * 1000.0)
    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))

    print(
        f"perf_smoke: n_items={engine._state.n_items} fit={fit_s:.1f}s "
        f"serve p50={p50:.1f}ms p95={p95:.1f}ms "
        f"(envelope fit<{FIT_SECONDS_MAX:.0f}s p95<{SERVE_P95_MS_MAX:.0f}ms)"
    )
    ok = fit_s < FIT_SECONDS_MAX and p95 < SERVE_P95_MS_MAX
    if not ok:
        print("PERF REGRESSION: outside the envelope.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
