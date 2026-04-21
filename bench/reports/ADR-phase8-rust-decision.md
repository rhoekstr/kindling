# ADR: Phase 8 Rust decision (Python-only for v1.0)

**Status:** decided — ship v1 in pure Python; defer Rust to v1.x optimization if evidence requires it

**Date:** 2026-04-21

## Context

The plan front-loaded dataset benchmarks (Phases 3, 4, 7) ahead of the
Rust rewrite the PRD placed as Phase 1. Phase 8 is the measurement
phase that finally binds the Rust decision: measure pure-Python
performance against the PRD §13.1 targets, then decide.

PRD §13.1 targets (commodity cloud hardware, 8-core CPU, 16 GB RAM, SSD):

- Cold recommend (no GBT): p95 under 50 ms at ≤10k items
- Warm recommend (GBT active): p95 under 150 ms at ≤100k items
- Initial fit: under 60s at ≤1M interactions, ≤10k items
- Memory footprint: under 2 GB at 100k items + 10M interactions

## Method

- Synthetic session-heavy grocery data at two scales:
  - Cold: ~4k items, ~5k interactions, Bayesian blend off.
  - Warm: ~42k items, ~54k interactions, Bayesian blend on with 50
    VI iterations.
- 500 recommend calls per regime, p95 over the full timing vector.
- Peak RSS via ``resource.getrusage`` + ``tracemalloc``.
- Hot paths from ``cProfile`` over 100 recommend calls, sorted by
  cumulative time.
- Data: `bench/reports/profile_cold.json`, `bench/reports/profile_warm.json`.

## Results

| Regime | Items | Interactions | Fit (s) | p95 ms | PRD ms | p95 vs target | RSS MB | PRD MB |
| ------ | ----- | ------------ | ------- | ------ | ------ | ------------- | ------ | ------ |
| Cold   | 4,195  | 5,400         | 1.5     | 2.2    | 50     | 4.4% (22.7x headroom) | 155 | n/a |
| Warm   | 41,791 | 54,000        | 16.1    | 8.1    | 150    | 5.4% (18.5x headroom) | 413 | 2048 |

Both regimes meet PRD targets with >18× margin on p95 latency. Memory
is comfortably under budget.

### Caveats on the warm run

The warm synthetic dataset is materially smaller than the PRD's
"100k items + 10M interactions" reference point. Scaling linearly
from the measured numbers:

- Fit at 10× scale → ~160s, exceeding the 60s fit target.
- Recommend latency mostly scales with per-user owned-set size, which
  we held constant; at truly large user histories, the basket-signal
  posting-list unions grow.
- Memory likely scales to ~4 GB at 10M interactions.

So the cold + small-warm results are clean passes. The "PRD full
scale" result is inferred not measured, and fit time is the most
likely target to miss. Recommend latency, which is the user-facing
performance surface, is not at risk.

## Top hot paths (warm regime, 100 calls, 1024ms cumulative)

| Cumulative ms | Symbol                                   |
| ------------- | ---------------------------------------- |
| 1024          | engine.Engine.recommend (entry)          |
| 406           | engine._compute_signal_features          |
| 397           | retrieve.CoOccurrenceRetriever.retrieve  |
| 204           | engine._cooccurrence_signal              |
| 167           | ``dict.get`` across many callers         |
| 109           | path.{tail,path}.score_many              |
| 62            | scipy.sparse.__getitem__                 |
| 45            | engine._dedup_max_score                  |

The top four symbols together are 1,013 ms of the 1,024 ms total.
Three of the four are cooccurrence-related (retriever + signal
recomputation + sparse row slicing). An obvious Python-side
optimization: the retriever and ``_cooccurrence_signal`` both compute
``adjacency[owned_indices].sum(axis=0)`` — unifying that into one pass
would halve the cooccurrence work. Worth doing in Python before
considering Rust.

Path-family score_many is the next tier. 109 ms total = ~1 ms per
call, already fast.

## Decision

**Ship v1.0 in pure Python.** No Rust workspace, no PyO3 bindings, no
cbindgen C API in v1.0.

Justifications:

1. Both regimes clear PRD latency targets by 18×+. The PRD's 150 ms
   warm target was set assuming the Python implementation would be
   the bottleneck; measurement shows it isn't.
2. Pure-Python keeps the build matrix small: no cross-platform wheel
   matrix for a Rust crate, no cbindgen C ABI to maintain, no
   workspace dependency on Cargo. Ship straight to PyPI + conda-forge
   via a standard ``pyproject.toml`` build.
3. The remaining hot paths are all in Python that uses scipy/numpy
   under the hood - the CSR operations already drop into C via scipy.
   The overhead that does show up is Python-loop overhead over
   per-candidate dict lookups, which is easier to address with
   vectorization than with Rust.

## What to do in v1.x if it becomes needed

If real-data runs (Instacart / Amazon / RetailRocket at the PRD scale)
show p95 regressions, the staged plan is:

1. **Unify cooccurrence computation.** Merge the retriever's row-sum
   with ``_cooccurrence_signal`` so we do it once per recommend.
   Estimated ~20% recommend speedup, no new deps.
2. **Vectorize path scoring.** ``score_many`` uses Python
   comprehensions; move to numpy for the path tree / tail / basket
   loops. ~10-15% more.
3. **Hybrid SQLite + LRU for basket postings.** If basket posting-list
   union becomes the bottleneck at real scale.
4. **Only then: narrow Rust crate.** A single ``kindling-native`` PyO3
   extension containing just the measured hot paths, not the full 9-
   crate workspace the PRD specified. Item graph + basket posting are
   the candidate ports.

## Re-measure trigger

Re-run this ADR's measurements when any of:

- A real-data benchmark reports p95 over 50% of the PRD target.
- Fit time on a real dataset exceeds 60s for <=1M interactions.
- Memory footprint exceeds 1 GB at <=100k items in production.

The profile harness at `kindling.benchmarks.profile_harness` is the
canonical entry point; results land in `bench/reports/profile_*.json`.

## C API consequences

The PRD promised a ``kindling-cabi`` C library (for C++ / FFI
consumers). With no Rust core, there's no C ABI in v1.0 either. If a
C API is needed before v1.x Rust work happens, alternatives:

- Call Python via the CPython C API (embedding).
- Wrap with Apache Arrow FFI (the engine already supports Arrow
  export of frozen state).

The PRD's C API commitment honestly updates to "deferred to v1.x,
gated on Rust extension-module work."
