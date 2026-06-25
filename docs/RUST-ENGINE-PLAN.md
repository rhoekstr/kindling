# Plan: optimizing the kindling engine toward a Rust-native core

## Why now

The metadata-kNN Rust port was a microcosm of the whole problem: moving one hot
path to Rust took the full 105k-item catalog kNN from "~4 min + memory thrash"
to **20s**. The very next wall — the full-scale H&M engine **fit** (565k users)
thrashing this box's RAM — sits in the *Python orchestration*, not the
algorithm: it thrashed even with smoothing **off**, in the base fit. The cost is
pandas DataFrames + NumPy intermediate copies + Python-level catalog building,
not the numerics (which are already Rust).

Three wins from going Rust-native:
1. **Memory** — compact native arrays instead of pandas/NumPy copies; fit data
   several× larger on the same hardware (the wall we just hit).
2. **Fit speed** — no Python groupby/map/dict-comprehension overhead.
3. **Serving + deployment** — no GIL (true parallel batch scoring), lower
   latency, and an option to ship a standalone binary with no Python runtime.

## What's already Rust vs. what isn't

| Already in `kindling_core` (Rust) | Still Python (`engine.py` orchestration) |
|---|---|
| `build_cooccurrence`, `directional_cooc` | ingestion: pandas validate / canonicalize / preprocess |
| `fit_ease` (faer Cholesky), `build_item_cosine` | catalog building (`item_to_idx`, `entity_to_idx`) |
| layered scoring, retrieval | channel assembly (trend / last-item / transitions / user-CF) |
| `metadata_knn` (new), repeat | cooc-transform + smoothing wiring, `EngineState` assembly |
| | recommend blend (z-norm channels, cold slots, top-N), persistence (pickle) |

**The gap is ingestion + orchestration** — exactly where the memory/perf cost
lives. The numeric kernels are done.

## Target architecture

A `kindling-engine` Rust crate that owns **fit → state → recommend** end to
end, built on the existing `kindling_core` primitives, with **PyO3 bindings
exposing the same `Engine` API** (`fit` / `recommend` / `recommend_for_items` /
`save` / `load` / `activation_plan`). The existing tests, eval harness, and
FastAPI serving keep working unchanged — only the internals move.

- Loaders, eval harness, and CLI **stay Python** (not hot; convenient).
- Optional later: a pure-Rust CLI + `axum` HTTP server for no-Python deploys.

## Phases (incremental, parity-gated)

### Phase 1 — Rust ingestion *(the memory win)*
Accept interactions as Arrow/NumPy arrays via PyO3 (zero-copy where possible),
or read files directly (polars). Build catalogs with `FxHashMap`; emit compact
`i32`/`f32` arrays. Replaces the pandas DataFrames (the biggest memory hog) and
the slow groupby/map. **This is what unblocks full-scale fits on this box.**

### Phase 2 — Rust fit pipeline
One Rust `fit(interactions, config) -> EngineState`: rating-signal detection,
base (cooc+wilson *or* EASE — both already Rust), channel builds (trend,
last-item, transitions, user-CF), metadata smoothing (kNN + the dose fit ported
to Rust), native `EngineState` assembly.

### Phase 3 — Rust recommend + batch
Port retrieve → base → z-normalized blend → cold slots → top-N → explanation.
Add a **rayon-parallel batch recommend** so full-catalog eval is fast (no GIL).

### Phase 4 — Rust persistence
Serialize `EngineState` with `bincode`/`rkyv` (compact, fast, mmap-able),
replacing pickle. Enables mmap loading for serving.

### Phase 5 — Bindings + differential parity
PyO3 `Engine` wrapping the Rust core, same API. A **differential test suite**:
the Rust engine must reproduce the frozen reference numbers (ml1m 0.2928,
beauty 0.0328, steam 0.0660, book 0.0318) within tolerance, plus per-rec parity
on fixtures. Drop-in replacement; existing suites stay green.

### Phase 6 *(optional)* — standalone Rust deployment
Pure-Rust CLI (`kindling bench/fit/serve`) + an `axum` server mirroring
`kindling.serving`. Single static binary, no Python runtime.

## Sequencing by value
1. **Phases 1–2** (ingestion + fit) — directly kills the memory wall; biggest ROI.
2. **Phase 3** (recommend/batch) — fast full-scale eval.
3. **Phases 4–5** — persistence + parity hardening.
4. **Phase 6** — only if no-Python deployment is wanted.

## Risks & mitigations
- **Parity drift** → differential testing at every phase; the PyO3 boundary lets
  us swap internals while keeping Python tests green; the `metadata_knn` port
  already proved byte-exact parity is achievable.
- **Scope creep** → loaders/eval/CLI stay Python; only the hot core moves.
- **Numeric ordering / f32–f64** → match the Python kernels (validated approach).
- **Build/naming** → already resolved (`kindling._core` via maturin).

## Definition of done
Full-window H&M (565k users) fits and serves on this box without thrashing;
the four reference numbers reproduce within tolerance; fit and batch-eval are
materially faster; the Python API is unchanged.

## Status (rust-engine-port branch)

Parity-first (reproduce the current engine exactly; the 4 reference numbers are
the gate), full-library + PyO3 the target. Differential harness: `bench/rust_parity.py`.

| piece | state |
|---|---|
| cooc, directional cooc, EASE (faer), cosine, metadata-kNN | ✅ already Rust |
| **cooc weight transform (wilson/cosine/jaccard)** | ✅ ported, **byte-exact** (this branch) |
| ⇒ the whole **base build** (cooc/EASE + transform) | ✅ Rust-capable |
| channel fit (trend_z, item_popularity, transitions, user-CF, last-item) | ⬜ next |
| native `EngineState` assembly | ⬜ |
| recommend (`_blend_channels` z-blend + retrieval + cold-slots) | ⬜ |
| persistence (bincode/rkyv) | ⬜ |
| ingestion (drop pandas — the memory win) | ⬜ |
| PyO3 `Engine` + full 4-dataset parity | ⬜ |

**Honest scope note:** the engine is ~1563 lines of orchestration over a large
`EngineState`; a full exact-parity Rust library is a multi-run effort, not a
single pass. This branch establishes the foundation — the base build is now
fully Rust-capable and the parity harness is the gate — with the remaining
phases above to follow, each parity-gated. The split that keeps parity cheap:
Python retains ingest + preprocess + activation-plan (resolve config); Rust
takes resolved arrays+config → fit (state) → recommend.
