# CLAUDE.md

The single reference for working in this repository — what kindling is, its architecture, the
invariants, and how to build/test it. **Canonical, version-controlled copy.** Keep it in sync with
the code (and with `docs/REFERENCE.md` / `docs/EXPERIMENTS.md` / `docs/LESSONS.md`).

## The rules that can't bend (READ FIRST)

1. **A wheel that imports is a wheel that works.** Runtime deps are **numpy / pandas / scipy only** —
   no PyTorch, no BLAS system deps. The linear algebra that matters (the EASE inversion) and the whole
   recommend path run in a **pure-Rust core** (`kindling_core`, via PyO3). **The v2 engine requires the
   Rust core — there is no Python fallback.** Don't add a heavy/native dependency to the runtime path;
   if it isn't numpy/pandas/scipy it's an optional extra, or it goes in Rust.
2. **Closed-form, gated channels beat speculative complexity.** Every channel is closed-form or a
   counting statistic; every channel is **activated by a measurable property of the data** (by regime,
   not configuration); every gate exists because the ungated version **measurably hurt somewhere**. New
   behavior earns its place against the bench, and the result — *especially a negative one* — is
   recorded in `docs/EXPERIMENTS.md`.

## Overview — what kindling is

A hybrid recommender that grows with your data: closed-form, no training loop, no GPU. One fused base
score per (user, item) from EASE / Wilson-cooccurrence plus auto-gated z-normalized channels (trend,
last-item, transitions, user-CF), blended natively. Single recommend is sub-millisecond; the batch
path runs in parallel with the GIL released. New/anonymous users are served from seed items (no
per-user training); an empty/all-unknown seed falls back to popularity. Apache-2.0 (Awry Labs, alpha).

## Architecture — where code lives

| Path | What |
|---|---|
| `src/kindling/` | the Python package (maturin `python-source`): `Engine` (fit / recommend / recommend_batch), `serving` (`KindlingServer`, `serving_app` FastAPI), loaders, Rust-binding glue |
| `native/kindling_core/` | the **Rust core** (`Cargo.toml`): EASE inversion, cooccurrence, channel blend, boost layer, cold-slots, recommend. Built as `kindling._core` (PyO3) |
| `bench/` | the **regression gate** (`bench/verify.py`) + frozen reports + growth-curve plots. `bench/experiments/` is provenance-only (lint-excluded) |
| `tests/` | unit, property (hypothesis), integration |
| `docs/` | `REFERENCE.md` (architecture + the channel gate table, §2) · `EXPERIMENTS.md` (the benchmark record, incl. negatives) · `LESSONS.md` · `RUST-ENGINE-PLAN.md` |

Channels activate by **regime**: EASE for catalogs ≤ 20k items, Wilson-normalized cooccurrence above;
trend needs timestamps; transitions need timestamps and not-a-rating-burst; user-CF only on
sparse-history data; rating-weighting only with true ratings. Each decision is made from the data at
`fit()` time — see `docs/REFERENCE.md` §2 for the gate table.

## Build & test

Mixed Python/Rust via **maturin** (build backend). A Rust toolchain is required.

```bash
pip install -e ".[dev]"          # builds the Rust core + installs the package + dev tooling
pip install -e ".[dev,bench]"    # + the benchmark harness

pytest                            # tests/ — note: filterwarnings=error (a warning fails the run)
ruff check . && mypy              # lint + strict typing (mypy strict on src/kindling)
python bench/verify.py            # the regression gate — run before claiming a perf/accuracy change
```

After editing Rust in `native/kindling_core/`, **rebuild** (`pip install -e .` or `maturin develop`)
before testing — a stale compiled `.so` is a silent-wrong-results trap. Test markers: `slow`,
`integration`, `bench`.

## Conventions

- **Python** — ruff (line length 100, broad select set) + **mypy strict** on `src/kindling`. Type
  hints + docstrings on public API. Lazy-import optional/heavy deps to keep import time clean.
- **Rust** — the core owns the numerics and the hot recommend path; keep the PyO3 surface narrow and
  release the GIL on the batch path.
- **The bench is the arbiter.** Accuracy/perf claims are made against `bench/verify.py` + the frozen
  reports, not vibes. Record experiments (including failures) in `docs/EXPERIMENTS.md`.
- **Commits** — branch from `main`; each logical change is its own commit; tests + bench green.

## Documentation set (keep current, don't proliferate)

| Doc | Update when you change… |
|---|---|
| **CLAUDE.md** (this) | the architecture map, the two invariants, build/test, conventions |
| `README.md` | the front-door overview, install, headline numbers |
| `docs/REFERENCE.md` | architecture + the channel gate table |
| `docs/EXPERIMENTS.md` | every benchmark run / experiment, especially negatives |
| `docs/LESSONS.md` | durable lessons the build taught |
| `docs/RUST-ENGINE-PLAN.md` | the Rust core plan / roadmap |
