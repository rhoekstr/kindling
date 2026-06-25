# Next steps — hardening & performance

Follow-on plan after the v0.2 production consolidation (PR #4). Two themes:
**hardening** (make it a real, robust, distributable library) and
**performance** (latency, memory, scale). Items are grounded in the actual
state observed during the consolidation, prioritized, and phased so the
"unblocks a real install" work lands first.

Legend — effort: S (<1d) · M (1–3d) · L (>3d). Each item lists why, the
acceptance criterion, and dependencies.

## Status (2026-06-24)

All of Phases A–C landed on branch `hardening`:

| Item | Status |
|---|---|
| A1 ship Rust core in wheel | **DONE** — unified maturin build; clean-venv install works |
| A2 CI regression gate | **DONE** — `check_gate.py` + CI `gate` job |
| A3 persistence | **DONE** — `Engine.save/load` (versioned pickle) |
| A4 single Rust crate | **DONE** — v1 `kindling_native` deleted |
| B1 large-catalog memory | **PARTIAL** — fixed the cap (real RAM via `os.sysconf` + OS-reserve); the streaming cooc-build (the real book-OOM fix) is **deferred** (Rust, unverifiable on 24 GB) |
| B2 retrieval-first latency | **DEFERRED (measured)** — latency is 7 ms @124 k items; the refactor only matters >200 k items, which B1 gates |
| B3 perf smoke | **DONE** — `bench/perf_smoke.py` |
| C1 mypy strict | **DONE** — 0 errors; CI mypy blocking |
| C2 edge-case robustness | **DONE** — already robust; locked in with tests |
| C3 coverage + golden anchor | **DONE** — golden-output regression test |

**Remaining follow-ups** (not done this run): the streaming/sparse cooc
build (B1's real fix), retrieval-first serving (B2, when a >200 k-item
memory-feasible catalog exists), a portable npz/Arrow persistence format
(A3 is pickle today), and the Rust tail/basket/path fast-path port (A4
left those on the Python fallback).

---

## Phase A — Foundational hardening (makes it actually shippable)

### A1. Ship the Rust core in the wheel  · **CRITICAL** · L
**Why.** The wheel build is `hatchling` with `packages=["src/kindling"]` —
it does **not** build or include `kindling_core`. `_native.py` has **no
Python fallback for v2** (`CORE_AVAILABLE=False` → the engine can't fit).
So `pip install kindling` yields an importable-but-dead package. This
directly violates the "a wheel that imports is a wheel that works" goal.
**Do.** Move to a Rust-aware build backend (maturin, or hatchling +
maturin for the extension); wire `native/kindling_core` into the wheel;
add `cibuildwheel` to produce platform wheels (macOS arm64/x86, manylinux).
**Accept.** `pip install` into a clean venv on each target platform →
`Engine().fit(df); .recommend(...)` works; CI builds + smoke-tests the wheel.
**Depends.** Pairs with A4 (crate cutover) — settle the crate set first.

### A2. Wire the CI regression gate  · S
**Why.** `bench/gates.toml` (baselines + 2% limit) and `bench/verify.py`
("the regression gate") exist but nothing enforces them — the CI `bench`
job runs without comparing to the gate.
**Do.** A small `bench/check_gate.py` that runs `verify.py` on the
CI-feasible dataset (ml1m), reads `gates.toml`, and exits non-zero on a
>limit NDCG drop. Cache `~/.cache/kindling` for ml1m. steam/book stay
local (documented in gates.toml).
**Accept.** A deliberate scoring regression fails CI; a no-op change passes.

### A3. v2 persistence (save / load a fitted engine)  · M
**Why.** `persist/` was v1-only and was deleted; **v2 cannot serialize a
fitted model.** That blocks any real deployment (fit once, serve many).
**Do.** Serialize `V2FitState` — the cooc/EASE CSRs + dense B, channel
state (trend_z, transitions, content), item index, profile — to a
versioned on-disk format (npz/Arrow + a JSON manifest with a schema
version). `Engine.save(path)` / `Engine.load(path)`.
**Accept.** Round-trip test: `load(save(engine))` recommends identically;
forward-compat manifest version check; a multi-GB model (steam) round-trips.

### A4. Single Rust crate cutover  · S–M
**Why.** `kindling_native` (the v1 crate) is still in `native/`, marked
"delete after cutover" — dead build surface now that v1 is gone.
**Do.** Delete `kindling_native`; confirm `kindling_core` is the only crate
`_native.py` references; simplify `_native.py` (drop the v1 branch).
**Accept.** Clean `cargo build`; wheel builds one extension; tests green.

---

## Phase B — Performance (the scale story)

### B1. Fix the large-catalog memory blow-up  · **HIGH** · L
**Why.** book-chrono **OOM-killed on 24 GB** — full-extension (~18 GB peak)
*and* warm-only (357k-item cooc Gram) both exit 137. The open-catalog
extension + Gram materialization don't scale; the "memory-aware cap" didn't
prevent it on this box.
**Do.** Profile the fit peak (where the 18 GB lives). Likely fixes:
chunked/streaming cooc accumulation; keep the cooc base sparse end-to-end
(no dense materialization on the >20k wilson path); cap the open-catalog
extension against *measured* available RAM, not an estimate; f32/int32
intermediates; consider mmap for the extension block.
**Accept.** amazon-book-chrono fits on a 32 GB box (and ideally 24 GB) and
produces the NDCG the capstone currently has to cite from REFERENCE.

### B2. Retrieval-first serving — bound latency by candidate budget  · **HIGH** · M
**Why.** `recommend` computes the **full-catalog EASE vector per query**
then `argpartition`s top-K — O(n_items) per call. 0.5 ms on ml1m (3.6k
items) but **linear in catalog size**; on 357k-item book it's the latency
cliff.
**Do.** Insert a real candidate-generation step (cooc/ANN top-N retrieve)
*before* scoring, so only `retrieval_budget` items are scored. The
plumbing (`retrieval_budget`, the cooc retriever) exists; the EASE path
currently bypasses it by scoring everything.
**Accept.** Serve p50/p95 roughly flat from 10k → 500k items; no NDCG@10
regression on the four datasets (the budget already holds the answers per
the gap-decomposition pool-recall).

### B3. Perf regression harness in CI  · M
**Why.** No guard against fit-time / latency / memory regressions (the old
`benchmarks/perf.py` was deleted). The session's transient slowdowns went
unnoticed until measured by hand.
**Do.** A lean perf bench: fit seconds, serve p50/p95, peak RSS on a fixed
synthetic-large fixture; freeze baselines; gate growth (e.g. 10%). Run on a
CI-sized fixture, not the OOM datasets.
**Accept.** A deliberate 2× slowdown fails the perf gate.

### B4. Fit-time optimization  · M
**Why.** EASE inversion dominates fit (steam ~250s; book minutes). It's the
cost behind the slow `verify` runs and the OOMs.
**Do.** Profile fit; ensure `faer` uses all cores for the Cholesky/Gram;
evaluate f32-storage / f64-compute tradeoffs already in place; remove
redundant catalog passes. Target steam EASE < 60s.
**Accept.** Measured fit-time drop on steam/ml1m with NDCG unchanged.

### B5. Port remaining hot paths to Rust  · M
**Why.** Channel blending, the cold-slot ranker, and user-CF k-NN may still
be Python/numpy; profiling A1/B2 will show which dominate serve/fit.
**Do.** Profile first; port only the measured hot loops into `kindling_core`.
**Accept.** Latency/throughput improvement attributable to the ported path.

---

## Phase C — Robustness depth

### C1. mypy strict clean  · M
**Why.** 18 pre-existing strict errors in `engine_v2` (indexing `ndarray |
None`, a redef) — master wasn't mypy-clean either; the CI mypy job is
effectively red.
**Do.** Fix the Optional-narrowing + redefs; then make `mypy` a real CI gate.
**Accept.** `mypy` exits 0; wired into CI lint.

### C2. Input validation & edge-case robustness  · M
**Why.** Real data is messy; the cold/new-user paths are the most exposed.
**Do.** Harden `ingest.contract` against empty frames, single user/item,
all-cold catalogs, NaN/duplicate interactions, non-monotonic timestamps,
huge id cardinality; clear exceptions, not crashes. Add edge-case tests
(incl. `recommend_for_items` with unknown/empty/duplicate seeds).
**Accept.** Each malformed input yields a documented error or graceful
degradation, covered by tests.

### C3. Test coverage uplift + golden regression  · M
**Why.** Coverage dropped with the v1 deletion (and the v1↔v2 differential
test is gone). Current: 14 unit + 6 integration + 4 activation.
**Do.** Run `pytest --cov`; target gaps — loaders, cold-slot ranker, the
activation gates across regimes (burst → transitions off; sparse → user_cf
on; >20k → wilson base). Add hypothesis property tests (already a dep) and a
**frozen golden-output** fixture (small dataset → exact recs) as the new
regression anchor.
**Accept.** Coverage target met (e.g. ≥85% of the core); golden test guards
scoring drift.

### C4. Public API surface & errors  · S
**Why.** The shipped surface (`Engine`, `recommend`, `recommend_for_items`,
`activation_plan`, `Recommendation`, `ActivationPlan`) needs a stability
contract and consistent errors.
**Do.** Document the public API + stability/deprecation policy; consistent
exceptions (fit-before-recommend, unknown entity, unfitted activation_plan);
`__all__` hygiene.
**Accept.** API reference page; error-path tests.

---

## Suggested sequencing

1. **A1 + A4** (packaging + crate cutover) — without these it isn't a real
   library; do them together.
2. **A2, C1** (CI gate + mypy) — cheap, make the pipeline trustworthy.
3. **B1, B2** (memory + retrieval-first) — the two real scale gaps; B1
   unblocks measuring book, B2 bounds serve latency.
4. **A3** (persistence) — needed before any serving deployment.
5. **B3, C2, C3** (perf gate, robustness, coverage) — depth.
6. **B4, B5, C4** (optimization + API polish) — last, profile-driven.

## Explicitly out of scope (decided in the consolidation)
Sequential/learned ranking, content cold-start, learned activation — all
closed as philosophy-bounded (see EXPERIMENTS.md). Re-port of the
repeat-consumption layer (+6% replenishment) is a *feature* follow-up, not
hardening/perf — track separately.
