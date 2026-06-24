# Production-transition worklog (autonomous run)

Branch: `prod-transition`. Started 2026-06-23 (overnight autonomous run).
Governs: [PRODUCTION-CONSOLIDATION.md](PRODUCTION-CONSOLIDATION.md).

**Run parameters (locked with the user up front):**
- Scope: complete all phases **including deletion** (Phase 5).
- Delivery: push branch + open **draft PR** on `origin` (rhoekstr/kindling).
- Behavior latitude: **adopt clear metric wins** from the z/boost sweep
  (record before/after); otherwise behavior-preserving except the
  intended v1→v2 default swap.
- Verification gate: **all four datasets** (ml1m, beauty, steam, book) +
  full pytest at each milestone. Unexplained regression = hard stop +
  notes. Crate-collapse (`kindling_native` removal) left as a flagged
  follow-up (Rust rebuild risk, not done unsupervised).
- Env: `.venv/bin/python` (editable install fixed); `CORE_AVAILABLE=True`;
  24 GB RAM (book ~18 GB peak — tight, run in background, OOM = note+continue).

**Capstone deliverable (user request, write last):** `docs/PRODUCTION-SYSTEM.md`
— clean honest description of the shipped system: what it is, what it
includes, value-add (and explicitly where it has none), noteworthy/novel
aspects, and *measured* performance statistics. Spec in PRODUCTION-CONSOLIDATION.md.

Reference metrics to hold (REFERENCE §3.3, EngineV2 defaults):
ml1m **0.2931** · beauty **0.0343** · steam **0.0660** · book-chrono **0.0318** (NDCG@10).

---

## Log

- **Setup** — branch `prod-transition` off master. 24 GB RAM confirmed.
  Env note: the editable install resolves `kindling` as a *namespace*
  package in this anaconda venv (`kindling.__file__` was `None`), so
  **direct `python` calls use `PYTHONPATH=src`**; `pytest` self-resolves
  via `pythonpath=["src"]` in pyproject. Standardized on these.
- **Verification harness** — built [`bench/verify.py`](../bench/verify.py):
  one entry point, the four documented per-dataset configs, reports
  NDCG@10/recall/MRR/HR. Doubles as the CI-minimal runner (Phase 6 keeper).
- **Baselines captured (my harness, seed=0, deterministic — the regression
  floor):**

  | dataset | NDCG@10 | recall@10 | MRR | HR@10 | fit | vs REFERENCE §3.3 |
  |---|---:|---:|---:|---:|---:|---|
  | movielens-1m | **0.2928** | 0.0611 | 0.4734 | 0.756 | 6s | 0.2931 ✓ exact |
  | amazon-beauty | **0.0328** | 0.0425 | 0.0432 | 0.094 | 13s | 0.0343 — gap = `user_cf` channel (see ⚠) |
  | steam | _(running)_ | | | | | 0.0660 |
  | amazon-book-chrono | _(pending, isolated)_ | | | | | 0.0318 |

  ⚠ **Open item for Phase 2 (default audit):** beauty reproduces 0.0328
  with λ=250 but REFERENCE's 0.0343 includes the `user_cf` channel — it
  may not be auto-firing under default config. Investigate whether the
  median-history gate (≤20) engages user_cf automatically on beauty.
- **Baseline test suite** — **416 passed / 7 failed / 1 skipped** (675s).
  ⚠ The 7 failures are ALL in `tests/unit/test_temporal_interaction.py`
  (e.g. `calibrate_kernel` returns `manual_fallback` ≠ `rating_burst_detected`)
  — a real pre-existing bug in `graph/temporal_interaction.py`, a **v1
  module in the delete set** (v2 uses its own Rust temporal layer). Not
  systemic to v2. **Gate from here = no NEW failures beyond these 7;** they
  vanish when the module is deleted in Phase 5.

### Phase 1 — experiment addendum (in progress)
- **Provenance audit** (subagent) — most addendum numbers TRACE to frozen
  artifacts; flagged a cluster of shipped-engine numbers that lived only
  in REFERENCE prose / runner stdout (would be lost on deletion): U1/U2
  channel progression, U3 cold-slot recovery, U4 academic baselines, plus
  a wrong Part V pointer and two uncovered ADRs.
- **Re-captured to frozen JSON** (the fix): `bench/capture_channel_ablation.py`
  → `channel_ablation_{movielens-1m,amazon-beauty}.json`. ml1m reproduces
  the progression near-exactly and **confirms +1.8% rating-weight**; beauty
  is messier (rating-weight neutral-to-negative cumulatively, user_cf the
  late lift, endpoint 0.0328 not 0.0343) — documented honestly in §II.2.
- **U3 cold-slot recovery** — capturing via `run_book_chrono` →
  `bench/reports/book_chrono_recovery.txt` (the isolated book run, also the
  book baseline). _(running, ~27 min)_.
- **EXPERIMENTS.md corrected** — fixed the rating-aware Part V pointer (it
  mis-pointed to persona ablations), added the channel-ablation +
  cold-slot artifacts, added the two uncovered ADRs (pair-index, phase-7),
  and added a "provenance honesty" note marking U4–U7 as REFERENCE
  synthesis (verdict sound, raw sweep not frozen).

### Dependency closure (Phase 3/5 foundation) — see `transition-deletion-map.md`
- Root coupling found: **`kindling/__init__.py` does `from kindling.engine
  import Engine` (v1)** — so importing the package drags in the entire v1
  stack. Rewriting `__init__` + deleting `engine.py` collapses it.
- v2's **true static footprint = 57 modules**; **83 deletion candidates**.
- CI utils (`metrics`/`parity`/`comparison`) currently import v1; `metrics
  → layer_scoring → basket_index` is keepable (so `MetricReport` survives);
  only `comparison.py` has a hard `from kindling.engine import Engine`
  (used by comparison arms, not `_load_dataset`) — strippable. `verify.py`
  already sidesteps this.
- `profile/*` is v1's; v2 profiles inline → Phase 4 ActivationPlan builds
  on engine_v2. `rerank.temperature` is v1-only (v2 uses `rerank.dpp`) →
  drop, don't port.

### Phase 3a — promote v2 to public Engine (the headline change)
- **`kindling/__init__.py` now exports `EngineV2 as Engine` + `RecommendationV2
  as Recommendation`.** `from kindling import Engine` resolves to the
  validated v2 stack (confirmed: `EngineV2` w/ `recommend_for_items`).
  **Ship == validated.**
- Blast-radius handled: 18 v1-API test files redirected to explicit
  `from kindling.engine import Engine` so they keep testing v1 until it's
  deleted in Phase 5. No test imports `Recommendation` from `kindling`; the
  4 `credible_interval` tests use explicit-v1 Engine → still valid.
- **Decision: drop credible-interval porting.** v2's `Recommendation` is
  (item_id, score, base_kind, explanation); credible intervals were a v1
  Bayesian-blend feature (blend being deleted) and aren't data-supported.
- README rewritten to v2 production reality (was stale "Phase 1, trivial").
- Verifying: full test suite post-swap (expect 416/7/1, no NEW failures);
  book baseline + U3 recovery still running (~27 min).
- **cold_impute removal (3b) mapped:** contained to engine_v2 constructor
  params + fit block L1292-1324 + recommend branch L2037-2042 + state
  field; default `cold_impute="content"` means removal is behavior-
  preserving (verify via steam staying 0.0660). Then `graph/cooc_impute.py`
  is deletable.

### Phase 3a/3b committed
- **3a `6e398ab`**: `__init__` → v2; 18 v1-API tests redirected to explicit
  v1; README rewritten. Tests 416/7/1 — the 7 are pre-existing v1/dead-module
  failures (gate, v1-persistence, v1-retrievers, temporal), confirmed via
  stash-to-baseline. No new failures.
- **3b `65259dc`**: removed `cold_impute`/`cold_impute_min_r2` knobs + the
  embedding-imputation paths from engine_v2; deleted `graph/cooc_impute.py`
  + `test_cooc_impute.py`. ml1m unchanged 0.2928; cold-slot content ranker
  smoke OK; import + v2-direct tests clean. (`bench/run_dataset_screen.py`
  still refs it — a delete-set script, goes in Phase 5.)

### Phase 5 deletion plan (refined; execute after book frees RAM)
Batches, leaf-first, ml1m+import after each, full pytest at the end:
1. **Leaf experiment code** (zero production importers): `benchmarks/*`
   experiment modules (keep `parity`, `metrics`, `comparison`, `baselines`,
   `gap_decomposition`, `layer_scoring` for CI); `bench/run_*.py` (keep
   `verify.py`, `capture_channel_ablation.py`, `run_book_chrono.py`);
   `dense_content.py`, `llm_enrich.py`; FPR PRD → archive.
2. **Strip v1 from CI-keep `comparison.py`** (`from kindling.engine import
   Engine`, `from kindling.personas import ...` → into the arms or drop).
3. **v1 + exclusive deps**: `engine.py`, `gate/*`, `personas/*`,
   `graph/{als_factors,cost_graph,item_cosine,lightgcn,persona_cooccurrence,
   session_cooccurrence,temporal_interaction}`, `profile/*`, `rank/*`,
   `rerank/{calibration,constraints,lift,temperature}`, `lifecycle/drift`,
   `outcomes/replay`, `retrieve/{interaction_neighborhood,interaction_network,
   policy,signal_retrievers}`, `blend/{bayesian,diagnostics,layered,
   layered_calibrator,outcome_builder,priors}` (verify `normalize` not
   needed by CI first).
4. **Test triage**: delete tests for each deleted module (incl. the 7
   pre-existing failures' files).
5. **Full pytest + 4-dataset gate**.

**Book run note:** the first book run OOM-killed (exit 137) under concurrent
load; lesson = book (18GB/24GB) must run ALONE. Resequenced: did Phase 5
deletion + verification while the machine was free, then launched book last
and alone.

### Phases 3b–6 + capstone — COMMITTED
- `65259dc` 3b cold_impute removal · `6e398ab` 3a promote v2.
- `af991c9` **Phase 5 deletion**: 175 files changed, 32,359 deletions; core
  130→40 modules. ml1m 0.2928 unchanged throughout.
- `03e7303` v2 integration test · `ef27810` **Phase 4 ActivationPlan**
  (`engine.activation_plan`).
- `388b046` Phase 6 config: gates.toml real baselines, version 0.2.0, ruff
  **739→0**, format clean. `a494ce9` docs rewrite (user/tuning/migration).
- **4-dataset gate (re-measured this run):** ml1m 0.2928, beauty 0.0328,
  steam 0.0660 — all unchanged post-deletion. Book = final run (in progress).
- Suite: **121 passed** (was 416/7); mypy: 18 pre-existing engine_v2 errors
  (master wasn't mypy-clean either — strict mode vs dynamic numpy; noted).
- Capstone `docs/PRODUCTION-SYSTEM.md` drafted; pending book NDCG + serve
  latency (fill after book frees RAM).

**REMAINING:** book NDCG + U3 capture → fill capstone + gates.toml + §U3;
measure serve latency; final pytest/ruff gate; push + draft PR.
