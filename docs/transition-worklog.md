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
