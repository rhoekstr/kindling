# Production Consolidation Plan

> The shift from an experimental research engine to a clean, robust,
> pip-installable library — built around the layers the benchmarks
> actually validated, with explicit activation detection, and with the
> full experimental record preserved as a documented addendum.
>
> Status: **planning**. Decisions locked 2026-06-23 (see "Decisions").
> This plan governs the sequence; `REFERENCE.md` remains the living
> architecture + experiment source of truth.

---

## 0. The finding that motivates this

**What ships is not what was validated.** `from kindling import Engine`
resolves to the v1 engine (`engine.py`) — a Bayesian-blend scorer with
every advanced subsystem off by default. But *every* modern benchmark
(`bench/run_warming_curve.py`, `run_gap_decomp.py`, `run_beauty_retrieval.py`, …)
instantiates `EngineV2` directly, and `REFERENCE.md` documents v2 as
"the engine." The validated stack is reachable only via the non-default
`Engine(use_v2_core=True)`.

Consolidation is therefore not primarily new capability. It is: **make
the validated engine the shipped engine, make the activation logic
explicit and inspectable, and delete everything the data retired —
keeping the writeup.**

## Decisions (locked)

1. **Dead code → delete, keep only the writeup.** Confirmed dead-end
   modules leave the tree entirely. Their knowledge survives as the
   curated experiment addendum (§4) + the frozen evidence in
   `bench/reports/` + git history. **Corollary (binding sequence): the
   writeup is a prerequisite for deletion, not a follow-up.** No module
   is deleted until its result is captured in the addendum and the ADR
   evidence it depends on is confirmed retained.
2. **Target = pip-installable library.** Clean `Engine` API, single Rust
   core, real CI gates, serialized activation plan, honest docs. The
   "a wheel that imports is a wheel that works" philosophy (REFERENCE §1).
   No serving layer in scope.

## Non-goals (explicitly out of scope — the data closed these)

- **Closing the ml1m ranking gap** (oracle 0.93 vs current 0.29). It is
  *sequential* headroom; capturing it requires training-based sequence
  models, out of philosophy. §7.2 is closed — do not reopen.
- **Content cold-start** (content channel / imputation / enrichment /
  grafting). Closed across four independent attempts (§4.6, §4.7, §4.9,
  §7.6). The shipped cold-start answer is the structural `cold_slots`
  mechanism, not a learned content ranker.
- **Learned ranking / learned gating** (LambdaRank, GBM reranker, gate
  MLP). They don't deploy — the internal holdout inverts the test
  ranking (§4.4, §7.2). The activation detector is **deterministic** by
  design *because the data proved deterministic beats learned here*.

---

## 1. Keep / delete inventory (dependency-verified)

Verified against actual import edges, not docstrings. Deleting v1
dissolves most coupling automatically: the dead-end retrievers
(lightgcn/als/personas) are reached only through
`retrieve/signal_retrievers.py` (a v1-path module), and the lone
`benchmarks → engine` leak is `blend/layered_calibrator.py`, which is
v1-only.

### KEEP — production core

| Area | Modules | Note |
|---|---|---|
| Engine | `engine_v2.py` → promoted to `Engine` | the validated stack |
| Rust core | `native/kindling_core` | sole crate post-cutover |
| Base/channels | `graph/cooc_transform.py` (wilson), EASE/trend/last-item/transitions (in core) | REF §2 |
| Conditional layers | `path/{basket_index,tail_index,_sessions}.py`, repeat module | session/replenishment-gated |
| Cold-start (shipped) | `cold_slots` path + `item_features.py` (cold-slot ranker only) | REF §4.8 — keep extractor, drop warm content *blending* |
| Serving | `recommend`, `recommend_for_items`, popularity fallback + EB shrinkage | REF §7.4 |
| Plumbing | `ingest/`, `preprocess.py`, `loaders/`, `persist/`, `explain/`, `lifecycle/` (decay/pruning if live) | |
| Minimal bench (for CI) | `benchmarks/{parity,metrics,gap_decomposition}.py` + one runner | gates + the "run-before-believing" diagnostic |
| Evidence archive | `bench/reports/ADR-*.md`, `consolidated/`, `parity/`, frozen result files | the chain-of-evidence; **writeup, not code** |

### DELETE — dead ends (writeup preserved in §4 addendum)

| Module(s) | Verdict / evidence | Coupling to resolve |
|---|---|---|
| `engine.py` (v1) + `blend/{bayesian,heuristic,layered_calibrator,decorrelate,diagnostics,priors,outcome_builder,likelihoods}` | superseded by v2 | removes the `benchmarks` import leak |
| `retrieve/signal_retrievers.py` (+ stack builders) | v1 retriever path | unblocks lightgcn/als/persona deletion |
| `gate/*` (learned MLP) | doesn't deploy (§4.4, §7.2) | v1-only, no other importers |
| `personas/*`, `graph/persona_cooccurrence.py` | zero LOO, collapses to cooc (§4.1) | |
| `graph/{lightgcn,als_factors,graph_mf,session_cooccurrence}.py` | identical-in-blend / no lift (§4.2, §4.3) | |
| `graph/cooc_impute.py` | imputation lowers NDCG (§4.9) | **remove dormant `cold_impute` path from engine_v2; keep `cold_slots` content ranker** |
| `llm_enrich.py`, `dense_content.py` | probe gate fails (§4.7) | self-contained pair |
| Force-projection (`force-projection-recommender-benchmark-prd.md` + any FPR code) | below popularity floor | |
| Most `benchmarks/*` + `bench/run_*.py` | experiment drivers | keep only the CI-minimal set above |

**Pre-deletion gate (per module):** result captured in §4 addendum ✓ ·
evidence ADR retained in `bench/reports/` ✓ · no production importer
(re-grep) ✓.

---

## 2. The activation-detection upgrade ("real added value layers with intelligent activation detection")

Today activation is (a) scattered across `fit()`, (b) partly fake — the
layered `z=2.5 / boost=3.0` are hardcoded but documented as "calibrated
via held-out NDCG sweep" (`engine_v2.py` docstring step 6 is
unimplemented), and (c) partly dead — the gate MLP is built but unused.

**Target:** one typed, inspectable `ActivationPlan`.

```
Profiler(interactions, metadata) → regime features:
    n_items, density(nnz), has_timestamps, rating_signal∈{binary,ratings},
    rating_burst, median_user_history, cold_item_fraction, has_metadata,
    deep_session_fraction, catalog_churn
        │
        ▼
ActivationPlan  (serializable, attached to the fitted model):
    base            = ease | wilson_cooc | cooc      [n_items ≤ 20k → ease]
    channels        = {trend, last_item, transitions, user_cf} with weights
    boost_layers    = {path, session, temporal}      [size/session-gated]
    cold            = {cold_slots, open_catalog}      [metadata + churn]
    repeat          = on|off                          [replenishment]
    each decision → {condition, measured_Δ, evidence_ref(REF §/ADR)}
```

Properties that make "robust results" real:
- **Inspectable:** `engine.activation_plan` returns the plan + the
  reason and measured justification for every on/off decision.
- **Serialized** with the model (`persist/`), so a loaded model can
  explain its own configuration.
- **Tested as decisions, not just metrics:** assertions like
  `ml1m → transitions OFF (rating-burst), user_cf OFF (dense)`;
  `steam → cold_slots=1, transitions ON`.
- **Surfaced in explanations:** "ranked by EASE base + active trend
  channel" rather than an opaque score.
- **Deterministic by evidence:** §4.4/§7.2 proved fixed cross-dataset
  gates beat per-fit/learned calibration — this is the honest form of
  the abandoned learned gate.

Also in scope: run the `(z, boost)` sweep **once** to either freeze a
per-regime value or replace the false "calibrated" claim with an honest
"fixed prior, evidence: …".

---

## 3. Phased sequence

Ordering reflects the binding corollary (**addendum before any code
change**) and risk (correctness-of-shipped before cleanup before polish).
**No code is modified until Phase 1 is signed off.**

**Phase 1 — Build & verify the experiment addendum (NO code changes).**
The writeup that makes deletion safe. Deliverable at
[`docs/EXPERIMENTS.md`](EXPERIMENTS.md). Sub-steps:
- **1a — Catalog.** Enumerate every experiment from all three sources:
  `bench/reports/ADR-*.md` (20), the ~150 frozen result artifacts under
  `bench/reports/`, and the `§`-tagged commits. *(done)*
- **1b — Evidence-provenance audit.** Confirm every number the addendum
  cites traces to a retained artifact. Flag any result that lives *only*
  in git history or uncaptured printed output — those must be re-captured
  to a file *before* their code becomes deletable.
- **1c — Retention manifest.** `bench/reports/` is writeup → retained in
  full; runner code (`bench/run_*.py`, `benchmarks/*`) is code →
  deletable. List any at-risk artifact explicitly (addendum Part V).
- **1d — Write the addendum** (Parts I–V: methodology · positive
  architecture record · rejected fence posts · open fronts · evidence
  map). *(draft complete)*
- **1e — Doc topology.** `EXPERIMENTS.md` becomes the canonical experiment
  record; plan `REFERENCE.md`'s slim-down to shipped architecture (its
  §4/§7 migrate out); resolve the two-`§`-numbering confusion (docx vs
  REFERENCE) in-doc. *(captured)*
- **1f — Sign-off.** Comprehensiveness + accuracy review **gates all
  later deletion.**

**Phase 2 — Freeze the contract (low code).** Public API surface
(`Engine`, `recommend`, `recommend_for_items`, `activation_plan`,
`explanation`, persistence). Run the `(z, boost)` sweep once; replace or
honest-document the false "calibrated" claim. **Default-knob freeze:**
every default in `REFERENCE §5` becomes the production contract with an
evidence reference. Decide the **supported-loader set** (realistic-tier
`steam`/`amazon_chrono` are load-bearing; others may be experimental).
Confirm single-crate cutover timing (`kindling_native` is marked "delete
after cutover" in `_native.py`).

**Phase 3 — Promote v2 to `Engine`.** Make the validated stack the
default export. **Rename mechanics:** `engine_v2.py → engine.py`,
`EngineV2 → Engine`, `RecommendationV2 → Recommendation`, drop the
`use_v2_core` flag. Remove the dormant `cold_impute` path from the engine
(keep `cold_slots`). Port only what data supports — credible
intervals/explanation, the conditional repeat multiplier (+6% grocery),
the temperature *knob* as opt-in diversity; **not** the failed learned
components. **Test triage:** classify every test as v1-only / v2-only /
shared before v1 leaves; one last v1↔v2 differential, then v1 is staged
for deletion.

**Phase 4 — Extract `Profiler → ActivationPlan`.** Unify the scattered
gates into the one component in §2; add introspection, serialization with
the model, and decision-level regression tests.

**Phase 5 — Delete (gated by Phase 1 sign-off).** Execute the §1
deletions module by module through the pre-deletion gate (result captured
✓ · ADR retained ✓ · no production importer ✓). Move historical PRDs
(`kindling_PRD_v08.docx`, `force-projection-…-prd.md`) to `docs/archive/`.
Collapse to the single Rust crate.

**Phase 6 — Production hardening (pip library).** Rewrite README (still
says "Phase 1, trivial algorithms"), user-guide + tuning-guide (still the
v1 seven-signal world); slim `REFERENCE.md` to shipped architecture.
Populate `gates.toml` with real baselines (§5) — and decide **CI-dataset
feasibility**: ml1m is CI-runnable; steam/book need large local caches,
so their gates are local/manual validation, not CI. Packaging + version
bump (0.0.1.dev0 → a real v2 line). Production-path test coverage.

**Capstone — `docs/PRODUCTION-SYSTEM.md` (write last, with measured
numbers).** A clean, honest description of the shipped system for someone
who never saw the experiment history:
1. **What it is** — one-paragraph definition + the design philosophy.
2. **What it includes** — the engine, the base/channels, the activation
   detection, cold-start/new-user serving, the Rust core, loaders, the
   public API surface. A component inventory, not aspirational.
3. **Value-add — and where it has none.** The honest two-sided claim:
   strongest personalized model on all four datasets, beats ALS
   everywhere, wins cold-*users* on cold-heavy catalogs; **does NOT** beat
   popularity in data-starved global regimes, has closed-as-bounded
   ranking/retrieval headroom (sequential / discriminative), and banks no
   value from content cold-start. State the non-value-add plainly.
4. **Noteworthy / novel** — the auto-gated regime activation (deterministic,
   because learned calibration provably doesn't deploy here); raw-cooc =
   popularity-in-costume → EASE/wilson pivot; the realistic-tier
   methodology; "a wheel that imports is a wheel that works"; closed-form
   no-training serving incl. anonymous users.
5. **Performance statistics** — the final four-dataset NDCG@10/recall/MRR/HR
   from `bench/verify.py`, fit times, serve latency, vs popularity/kNN/ALS,
   and the cold-user segment slices. Real measured numbers, not REFERENCE
   carryover.

---

## 4. The experiment addendum

Built at [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) — the curated record that
replaces the deleted code as the surviving knowledge. Three-part spine:
the **methodology** (academic vs realistic tier; gap-decomposition) that
makes the verdicts legible; the **positive record** (Phases 1–8 build +
the 2026-06 EASE/wilson pivot — how the engine got its shape); and the
**negative record** (the fence posts: personas · ALS/LightGCN/graph-MF as
signals · content/enrichment/imputation/grafting · learned gate/ranker ·
per-fit calibration · score-norm default · force-projection). Plus the
open fronts and a full evidence map. `bench/reports/` (ADRs + frozen
metrics) remains the deep chain-of-evidence beneath it, retained in full.

### 4.1 Re-examination — additional steps this surfaced

Detailing the addendum-first pass exposed steps the first draft missed,
now folded into the phases above:

| Step | Phase | Why it matters |
|---|---|---|
| Evidence-provenance audit | 1b | "Keep only the writeup" fails silently if a cited result lives only in printed output / git — must re-capture before deletion |
| Retention manifest (`bench/reports/` = writeup) | 1c | Draws the precise code/writeup line so deletion can't take evidence with it |
| Doc topology + slim `REFERENCE.md` | 1e/6 | Two `§`-numbering systems exist (docx vs REFERENCE); the canonical record must be unambiguous |
| Default-knob freeze with evidence | 2 | Promoting v2 makes its defaults the contract — each needs a justification of record |
| Supported-loader set | 2 | Not all 12 loaders are production; realistic-tier ones are load-bearing for the claims |
| Rename mechanics | 3 | `engine_v2/EngineV2/RecommendationV2/use_v2_core` → clean names is a discrete, reviewable step |
| Remove dormant `cold_impute` (keep `cold_slots`) | 3 | The one dead-end path entangled with a *shipped* feature — needs surgical, not blanket, removal |
| Test triage (v1/v2/shared) | 3 | Deleting v1 will break/remove tests; must be classified first |
| CI-dataset feasibility | 6 | steam/book gates can't run in CI (large caches) — they're local/manual validation |
| Version bump + packaging | 6 | 0.0.1.dev0 → a real v2 line for a pip release |
| Archive historical PRDs | 5 | docx + FPR PRD are vision/negative-result, not operative — `docs/archive/` |

## 5. CI baseline targets (Phase 6 — replaces the `0.0` placeholders)

| dataset | NDCG@10 | notes |
|---|---:|---|
| movielens-1m | 0.2931 | rating-weighted EASE |
| amazon-beauty | 0.0343 | λ=250, +user_cf |
| steam (realistic) | 0.0660 | open-catalog, cold_slots=1 |
| amazon-book-chrono | 0.0318 | timestamps activate trend/transitions |

Regression limits per `bench/gates.toml` (currently 2% relative drop).
