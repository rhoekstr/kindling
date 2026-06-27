# The held-out repeat gate

On repeat-regime datasets (grocery, retail replenishment) the most valuable
recommendation is often the thing the user is about to **re-buy** — which the
default seen-item mask hides. kindling's repeat module re-surfaces reorders (the
personal-frequency layer + a REPLENISH timing multiplier; see
[REPEAT-FREQ-LAYER.md](REPEAT-FREQ-LAYER.md)). The hard problem is **when to turn
it on**, and this doc is about the gate that decides.

## Why a repeat-rate threshold doesn't work

The obvious gate — "enable repeats when the duplicate-interaction rate is high
enough" — fails, because re-logging is not repurchase:

| dataset | repeat rate | repeat-targets | repeat module under fair eval |
|---|---:|---:|---|
| tafeng / hm / dunnhumby / instacart / gowalla | 8–59% | 8–53% | **helps** (+47% to ~10×) |
| **steam** | 12% | **30%** | **hurts (−4%)** |

Steam has a *higher* repeat-target rate than Ta-Feng, yet the module hurts it:
its "repeats" are re-logged game sessions, not purchase intent, and the
frequency-reordering ranks worse than the plain EASE base. No count of
duplicates separates a habit (groceries) from a coincidence (re-logs).

## The gate: test it on a held-out

`Engine._apply_repeat_gate` keeps the module only if recommending reorders
**strictly improves** a held-out NDCG@10 (`repeat_recommend="auto"`; `True`
forces it on, ungated). A low `repeat_min_rate=0.05` pre-filter builds the
profile; the gate makes the real decision. Three properties make it trustworthy:

1. **It mirrors the benchmark's split protocol.** This was the lesson. A first
   version held out each user's most-recent items — which over-represents the
   re-logs and *confidently kept Steam*. The benchmark uses a **chronological
   global** split (most-recent fraction by global time), so the gate now does the
   same: hold out the most-recent 15% of interactions globally, predict them from
   the rest. With the matched protocol the gate cleanly declines Steam and keeps
   the genuine grocery logs. *A held-out gate that doesn't match the eval lies.*
2. **It's leak-free.** The full-train reorder profile contains the held-out tail,
   so "recommend ON" would trivially re-surface the targets. The gate rebuilds a
   profile on the held-out *history only* before comparing.
3. **It's faithful.** The held-out profile is rebuilt **with real timestamps**
   (and `now_ts` = the split point), so the REPLENISH timing multiplier behaves
   exactly as it does at serve time — which is what makes it suppress the
   just-played games that sink Steam.

The Rust `EngineState` exposes `set_repeat_active` / `set_repeat_profile` so the
gate compares ON vs OFF (and swaps the leak-free profile in and out) without a
refit.

## Decisions it makes (defaults)

| dataset | decision | held-out NDCG on/off |
|---|---|---|
| steam | **decline** (reference 0.0659 preserved) | 0.076 / 0.083 |
| tafeng | keep | 0.116 / 0.074 |
| hm | keep | 0.017 / 0.017 |
| dunnhumby | keep | 0.587 / 0.087 |
| instacart | keep, **ungated** (no timestamps → gate can't run; pre-filter on) | — |
| ml1m / beauty / book | n/a (≈0% repeat, pre-filter off) | — |

Because the gate declines Steam, the repeat work is NDCG-neutral on the discovery
reference numbers — steam stays base-only at 0.0659.

## Evaluating fairly

The repeat module only earns credit under **repeat-aware** eval (reorders count).
The warming-curve harness takes `REPEAT_AWARE=1` for this; the growth grid marks
repeat-aware rows `⟳` and leaves discovery rows on the standard exclude-seen
objective. Regression coverage: `tests/unit/test_repeat_gate.py` locks both
directions (keep true-repeat / decline useless-repeat) and the native toggle.
