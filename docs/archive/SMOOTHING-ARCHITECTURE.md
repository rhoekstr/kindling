# Metadata-smoothed cooc base — validated architecture (redesign spec)

The large-catalog (>20k items, EASE too costly) base architecture, validated on
H&M. This is the **redesign target** layered on *after* the Rust parity port;
the current engine keeps `metadata_smoothing` default-off so reference numbers
are unchanged. Experiments: `bench/experiments/smoothing_*.py`.

## What it is

Augment the wilson-cooc item-item base with a **content/metadata item-item
graph** (a hybrid / content-augmented item-item CF — not novel as an algorithm;
see the conversation record). Cold items have no co-occurrence, so they *borrow*
signal from metadata-similar hot items; hot items keep their own signal.

    base = wilson_cooc + M,   where  M[i,j] = sim(meta_i, meta_j) · cap · base_max
                              over each item's top-k metadata neighbours (kNN)

- **kNN** is the Rust `metadata_knn` (inverted index, rayon) — full catalogs, no subsampling.
- **Dose** = a fixed `cap` (the grounded/predicted weight under-doses ~15–30×).
- **Gate** = the slope of `cooc ~ metadata_sim`: only apply when `slope > 0`.

## The validated stack (H&M 50k, ndcg/recall@12)

| component | weight | result | role |
|---|---|---|---|
| plain wilson cooc | — | 0.0088 | baseline |
| **+ metadata smoothing (all items)** | cap≈0.1 | **0.0143 (+63%)** | the lever: hot→cold transfer |
| **+ hot-cooc reinforcement layer** | α≈0.5 | 0.0149 (+4%) | re-sharpen hot ranking, additively |
| **+ time-weighted cooc layer** | α≈0.25 | +2% | recency, *layered* not baked in |
| + trend / transitions / user-CF | existing | — | warm lift (orthogonal to smoothing) |

Full engine (channels on) + smoothing(cap0.1) on H&M: **+18% overall NDCG**
(cold tiers +21%/+22%). The base is the lever; layers are low-weight fine-tuning
that hurt if over-applied.

## Key findings (the load-bearing detail)

1. **All-items, not cold-only.** Smoothing must lift *all* items. Cold-only
   ("route cold→smoothed, hot→unsmoothed") **backfires** — lifted cold items
   leapfrog the un-boosted hot ones (the flood). Smoothing-all lifts warm too,
   and that warm bump is what keeps warm ahead. (`smoothing_hybrid.py`)
2. **Hot-cooc reinforcement is additive, not a split.** Adding back a pure-cooc
   layer with cold rows zeroed (`α≈0.5`) sharpens hot ranking with *no* cold
   dilution; `α≥1` reverts toward the flood. (`smoothing_hotlayer.py`)
3. **Decay belongs in a layer, not the base.** Time-decayed cooc ≈ plain cooc as
   a *base* (no-op); a time-weighted cooc *layer* (`α≈0.25`) adds a small win.
   (`smoothing_arch.py`)
4. **Complementary to the channels.** The channels lift *warm* (cold untouched);
   smoothing lifts *cold* (and warm a bit) — so they stack: full engine +
   smoothing is the best of all. (`smoothing_hybrid.py`)
5. **Dose drivers.** Optimal `cap` rises with metadata→cooc **fidelity**
   (fit slope/R²) and **cold-tail fraction**, ~monotone in both: low fidelity →
   0 (off); rich metadata + heavy tail → ~0.1. So a short, *centered* sweep
   (warm-started from those two measurable properties), not a blind grid.
   (`smoothing_cap_drivers.py`)
6. **Where it pays.** Needs (a) a base the channels haven't saturated and
   (b) rich, **graded** metadata. Binary single-category metadata (retailrocket)
   degenerates; thin metadata (book, 87% just 'Books') is dead.

## Engine knobs (already wired, default-off)

`metadata_smoothing` (off|on|auto|cooc|ease), `metadata_smoothing_cap` (≈0.1),
`metadata_smoothing_family` (logistic default), `metadata_smoothing_topk`.
The hot-cooc and time-cooc layers are **not yet wired** — they're the redesign
follow-up after the Rust parity port.
