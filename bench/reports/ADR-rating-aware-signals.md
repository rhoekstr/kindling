# ADR: rating-aware positive signals + centralized preprocessor

**Date:** 2026-04-24
**Status:** shipped. Rating handling is now a first-class preprocessing stage.
**Related:** [ADR-signal-audit.md](ADR-signal-audit.md)

## What shipped

**New centralized preprocessor.** `src/kindling/preprocess.py` attaches a
single `_interaction_weight` column to the interaction DataFrame at
`Engine.fit` time. Every positive-signal builder reads this column
instead of hard-coding ones. One code path for rating-aware logic
instead of scattering it across five signal files.

**Auto-detection + override.** `Engine(use_ratings=None)` (default)
auto-detects from the presence of a numeric rating column.
`use_ratings=True` forces on (raises on missing column).
`use_ratings=False` forces off (ignores column even if present).

**Weight transform.** When ratings are used:
```
w = max(0, (rating - threshold) / (scale_max - threshold))
```
Defaults: threshold=2.5, scale_max=5.0. On ML-1M's 1-5 stars this
gives:
- Rating 1/2 → weight 0 (dropped from positive signals; cost graph
  handles these as negatives instead)
- Rating 3 → 0.2
- Rating 4 → 0.6
- Rating 5 → 1.0

NaN ratings → 1.0 (implicit positive, unchanged from binary).

**ML-1M loader now preserves the rating column.** Previously it dropped
ratings < 4 entirely — Phase-1 binary-implicit shim that noted "later
phases may re-introduce them." This commit re-introduces them so the
preprocessor can see the data.

## Per-signal audit — who respects weights now

| signal | rating-aware | details |
|---|---|---|
| cooccurrence | **Yes** | Bipartite U-matrix uses per-row weight; cooc = Uᵀ U weights scale cleanly. |
| item_item_cosine | **Yes (inherited)** | Derived from the weighted cooc matrix. |
| als_factor | **Yes** | Weighted implicit feedback passed to `implicit` library's ALS. |
| persona (cluster) | **Yes** | `build_user_vectors` now uses weighted entries. |
| popularity (lift + cold-start fallback) | **Yes** | `compute_population_baselines` uses weighted sum. |
| cost_population / cost_entity / cost_context | Already rating-aware | Reads `rating < 2.5` for negative flagging (unchanged). |
| path_full / path_tail / path_basket | **No** (flagged) | Sequence structure - rating-weighting for these requires per-session weights (follow-up). |
| repeat | N/A | Interval-based, rating-independent by design. |

## Measurement — ML-1M (NDCG@10, 500 eval entities, full 100% data)

| config | NDCG | Recall@10 | MRR | zero-weight rows | uses_ratings |
|---|---|---|---|---|---|
| prior Phase-1 (rating>=4 filter) | 0.183 | 0.047 | 0.319 | — | False (loader-dropped) |
| **auto (use_ratings=None)** | **0.288** | 0.047 | **0.456** | 146k | True |
| forced False | 0.291 | 0.045 | 0.454 | 0 | False |
| forced True | 0.288 | 0.047 | 0.456 | 146k | True |

## Two honest findings

### 1. The NDCG lift from 0.183 → 0.29 comes from the LOADER CHANGE, not the rating transform

Re-introducing 1-3 star ratings (previously dropped) gives the engine
~2× more training data, which alone lifts MRR from 0.32 to 0.46 and
NDCG from 0.18 to 0.29. This is a real, measurable win from a
principled architectural cleanup — but it's not from the rating-
weighting mechanism per se. It's from "stop throwing away 60% of the
data."

### 2. Rating-weighted vs. binary-on-full-data is a wash on ML-1M

0.288 (weighted) vs 0.291 (binary) — 1% difference, probably within
noise. On ML-1M specifically, rating-weighting doesn't move the
needle because:

- Most user ratings are 3-5 (83%). The weighted transform maps these
  to [0.2, 1.0], but the weighting compounds with cooc's popularity
  bias (high-rated movies are also heavily co-rated) rather than
  adding genuinely new information.
- The 17% of 1-2 star ratings are dropped in both rating-weighted
  mode (zero weight) and cost-graph treatment. Binary-on-full-data
  includes them as positives; rating-weighted excludes them
  entirely from positive signals. End result on NDCG: very similar.

**This is consistent with the broader signal-audit finding**
([ADR-signal-audit.md](ADR-signal-audit.md)): on ML-1M, cooccurrence
is load-bearing and other variations of the same information don't
add much.

### Sanity: grocery is a perfect no-op

Grocery has no rating column; `use_ratings=None` auto-resolves to
False; NDCG 0.3197 identical to before. Backward compat preserved.

## What this architecture buys us

The immediate NDCG lift on ML-1M is smaller than we'd hoped, but:

1. **One place to own rating logic.** Any future signal (including
   LightGCN, queued next) reads weights from the same column. No
   per-signal copy-paste of `max(0, (rating - 2.5) / 2.5)`.
2. **Dataset-aware defaults.** Auto-detection means users on
   grocery/Amazon/RetailRocket-style data don't have to do anything;
   ML-1M users don't have to do anything; anyone with a real
   preference-strength rating gets the weighting transparently.
3. **Clean override surface.** Researchers can force ratings on/off
   to test ablations without touching loader code.
4. **Paths as a follow-up** — we identified that path signals don't
   yet respect weights. Next iteration: `SessionSequence.weight` =
   mean of session's interaction weights; multiplied into path/tail/
   basket counts.

## Deferred to follow-up

- **Path-family rating-awareness.** Sessions with mostly-low ratings
  should contribute less to the path tree / tail / basket counts.
- **HKV confidence formulation for ALS.** `implicit` natively
  supports `confidence = 1 + alpha * weight` semantics; current
  code uses weight directly as confidence. Could move closer to
  the academic formulation.
- **Eval-side rating handling.** Test positives are binary regardless
  of test rating — a 1-star test rating counts as a hit. Could
  extend to weighted-positive NDCG where 5-star test hits count
  more than 1-star.
- **LightGCN signal** (next up): the centralized preprocessor makes
  adding this cleanly much easier — it'll read the same weight column.

## What's shipping in this commit

- `src/kindling/preprocess.py` (new module) with
  `preprocess_interactions` + `InteractionContext` + `weights_of`.
- `Engine(use_ratings=...)` kwarg, plumbed through `fit()`.
- Signal builders updated: `build_item_graph`,
  `build_als_factors`, `compute_population_baselines`,
  `build_user_vectors` (persona).
- `loaders/movielens.py`: `to_interactions` now preserves the
  rating column by default (`drop_below_threshold=False`).
- Tests: `test_rating_weights.py` (6) for the full stack.
- Deleted obsolete `src/kindling/graph/weights.py`.
- Full suite: 283 passed, 1 skipped.
