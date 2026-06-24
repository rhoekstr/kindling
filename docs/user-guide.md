# kindling — User Guide

A practical guide to the kindling recommender (v0.2, the consolidated v2
engine). For the architecture and the empirical record see
[`REFERENCE.md`](REFERENCE.md); for a one-page system overview see
[`PRODUCTION-SYSTEM.md`](PRODUCTION-SYSTEM.md).

## Install

```bash
pip install -e ".[dev]"        # core + dev tooling
pip install -e ".[dev,bench]"  # + benchmark harness
```

## Fit and recommend

`kindling` takes a long-format interaction DataFrame with `entity_id` and
`item_id` (and, ideally, `timestamp` and/or `rating`). It infers the index
and chooses the scoring layers from the data — there is nothing to
configure for the common case.

```python
from kindling import Engine
from kindling.loaders import movielens

interactions = movielens.load_1m()   # entity_id, item_id, timestamp, rating

engine = Engine()
engine.fit(interactions)

for rec in engine.recommend(entity_id=42, n=10):
    print(rec.item_id, rec.score, rec.explanation)
```

A `Recommendation` carries `item_id`, `score`, `base_kind` (which base
scored it), and an `explanation`. For just the ids:
`[r.item_id for r in engine.recommend(entity_id=42, n=10)]`.

## New / anonymous users

Brand-new users absent from training are served with **no per-user
training** from whatever items they've touched; a zero/all-unknown seed
set falls back to popularity:

```python
engine.recommend_for_items(seed_item_ids=[101, 205], n=10)  # personalized from seeds
engine.recommend_for_items(seed_item_ids=[], n=10)          # → popularity fallback
```

The closed-form base scores any seed set immediately, so a user who just
interacted with a few items gets personalized recommendations on the spot.

## Understanding what activated

The engine gates each layer on a measurable property of your data. Inspect
the decisions after fit:

```python
print(engine.activation_plan.summary())
# Regime: 6,011 users x 3,883 items, median history 23, timestamps=True, ...
# Base: ease (lambda=750), rating-weighted
# Channels:
#   [ON ] trend x0.5 — recent-window popularity; needs timestamps
#   [ON ] last_item x0.25 — EASE row of the newest item ...
#   [off] transitions — ... (burst detected → off)
#   [off] user_cf — ... (median 23 > gate 20)
```

`engine.activation_plan.active_channels` returns just the active layer
names. This is how the system explains *why* it is configured the way it
is for your dataset.

## Getting good results

1. **Add a `timestamp` column.** It activates the trend and (on session
   data) transition channels; nearly every signal improves with it.
2. **Keep `rating` if you have real ratings.** The base becomes
   rating-weighted (preference intensity), worth a few % NDCG.
3. **Inspect `activation_plan`** to confirm the engine detected your regime
   correctly (catalog size → EASE vs cooc base; sparse history → user-CF).
4. **For cold/churning catalogs**, set `cold_slots=1` and keep
   `open_catalog=True` so metadata-only items can be recommended.
5. **Benchmark your own data** with `bench/verify.py` (`DATASET=...`) before
   comparing metrics to a previous stack.

## Cold-start

- **Cold *users*** (short history): handled automatically — kindling is the
  strongest personalized model on cold-heavy catalogs (see §3.5 of
  REFERENCE), and `recommend_for_items` serves anonymous users.
- **Cold *items*** (no interactions): the reserved `cold_slots` mechanism
  surfaces metadata-only items ranked by content similarity + release
  recency. This is the shipped cold-item answer; a learned content ranker
  was tried across four programs and retired (see EXPERIMENTS.md §4).
