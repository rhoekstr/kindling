# Migrating to kindling

Short mappings from the two packages kindling most often replaces.

## From LightFM

LightFM's `Dataset.build_interactions` yields a sparse COO matrix indexed
by internal integer ids. kindling accepts a long-format DataFrame with
raw ids and does the indexing for you.

| LightFM                                       | kindling                                       |
| --------------------------------------------- | ---------------------------------------------- |
| `Dataset().fit(users, items)`                 | — (kindling infers the index from interactions) |
| `dataset.build_interactions((u, i) for ...)`  | `pd.DataFrame({"entity_id": ..., "item_id": ...})` |
| `LightFM(loss="warp")`                        | `Engine()` (no training loop; closed-form base + auto-gated channels) |
| `model.fit(interactions, epochs=10)`          | `engine.fit(df)`                               |
| `model.predict(uid, np.arange(n_items))`      | `engine.recommend(entity_id=..., n=10)`        |
| `model.user_biases` / `model.item_biases`     | `engine.activation_plan` (which layers fired + why) |

Notes:

- LightFM's timestamped recency (via user/item features) becomes
  automatic trend/transition signals once you include a `timestamp` column.
- LightFM returns scalar scores; kindling returns `Recommendation`
  objects with `score`, `base_kind`, and `explanation`. If you only need
  the ids, use `[r.item_id for r in engine.recommend(...)]`.
- WARP sampling has no direct analog — kindling has no training objective;
  the EASE/wilson base is closed-form and the channels are counting
  statistics, all gated by regime (see `engine.activation_plan`).

## From Surprise (scikit-surprise)

Surprise is a ratings-prediction framework; kindling is a ranking
framework. The straight port is to treat Surprise's top-N evaluation
path as the kindling default and leave `rating` as metadata.

| Surprise                                        | kindling                                    |
| ----------------------------------------------- | ------------------------------------------- |
| `Reader(rating_scale=(1, 5))`                   | — (infer from data)                         |
| `Dataset.load_from_df(df, reader)`              | pass the DataFrame directly                 |
| `SVD().fit(trainset)`                           | `Engine().fit(df)`                          |
| `algo.predict(uid, iid).est`                    | `engine.recommend(entity_id=uid, n=...)` and look for `iid` |
| `algo.get_neighbors(iid, k=10)`                 | `engine.recommend_for_items(seed_item_ids=[iid], n=10)` (item-item from a seed) |

Notes:

- Surprise's `rating` column becomes **preference intensity** in kindling
  when real ratings are detected — the EASE base is rating-weighted (worth
  a few % NDCG). No configuration needed; confirm via
  `engine.activation_plan` (`rating-weighted`).
- Surprise's cross-validation splits are random; kindling's benchmark
  harness splits chronologically, which changes baseline numbers. The
  `bench/reports/` directory has reference NDCG/Recall/MRR on the four
  supported datasets for comparison.

## General advice

1. Start with `Engine()` and no overrides — the defaults transfer across
   datasets better than per-fit tuning (EXPERIMENTS.md §4.4).
2. Add a `timestamp` column to your interaction frame. Nearly every
   signal improves with it; trend and transitions depend on it.
3. Inspect `engine.activation_plan.summary()` once fit completes to see
   which base + channels the engine selected for your regime, and why.
4. Run your own benchmark via `DATASET=<name> python bench/verify.py`
   before comparing wall-time or metrics to your previous stack.
