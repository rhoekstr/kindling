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
| `LightFM(loss="warp")`                        | `Engine()` (learned blend, no single loss)     |
| `model.fit(interactions, epochs=10)`          | `engine.fit(df)`                               |
| `model.predict(uid, np.arange(n_items))`      | `engine.recommend(entity_id=..., n=10)`        |
| `model.user_biases` / `model.item_biases`     | `engine.posterior_summary()` + `engine.item_graph` |

Notes:

- LightFM's timestamped recency (via user/item features) becomes an
  automatic path/decay signal once you include a `timestamp` column.
- LightFM returns scalar scores; kindling returns `Recommendation`
  objects with `score`, `credible_interval`, and `explanation`. If you
  only need the ids, use `[r.item_id for r in engine.recommend(...)]`.
- WARP sampling has no direct analog - kindling uses a listwise
  calibration likelihood by default. If you want a pairwise objective,
  swap the likelihood in the blend (see `src/kindling/blend/likelihoods.py`).

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
| `algo.get_neighbors(iid, k=10)`                 | `engine.item_graph.neighbors(iid, k=10)`    |

Notes:

- Surprise's `rating` column becomes a *negative* signal in kindling: low
  ratings populate the cost graph rather than boosting the positive
  score. If you want ratings to drive the positive side, derive an
  interaction weight and replicate rows, or swap to a ratings-weighted
  retriever.
- Surprise's cross-validation splits are random; kindling's benchmark
  harness splits chronologically, which changes baseline numbers. The
  `bench/reports/` directory has reference NDCG/Recall/MRR on the four
  supported datasets for comparison.

## General advice

1. Start with `Engine()` and no overrides - the defaults are tuned on
   the four reference datasets.
2. Add a `timestamp` column to your interaction frame. Nearly every
   signal improves with it; the path family depends on it.
3. Inspect `engine.posterior_summary()` once fit completes. If
   credible intervals are wide on all signals, you are in the cold
   regime and the heuristic blend is in use.
4. Run your own benchmark via `python -m kindling.benchmarks.harness
   --dataset <your-loader>` before comparing wall-time or metrics to
   your previous stack.
