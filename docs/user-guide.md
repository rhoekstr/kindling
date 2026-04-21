# kindling user guide

A hybrid recommender system that grows with your data.

## Install

```bash
pip install kindling
```

With the optional Rust extension (modest speedup):

```bash
pip install kindling[native]  # when the wheel exists for your platform
# or build from source:
pip install kindling
# then:
git clone ...
cd kindling/native
maturin develop --release
```

With optional dependencies:

```bash
pip install kindling[lightgbm]   # activates LightGBMRanker
pip install kindling[arrow]      # enables export_arrow() (pyarrow)
```

## Quickstart

```python
import pandas as pd
from kindling import Engine

interactions = pd.DataFrame({
    "entity_id": [...],
    "item_id":   [...],
    "timestamp": [...],      # optional but recommended
    "action_type": [...],    # optional: add/remove/rate for cost graph
})

engine = Engine()
engine.fit(interactions)

for rec in engine.recommend(entity_id="customer_42", n=10):
    print(rec.item_id, rec.score, rec.explanation.primary)
    print("  credible:", rec.credible_interval)
```

## Core concepts

- **Entity**: the subject of recommendations (user, customer, account).
- **Item**: what we recommend (product, movie, article).
- **Signal**: one dimension of the scoring stack. kindling ships seven:
  three path signals (full / tail / basket), cooccurrence, and three
  cost signals (population / entity / context).
- **Blend**: how signals combine. The Bayesian blend (default) learns
  weights from outcome data; the heuristic blend uses fixed weights
  until the posterior is confident.
- **Credible interval**: a Bayesian range on each recommendation's
  score, derived from the posterior over blend weights. *Not* a
  frequentist confidence interval; conformal prediction for frequentist
  coverage arrives in v1.x.

## Input format (PRD §4)

Required columns: `entity_id`, `item_id`. Optional columns the engine
activates automatically:

| Column         | Enables                                       |
| -------------- | --------------------------------------------- |
| `timestamp`    | Time decay, session inference, path signals, drift |
| `session_id`   | Explicit sessions (skips GMM inference)       |
| `action_type`  | Cost graph (`remove`, `negative_rating`)       |
| `rating`       | Low ratings populate the cost graph            |

## Recommendation controls

```python
recs = engine.recommend(
    entity_id="customer_42",
    n=5,

    # Stage 3 re-rank knobs (all optional):
    diversity=0.5,                        # DPP weight in [0, 1]
    temperature=[0.0, 0.25, 0.5, 0.75, 1.0],  # per-position novelty
    temperature_solver="beam",            # beam | greedy | dpp
    calibration_weight=0.3,               # Steck 2018 category calibration
    emphasis="distinctive",               # lift rare items
    lift_weight=1.0,                      # 0..1 lift intensity
    constraints=[lambda item: item != "out_of_stock"],
)
```

## Tuning guide

See [tuning-guide.md](tuning-guide.md) for decay half-life, temperature,
blend weights, constraint design, and the coverage U-shape that
surfaces at mid temperature.

## Observability

- `engine.posterior_summary()` — posterior mean, credible interval, VI
  diagnostics, simple-reporter warning if any.
- `engine.drift_report()` — item-graph drift, community stability,
  estimated retention horizon.
- `engine.data_density()` — item / entity / interaction counts.
- `engine.preserved_aggregates` — ledger of what pruning has removed.

## Persistence

```python
engine.save("customer_recs.kndl")
loaded = Engine.load("customer_recs.kndl")

# Cross-language interop via Apache Arrow IPC:
engine.export_arrow("customer_recs.arrow")
```

User-supplied constraint closures can't be pickled and are not
restored. Pass constraints per call on the loaded engine instead.

## Migration from LightFM / Surprise

See [migration-guide.md](migration-guide.md).
