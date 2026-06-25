# kindling

A hybrid recommender that grows with your data — closed-form, no training
loop, no GPU. One fused base score per (user, item) built from EASE /
wilson-cooccurrence plus auto-gated z-normalized channels (trend,
last-item, transitions, user-CF), with a Rust core for the numerics.

**Design goals (learned the hard way — see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)):**
1. **A wheel that imports is a wheel that works.** numpy / pandas / scipy
   only; the linear algebra that matters (the EASE inversion) runs on a
   pure-Rust core (`kindling_core`). No PyTorch, no BLAS system deps.
2. **Closed-form shallow models, gated per dataset, beat speculative
   complexity.** Every channel is closed-form or a counting statistic;
   every channel is activated by a measurable property of the data; every
   gate exists because the ungated version measurably hurt somewhere.

## Install

```bash
pip install kindling              # the engine — one wheel, Rust core included
pip install 'kindling[serve]'     # + the HTTP serving harness (FastAPI)
pip install 'kindling[baselines]' # + ALS/BPR/item-kNN for the eval harness
```

A single wheel ships the pure-Python package **and** the `kindling._core` Rust
extension — numpy / pandas / scipy are the only runtime deps. From a checkout,
`pip install -e ".[dev]"` gets the lint / type-check / test tooling.

## Quickstart

```python
from kindling import Engine
from kindling.loaders import movielens

interactions = movielens.load_1m()       # entity_id, item_id, timestamp[, rating]

engine = Engine()
engine.fit(interactions)

for rec in engine.recommend(entity_id=42, n=10):
    print(rec.item_id, rec.score, rec.explanation)
```

**New / anonymous users** (absent from training) are served from ad-hoc
seed items with no per-user training — and a zero/all-unknown seed set
falls back to popularity:

```python
engine.recommend_for_items(item_ids=[101, 205], n=10)   # personalized from seeds
engine.recommend_for_items(item_ids=[], n=10)           # → popularity fallback
```

## Prove it, then serve it

The `kindling` command wraps the same realistic-tier benchmark the project
validates itself with, plus a fit/serve path — see [`docs/HARNESS.md`](docs/HARNESS.md).

```bash
# Does it beat popularity (and ALS/BPR/item-kNN) on YOUR data, by warmth bucket?
kindling bench --data interactions.csv --all-baselines

# Fit, save, and serve over HTTP (POST /recommend, /recommend_for_items, /batch).
kindling fit   --data interactions.csv --out engine.kindling
kindling serve --model engine.kindling --port 8000
```

```python
from kindling.serving import create_app
app = create_app(engine)   # a FastAPI app — mount it, add auth, deploy it
```

## Intelligent activation

Channels turn on by *regime*, not configuration. The base is EASE for
catalogs ≤ 20k items and wilson-normalized cooccurrence above that;
the trend channel needs timestamps; transitions additionally need the
data not to be a rating-burst; user-CF activates only on sparse-history
data; rating-weighting engages only when true ratings are present. Each
decision is made from the data at `fit()` time. See
[`docs/REFERENCE.md`](docs/REFERENCE.md) §2 for the gate table.

## Where it stands (full-ranking NDCG@10, engine defaults)

| dataset | NDCG@10 | notes |
|---|---:|---|
| movielens-1m | 0.293 | rating-weighted EASE |
| amazon-beauty | 0.033 | + user-CF channel |
| steam (realistic tier) | 0.066 | open-catalog + cold slots |
| amazon-book-chrono | 0.032 | timestamps activate trend/transitions |

Strongest personalized model on all four; beats implicit ALS everywhere;
wins cold-*user* buckets on cold-heavy catalogs. The full benchmark
record — including the negative results, which are half the value — is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Project layout

```
src/kindling/          library source (engine, channels, Rust bindings, loaders)
  harness/             eval harness (reusable realistic-tier benchmark)
  serving/             FastAPI serving app
  cli.py               the `kindling` console command
native/kindling_core/  Rust numeric core (EASE, cooccurrence, layered scoring)
bench/                 regression gate (bench/verify.py) + frozen reports
docs/                  REFERENCE.md (architecture) · EXPERIMENTS.md (record) · HARNESS.md
tests/                 unit, property, integration
```

## License

Apache 2.0.
