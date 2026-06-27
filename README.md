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
pip install -e ".[dev]"      # dev tooling
pip install -e ".[dev,bench]"  # + benchmark harness
```

## Quickstart

```python
from kindling import Engine
from kindling.loaders import movielens

interactions = movielens.load_1m()       # entity_id, item_id, timestamp[, rating]

engine = Engine()
engine.fit(interactions)

for rec in engine.recommend(entity_id=42, n=10):
    print(rec.item_id, rec.score, rec.base_kind)

# Many users at once — runs in parallel in the Rust core (GIL released).
batches = engine.recommend_batch([42, 99, 7], n=10)
```

Recommendation is served end-to-end by the Rust core (`kindling_core`): the
EASE/cooc base, the channel blend, the boost layer, and cold-slots all run
natively. Single recommend is sub-millisecond; batch is the parallel path.

**New / anonymous users** (absent from training) are served from ad-hoc
seed items with no per-user training — and a zero/all-unknown seed set
falls back to popularity:

```python
engine.recommend_for_items(item_ids=[101, 205], n=10)   # personalized from seeds
engine.recommend_for_items(item_ids=[], n=10)           # → popularity fallback
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

> Full results — discovery growth **and** the repeat-regime dominance — in [`docs/RESULTS.md`](docs/RESULTS.md).

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

On **repeat-regime** datasets (grocery/retail), a held-out gate turns on reorder
recommendation; under repeat-aware eval kindling separates from the field —
e.g. Dunnhumby 0.48 NDCG@10 vs ~0.05 for every baseline — while correctly
*declining* on fake-repeat data like Steam (re-logs aren't repurchase). See
[`docs/REPEAT-GATE.md`](docs/REPEAT-GATE.md). An opt-in EASE+ (EDLAE) base is
available but off by default ([`docs/EASE-VARIANTS-ASSESSMENT.md`](docs/EASE-VARIANTS-ASSESSMENT.md)).

### Growth curves

How accuracy grows from cold to hot, against the standard baselines
(`bench/plot_growth_curves.py`):

![growth curves](bench/reports/growth_curves_grid.png)

### Serving performance (native engine, `bench/final_state_perf.py`)

| dataset | fit | single recommend p50 | batch throughput | NDCG@10 |
|---|---:|---:|---:|---:|
| movielens-1m | 4.2 s | 0.17 ms | 15.4k recs/s | 0.2928 |
| amazon-beauty | 13.1 s | 1.21 ms | 3.0k recs/s | 0.0328 |
| steam | 110 s | 5.81 ms | 0.8k recs/s | 0.0659 |

The recommend path is pure Rust with the GIL released for the batch path —
single recommend dropped from ~200 ms (the earlier Python path) to
sub-millisecond, with byte-identical rankings.

### Serving

Persist a fit as a self-contained artifact and serve it with no re-fit:

```python
from kindling.serving import KindlingServer
KindlingServer.from_engine(engine).save("artifact/")
# ── in the serving process ──
server = KindlingServer.load("artifact/")
server.recommend("user-42", n=10)
```

A FastAPI example (`kindling.serving_app`) ships behind the optional
`serve` extra: `pip install 'kindling[serve]'`.

## Project layout

```
src/kindling/      library source (engine, serving, Rust bindings, loaders)
native/kindling_core/  Rust core (EASE, cooccurrence, channel blend, recommend)
bench/             regression gate (bench/verify.py) + frozen reports + plots
docs/              RESULTS.md (what it brings) · REFERENCE.md (architecture) ·
                   EXPERIMENTS.md (record) · LESSONS.md (what the build taught)
tests/             unit, property, integration
```

## License

Apache 2.0.
