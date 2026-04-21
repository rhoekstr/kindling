# kindling

A hybrid recommender system that grows with your data.

**Status:** pre-alpha. Phase 1 of the [implementation plan](../../../.claude/plans/read-this-prd-ponder-fluffy-turing.md) — a runnable end-to-end pipeline and benchmark harness on MovieLens-1M. Algorithms are intentionally trivial at this stage; the point of Phase 1 is the scaffold, not the recommendations.

## Install (dev)

```bash
pip install -e ".[dev,bench]"
pre-commit install
```

## Quickstart

```python
from kindling import Engine
from kindling.loaders import movielens

interactions = movielens.load_1m()

engine = Engine()
engine.fit(interactions)

recs = engine.recommend(entity_id=42, n=10)
for rec in recs:
    print(rec.item_id, rec.score, rec.explanation.primary)
```

## Running the benchmark harness

```bash
python -m kindling.benchmarks.harness --dataset movielens-1m
```

See `bench/gates.toml` for the CI regression thresholds.

## Project layout

```
src/kindling/    # library source
tests/           # unit, property, integration
bench/           # regression gates + frozen reports
docs/            # (later) user guide, cookbook, tuning guide
```

## License

Apache 2.0.
