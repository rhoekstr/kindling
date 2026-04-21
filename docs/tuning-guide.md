# kindling tuning guide

Most of kindling's knobs are data-adaptive: leave them at defaults and the
engine will learn reasonable values from your interactions. This guide
covers the knobs worth thinking about, grouped by what you're trying to
change.

## Decay half-life

```python
from kindling import Engine
from kindling.lifecycle.decay import ExponentialDecay

engine = Engine(decay=ExponentialDecay(half_life_days=90))
```

- Default: `ExponentialDecay(half_life_days=180)`.
- Shorter half-life (30-60 days) for fast-moving catalogs (news, trending
  products). Old interactions fade quickly so recent behavior dominates.
- Longer half-life (365+ days) for stable catalogs (movies, books). Old
  interactions still carry signal.
- `LinearDecay(zero_at_days=...)` and `NoDecay()` ship as alternates.
- `CustomDecay(fn, name=...)` lets you supply any monotonic non-increasing
  function. kindling checks `decay(0) == 1.0` and monotonicity at
  construction.

The decay applies consistently to every structure that cares about age -
path trees, item graph, cost graph, basket index. Changing it is one
dial, not seven.

## Temperature (novelty vs. relevance trade-off)

Four input forms, all equivalent internally:

```python
engine.recommend(entity_id=..., temperature=0.3)                    # scalar
engine.recommend(entity_id=..., temperature=[0.0, 0.2, 0.5, 0.8, 1.0])  # per-position
engine.recommend(entity_id=..., temperature="balanced")             # named profile
engine.recommend(entity_id=..., temperature={0: 0.0, 4: 1.0})       # sparse dict
```

Named profiles live in `kindling.rerank.temperature`:

- `"balanced"` = `[0.0, 0.25, 0.5, 0.75, 1.0]` - reliable at the top,
  exploratory at the bottom.
- `"explore_tail"` = `[0.0, 0.0, 0.5, 1.0, 1.0]` - two safe picks, then
  sharply novel.
- `"conservative"` = `[0.0, 0.0, 0.0, 0.25, 0.5]` - argmax-dominant.

Solvers (`temperature_solver=`):

- `"beam"` (default, width 10) - near-optimal, fast.
- `"greedy"` - fastest; can be suboptimal when high- and low-temperature
  positions compete for the same items.
- `"dpp"` - DPP with position-dependent quality. Use when diversity is
  the dominant constraint.

### The coverage U-shape

Catalog coverage is not monotonic in τ. At τ=0 coverage is low (argmax
only). At τ=1 coverage is also low (novelty collapses toward a single
tail mode). Maximum catalog coverage appears at mid τ (0.4-0.6 on the
benchmark datasets). If you care about coverage, sweep τ and look for
the peak on your data - the Phase 7 reports under `bench/reports/` show
the shape on the four reference datasets.

## Blend weights

The default is the Bayesian blend - kindling learns weights from outcome
data and reports credible intervals. You don't set the weights directly.
If the posterior variance is too wide to trust (cold start, sparse
outcomes), the engine falls back to the heuristic blend with fixed
weights until the posterior tightens.

Inspect the posterior:

```python
summary = engine.posterior_summary()
print(summary.posterior_mean)          # per-signal weight
print(summary.credible_interval)       # Bayesian CI per signal
print(summary.warnings)                # e.g., simple-reporter calibration
```

If you want to override weights (disabling the learned blend):

```python
from kindling.blend.heuristic import HeuristicBlend

engine = Engine(blend=HeuristicBlend(weights={
    "path_full": 0.2,
    "path_tail": 0.15,
    "path_basket": 0.15,
    "cooccurrence": 0.3,
    "cost_population": 0.1,
    "cost_entity": 0.05,
    "cost_context": 0.05,
}))
```

The seven signals are fixed in v1. The prior coefficients for the
learned blend live in `src/kindling/blend/priors.toml` and are tunable
per install - see the file header comment for the mapping.

## Constraint design

Constraints apply between retrieval and ranking (plan departure from
PRD §7.6) so filtered items never reach the ranker:

```python
in_stock = set(catalog.available_now())

recs = engine.recommend(
    entity_id=...,
    constraints=[
        lambda item: item in in_stock,
        lambda item: not catalog[item].is_adult,
    ],
)
```

Guidelines:

- Predicates should be cheap and total (no I/O, no exceptions). A slow
  predicate runs once per candidate; a failing predicate short-circuits
  that candidate.
- If a predicate needs external state, close over a cached snapshot
  rather than querying live. The retrieval budget is typically 200-1000
  candidates.
- Order predicates from most- to least-restrictive. The short-circuit
  order matters for wall-time, not for correctness.
- Constraints do not persist with `engine.save(...)`. Pass them per call
  on the loaded engine.

## Diversity and calibration

- `diversity=0.5` (default 0) turns on DPP greedy MAP with cosine-kernel
  similarity. Higher = more diverse at the cost of per-item relevance.
- `calibration_weight=0.3` turns on Steck 2018 category calibration.
  Requires categorical metadata on items; configure via
  `Engine(category_index=...)`.
- `emphasis="distinctive"` with `lift_weight` in `[0, 1]` promotes items
  with above-average personal lift vs. the population baseline. Useful
  when you want the list to feel personal, not merely popular.

## When to retrain

Default retrain cadence: weekly for most production loads. Signals for
"retrain now":

```python
drift = engine.drift_report()
if drift.concerning:
    engine.fit(fresh_interactions)
```

`drift_report()` uses item-graph Spearman + neighbor overlap as defaults.
See `src/kindling/lifecycle/drift.py` for thresholds. The drift baseline
is calibrated at first retrain; a 3× baseline deviation flags as
concerning.
