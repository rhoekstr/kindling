# ADR: signal audit — which of the 9 signals earn their compute?

**Date:** 2026-04-23
**Status:** shipped (harness); signals NOT pruned yet pending real-session data
**Related:** [ADR-growth-curves.md](ADR-growth-curves.md), [ADR-lightgbm-warm-regime.md](ADR-lightgbm-warm-regime.md)

## What ran

`kindling.benchmarks.signal_ablation` on two datasets, three fractions
each. Per (dataset × fraction) we fit the Engine once and evaluate
under:

- **full** — all 9 signals active (baseline).
- **`-<signal>`** — leave-one-out: mask one signal, keep the other 8.
  Delta vs. full = that signal's marginal contribution.
- **`only_<family>`** — mask everything except a family. Measures each
  family's standalone accuracy floor.

Also recorded: posterior weights per fraction (shows how the blend
adapts across the growth curve).

Reports:
[signal_ablation_grocery.json](signal_ablation_grocery.json),
[signal_ablation_movielens.json](signal_ablation_movielens.json).

## Headline finding

**Cooccurrence is the only signal whose LOO impact exceeds noise on either dataset.**

### synthetic-grocery-deep (full dataset, 500 eval entities)

| config | NDCG @10 | LOO delta vs full |
| ------ | --------: | ----------------: |
| full | 0.3197 | — |
| -path_full | 0.3192 | −0.0005 |
| -path_tail | 0.3200 | +0.0003 |
| -path_basket | 0.3198 | +0.0001 |
| **-cooccurrence** | **0.2223** | **−0.0974** |
| -item_item_cosine | 0.3194 | −0.0003 |
| -als_factor | 0.3197 | 0.0000 |
| -cost_population | 0.3194 | −0.0003 |
| -cost_entity | 0.3194 | −0.0003 |
| -cost_context | 0.3196 | −0.0001 |
| only_cooc | 0.3196 | −0.0001 (matches full) |

### movielens-1m (full dataset, 500 eval entities)

| config | NDCG @10 | LOO delta vs full |
| ------ | --------: | ----------------: |
| full | 0.1826 | — |
| -path_full | 0.1827 | +0.0001 |
| -path_tail | 0.1827 | +0.0001 |
| -path_basket | 0.1830 | +0.0004 |
| **-cooccurrence** | **0.1530** | **−0.0296** |
| -item_item_cosine | 0.1826 | 0.0000 |
| -als_factor | 0.1823 | −0.0003 |
| -cost_population | 0.1827 | +0.0001 |
| -cost_entity | 0.1826 | 0.0000 |
| -cost_context | 0.1827 | +0.0001 |
| only_cooc | 0.1828 | +0.0002 (matches full) |

**Cooccurrence alone reproduces the full blend on both datasets.**
Every other signal can be deleted with ≤0.05% NDCG impact.

## The posterior is not tracking signal value

Posterior weights per fraction (constant because we never feed outcomes
back → prior-dominated):

**grocery-deep:** path_full **0.37**, path_tail 0.20, path_basket 0.11,
cooc **0.11**, cosine 0.11, als 0.05, costs 0.02 each.

**ml-1m:** path_full 0.02, path_tail 0.02, path_basket 0.02, cooc **0.02**,
cost_* 0.10-0.14 each, cosine **0.30**, als **0.27**.

Cross-referencing with the NDCG-impact table:

- On grocery-deep, **path_full has 37% posterior weight but ±0.1%
  NDCG impact**. Cooc has 11% weight and 30% NDCG impact.
- On ML-1M, **cosine has 30% posterior weight but 0% NDCG impact**.
  Cooc has 2% weight and 16% NDCG impact.

This is a signal-scale problem. The prior weights each signal as a
fraction of the total, but cooc's raw *values* are far more
discriminative than path/cosine/als/cost values on these candidates.
Weight × value = contribution; cooc's contribution dominates because
its values are larger, even at small posterior weight.

## What we do NOT conclude

**We do not conclude "drop the other 8 signals."** Two reasons:

1. **The posterior never adapts** because we never feed outcomes back
   (`engine.report_outcomes()` is never called in these runs). The
   Bayesian blend was designed as a warm-regime tool where outcome
   feedback tightens weights around useful signals. We've only
   measured the cold regime.
2. **We haven't measured real session data.** RetailRocket, Instacart,
   Amazon — the datasets kindling was designed for — aren't locally
   cached. Path signals might earn their keep on those and fail on
   synthetic-grocery.

**We do conclude** that on the datasets we have access to, the
pay-off is:

- **cooc**: essential, irreplaceable, worth its compute.
- **path_***: zero measurable NDCG impact. Still worth keeping in v1
  pending real-session validation.
- **item_item_cosine**: zero measurable NDCG impact on either dataset.
  Most-obvious candidate for deletion if nothing changes on real data.
- **als_factor**: zero measurable NDCG impact. HNSW-over-ALS as a
  retriever (queued separately) has more architectural promise than
  als-as-feature-signal.
- **cost_***: zero measurable NDCG impact, but these are meant for the
  negative-signal pathway (explicit `remove`, low-rating) which the
  current eval harness doesn't exercise. Not deletable on this
  evidence.

## What we change in this commit

- `src/kindling/benchmarks/signal_ablation.py` — the harness.
- Reports written to `bench/reports/signal_ablation_*.json`.
- **No code changes to the engine.** Deletion of any signal is blocked
  on two things: (a) an outcome-fed evaluation that exercises the
  warm regime, (b) measurements on RetailRocket/Instacart.

## Action items (queued, in priority order)

1. **Real session data** — once RetailRocket / Instacart / Amazon are
   available, rerun this harness. If path signals still fail to earn
   their keep, drop at least one (basket or tail) per family-ablation
   data.
2. **Outcome-fed eval** — build a harness that feeds reported outcomes
   back between `fit` and `recommend`, so the Bayesian posterior
   actually adapts. Re-measure; signals that adapt upward deserve to
   stay.
3. **Prior rebalancing** — the mismatch between posterior weight and
   NDCG impact means our data-characteristic priors are wrong. In
   particular, the ML-1M prior assigns 2% weight to the signal that
   carries 16% of NDCG and 30% to one that carries 0%. `priors.toml`
   needs a revision once we have enough real-data evidence.
4. **HNSW-over-ALS retriever** — converts the "als isn't helping as a
   feature" finding into a retrieval move. Retrieval augmentation is
   the right use for latent factors on these datasets.

## Why we didn't just prune now

Shipping signal deletions off two synthetic/rating datasets without
session data would be premature. The PRD's novelty claim is the
Bayesian blend on session-rich data — we don't have session-rich data
yet. The honest path is "build the instrumentation, publish the
finding, queue the real test." This ADR is that instrumentation.
