# ADR: LightGBM LambdaRank as warm-regime scorer — shipped OFF by default

**Date:** 2026-04-21
**Status:** shipped disabled; retraining fix queued
**Related:** [ADR-signals-and-growth.md](ADR-signals-and-growth.md)
item "LightGBM missing"

## What shipped

- `LightGBMRanker.score_features(features)` replaces the old
  NotImplementedError stub; takes the (n_candidates, k_signals)
  feature matrix and returns raw LambdaRank output.
- `Engine.fit` optionally trains the ranker (`use_ranker=True`) using
  per-entity last-item holdout + `ranker_negatives_per_positive=99`
  uniformly-sampled negatives.
- `Engine.recommend` routes through the ranker when fitted; Bayesian
  posterior mean remains for explanation + credible-interval surfaces.
- Persistence round-trips the fitted ranker.
- New `tests/unit/test_ranker_training.py` locks in: trains when
  lightgbm is installed, recommends with ranker scores, opt-out works,
  skips on too-small datasets.

## Why the default is `use_ranker=False`

Measured on the growth curve with the ranker active:

### ML-1M (ratings)

| frac | kindling (ranker OFF) | kindling (ranker ON) | popularity | kNN |
| ---- | --------------------: | -------------------: | ---------: | ----: |
| 0.10 | 0.116 | 0.116 | 0.116 | 0.000 |
| 0.30 | 0.115 | 0.110 | 0.116 | 0.016 |
| 0.60 | 0.133 | 0.111 | 0.133 | 0.071 |
| 1.00 | **0.183** | **0.112** | 0.172 | 0.196 |

### Grocery-deep (sessions)

| frac | kindling (ranker OFF) | kindling (ranker ON) | popularity | kNN |
| ---- | --------------------: | -------------------: | ---------: | ----: |
| 0.10 | 0.076 | 0.078 | 0.041 | 0.078 |
| 0.30 | 0.128 | 0.104 | 0.039 | 0.121 |
| 0.60 | 0.190 | 0.160 | 0.048 | 0.188 |
| 1.00 | **0.320** | **0.247** | 0.060 | 0.320 |

The ranker **actively destroys accuracy** at every fraction ≥30% on
both datasets. At full ML-1M, NDCG collapsed 0.183 → 0.112 (−39%). On
full grocery-deep, 0.320 → 0.247 (−22%).

Fit time also ballooned — ML-1M full went from 123s (no ranker) to 419s
(with ranker), because we build features for up to 2000 entities ×
100 candidates = 200k feature rows every fit.

## Root cause

**The training distribution doesn't match the inference distribution.**

- **Training:** for each entity, positive = their last item, negatives
  = 99 items drawn uniformly at random from the catalog. Almost all
  random items have a cooccurrence score of zero or near-zero with
  the entity's history. Positives have non-trivial cooc/cosine/ALS
  scores by construction.
- **LambdaRank learns:** the trivially-discriminating rule
  "`cooccurrence > ε → positive`" (or an equivalent tree over cosine,
  ALS). This rule is nearly perfect on the training set.
- **Inference:** candidates come from the retriever, which surfaces
  the top ~500 items *by cooccurrence already*. Almost every candidate
  scored by the ranker has `cooccurrence > ε` — the feature the ranker
  is trivially splitting on is useless at inference time because every
  candidate already has it.

The Bayesian posterior mean isn't trying to learn a hard classifier —
it's interpolating between signals with known relative weights. So it
doesn't collapse the way LambdaRank does on this input.

## The fix (queued)

Replace the training generator with **retrieved-candidate negatives**:

1. For each training entity, run the engine's retrieve step against
   the entity's interactions *with the last item held out*.
2. Take the top-K retrieved candidates. Positive: if the held-out
   last item is in that set, its features. Negatives: the other
   retrieved candidates (they're the items the retriever would propose
   at inference time).
3. Train LambdaRank to rank the positive above the other retrieved
   candidates — that's the inference distribution.

This requires either a re-fit on hold-one-out data or accepting the
leakage of fitting signals on all data and using them to generate the
training features anyway. Either is more expensive than the current
approach; neither is more than a day of work.

## Smaller fixes we could bundle with the retraining

- **Much larger training set.** 2000 entities is too small for a 200-
  tree LambdaRank. With retrieved-candidate negatives + 10k+ entities
  the model should start generalizing.
- **Stratify by entity activity.** Right now random-sampled entities
  include users with 2 interactions (where "last item" is nearly
  random) and users with 500 (where "last item" has meaning). Filter
  to entities with ≥10 interactions.
- **Regularize harder.** Current `num_leaves=31, n_estimators=200` is
  aggressive for 200k rows. Halving n_estimators and bumping
  min_child_samples would reduce overfitting.
- **Use the BayesianBlend mean as a feature.** The blend is already a
  good scorer; feeding it as a 10th feature lets LambdaRank learn
  "stick with the blend unless feature X is extreme."

## Verdict

Shipping the code with `use_ranker=False` default so users can opt in,
but the cost (massive NDCG regression) means it's not production-ready
until the training generator is fixed. This is documented, tested, and
persistence works — what's missing is the inference-matching training
distribution.

---

## Update 2026-04-23: training fix applied, still doesn't help

Three variants measured after the initial ADR:

| Variant | ML-1M @1.0 | grocery-deep @1.0 |
| ------- | ---------: | ----------------: |
| baseline (no ranker) | **0.183** | **0.320** |
| v1 random negatives | 0.112 | 0.247 |
| v2 retrieved-candidate negatives, positive always included | 0.053 | 0.252 |
| v3 retrieval-hit-only filter + blend-mean as 10th feature | 0.101 | 0.215 |

None of the training-distribution fixes recovered baseline NDCG.
Reports: [growth_grocery_ranker_v3.json](growth_grocery_ranker_v3.json),
[growth_ml1m_ranker_v3.json](growth_ml1m_ranker_v3.json).

### Why the ranker can't help right now

The signal-audit ADR showed `only_cooc` reproduces the full blend on
both datasets — meaning the other 8 signals carry **no independent
information** beyond what cooccurrence already encodes. LambdaRank can
only extract what's in its feature space; if that space is effectively
1-dimensional (cooc), a linear blend is already optimal and a tree
ensemble just adds overfitting noise.

The training-distribution mismatch was real — fixing it made grocery
slightly less bad (0.247 → 0.252). But it's a second-order problem on
top of a first-order "features are redundant" problem. You cannot
ranker your way out of a feature space that doesn't carry signal.

### What this teaches us

The architecture decision isn't "better LambdaRank training" — it's
**more independent signals**. Three queued items each add a genuinely
new information source:

1. **HNSW over ALS factors as a retriever.** Produces candidates the
   cooccurrence graph doesn't surface. Changes the input distribution
   the blend/ranker scores, rather than adding another redundant
   score-layer on the same candidates.
2. **Persona signal** (per `kindling_PRD_supplement_persona_signal.md`).
   Group-level taste matching, cold-start for new items. Captures
   structure the cooc/path signals can't. Proposed HDBSCAN-based
   clustering + TF-IDF matching.
3. **Outcome-fed blend adaptation.** The Bayesian blend is currently
   prior-dominated because the harness never calls
   `report_outcomes()`. With real outcome feedback the blend's
   weights would actually adapt to signal quality. The signal-audit
   ADR shows that right now the posterior weights miscorrelate with
   signal value (cosine 30% weight, 0% NDCG impact on ML-1M).

Once any of (1), (2), (3) lands and the ablation shows independent
information in the feature space, LambdaRank gets a chance to earn
its compute. Until then, it's a deadweight option kept off by default.

### Code state after this update

- Training generator upgraded to retrieved-candidate negatives with
  retrieval-hit filtering (`src/kindling/engine.py::_fit_ranker`).
- Blend posterior mean included as a 10th feature so LambdaRank has a
  floor to build on.
- Regularization tightened: `num_leaves=15, n_estimators=150`.
- None of the above made LambdaRank beat the linear blend given the
  current feature space.
- `use_ranker=False` remains the default. The code is preserved for
  re-evaluation once independent signals arrive.
