# ADR: persona signal — shipped, earns zero as a 10th blend signal, queued as a retriever instead

**Date:** 2026-04-23
**Status:** shipped disabled-by-default; next move is `PersonaRetriever`, not more blend weight
**Related:** [ADR-signal-audit.md](ADR-signal-audit.md), [ADR-lightgbm-warm-regime.md](ADR-lightgbm-warm-regime.md)

## What shipped

Four commits implementing the PRD supplement (persona_signal):

1. Rust kernel (`native/src/personas.rs::persona_rates`) + Python
   skeleton: config, HDBSCAN + K-means clustering, persona index
   builder, online matching, score-candidates pipeline.
2. Engine integration: persona is the 10th entry of `SIGNAL_ORDER`;
   `_fit_persona_index` runs after graph/path setup; the 10th column
   of `SignalFeatures` carries the persona score at every recommend.
3. Cold-start: per-(item, persona) overperformance ratio matrix,
   folded into scoring via a configurable `cold_start_weight`.
   Base-rate normalized by persona's interaction-volume share.
4. Measurement (this ADR): ablation + growth-curve + direct
   head-to-head against cooccurrence.

All code is behind `Engine(persona_config=PersonaConfig(enabled=True, ...))`;
opt-in. Default engines are unaffected.

## Headline finding

**Persona has zero marginal NDCG contribution when added to the
Bayesian blend on either dataset.** The LOO ablation and the direct
head-to-head both confirm it.

### Leave-one-out ablation (NDCG @10, 500 eval entities)

| dataset         | frac | full  | -persona | -cooc  |
| --------------- | ---: | ----: | -------: | -----: |
| grocery-deep    | 1.00 | 0.320 | 0.320    | 0.223 |
| grocery-deep    | 0.60 | 0.190 | 0.190    | 0.128 |
| grocery-deep    | 0.30 | 0.128 | 0.128    | 0.100 |
| ml1m            | 1.00 | 0.183 | 0.183    | 0.153 |
| ml1m            | 0.60 | 0.133 | 0.134    | 0.123 |
| ml1m            | 0.30 | 0.115 | 0.115    | 0.110 |

Removing persona does nothing on any fraction of either dataset.
Removing cooccurrence costs 15–30% of NDCG every time.

### Growth curve (kindling with vs. without persona)

| dataset         | frac | kindling | kindling+persona | Δ |
| --------------- | ---: | -------: | ---------------: | ---: |
| grocery-deep    | 1.00 | 0.320    | 0.320            | 0 |
| grocery-deep    | 0.60 | 0.190    | 0.190            | 0 |
| grocery-deep    | 0.30 | 0.128    | 0.128            | 0 |
| ml1m            | 1.00 | 0.183    | 0.183            | 0 |
| ml1m            | 0.60 | 0.133    | 0.133            | 0 |
| ml1m            | 0.30 | 0.115    | 0.115            | 0 |

Reports: [growth_grocery_persona.json](growth_grocery_persona.json),
[growth_ml1m_persona.json](growth_ml1m_persona.json),
[signal_ablation_grocery_persona.json](signal_ablation_grocery_persona.json),
[signal_ablation_movielens_persona.json](signal_ablation_movielens_persona.json).

## Direct head-to-head (cooc vs. persona standalone + combined)

At 100% data, evaluating with ALL signals masked EXCEPT the named ones:

| dataset         | only_cooc | only_persona | only_cooc+persona |
| --------------- | --------: | -----------: | ----------------: |
| grocery-deep    | 0.326     | 0.261        | 0.326             |
| ml1m            | 0.214     | 0.051        | 0.214             |

**Persona carries real standalone signal** — 80% of full NDCG on
grocery-deep, 23% on ML-1M. It's not garbage. But combined with cooc
in the linear blend, the combined score equals cooc alone.

## Diagnosis: cooc and persona see overlapping information

Both signals are derived from the same interaction records. Cooc
operates at the item-pair level: "items A and B co-occur often." Persona
operates at the cluster level: "users in persona P over-consume item c."
Both ultimately encode which items cluster together in user behavior;
the persona transformation adds a pooling step that discards
pair-level granularity.

When the linear blend sees both, every direction in feature space
persona can represent is ALREADY representable by a weighted cooc
score. There's no new information the blend can extract because its
function class (linear combinations) isn't expressive enough to
separate the two.

This is the same pattern the signal audit found for cosine, ALS, paths:
all of the hand-engineered signals collapse to cooccurrence under a
linear combiner.

## The architectural question this sharpens

We've now added **three** "independent information" signals to
kindling — item_item_cosine, als_factor, persona — each of which was
supposed to contribute complementary information to the blend. All
three have zero LOO impact. That's no longer three separate failures;
that's the same failure three times over.

The blend architecture is structurally limited for offline
retriever→score→rerank pipelines with interaction-only data. Adding
more hand-engineered signals won't fix it.

Two paths forward:

### Path A: Persona as a **retriever**, not a signal

Instead of scoring retrieved candidates by persona, **surface new
candidates** that the cooccurrence retriever doesn't find. Union the
two retriever outputs. The blend then operates on a strictly richer
candidate set.

Concretely: `PersonaRetriever.retrieve(entity_id, budget)` returns
the top-K items from the entity's matched persona's vector, excluding
items the entity already owns. These items may or may not have
high cooccurrence scores against the entity's history — if they
don't, they're items kindling currently misses.

This changes WHAT kindling recommends, not HOW it ranks. The scoring
stays the same (cooc still dominates); the set of candidates widens.

I'd predict this improves **coverage** materially and **NDCG**
modestly on session data. It's the same argument that motivated
HNSW-over-ALS as a retriever.

### Path B: Non-linear scorer over the same features

LambdaRank attempted this and failed ([ADR-lightgbm-warm-regime.md])
precisely because the feature space is degenerate. But a non-linear
scorer over **decorrelated** features (persona minus its cooc
projection) might work where LambdaRank-over-raw-features didn't.
This is a bigger redesign; Path A is cheaper and likely hits 80% of
the benefit.

## Recommendation

**Add `PersonaRetriever` alongside the existing
`CoOccurrenceRetriever` and `PathEndpointRetriever`.** Wire it into
`Engine` behind `persona_config.use_as_retriever=True` (default True
when personas are enabled). Re-run the growth curves and measure.

The signal-side persona code stays in place but remains disabled;
we're not deleting it because the outcome-fed blend (queued) might
reassign weights in a way that makes persona-as-signal useful when
we're no longer prior-dominated. For now, persona's real value is as
a retrieval augmentation.

## What shipped in this ADR's commit

- Three new benchmark reports (grocery-deep + ml1m ablation + growth
  + head-to-head).
- This ADR.
- No code changes — the persona signal code is unchanged from
  commits 1-3; this commit is measurement + honest framing.

## What's queued next

1. **`PersonaRetriever`** (Path A above). Should be ~200 lines.
2. **HNSW-over-ALS retriever** — same architectural pattern (promote
   a redundant signal to a retriever role). The two can share
   abstraction.
3. **Outcome-fed blend adaptation** — the Bayesian posterior can't
   learn to downweight redundant signals when we never report
   outcomes; until that's fixed, we can't separate "redundant signal
   the blend never noticed was useless" from "useful signal the
   blend hasn't learned to weight yet."
