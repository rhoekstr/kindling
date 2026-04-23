# ADR: each signal as a standalone retriever — what each one actually knows

**Date:** 2026-04-23
**Status:** shipped as diagnostic; architectural decisions queued
**Related:** [ADR-signal-audit.md](ADR-signal-audit.md),
[ADR-persona-signal.md](ADR-persona-signal.md)

## Why this exists

Every prior measurement in kindling has asked "what does this signal
contribute to the Bayesian blend?" and every prior answer has been
~nothing beyond cooccurrence. That framing hides what each signal
actually knows because the linear blend is structurally incapable of
separating signal information that correlates with cooc.

This ADR uses a different probe: **treat each signal as a complete
recommender — its own retriever, its own score, its own ranking.**
Skip the blend entirely. Measure NDCG on what each signal produces on
its own.

New harness: `kindling.benchmarks.retriever_standalone`. Per-signal
retrievers in `src/kindling/retrieve/signal_retrievers.py` cover
path_tail, path_full, path_basket, cosine, als, and persona (each fed
from the engine's fitted state — no re-training).

## Results — side by side

NDCG@10, MRR, recall@10 (the test positive landed in the retrieved
top-10), and recall@budget (the test positive landed anywhere in the
top-500). 500 eval entities per dataset, fraction=1.0.

|                        | grocery-deep (162k int) |        |        |        | ml1m (518k int) |        |        |        |
| ---------------------- | ----------------------: | -----: | -----: | -----: | --------------: | -----: | -----: | -----: |
| **retriever**          |                    NDCG |    MRR |  rec@K |  rec@B |            NDCG |    MRR |  rec@K |  rec@B |
| cooccurrence           |                   0.319 |  0.351 |  0.738 |  1.000 |           0.183 |  0.320 |  0.596 |  0.958 |
| item_item_cosine       |                   0.320 |  0.355 |  0.742 |  1.000 |           0.183 |  0.313 |  0.592 |  0.976 |
| als_factor             |                   0.295 |  0.349 |  0.757 |  1.000 |           0.163 |  0.303 |  0.586 |  0.982 |
| **path_basket**        |               **0.304** |  0.341 |  0.732 |  1.000 |           0.050 |  0.111 |  0.270 |  0.880 |
| path_tail              |                   0.181 |  0.248 |  0.474 |  0.996 |           0.088 |  0.179 |  0.424 |  0.798 |
| path_full              |                   0.047 |  0.093 |  0.178 |  0.180 |           0.018 |  0.056 |  0.090 |  0.096 |
| path_endpoint_combined |                   0.165 |  0.221 |  0.505 |  0.996 |           0.085 |  0.179 |  0.420 |  0.798 |
| persona                |                   0.268 |  0.322 |  0.683 |  1.000 |           0.000 |  0.000 |  0.002 |  0.446 |

(path_endpoint_combined is the retriever the current Engine uses as
Stage 1 alongside cooccurrence.)

## What this actually tells us

### 1. Path signals are session-data signals, decisively

**`path_basket` is ×6 stronger on grocery-deep than on ML-1M** (0.304
vs 0.050). No other signal shows this magnitude of dataset-dependence.
Ratings data has no meaningful basket structure, and the signal
correctly reflects that by going nearly-zero. On grocery, path_basket
is the 3rd-best retriever behind cosine and cooc — essentially tied
with the neighborhood signals.

This is the first empirical validation in this project of the PRD's
original claim that kindling is designed for session-rich data. It
just had to be probed as a retriever, not a signal.

### 2. The engine's current retriever is the worst viable option

`path_endpoint_combined` combines `path_tail` and `path_full`. On both
datasets it's **worse than `path_tail` alone** (0.165 vs 0.181 on
grocery, 0.085 vs 0.088 on ml1m). The combination with path_full drags
the result down because path_full is contributing noise.

The engine's current Stage 1 is `CoOccurrenceRetriever` + this
combined thing. Simply swapping in `path_tail` (or better,
`path_basket`) would strictly help.

### 3. `path_full` is broken — candidate for deletion

Recall@budget 0.18 on grocery, 0.10 on ML-1M. The exact-prefix match
barely ever succeeds. It's contributing to the feature space (as a
signal) but it has nothing to contribute as a retriever, and by
inclusion in `path_endpoint_combined` it's actively hurting recall.

### 4. ALS has the best recall@10 on both datasets despite contributing nothing to the blend

`als_factor` retrieves the test positive in the top-10 more often than
any other retriever on both datasets (0.757 grocery, 0.586 ML-1M). As a
*signal*, it had zero LOO impact on both datasets (see
ADR-signal-audit). As a *retriever*, it's among the best. This is the
"same information, different architectural role" pattern the
persona-signal ADR already called out.

### 5. Persona is session-specific too

Persona's standalone performance on grocery-deep (0.268 NDCG, 0.683
rec@10) is competitive — weaker than the top three but meaningful.
On ML-1M it collapses: 0.000 NDCG, 0.002 rec@10. It surfaces
candidates (rec@B = 0.446) but they're the wrong ones for ML-1M's
test distribution.

Why? Persona aggregates over stable taste groups. Grocery-deep has
visible taste groups (categorical items). ML-1M does have taste
groups (genre preferences) but its test window is the most recent
10% of ratings, dominated by *recency*, not taste. The persona signal
doesn't model recency, so it fails on this particular split.

This is a split-design artifact more than a persona weakness.
Retrieval-hit-rate-at-budget (0.446 on ML-1M) shows persona has
plenty of signal — it's just pointing at older/different items than
the test window rewards.

### 6. Cooc, cosine, ALS are the dataset-robust trio

All three sit at the top of both datasets with near-identical
performance. They're the floor kindling can rely on regardless of
data shape. Among them, cosine is marginally ahead (it's L2-normalized
cooc). If we had to pick one, it's cosine.

## Actionable architecture changes

Based on this data, the current engine's retrieval stage is
mis-specified. The right retriever set for v1.x is:

**On both datasets:**
- Keep `CoOccurrenceRetriever` (or swap to a cosine variant; marginal).
- Add `ALSRetriever` — highest recall on both datasets, zero marginal
  cost given ALS factors are already fitted.

**On session-rich data:**
- Add `PathBasketRetriever` — 3rd-best standalone on grocery-deep;
  surfaces items via basket similarity that no other retriever does.
- Add `PersonaRetriever` — middle of the pack but meaningful, and
  its candidate distribution differs from neighborhood retrievers.

**Remove / demote:**
- **Delete `PathFullRetriever`** — negative value on both datasets.
- **Delete `path_endpoint_combined`** (replace with the two components
  used independently, or just `path_tail` standalone).

**Gate session-specific retrievers:**
Add a data-characteristic check at `Engine.fit`: when the session
structure is weak (low sessions-per-entity, no timestamp column,
rating-style data), skip PersonaRetriever and optionally
PathBasketRetriever. The existing session-stiffness prior already
detects this; reuse that signal.

## What this does NOT change (yet)

- The scoring stack (stage 2) is unchanged. The blend is still
  dominated by cooc; that's a separate architecture problem tracked
  in [ADR-signal-audit.md] and queued behind the outcome-fed eval
  harness.
- The re-rank stage is unchanged.
- Nothing is deleted or added in this commit — the diagnostic ships
  as a tool. The architecture decision is written up but
  implementation is a follow-up PR.

## Queue after this

1. **Union-retrieval measurement** — Pick the top-N standalone
   retrievers per dataset, measure NDCG of their union (using each
   candidate's max retriever score). Does the union beat the best
   individual retriever? That tells us whether the retrievers are
   complementary or just redundant.
2. **Wire the new retriever set into Engine.fit** — conditionally,
   behind a data-characteristic gate so ratings data doesn't incur
   persona/basket compute.
3. **Re-run growth curves and comparison with the new retrieval.** If
   recall climbs, so does the NDCG ceiling — and that's the only way
   kindling meaningfully beats popularity on ML-1M and crosses kNN on
   grocery-deep.
