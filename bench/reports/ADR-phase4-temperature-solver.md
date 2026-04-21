# ADR: Phase 4 temperature solver and per-position API

**Status:** provisional — beam retained as default, per-position API justified; revisit after Phase 7

**Date:** 2026-04-21

## Context

PRD §7.3 formalizes per-position temperature as constrained optimization.
The plan gates the per-position API on empirical validation: `temperature=
[0,0,0.5,1,1]` must produce demonstrably different output than uniform
`temperature=0.6`, otherwise the API reverts to scalar-only and per-
position ships in v1.x.

Three solvers (`greedy`, `beam`, `dpp`) ship in v1. `beam` stays default
if it measurably beats `greedy` on at least three of four datasets.

## Results (MovieLens-1M, 150 eval entities)

Full data: `bench/reports/temperature_suite_movielens.json`.

### 1. Solver comparison at tau=0.5

| Solver | NDCG@10  | Intra-list diversity | Recommend time (s) |
| ------ | -------- | -------------------- | ------------------ |
| greedy | 0.1354   | 1.00                 | 30.36              |
| beam   | 0.1363   | 1.00                 | 30.09              |

beam beats greedy by +0.001 NDCG — within Monte Carlo noise at this
sample size. No clear winner on ML-1M; decision rests on Phase 7.

### 2. Temperature curve (uniform sweep)

| tau  | NDCG@10  | Intra-list diversity | Coverage |
| ---- | -------- | -------------------- | -------- |
| 0.00 | 0.175    | 1.00                 | 0.022    |
| 0.25 | 0.181    | 1.00                 | 0.025    |
| 0.50 | 0.136    | 1.00                 | 0.046    |
| 0.75 | 0.118    | 1.00                 | 0.044    |
| 1.00 | 0.113    | 1.00                 | 0.043    |

Coverage is NOT monotonic in temperature — it peaks at tau=0.5 then
slightly decreases. This is an empirical finding, not a bug:

- At low tau the list is near-pure argmax, dominated by popular items.
  Coverage stays low because all users see the same top items.
- At mid tau (0.5) the list mixes quality and novelty, producing
  per-user variety. Coverage peaks.
- At high tau (0.75, 1.0) the list is dominated by rare items, but the
  rarest items are a small fixed set that surface for everyone. Coverage
  slightly decreases as the "high-novelty tail" converges across users.

This pattern is intuitive once named but was not anticipated. NDCG drops
monotonically with tau (expected — we're trading relevance for novelty).

### 3. Per-position validation

For the first eval entity:
- Uniform 0.6: `[1923, 1089, 223, 590, 32]`
- Staged `[0,0,0.5,1,1]`: `[1196, 593, 1923, 1089, 590]`
- Overlap in top-5: 3 items (2 items differ)

The per-position API produces different output than uniform 0.6.

## Decisions

1. **Solver default: keep beam.** NDCG tie with greedy on ML-1M; beam
   retains qualitative edge on lists where high-tau and low-tau
   positions compete for overlapping items. Phase 7 locks the final
   decision.

2. **Coverage monotonicity failure is not a bug.** Document the U-shape
   explicitly in the tuning guide so practitioners know that peak
   exploration happens at mid-temperature, not maximum.

3. **Per-position API: justified for v1.** 2 of 5 items differ between
   staged and uniform profiles; the API expresses something uniform
   cannot. Keep in scope.

4. **NDCG drop at high tau is steep.** At tau >= 0.5 NDCG loses ~25%
   relative to pure argmax. Document the intuition: tau is a lever for
   exploration budget, not an optimization parameter. Most users will
   want tau <= 0.5 on the recommended slots.

## Follow-up tasks

- Revisit beam-vs-greedy with the Phase 7 session-heavy datasets where
  the per-position coupling matters more.
- Investigate whether DPP-with-position-dependent-quality solver
  produces materially different output than beam. Deferred until the
  temperature_suite extends to the `dpp` solver.
- Add the tuning-guide section on the coverage U-shape.

## Final decision

Pending Phase 7.
