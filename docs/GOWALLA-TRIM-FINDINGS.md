# Gowalla super-consumer trimming (Stage 6)

**Hypothesis (tested):** Gowalla's NDCG peaks before full data and dips at 100%;
it has the highest interaction concentration of the repeat datasets (gini 0.70,
top-1% share 18%); maybe super-consumers (huge check-in counts) inject spurious
co-occurrence edges that pollute the cooc base, and per-user *trimming* would fix
the dip.

**Result: refuted.** Capping each user's training interactions monotonically
*hurts* — full data is best (`bench/gowalla_trim.py`, exclude-seen eval, repeat
off):

| per-user cap | train rows | NDCG@10 |
|---|---:|---:|
| none (full) | 5.76M | **0.0263** |
| 200 | 4.45M | 0.0258 |
| 100 | 3.58M | 0.0207 |
| 50 | 2.61M | 0.0198 |
| 25 | 1.72M | 0.0170 |

Gowalla profile: 107k users, 1.21M items, median 22 / p99 547 / max 1957
interactions per user.

## Reading

- **No trimming mechanism to implement.** Super-consumers' check-ins are signal,
  not noise — removing any of them degrades the base. The concentration metric
  (gini/top-1%) does not indicate harmful pollution here.
- The "dip past 75%" is therefore **not** base pollution. Combined with the
  Stage 5 fair comparison (on Gowalla the personal-frequency "revisit" baseline
  beats kindling's repeat module 0.248 vs 0.147), Gowalla's behavior is a
  **repeat / revisit-frequency** phenomenon (37% of its interactions are repeats),
  not a super-consumer-concentration one.
- **Right lever:** the same as Stage 5 — a frequency-aware repeat/revisit score,
  not trimming. The earlier intuition that the dip implied "trim the super-hot
  set" does not hold for Gowalla; the experiment closes that direction.

**Net:** trimming direction closed (negative result). The Gowalla opportunity is
folded into the repeat-module frequency fix (docs/REPEAT-AWARE-FINDINGS.md).
