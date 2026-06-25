# Real-world validation: RetailRocket (production e-commerce churn)

> The deepest real-world test so far: not an academic split but a raw
> e-commerce clickstream (RetailRocket: view / add-to-cart / transaction),
> with the cold-start-under-churn regime that 5-core academic preprocessing
> deletes entirely. Does kindling add real value here?

## The regime (why it's the right test)

- **1.27M users, 224k items, 2.48M events** — but **~2 events/user**.
- Of users with a held-out item, **86% have ≤4 training interactions, median = 1.**
- Timestamps + add-to-cart-without-purchase soft-negatives + churn.

This is the genuinely data-starved cold regime. Per the warming benchmark
(REFERENCE §3.5) the **popularity prior is the bar to beat** here — a trained
MF (ALS) can barely serve 1-event users at all. So the meaningful question is
*does personalization beat popularity, and for which users?*

## Result

Realistic tier — no k-core, chronological 90/10 split, full-catalog ranking,
k=20, sliced by user history length. `bench/validate_retailrocket.py` →
`bench/reports/validate_retailrocket.json`. kindling: wilson-cooc base, **all
three channels active** (trend + transitions + user_cf), 100s CPU fit, no
training.

| user history | n | kindling NDCG@20 | popularity | kindling Recall@20 | popularity |
|---|---:|---:|---:|---:|---:|
| **1–4** (86% of users) | 5,836 | **0.0274** | 0.0037 | **0.0475** | 0.0084 |
| 5–19 | 786 | **0.0233** | 0.0024 | **0.0456** | 0.0057 |
| 20+ | 159 | **0.0276** | 0.0051 | **0.0293** | 0.0039 |
| **all** | 6,781 | **0.0269** | 0.0036 | **0.0469** | 0.0080 |

**kindling beats popularity in every bucket — including the coldest (1–4
interactions): 7.4× on NDCG, 5.7× on Recall.** Overall ~7.5× popularity.

## Reading

- **The value-add holds on real production churn data.** This is the steam
  realistic-tier finding (wins cold *users* on a cold-heavy catalog, beating
  even the popularity prior) **reproduced on a genuine e-commerce log** — the
  regime kindling was designed for, confirmed outside the benchmark suite.
- **Why kindling wins here** when the warming benchmark said "popularity wins
  the data-starved regime": RetailRocket is cold-*users* on a *rich catalog*,
  not globally data-starved. 2.48M events give dense co-occurrence structure;
  kindling personalizes from even a single seed item by leaning on that
  structure (+ trend/transition/user-CF, all active on this timestamped,
  sparse-history data). Popularity ignores the seed entirely.
- **Honest scope.** This predicts the next *viewed* item (views dominate the
  stream); absolute NDCG is low (full-ranking over 224k items, ~1-event
  users). Popularity is a deliberately weak, but *relevant*, baseline — it's
  the cold-start bar the warming study established, and trained MF can't serve
  these users at all. A personalized item-kNN baseline would sit closer to
  kindling (kindling *is* a normalized cooc-kNN with channels); the headline
  is that **personalization adds ~7× value over the non-personalized prior on
  real cold-start traffic.**

## Cost

100s CPU fit on 2.48M events / 224k items, no GPU, no training loop — and it
fit on a 24 GB box (the memory-cap hardening from the B1 work held; 224k
items, base-only, no extension).
