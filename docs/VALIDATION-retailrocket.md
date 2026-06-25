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

## Result — vs the standard baselines

Realistic tier — no k-core, chronological 90/10 split, full-catalog ranking,
k=20, sliced by user history length. kindling vs popularity, item-item kNN
(cosine), implicit **ALS** (the industry-standard trained MF), and **BPR** —
the same baseline set as the warming benchmark (REFERENCE §3.5).
`bench/validate_retailrocket_baselines.py` →
`bench/reports/validate_retailrocket_baselines.json`. kindling: wilson-cooc
base, **all three channels active** (trend + transitions + user_cf), 99s CPU
fit, no training (vs ALS 72s + BPR 44s of iterative training).

NDCG@20:

| user history | n | **kindling** | item-kNN | ALS | BPR | popularity |
|---|---:|---:|---:|---:|---:|---:|
| **1–4** (86% of users) | 6,888 | **0.0263** | 0.0139 | 0.0107 | 0.0049 | 0.0039 |
| 5–19 | 916 | **0.0261** | 0.0112 | 0.0079 | 0.0104 | 0.0027 |
| 20+ | 196 | **0.0196** | 0.0054 | 0.0156 | 0.0107 | 0.0036 |
| **all** | 8,000 | **0.0261** | 0.0134 | 0.0105 | 0.0057 | 0.0037 |

**kindling wins every bucket against every baseline** — overall ~2× item-kNN,
**2.5× ALS**, 4.6× BPR, 7× popularity. The margin is largest on the cold users
that dominate this data (1–4 interactions, 86% of users), where the trained
models (ALS/BPR) have too little per-user signal to factorize. On the warmest
sliver (20+, n=196) ALS narrows the gap (trained MF improves with data) but
kindling still leads. *(A popularity-only slice, with Recall too, is in
`bench/validate_retailrocket.py` / `validate_retailrocket.json`.)*

## Reading

- **The value-add holds on real production churn data — against trained
  models, not just popularity.** kindling beats the industry-standard trained
  MF (ALS) by **2.5×** overall and item-kNN by **2×**, with the largest gap on
  the cold users that make up 86% of the data. This is the steam
  realistic-tier finding ("beats ALS everywhere, especially cold users")
  **reproduced on a genuine e-commerce log** — the regime kindling was
  designed for, confirmed outside the benchmark suite.
- **Why the trained models lose here.** RetailRocket is cold-*users* on a
  *rich catalog*. ALS/BPR factorize a user×item matrix — with ~1 interaction
  per user there's almost nothing to factorize, so they regress toward weak
  per-user factors. kindling instead leans on the *catalog's* dense
  co-occurrence structure (2.48M events) + trend/transition/user-CF channels,
  so it personalizes from even a single seed item. ALS only closes the gap on
  the rare warm users (20+), exactly where factorization has enough signal.
- **Honest scope.** This predicts the next *viewed* item (views dominate the
  stream); absolute NDCG is low (full-ranking over 224k items, mostly
  ~1-event users — the task is genuinely hard). The ranking across methods is
  the signal, and it's unambiguous: the no-training shallow stack is the
  strongest model on this real cold-start traffic, at no GPU and no training
  loop.

## Cost

100s CPU fit on 2.48M events / 224k items, no GPU, no training loop — and it
fit on a 24 GB box (the memory-cap hardening from the B1 work held; 224k
items, base-only, no extension).
