# Real-world validation: H&M (fashion retail purchases)

> Second real production dataset, a different retail domain (fashion) and a
> *warmer* regime than RetailRocket — plus rich, readable product metadata.
> H&M Personalized Fashion Recommendations (Kaggle competition): real
> purchase transactions with timestamps + 25 columns of article metadata.

## The regime

- Full data: **31.8M transactions, ~1.37M customers, ~105k articles**,
  2018-09 → 2020-09. This validation uses a **recent ~3.7-month window**
  (≥2020-06-01): 4.66M transactions, 565k customers, 44k articles.
- **~8 events/user** — warmer than RetailRocket's ~2, so trained MF has more
  per-user signal to work with. A genuinely different point on the warmth
  spectrum.

## Result — vs the standard baselines

Realistic tier — chronological split, full-catalog ranking, k=12 (H&M's
competition metric is MAP@12), sliced by user history length. kindling vs
popularity / item-kNN / implicit ALS / BPR. `bench/validate_hm.py` →
`bench/reports/validate_hm.json`. kindling: wilson-cooc base, channels
trend + user_cf active, 92s CPU fit, no training.

NDCG@12:

| user history | n | **kindling** | item-kNN | ALS | BPR | popularity |
|---|---:|---:|---:|---:|---:|---:|
| 1–4 | 2,402 | **0.0173** | 0.0113 | 0.0090 | 0.0074 | 0.0069 |
| 5–19 | 4,337 | **0.0138** | 0.0079 | 0.0083 | 0.0055 | 0.0056 |
| 20+ | 1,261 | **0.0090** | 0.0061 | 0.0063 | 0.0042 | 0.0038 |
| **all** | 8,000 | **0.0141** | 0.0086 | 0.0082 | 0.0059 | 0.0057 |

**kindling is the strongest model in every bucket** — ~1.6× item-kNN, ~1.7×
ALS, ~2.4× BPR overall. The margin is smaller than RetailRocket's (here the
warmer ~8-event users give the trained models more to factorize), and on the
warm 20+ slice ALS/item-kNN draw close — but kindling still leads everywhere.

## Reading — the two-dataset pattern

| dataset | regime | kindling vs ALS (all users) |
|---|---|---|
| RetailRocket | clickstream, ~2 events/user (extreme cold) | **2.5×** |
| H&M | fashion purchases, ~8 events/user (warmer) | **1.7×** |

The same story on two independent real production datasets: **kindling beats
the industry-standard trained MF — and item-kNN and BPR — at no GPU and no
training loop, margin largest on cold users and narrowing (but still winning)
as users warm.** That warm-convergence is exactly what theory predicts
(factorization needs interactions-per-user), and it's why kindling's edge is
biggest precisely where real catalogs hurt: the cold majority.

## Note on the content / cold-slot path

The article metadata (product type, colour, department, garment group) was
passed to `fit`, but the headline comparison is *warm ranking*, where the
content channel is cold-gated off by design (§4.6) — so it doesn't move these
numbers. Exercising the content cold-slot mechanism (§4.8) on H&M's rich
metadata is a separate cold-*item* recovery probe (surface never-purchased
articles by content similarity); the warm model comparison above is the
"is it meaningful vs real models" answer.
