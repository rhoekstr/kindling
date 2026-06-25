# Real-world validation — summary

Does kindling's value-add hold **beyond the four datasets it was tuned on**
(ml1m, amazon-beauty, amazon-book, steam)? Three independent external tests —
one academic-with-published-baselines, two real production datasets — say
**yes, decisively.** Details per dataset:
[yelp2018](VALIDATION-yelp2018.md) · [RetailRocket](VALIDATION-retailrocket.md)
· [H&M](VALIDATION-hm.md).

## 1. Generalization (yelp2018, vs published GNNs)

The exact LightGCN/NGCF academic split, a new domain (local business). In its
*weakest* config (no timestamps → channels off, cooc base only):

| | NDCG@20 |
|---|---:|
| BPR-MF | 0.0445 |
| Mult-VAE | 0.0450 |
| **kindling** | **0.0459** |
| NGCF | 0.0477 |
| LightGCN | 0.0530 |

Beats the classic trained baselines, **87% of LightGCN**, at ~400× less
compute and zero training. Competitive with GNNs, honestly not SOTA.

## 2. Real production data, vs the standard trained models

Two real retail logs, realistic tier (no k-core, chronological, full-ranking,
sliced by user history), kindling vs popularity / item-kNN / implicit ALS /
BPR. **kindling is the strongest model in every history bucket on both.**

NDCG, all users:

| dataset | regime | kindling | item-kNN | ALS | BPR | popularity |
|---|---|---:|---:|---:|---:|---:|
| **RetailRocket** | clickstream, ~2 events/user | **0.0261** | 0.0134 | 0.0105 | 0.0057 | 0.0037 |
| **H&M** | fashion purchases, ~8 events/user | **0.0141** | 0.0086 | 0.0082 | 0.0059 | 0.0057 |

- **RetailRocket** (extreme cold): kindling **2.5× ALS**, 2× item-kNN, 7× pop.
- **H&M** (warmer): kindling **1.7× ALS**, 1.6× item-kNN.

## The pattern (and why it's the right one)

kindling's edge is **largest on cold users and narrows as users warm** — on
both datasets, and across the warmth slices within each. That's exactly what
factorization theory predicts: ALS/BPR need interactions-per-user to learn
good factors, so they're weakest precisely where real catalogs hurt (the cold
majority), while kindling leans on the *catalog's* dense co-occurrence
structure + its auto-gated channels and personalizes from a single seed.

Real recommender traffic is cold-heavy, so kindling's advantage shows up
where it matters — and it does it at **no GPU, no training loop** (CPU fits of
90–120s vs the trained models' iterative training).

## Honest scope

- These are **standard, untuned** baseline configs (factors=64, the
  warming-benchmark settings). A heavily-tuned ALS could close some gap; the
  point is the no-training shallow stack is **stronger out-of-the-box** on
  real cold-start traffic.
- Absolute NDCG is low (full-ranking over 40k–224k items, cold users — the
  task is genuinely hard); the cross-method ranking is the signal.
- yelp2018 runs kindling's floor (no timestamps); the production datasets
  exercise the channels (trend / transitions / user-CF light up by regime).
- The content / cold-*item* path (H&M's rich metadata, §4.8) is a separate
  cold-item-recovery probe, not part of these warm-ranking comparisons.

## Reproduce

```
python bench/validate_yelp2018.py              # vs published GNN baselines
python bench/validate_retailrocket_baselines.py   # vs ALS/BPR/item-kNN
python bench/validate_hm.py                    # vs ALS/BPR/item-kNN (needs Kaggle H&M)
```
(The production comparisons need the `baselines` extra: `pip install -e ".[baselines]"`.)
