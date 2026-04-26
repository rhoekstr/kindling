# ALS ablation — movielens-1m

- users evaluated: 200
- train / test: 900,188 / 100,021
- k = 10
- timestamp: 2026-04-26T01:16:43

## Quality

| metric | A: ALS→HDBSCAN | B: SVD→HDBSCAN | C: SVD→HDBSCAN + ALS-boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.2567 | 0.2602 | 0.2602 |
| `mrr` | 0.4256 | 0.4222 | 0.4222 |
| `recall_at_k` | 0.0439 | 0.0430 | 0.0430 |
| `hit_rate` | 0.6700 | 0.6700 | 0.6700 |
| `coverage` | 0.0454 | 0.0408 | 0.0408 |

## Fit timing + persona count

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 14.85 | 14.87 | 15.20 |
| personas_found | 9 | 3 | 3 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.74 | 0.75 | 0.77 |
| `p95_ms` | 3.04 | 3.10 | 3.10 |
| `p99_ms` | 4.58 | 4.44 | 4.50 |

## Reading guide

- **A vs B**: does ALS-quality matter for clustering input?
- **B vs C**: does ALS-as-boost contribute lift?
- **A vs C**: should ALS feed HDBSCAN, or only the boost layer?
