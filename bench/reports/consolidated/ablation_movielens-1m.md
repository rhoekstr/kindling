# ALS ablation — movielens-1m

- users evaluated: 500
- train / test: 900,188 / 100,021
- k = 10
- timestamp: 2026-05-03T21:55:14

## Quality

| metric | A: ALS→HDBSCAN | B: SVD→HDBSCAN | C: SVD→HDBSCAN + ALS-boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.2521 | 0.2529 | 0.2529 |
| `mrr` | 0.4202 | 0.4180 | 0.4180 |
| `recall_at_k` | 0.0442 | 0.0442 | 0.0442 |
| `hit_rate` | 0.6700 | 0.6820 | 0.6820 |
| `coverage` | 0.0506 | 0.0435 | 0.0435 |

## Fit timing + persona count

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 16.21 | 16.13 | 16.41 |
| personas_found | 9 | 3 | 3 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.71 | 0.73 | 0.78 |
| `p95_ms` | 2.77 | 2.86 | 2.93 |
| `p99_ms` | 4.16 | 4.20 | 4.27 |

## Reading guide

- **A vs B**: does ALS-quality matter for clustering input?
- **B vs C**: does ALS-as-boost contribute lift?
- **A vs C**: should ALS feed HDBSCAN, or only the boost layer?
