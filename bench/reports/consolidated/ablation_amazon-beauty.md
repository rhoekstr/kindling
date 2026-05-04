# ALS ablation — amazon-beauty

- users evaluated: 500
- train / test: 178,651 / 19,851
- k = 10
- timestamp: 2026-05-03T21:56:10

## Quality

| metric | A: ALS→HDBSCAN | B: SVD→HDBSCAN | C: SVD→HDBSCAN + ALS-boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.0290 | 0.0290 | 0.0296 |
| `mrr` | 0.0407 | 0.0407 | 0.0417 |
| `recall_at_k` | 0.0327 | 0.0327 | 0.0332 |
| `hit_rate` | 0.0860 | 0.0860 | 0.0880 |
| `coverage` | 0.1248 | 0.1248 | 0.1212 |

## Fit timing + persona count

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 69.34 | 91.23 | 91.35 |
| personas_found | 2 | 2 | 2 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.23 | 0.22 | 0.25 |
| `p95_ms` | 0.32 | 0.31 | 0.33 |
| `p99_ms` | 0.39 | 0.39 | 0.42 |

## Reading guide

- **A vs B**: does ALS-quality matter for clustering input?
- **B vs C**: does ALS-as-boost contribute lift?
- **A vs C**: should ALS feed HDBSCAN, or only the boost layer?
