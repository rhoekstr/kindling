# ALS ablation — amazon-beauty

- users evaluated: 200
- train / test: 178,651 / 19,851
- k = 10
- timestamp: 2026-04-26T01:12:17

## Quality

| metric | A: ALS→HDBSCAN | B: SVD→HDBSCAN | C: SVD→HDBSCAN + ALS-boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.0372 | 0.0372 | 0.0367 |
| `mrr` | 0.0384 | 0.0384 | 0.0376 |
| `recall_at_k` | 0.0582 | 0.0582 | 0.0582 |
| `hit_rate` | 0.1000 | 0.1000 | 0.1000 |
| `coverage` | 0.0675 | 0.0675 | 0.0660 |

## Fit timing + persona count

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 66.05 | 86.73 | 87.10 |
| personas_found | 2 | 2 | 2 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.27 | 0.26 | 0.29 |
| `p95_ms` | 0.39 | 0.35 | 0.39 |
| `p99_ms` | 0.55 | 0.51 | 0.53 |

## Reading guide

- **A vs B**: does ALS-quality matter for clustering input?
- **B vs C**: does ALS-as-boost contribute lift?
- **A vs C**: should ALS feed HDBSCAN, or only the boost layer?
