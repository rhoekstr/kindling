# Graph-MF ablation — amazon-beauty

- users evaluated: 500
- train / test: 178,651 / 19,851
- k = 10
- timestamp: 2026-05-08T17:34:04

## Quality

| metric | A: cooc base | B: graph_mf base | C: graph_mf boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.0290 | 0.0253 | 0.0292 |
| `mrr` | 0.0407 | 0.0370 | 0.0411 |
| `recall_at_k` | 0.0327 | 0.0304 | 0.0331 |
| `hit_rate` | 0.0860 | 0.0900 | 0.0880 |
| `coverage` | 0.1248 | 0.0807 | 0.1177 |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 76.42 | 73.43 | 73.50 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.22 | 0.11 | 0.25 |
| `p95_ms` | 0.32 | 0.19 | 0.34 |
| `p99_ms` | 0.42 | 0.25 | 0.43 |

## State

- B graph kind: `directional_inferred`
- C graph kind: `directional_inferred`
