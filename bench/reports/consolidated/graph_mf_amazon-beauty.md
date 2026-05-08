# Graph-MF ablation — amazon-beauty

- users evaluated: 500
- train / test: 178,651 / 19,851
- k = 10
- timestamp: 2026-05-08T17:06:14

## Quality

| metric | A: cooc base | B: graph_mf base | C: graph_mf boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.0290 | 0.0253 | 0.0292 |
| `mrr` | 0.0407 | 0.0370 | 0.0411 |
| `recall_at_k` | 0.0327 | 0.0304 | 0.0331 |
| `hit_rate` | 0.0860 | 0.0900 | 0.0880 |
| `coverage` | 0.1248 | 0.0806 | 0.1177 |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 117.43 | 115.57 | 80.55 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.25 | 0.13 | 0.29 |
| `p95_ms` | 0.42 | 0.26 | 0.61 |
| `p99_ms` | 0.64 | 0.34 | 0.81 |

## State

- B graph kind: `co-ownership`
- C graph kind: `co-ownership`
