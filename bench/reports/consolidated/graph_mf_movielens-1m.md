# Graph-MF ablation — movielens-1m

- users evaluated: 500
- train / test: 900,188 / 100,021
- k = 10
- timestamp: 2026-05-08T17:05:09

## Quality

| metric | A: cooc base | B: graph_mf base | C: graph_mf boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.2521 | 0.2163 | 0.2517 |
| `mrr` | 0.4202 | 0.3620 | 0.4172 |
| `recall_at_k` | 0.0442 | 0.0427 | 0.0442 |
| `hit_rate` | 0.6700 | 0.6620 | 0.6680 |
| `coverage` | 0.0506 | 0.2276 | 0.0506 |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 16.93 | 18.07 | 21.31 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.73 | 0.43 | 0.99 |
| `p95_ms` | 2.85 | 1.49 | 4.33 |
| `p99_ms` | 4.43 | 2.16 | 9.37 |

## State

- B graph kind: `co-ownership`
- C graph kind: `co-ownership`
