# Graph-MF ablation — movielens-1m

- users evaluated: 500
- train / test: 900,188 / 100,021
- k = 10
- timestamp: 2026-05-08T17:32:45

## Quality

| metric | A: cooc base | B: graph_mf base | C: graph_mf boost |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.2521 | 0.2147 | 0.2517 |
| `mrr` | 0.4202 | 0.3604 | 0.4172 |
| `recall_at_k` | 0.0442 | 0.0428 | 0.0442 |
| `hit_rate` | 0.6700 | 0.6660 | 0.6680 |
| `coverage` | 0.0506 | 0.2284 | 0.0506 |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 27.30 | 22.89 | 20.95 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.91 | 0.44 | 0.79 |
| `p95_ms` | 5.70 | 1.60 | 3.10 |
| `p99_ms` | 9.02 | 2.28 | 4.81 |

## State

- B graph kind: `directional_inferred`
- C graph kind: `directional_inferred`
