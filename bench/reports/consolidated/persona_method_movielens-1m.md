# Persona-method ablation — movielens-1m

- users evaluated: 500
- train / test: 900,188 / 100,021
- k = 10
- timestamp: 2026-05-08T21:25:18
- signal_kind: ratings

## Quality

| metric | A: no personas | B: SVD+HDBSCAN | C: Louvain |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.2561 | 0.2529 | 0.2562 |
| `mrr` | 0.4199 | 0.4180 | 0.4198 |
| `recall_at_k` | 0.0442 | 0.0442 | 0.0443 |
| `hit_rate` | 0.6820 | 0.6820 | 0.6820 |
| `coverage` | 0.0348 | 0.0435 | 0.0348 |

## Persona structure

| stage | A | B | C |
|---|---:|---:|---:|
| n_personas | 0 | 3 | 1 |
| persona_method_used | `none` | `hdbscan_factors` | `louvain_graph` |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 12.77 | 15.49 | 15.68 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.70 | 0.72 | 0.76 |
| `p95_ms` | 2.75 | 2.90 | 3.04 |
| `p99_ms` | 4.09 | 4.22 | 4.34 |
