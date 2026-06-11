# Persona-method ablation — movielens-1m

- users evaluated: 500
- train / test: 900,188 / 100,021
- k = 10
- timestamp: 2026-05-09T11:38:22
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
| fit_s | 11.86 | 15.81 | 16.19 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.73 | 0.75 | 0.77 |
| `p95_ms` | 3.00 | 2.98 | 2.88 |
| `p99_ms` | 4.22 | 4.40 | 4.33 |
