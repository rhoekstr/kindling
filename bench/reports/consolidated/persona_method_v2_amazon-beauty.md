# Persona-method ablation — amazon-beauty

- users evaluated: 500
- train / test: 178,651 / 19,851
- k = 10
- timestamp: 2026-05-09T11:39:14
- signal_kind: ratings

## Quality

| metric | A: no personas | B: SVD+HDBSCAN | C: Louvain |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.0290 | 0.0290 | 0.0295 |
| `mrr` | 0.0407 | 0.0407 | 0.0400 |
| `recall_at_k` | 0.0327 | 0.0327 | 0.0344 |
| `hit_rate` | 0.0860 | 0.0860 | 0.0860 |
| `coverage` | 0.1258 | 0.1248 | 0.1287 |

## Persona structure

| stage | A | B | C |
|---|---:|---:|---:|
| n_personas | 0 | 2 | 4 |
| persona_method_used | `none` | `hdbscan_factors` | `louvain_graph` |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 38.70 | 94.38 | 42.27 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.08 | 0.24 | 0.22 |
| `p95_ms` | 0.16 | 0.35 | 0.33 |
| `p99_ms` | 0.25 | 0.42 | 0.43 |
