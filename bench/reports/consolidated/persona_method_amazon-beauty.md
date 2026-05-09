# Persona-method ablation — amazon-beauty

- users evaluated: 500
- train / test: 178,651 / 19,851
- k = 10
- timestamp: 2026-05-08T21:26:10
- signal_kind: ratings

## Quality

| metric | A: no personas | B: SVD+HDBSCAN | C: Louvain |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.0290 | 0.0290 | 0.0297 |
| `mrr` | 0.0407 | 0.0407 | 0.0390 |
| `recall_at_k` | 0.0327 | 0.0327 | 0.0371 |
| `hit_rate` | 0.0860 | 0.0860 | 0.0900 |
| `coverage` | 0.1258 | 0.1248 | 0.1302 |

## Persona structure

| stage | A | B | C |
|---|---:|---:|---:|
| n_personas | 0 | 2 | 4 |
| persona_method_used | `none` | `hdbscan_factors` | `louvain_graph` |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 37.64 | 90.78 | 38.96 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 0.08 | 0.22 | 0.20 |
| `p95_ms` | 0.22 | 0.32 | 0.30 |
| `p99_ms` | 0.31 | 0.39 | 0.40 |
