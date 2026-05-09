# Persona-method ablation — amazon-book

- users evaluated: 500
- train / test: 2,380,730 / 603,378
- k = 10
- timestamp: 2026-05-08T21:29:01
- signal_kind: binary

## Quality

| metric | A: no personas | B: SVD+HDBSCAN | C: Louvain |
|---|---:|---:|---:|
| `ndcg_at_k` | 0.0253 | 0.0259 | 0.0239 |
| `mrr` | 0.0563 | 0.0570 | 0.0533 |
| `recall_at_k` | 0.0246 | 0.0256 | 0.0233 |
| `hit_rate` | 0.1400 | 0.1440 | 0.1380 |
| `coverage` | 0.0138 | 0.0143 | 0.0142 |

## Persona structure

| stage | A | B | C |
|---|---:|---:|---:|
| n_personas | 0 | 25 | 5 |
| persona_method_used | `none` | `hdbscan_factors` | `louvain_graph` |

## Fit timing

| stage | A | B | C |
|---|---:|---:|---:|
| fit_s | 778.93 | 1086.82 | 1106.94 |

## Recommend latency (ms)

| stage | A | B | C |
|---|---:|---:|---:|
| `p50_ms` | 1.07 | 1.05 | 2.61 |
| `p95_ms` | 2.64 | 2.55 | 9.68 |
| `p99_ms` | 4.85 | 4.98 | 21.44 |
