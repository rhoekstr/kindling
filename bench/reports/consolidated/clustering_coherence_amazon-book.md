# Clustering × coherence sweep — amazon-book

- users evaluated: 500
- train / test: 2,380,730 / 603,378
- k = 10
- coherence_filter_percentile: 0.50
- timestamp: 2026-05-09T13:20:13

## Coherence distribution per variant

| variant | n_personas | n_kept | mean | median | p25 | p75 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hdbscan_svd` | 25 | 13 | 0.450 | 0.381 | 0.246 | 0.551 | 0.142 | 1.367 |
| `hdbscan_als` | 29 | 15 | 0.514 | 0.410 | 0.283 | 0.619 | 0.173 | 1.414 |
| `louvain_raw` | 5 | 3 | 0.161 | 0.127 | 0.074 | 0.207 | 0.062 | 0.334 |
| `louvain_log_prune` | 3 | 2 | 0.114 | 0.067 | 0.065 | 0.140 | 0.062 | 0.214 |
| `louvain_cosine` | 4 | 2 | 0.104 | 0.090 | 0.068 | 0.126 | 0.065 | 0.172 |
| `louvain_gamma_2` | 6 | 3 | 0.104 | 0.099 | 0.079 | 0.118 | 0.066 | 0.165 |
| `dc_sbm` | 7 | 4 | 0.114 | 0.118 | 0.083 | 0.128 | 0.068 | 0.190 |

## Persona vs cooc differentiation

How different are persona-cooc top-K recs from global cooc top-K?
- `jaccard@K`: set overlap of top-K (1 = identical, 0 = disjoint)
- `kendall_tau`: rank agreement on shared items (-1 to 1)
- `rank_shift`: mean global-cooc rank of items unique to persona top-K (large = persona surfaces things cooc would have ranked far down)
- `frac_identical`: fraction of users where persona top-K = cooc top-K

| variant | jaccard@K | kendall_tau | rank_shift | frac_identical | n_users |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hdbscan_svd` | 0.313 | 0.412 | 88 | 1.28% | 390 |
| `hdbscan_als` | 0.321 | 0.216 | 91 | 0.00% | 450 |
| `louvain_raw` | 0.803 | 0.778 | 30 | 37.78% | 90 |
| `louvain_log_prune` | 0.784 | 0.806 | 26 | 41.67% | 60 |
| `louvain_cosine` | 0.738 | 0.728 | 76 | 33.33% | 60 |
| `louvain_gamma_2` | 0.616 | 0.578 | 51 | 18.89% | 90 |
| `dc_sbm` | 0.726 | 0.708 | 20 | 25.00% | 120 |

## Quality (with coherence filter applied)

| metric | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ndcg_at_k` | 0.0253 | 0.0251 | 0.0249 | 0.0242 | 0.0253 | 0.0249 | 0.0250 |
| `mrr` | 0.0560 | 0.0563 | 0.0553 | 0.0547 | 0.0561 | 0.0549 | 0.0552 |
| `recall_at_k` | 0.0247 | 0.0243 | 0.0242 | 0.0233 | 0.0248 | 0.0245 | 0.0245 |
| `hit_rate` | 0.1380 | 0.1360 | 0.1380 | 0.1400 | 0.1420 | 0.1360 | 0.1340 |
| `coverage` | 0.0142 | 0.0142 | 0.0139 | 0.0138 | 0.0140 | 0.0142 | 0.0141 |

## Fit timing (s)

| stage | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fit_s | 979.63 | 1029.85 | 952.82 | 1171.95 | 952.06 | 1106.29 | 1115.86 |

## Recommend latency (ms)

| stat | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p50_ms` | 0.89 | 0.93 | 0.92 | 1.98 | 0.93 | 1.08 | 1.13 |
| `p95_ms` | 1.90 | 2.05 | 1.94 | 4.19 | 1.92 | 3.15 | 3.22 |
| `p99_ms` | 3.20 | 3.50 | 3.22 | 6.62 | 3.22 | 4.96 | 5.10 |

## Variant configs

- `hdbscan_svd`: `{'persona_method': 'hdbscan_factors', 'use_als': 'force_off'}`
- `hdbscan_als`: `{'persona_method': 'hdbscan_factors', 'use_als': 'force_on'}`
- `louvain_raw`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'raw', 'louvain_min_edge_percentile': 0.0}`
- `louvain_log_prune`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'log', 'louvain_min_edge_percentile': 0.05}`
- `louvain_cosine`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'cosine'}`
- `louvain_gamma_2`: `{'persona_method': 'louvain_graph', 'louvain_resolution': 2.0}`
- `dc_sbm`: `{'persona_method': 'dc_sbm', 'louvain_weight_transform': 'raw', 'dc_sbm_warmstart_resolution': 2.0, 'dc_sbm_min_internal_fraction': 0.0, 'dc_sbm_max_passes': 12}`
