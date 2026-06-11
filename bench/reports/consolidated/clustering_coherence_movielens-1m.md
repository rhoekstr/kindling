# Clustering × coherence sweep — movielens-1m

- users evaluated: 500
- train / test: 900,188 / 100,021
- k = 10
- coherence_filter_percentile: 0.50
- timestamp: 2026-05-09T13:11:54

## Coherence distribution per variant

| variant | n_personas | n_kept | mean | median | p25 | p75 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hdbscan_svd` | 3 | 2 | 17.190 | 16.996 | 11.300 | 22.983 | 5.605 | 28.970 |
| `hdbscan_als` | 9 | 5 | 23.417 | 24.229 | 23.102 | 24.917 | 6.378 | 38.152 |
| `louvain_raw` | 1 | 0 | 5.234 | 5.234 | 5.234 | 5.234 | 5.234 | 5.234 |
| `louvain_log_prune` | 1 | 0 | 5.234 | 5.234 | 5.234 | 5.234 | 5.234 | 5.234 |
| `louvain_cosine` | 1 | 0 | 5.234 | 5.234 | 5.234 | 5.234 | 5.234 | 5.234 |
| `louvain_gamma_2` | 3 | 2 | 6.016 | 5.753 | 5.699 | 6.201 | 5.645 | 6.649 |
| `dc_sbm` | 43 | 0 | 155.954 | 155.407 | 88.247 | 215.920 | 5.234 | 389.181 |

## Persona vs cooc differentiation

How different are persona-cooc top-K recs from global cooc top-K?
- `jaccard@K`: set overlap of top-K (1 = identical, 0 = disjoint)
- `kendall_tau`: rank agreement on shared items (-1 to 1)
- `rank_shift`: mean global-cooc rank of items unique to persona top-K (large = persona surfaces things cooc would have ranked far down)
- `frac_identical`: fraction of users where persona top-K = cooc top-K

| variant | jaccard@K | kendall_tau | rank_shift | frac_identical | n_users |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hdbscan_svd` | 0.020 | 0.083 | 176 | 0.00% | 60 |
| `hdbscan_als` | 0.123 | 0.163 | 119 | 0.00% | 150 |
| `louvain_raw` | 0.000 | 0.000 | 0 | 0.00% | 0 |
| `louvain_log_prune` | 0.000 | 0.000 | 0 | 0.00% | 0 |
| `louvain_cosine` | 0.000 | 0.000 | 0 | 0.00% | 0 |
| `louvain_gamma_2` | 0.499 | 0.455 | 37 | 0.00% | 60 |
| `dc_sbm` | 0.000 | 0.000 | 0 | 0.00% | 0 |

## Quality (with coherence filter applied)

| metric | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ndcg_at_k` | 0.2535 | 0.2569 | 0.2561 | 0.2561 | 0.2561 | 0.2582 | 0.2561 |
| `mrr` | 0.4195 | 0.4205 | 0.4199 | 0.4199 | 0.4199 | 0.4277 | 0.4199 |
| `recall_at_k` | 0.0451 | 0.0443 | 0.0442 | 0.0442 | 0.0442 | 0.0443 | 0.0442 |
| `hit_rate` | 0.6860 | 0.6800 | 0.6820 | 0.6820 | 0.6820 | 0.6900 | 0.6820 |
| `coverage` | 0.0413 | 0.0392 | 0.0348 | 0.0348 | 0.0348 | 0.0454 | 0.0348 |

## Fit timing (s)

| stage | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fit_s | 15.41 | 15.74 | 13.53 | 13.62 | 13.71 | 19.42 | 25.11 |

## Recommend latency (ms)

| stat | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p50_ms` | 0.77 | 0.74 | 0.73 | 0.75 | 0.74 | 0.89 | 0.93 |
| `p95_ms` | 3.19 | 2.89 | 2.92 | 3.01 | 2.94 | 4.48 | 4.88 |
| `p99_ms` | 4.44 | 4.22 | 4.32 | 4.30 | 4.21 | 7.68 | 8.98 |

## Variant configs

- `hdbscan_svd`: `{'persona_method': 'hdbscan_factors', 'use_als': 'force_off'}`
- `hdbscan_als`: `{'persona_method': 'hdbscan_factors', 'use_als': 'force_on'}`
- `louvain_raw`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'raw', 'louvain_min_edge_percentile': 0.0}`
- `louvain_log_prune`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'log', 'louvain_min_edge_percentile': 0.05}`
- `louvain_cosine`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'cosine'}`
- `louvain_gamma_2`: `{'persona_method': 'louvain_graph', 'louvain_resolution': 2.0}`
- `dc_sbm`: `{'persona_method': 'dc_sbm', 'louvain_weight_transform': 'raw', 'dc_sbm_warmstart_resolution': 2.0, 'dc_sbm_min_internal_fraction': 0.0, 'dc_sbm_max_passes': 12}`
