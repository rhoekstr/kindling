# Clustering × coherence sweep — amazon-beauty

- users evaluated: 500
- train / test: 178,651 / 19,851
- k = 10
- coherence_filter_percentile: 0.50
- timestamp: 2026-05-09T13:14:05

## Coherence distribution per variant

| variant | n_personas | n_kept | mean | median | p25 | p75 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hdbscan_svd` | 2 | 1 | 0.032 | 0.032 | 0.021 | 0.043 | 0.010 | 0.055 |
| `hdbscan_als` | 2 | 1 | 0.032 | 0.032 | 0.021 | 0.043 | 0.010 | 0.055 |
| `louvain_raw` | 4 | 2 | 0.035 | 0.025 | 0.017 | 0.042 | 0.009 | 0.079 |
| `louvain_log_prune` | 4 | 2 | 0.355 | 0.050 | 0.023 | 0.381 | 0.009 | 1.311 |
| `louvain_cosine` | 5 | 2 | 0.228 | 0.065 | 0.015 | 0.395 | 0.010 | 0.656 |
| `louvain_gamma_2` | 9 | 5 | 0.131 | 0.029 | 0.013 | 0.051 | 0.009 | 0.889 |
| `dc_sbm` | 96 | 2 | 0.914 | 0.000 | 0.000 | 0.222 | 0.000 | 18.051 |

## Persona vs cooc differentiation

How different are persona-cooc top-K recs from global cooc top-K?
- `jaccard@K`: set overlap of top-K (1 = identical, 0 = disjoint)
- `kendall_tau`: rank agreement on shared items (-1 to 1)
- `rank_shift`: mean global-cooc rank of items unique to persona top-K (large = persona surfaces things cooc would have ranked far down)
- `frac_identical`: fraction of users where persona top-K = cooc top-K

| variant | jaccard@K | kendall_tau | rank_shift | frac_identical | n_users |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hdbscan_svd` | 0.000 | 0.000 | 7990 | 0.00% | 30 |
| `hdbscan_als` | 0.000 | 0.000 | 7990 | 0.00% | 30 |
| `louvain_raw` | 0.925 | 0.961 | 148 | 75.00% | 60 |
| `louvain_log_prune` | 0.674 | 0.509 | 21 | 31.67% | 60 |
| `louvain_cosine` | 0.713 | 0.654 | 92 | 31.67% | 60 |
| `louvain_gamma_2` | 0.758 | 0.781 | 110 | 54.00% | 150 |
| `dc_sbm` | 0.055 | -0.004 | 3590 | 0.00% | 60 |

## Quality (with coherence filter applied)

| metric | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ndcg_at_k` | 0.0290 | 0.0290 | 0.0291 | 0.0291 | 0.0290 | 0.0291 | 0.0294 |
| `mrr` | 0.0407 | 0.0407 | 0.0407 | 0.0407 | 0.0407 | 0.0407 | 0.0413 |
| `recall_at_k` | 0.0327 | 0.0327 | 0.0324 | 0.0324 | 0.0327 | 0.0324 | 0.0332 |
| `hit_rate` | 0.0860 | 0.0860 | 0.0860 | 0.0860 | 0.0860 | 0.0860 | 0.0880 |
| `coverage` | 0.1248 | 0.1248 | 0.1260 | 0.1258 | 0.1258 | 0.1266 | 0.1239 |

## Fit timing (s)

| stage | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fit_s | 97.44 | 70.81 | 38.22 | 38.41 | 39.17 | 39.54 | 40.16 |

## Recommend latency (ms)

| stat | hdbscan_svd | hdbscan_als | louvain_raw | louvain_log_prune | louvain_cosine | louvain_gamma_2 | dc_sbm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p50_ms` | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |
| `p95_ms` | 0.16 | 0.16 | 0.21 | 0.16 | 0.17 | 0.21 | 0.16 |
| `p99_ms` | 0.23 | 0.23 | 0.26 | 0.22 | 0.24 | 0.27 | 0.24 |

## Variant configs

- `hdbscan_svd`: `{'persona_method': 'hdbscan_factors', 'use_als': 'force_off'}`
- `hdbscan_als`: `{'persona_method': 'hdbscan_factors', 'use_als': 'force_on'}`
- `louvain_raw`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'raw', 'louvain_min_edge_percentile': 0.0}`
- `louvain_log_prune`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'log', 'louvain_min_edge_percentile': 0.05}`
- `louvain_cosine`: `{'persona_method': 'louvain_graph', 'louvain_weight_transform': 'cosine'}`
- `louvain_gamma_2`: `{'persona_method': 'louvain_graph', 'louvain_resolution': 2.0}`
- `dc_sbm`: `{'persona_method': 'dc_sbm', 'louvain_weight_transform': 'raw', 'dc_sbm_warmstart_resolution': 2.0, 'dc_sbm_min_internal_fraction': 0.0, 'dc_sbm_max_passes': 12}`
