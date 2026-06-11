# Coherence-filter percentile sweep — amazon-beauty

- best variant: `dc_sbm`
- users evaluated: 500
- train / test: 178,651 / 19,851
- k = 10
- timestamp: 2026-05-09T19:13:00
- variant kwargs: `{'persona_method': 'dc_sbm', 'louvain_weight_transform': 'raw', 'dc_sbm_warmstart_resolution': 2.0, 'dc_sbm_min_internal_fraction': 0.0, 'dc_sbm_max_passes': 12, 'dc_sbm_init_mode': 'louvain'}`

## Quality

| metric | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| `ndcg_at_k` | 0.0265 | 0.0294 | 0.0294 | 0.0290 |
| `mrr` | 0.0348 | 0.0413 | 0.0413 | 0.0407 |
| `recall_at_k` | 0.0325 | 0.0332 | 0.0332 | 0.0327 |
| `hit_rate` | 0.0760 | 0.0880 | 0.0880 | 0.0860 |
| `coverage` | 0.1137 | 0.1239 | 0.1239 | 0.1233 |

## Persona structure

| stage | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| n_personas | 96 | 96 | 96 | 96 |
| n_kept | 96 | 2 | 2 | 1 |
| mean_coherence | 0.914 | 0.914 | 0.914 | 0.914 |

## Differentiation vs cooc

| stat | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| `jaccard@K` | 0.178 | 0.055 | 0.055 | 0.043 |
| `kendall_tau` | 0.150 | -0.004 | -0.004 | -0.056 |
| `rank_shift` | 5522 | 3590 | 3590 | 5814 |
| `frac_identical` | 11.17% | 0.00% | 0.00% | 0.00% |
| `n_users_sampled` | 188 | 60 | 60 | 30 |

## Fit timing (s)

| stage | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| fit_s | 43.32 | 48.93 | 46.07 | 45.27 |
