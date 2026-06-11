# Coherence-filter percentile sweep — movielens-1m

- best variant: `louvain_gamma_2`
- users evaluated: 500
- train / test: 900,188 / 100,021
- k = 10
- timestamp: 2026-05-09T19:11:42
- variant kwargs: `{'persona_method': 'louvain_graph', 'louvain_resolution': 2.0}`

## Quality

| metric | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| `ndcg_at_k` | 0.2547 | 0.2582 | 0.2582 | 0.2509 |
| `mrr` | 0.4356 | 0.4277 | 0.4277 | 0.4191 |
| `recall_at_k` | 0.0414 | 0.0443 | 0.0443 | 0.0429 |
| `hit_rate` | 0.6880 | 0.6900 | 0.6900 | 0.6780 |
| `coverage` | 0.0541 | 0.0454 | 0.0454 | 0.0397 |

## Persona structure

| stage | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| n_personas | 3 | 3 | 3 | 3 |
| n_kept | 3 | 2 | 2 | 1 |
| mean_coherence | 6.016 | 6.016 | 6.016 | 6.016 |

## Differentiation vs cooc

| stat | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| `jaccard@K` | 0.484 | 0.499 | 0.499 | 0.393 |
| `kendall_tau` | 0.302 | 0.455 | 0.455 | 0.360 |
| `rank_shift` | 36 | 37 | 37 | 45 |
| `frac_identical` | 0.00% | 0.00% | 0.00% | 0.00% |
| `n_users_sampled` | 90 | 60 | 60 | 30 |

## Fit timing (s)

| stage | pct=0.00 | pct=0.25 | pct=0.50 | pct=0.75 |
| --- | ---: | ---: | ---: | ---: |
| fit_s | 18.91 | 17.48 | 17.20 | 15.43 |
