# kindling — Tuning Guide

Most of kindling's behaviour is **auto-gated**: the engine picks the base
scorer and the active channels from measurable properties of your data, so
the defaults are the right starting point on almost every dataset. Confirm
what it chose with `engine.activation_plan.summary()`. This guide covers
the few knobs worth thinking about. Full table: [`REFERENCE.md`](REFERENCE.md) §5.

## The defaults are the tuning

The single most important fact: per-fit / learned calibration of these
weights was tried and **rejected** — the internal holdout's drift structure
inverts the test ranking, so per-dataset tuning transfers *worse* than the
fixed cross-dataset defaults (EXPERIMENTS.md §4.4, §7.2). Reach for a knob
only when you have a specific, measured reason.

## Base scorer

| knob | default | when to touch |
|---|---|---|
| `base_scorer` | `"auto"` | force `"ease"` / `"cooc"` only for experiments |
| `ease_max_items` | 20 000 | the EASE/cooc gate; raise only with more RAM/patience |
| `ease_lambda` | auto (`20·nnz/n_items`) | beauty-like catalogs measure slightly better at ~250 |

Above `ease_max_items` the base switches to wilson-normalized
co-occurrence automatically (it removes popularity cheaply and, on
large sparse catalogs, beats low-rank EASE at a fraction of the cost).

## Channels (all auto-gated)

| knob | default | gate |
|---|---|---|
| `trend_alpha` | 0.5 | needs timestamps; 0 to disable |
| `last_item_alpha` | 0.25 | needs EASE base; 0.5 overshoots everywhere measured |
| `transition_alpha` | 0.25 | needs timestamps AND not a rating-burst (auto-off on burst data like ml1m) |
| `user_cf_alpha` / `user_cf_history_gate` | 1.0 / 20 | activates only on sparse-history data (median ≤ gate) |
| `content_alpha` | 0.0 | content blending stays off; the cold-slot path is the content channel |

If `activation_plan` shows a channel `off` that you expected on, check the
gate reason it prints — it's almost always a missing `timestamp` column, a
rating-burst, or a history length on the wrong side of the gate.

## Cold-start / open catalog

| knob | default | when to touch |
|---|---|---|
| `open_catalog` | `True` | metadata-only items become recommendable candidates |
| `cold_slots` | 0 | set `1` on churning catalogs to reserve a top-K slot for cold items |
| `cold_recency_beta` | 2.0 | release-recency prior in the cold-slot ranker; 0 disables |

## Retrieval

| knob | default | when to touch |
|---|---|---|
| `retrieval_budget` | 500 | the gap-decomposition diagnostic shows little headroom from raising it alone |

## Diagnosing a disappointing result

1. `engine.activation_plan.summary()` — did the right base + channels turn on?
2. `bench/run_gap_decomp.py` — is the system **ranking-bound** (the right
   items are retrieved but ranked poorly) or **retrieval-bound** (the right
   items never reach the candidate pool)? The fix differs entirely, and the
   diagnostic tells you which wall you're against before you tune anything.
