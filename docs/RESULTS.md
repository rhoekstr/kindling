# Results — what kindling brings

kindling is a self-configuring, closed-form recommender (no training loop, no GPU)
with a Rust core that serves in **sub-millisecond** time. It earns its keep in two
distinct regimes, and the honest picture is that it's *exceptional* in one of them.

## 1. Discovery — strongest personalized model, grows with the data

On standard "predict the held-out future, exclude already-seen" benchmarks,
kindling is the strongest *personalized* model on every reference dataset and
beats implicit ALS everywhere:

| dataset | NDCG@10 | base |
|---|---:|---|
| movielens-1m | 0.2928 | rating-weighted EASE |
| amazon-beauty | 0.0336 | EASE + user-CF |
| steam | 0.0659 | wilson-cooc + cold slots |
| amazon-book-chrono | 0.0318 | wilson-cooc + trend/transitions |

The design thesis is visible in the growth curves
([`bench/reports/growth_curves_grid.png`](../bench/reports/growth_curves_grid.png)):
at the very coldest data, plain popularity is competitive, and kindling *leans on
that prior*, then pulls away as signal accumulates — the channels turn themselves
on only when the data warrants. It wins the cold-*user* buckets (few interactions,
mature catalog) outright on cold-heavy catalogs.

## 2. Repeat regimes — it separates from the entire field

This is where kindling is in a different class. On grocery/retail/check-in logs,
the most valuable recommendation is the thing the user is about to *re-buy* —
which the standard eval hides. Evaluated fairly (reorders credited), with the
repeat module auto-enabled by the held-out gate:

| dataset | **kindling** | cooc | popularity | ALS | item-kNN | kindling vs best baseline |
|---|---:|---:|---:|---:|---:|---:|
| dunnhumby | **0.478** | 0.034 | 0.036 | 0.052 | 0.046 | **9×** |
| instacart | **0.372** | 0.017 | 0.023 | 0.015 | — | **16×** |
| gowalla | **0.173** | 0.010 | 0.004 | 0.013 | — | **13×** |
| tafeng | **0.140** | 0.049 | 0.083 | 0.013 | 0.068 | **1.7×** |
| retailrocket | **0.092** | 0.013 | 0.003 | 0.007 | — | **7×** |
| hm | **0.020** | 0.013 | 0.005 | 0.010 | 0.013 | **1.6×** |

No baseline comes close. kindling stacks the personal-frequency signal (what you
re-buy), a timing multiplier (suppress just-bought, surface due), and new-item
discovery — none of which the baselines have.

### It turns itself on correctly

The repeat module isn't always right: on **steam**, the "repeats" are re-logged
game sessions, not repurchase intent, and recommending them *hurts* — even though
steam has a higher duplicate rate than tafeng. A repeat-rate threshold can't tell
these apart. kindling's **held-out repeat gate** does: it tests the module on a
held-out built from the benchmark's own chronological-global split and keeps it
only if it strictly helps — declining steam (its reference number is untouched)
and keeping the genuine grocery logs, automatically. See
[REPEAT-GATE.md](REPEAT-GATE.md).

## 3. Speed — closed-form fit, microsecond serving

- **Serving:** sub-millisecond single recommend (native Rust, GIL-released batch),
  e.g. ml1m 0.17 ms p50; persists to a self-contained artifact for re-fit-free
  serving (`KindlingServer` + a FastAPI example).
- **Fit:** one closed-form pass — no epochs, no learning rate. The slowest of the
  *classical* baselines, but vastly faster than a trained GCN: on ml1m, full fit
  is **9 s vs LightGCN's 1,860 s (~200×)** — and kindling is *more accurate* there
  (0.309 vs 0.276). LightGCN only leads on the cold/sparse end, at 100–200× the
  fit cost.

## 4. Where it doesn't win (the honest part)

- **Data-starved cold *system*:** when the whole dataset is tiny, popularity (the
  global prior) wins — there's little personalization signal for *anyone* to
  extract. kindling leans on popularity there by design rather than fighting it.
- **EASE+ (EDLAE):** a denoising base variant that beats plain EASE on 3 datasets
  and *loses* on steam; the held-out δ search couldn't reliably pick, so it ships
  **opt-in** (`ease_denoise`), not as the default. See
  [EASE-VARIANTS-ASSESSMENT.md](EASE-VARIANTS-ASSESSMENT.md).

The full experiment record — including the many negative results, which are half
the value — is in [EXPERIMENTS.md](EXPERIMENTS.md) and [LESSONS.md](LESSONS.md).
