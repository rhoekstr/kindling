# Grafting revival — findings (metadata→cooc smoothing on the clean engine)

Re-examination of the "edge grafting DEAD" fence post (EXPERIMENTS §4.7) on the
clean wilson-cooc base, on production datasets with **rich cold metadata**.
Driver: `bench/run_graft_revisit.py`. All numbers: realistic tier, NDCG@20,
50k-user subsample (40k for H&M-grounded), ~2000-user eval, by item-warmth tier.

## 1. Gate — does metadata predict co-occurrence structure?

Spearman / R² of metadata-cosine vs cooc weight on held-out warm pairs.
ALIVE iff Spearman ≥ 0.20 or R²_lin ≥ 0.10.

| dataset | metadata | Spearman | R²_lin | R²_binned | verdict |
|---|---|---:|---:|---:|---|
| amazon-book | coarse categories | 0.099 | 0.021 | 0.013 | DEAD |
| retailrocket | single `category` | 0.157 | 0.123 | **0.000** | alive?* |
| steam | community tags | 0.212 | 0.088 | 0.056 | ALIVE |
| H&M | type/colour/dept/section/garment + text | 0.226 | 0.132 | 0.073 | ALIVE |

*retailrocket: only **3%** of pairs share any metadata (sparse single category);
R²_lin = 0.123 but R²_binned = 0.000 — **linear R² is fooled by sparse binary
metadata.** The gate metric itself needs a more robust model (see §4).

## 2. Cold-only grafting still floods; ALL-ITEMS smoothing fixes it

Cold-only grafting (patch cold rows) leapfrogs warm items → floods (H&M @R²:
**−32%** NDCG). Smoothing **every** item — `C_aug = C_obs + λ·max_obs·M`, M =
metadata-kNN graph — keeps warm items' real-cooc lead. Net result is a **strict
win** at a gentle dose:

| dataset | best NDCG Δ | cold tiers (0 → x) | warm 21+ | coverage |
|---|---:|---|---:|---:|
| H&M | **+12%** (cap≈0.02–0.035) | 3-5,6-10: 0 → ~0.01 | preserved | +30–60% |
| steam | **+3%** (cap≈0.02) | 11-20 doubles | preserved/up | +20–50% |

Net-positive zone: cap ≲ 0.05–0.075 on both. NDCG-optimal cap ≈ **0.02–0.035 on
both** despite different R² (0.13 vs 0.088) → the optimum is roughly
dataset-invariant, **not** proportional to R². So R² is better as the **gate**
than as the **dose**.

## 3. Grounding the cap — impute the fitted prediction (no hand-set cap)

Fit `cooc_weight ~ metadata_sim` on the kNN-edge population (88–94% of edges
have obs=0) and impute the prediction `f̂(sim)` as the edge weight. The mean
imputed weight / max_obs is the **effective cap**, chosen by the data:

| dataset | model | NDCG Δ | eff_cap | note |
|---|---|---:|---:|---|
| steam | OLS | +3% | 0.0074 | = cap-sweep peak |
| steam | Poisson | +3% | 0.0074 | ≈ OLS (warm 21+ slightly higher) |
| H&M | OLS | +11% | 0.0031 | ≈ cap-sweep peak (+12%) |
| H&M | Poisson | +11% | 0.0031 | identical to OLS |

The imputed dose **self-selects the empirical optimum** with no tuning — and at a
*lower average* dose than the uniform cap (0.003 vs ~0.025) because it
concentrates weight on high-confidence edges. **The cap is grounded as
E[cooc | metadata_sim].**

## 4. OLS vs Poisson

Empirically **identical for the dose** on both datasets (predicted weights are
tiny and the top-k selection is the same). Poisson is more principled (count
link, never negative) but banks no measurable edge here. Its real value is at
the **gate fidelity metric**: R²_lin is misleading on sparse binary metadata
(retailrocket), where a Poisson-deviance / rank-based measure would be more
robust for the go/no-go decision.

## Status & next steps

- **Validated at the base-cooc level on 2 datasets (H&M, steam), single
  subsamples.** Strict-win mechanism: all-items, R²-gated, prediction-grounded
  dose.
- **Next:** (a) wire into `Engine` (cooc-base path, >20k items) as a gated,
  default-off `metadata_smoothing` feature; re-run `bench/verify` + H&M/steam
  validation through the full stack. (b) repeated-subsample / larger-eval
  confirmation. (c) robustify the gate metric (Poisson deviance) for the
  sparse-metadata regime.
