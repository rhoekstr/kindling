# EASE-variant assessment (Stage 3)

**Question:** should kindling swap its EASE base for an EASE variant (EDLAE / RLAE
/ ADMM-SLIM / SLIM)? **Recommendation: no — not for accuracy.** Details below.
*No engine change was made; this is an assessment only.*

## Numbers (ml-1m, NDCG@10)

Two eval families (not directly cross-comparable — compare lifts *within* a block
against that block's EASE anchor):

**DIY closed forms** (`bench/ease_variants.py`, valid-masked eval, λ-tuned):

| variant | NDCG@10 | vs EASE |
|---|---:|---:|
| EASE (Steck '19) | 0.3062 | — |
| EDLAE (denoising, our impl) | 0.3071 | **+0.3%** |
| RLAE (relaxed diag, our impl) | 0.3067 | +0.16% |

**RecBole** (`bench/recbole_runner.py`, RecBole eval):

| variant | NDCG@10 | vs EASE | fit |
|---|---:|---:|---:|
| EASE | 0.3022 | — | 1.5s |
| **ADMM-SLIM** | **0.3110** | **+2.9%** | 19.7s |
| SLIMElastic | 0.2971 | −1.7% | 3.4s |

## Findings

1. **ADMM-SLIM is the only meaningful win** (+2.9% over EASE) — but it's iterative
   (**13× the fit time**, 19.7s vs 1.5s) and L1-sparse, which cuts against
   kindling's "fast closed-form fit, no training loop" identity.
2. **SLIMElastic is worse** than EASE (−1.7%).
3. **EDLAE / RLAE give marginal lifts** (+0.3% / +0.16%). Caveat: these are *our*
   closed forms (a popularity-scaled diagonal ridge; a relaxed zero-diagonal),
   not Steck's exact derivations — a canonical EDLAE could land closer to
   ADMM-SLIM. Even so, the EASE-family frontier on ml-1m is only ~3% above EASE.

## Why not swap (for accuracy)

- The whole EASE family sits within **~3%** of plain EASE on ml-1m; the best
  (ADMM-SLIM) costs 13× fit and an iterative solver.
- kindling's accuracy lever is **not the base** — it's the channels (trend /
  user-CF / last-item / transitions), the repeat module, and the boost
  calibrator stacked *on top* of the base. A ≤3% base bump is small next to those
  system-level contributions, and a base swap would complicate the native EASE
  path (which is byte-exact today).
- Held-out λ auto-search already captures most of the base headroom for free.

## The one angle worth revisiting (separately)

ADMM-SLIM's **sparsity** — not its accuracy — could let kindling push the EASE
gate *beyond* the current 20k-item cap (where it falls back to wilson-cooc),
since a sparse item-item solve scales where the dense O(n³) EASE inversion can't.
That's a **scaling** play (extend EASE to larger catalogs), not an accuracy play,
and would be its own investigation if catalog-scale EASE becomes a priority.

**Decision pending discussion (per the no-rewire instruction): keep the base as
EASE^R + wilson-cooc; do not adopt a variant for accuracy.**
