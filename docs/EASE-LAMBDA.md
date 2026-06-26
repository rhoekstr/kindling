# EASE λ — is the auto-heuristic leaving accuracy on the table?

The EASE base auto-selects its L2 regularization when `ease_lambda=None`:

```
λ_auto = 20 · n_interactions / n_items     (≈ 20× the mean Gram diagonal)
```

A RecBole comparison on a *static random split* of ml-1m suggested λ_auto was ~4×
too high (optimal there ≈ 0.2× the heuristic). This note chased whether that's a
real flaw on kindling's own (chronological) benchmark. **It isn't** — the
heuristic is well-calibrated; the chase turned into a validation of it.

## λ-optimum is protocol-dependent

| protocol | optimal λ vs heuristic | why |
|---|---|---|
| random-split (RecBole) | ≈ **0.2×** | random held-out rewards sharp item-item structure → less regularization |
| chronological (kindling) | ≈ **2.5×** | predicting the *future* rewards smoother, popularity-leaning scores → more regularization |

The optimal λ swings ~10× with the eval protocol, so **no fixed multiplier can be
right everywhere.** The heuristic sits sensibly between the two and is close to
optimal on the protocol kindling targets (chronological, predict-the-future).

## A held-out search confirms the heuristic

A leave-last-out held-out search on the train split — exactly the kind of cheap
auto-tuner one would add — independently lands on **1× the heuristic** for the
EASE base, across every proxy tried:

| held-out | metric | best λ multiplier |
|---|---|---|
| last-1 item | recall@10 / NDCG@10 | 1.0× |
| last-3 items | recall@10 / NDCG@10 | 1.0× |
| last-10% | recall@10 / NDCG@10 | 1.0× |

Every proxy *decreases* with higher λ. So for EASE itself, the heuristic is
already optimal — a search confirms it rather than improving it.

## The only headroom is a channel interaction (≈ 0.7%, likely noise)

The *full* default pipeline (EASE + trend/last-item/user-CF channels) peaks at
~2.5× the heuristic — ml-1m NDCG@10 0.2928 → 0.2949. That gain comes from how the
EASE base interacts with the channels, not from EASE in isolation, so a cheap
EASE-level held-out search can't see it. At +0.0021 NDCG over 500 eval users it's
plausibly within eval noise, and capturing it would require re-fitting the whole
pipeline per λ candidate (~n× fit time) — a poor trade for the gain.

## Decision

- **Keep the heuristic as the default.** It's EASE-optimal on kindling's protocol,
  validated independently by a held-out search.
- **Ship the search opt-in** (`ease_lambda_search=True | "auto"`, default `False`).
  It's a tool for *off-distribution* data where the heuristic might genuinely be
  off — there it would adapt λ — but defaulting it on would triple fit time only
  to re-confirm the heuristic on the data we actually have.
- **The RecBole "4× too high" was a protocol artifact**, not a calibration bug —
  see `docs/RECBOLE-COMPARISON.md`.
