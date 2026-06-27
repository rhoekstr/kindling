# Fair repeat-aware comparison (Stage 5)

**Corrects an earlier overclaim.** When the repeat module was wired, it was scored
against baselines that still *masked* seen items — so "kindling +465% on dunnhumby"
compared kindling (allowed to recommend reorders) to baselines that weren't. This
is the fair version: every model gets a repeat path, scored on the repeat-aware
eval (the full next basket, reorders included).

## Numbers (NDCG@10, include-seen eval, `bench/repeat_aware_compare.py`)

| dataset | kindling (repeat on) | personal_freq | global_pop |
|---|---:|---:|---:|
| dunnhumby | 0.260 | **0.469** | 0.249 |
| tafeng | **0.132** | 0.109 | 0.120 |
| instacart | 0.040 | **0.391** | 0.098 |
| gowalla | 0.147 | **0.248** | 0.005 |

`personal_freq` = the "buy it again" gold standard: recommend the user's own
most-frequently-bought items, ranked by train count.

## Finding

**The trivial personal-frequency baseline beats kindling's repeat module on 3 of 4
repeat datasets, often by a wide margin** (instacart 10×, dunnhumby ~1.8×, gowalla
~1.7×). kindling wins only tafeng, the lowest-repeat of the four.

Root cause: kindling's repeat module re-surfaces reorders but ranks them by
**base co-occurrence affinity × REPLENISH timing multiplier** — it never uses
**reorder frequency**, which is the dominant signal on high-repeat logs (how often
you buy milk predicts the next basket far better than what milk co-occurs with).
The per-(user, item) count *is* computed in the fit profile
(`fit_repeat_profile`), but the recommend path discards it.

## Implication

The repeat module as shipped is **not production-grade for grocery/replenishment**.
It works (re-surfaces reorders, beats masked baselines) but loses to the obvious
frequency baseline. To be competitive it must rank reorders primarily by
**reorder frequency** (the count already in the profile), with the timing
multiplier as a *modulation* (suppress just-bought, surface due) rather than the
co-occurrence affinity as the base. A frequency-driven repeat score would likely
match or beat personal_freq (it adds timing on top) — but that is a **rewire**,
left for discussion per the overnight scope.

**Net:** the repeat wiring was a correct first step that exposed the real
requirement. The headline repeat numbers should be read as "reorders now surface,"
not "kindling beats the repeat baselines" — it does not yet.
