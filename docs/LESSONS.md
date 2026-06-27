# Lessons learned

What ~two months and 170+ commits of building kindling actually taught — the
methods that paid off and the ideas that didn't, written down so the next
project (or the next person) doesn't re-tread them. The blow-by-blow benchmark
record lives in [`EXPERIMENTS.md`](EXPERIMENTS.md); this is the synthesis.

## 1. The negative results were the deliverable

More experiments closed *negative* than positive, and that was the point. Three
stand out:

- **Embedding imputation (a 3-phase program, all negative).** Content embeddings
  to fill cold items looked obviously right and was wired end to end. A bench
  positive control passed; it still didn't transfer (an EASE scale mismatch),
  and the deeper versions (low-rank EASE-beyond-gate, demand-aware extension)
  never beat the plain stack. Verdict: content is a *structurally weak*
  supplement in the regimes we had — no production value. Closing it explicitly
  was worth more than leaving it as a maybe.
- **Force-directed / projection layouts (FPR).** A geometric retrieval idea that
  was dead on top-K accuracy — below the popularity floor. Knowing it was dead
  stopped a whole PRD.
- **Edge grafting (metadata → cooc).** Alive-but-marginal on rich-metadata
  catalogs, dead on thin ones. A tool, not a universal fix.

The habit that made this cheap: write the repro, run it, and **record the
verdict — including the cost of not having the right data to test an idea**, so
"we couldn't test it" is never confused with "it doesn't work."

## 2. Diagnose where the ceiling is before trying to raise it

The single highest-leverage tool was **gap decomposition** — splitting the loss
into *retrieval-bound* (the right item isn't in the candidate pool) vs
*ranking-bound* (it's in the pool but mis-ranked). It showed that raw
co-occurrence scoring was degenerating toward a popularity ranking, which drove
the pivot to **EASE** (a closed-form inverse-Gram reweighting that subtracts the
popularity/redundancy structure). It also told us *per dataset* whether to
attack retrieval or ranking — so effort went where the headroom was.

## 3. Closed-form, gated per regime, beat speculative complexity

Every surviving piece is closed-form or a counting statistic: EASE / wilson
co-occurrence base, z-normalized channels (trend, last-item, transitions,
user-CF), a content-gated cold channel, reserved cold slots. Every channel is
**activated by a measurable property of the data** (timestamps → trend;
not-a-rating-burst → transitions; sparse history → user-CF; true ratings →
rating-weighting), and **every gate exists because the ungated version
measurably hurt somewhere**. No training loop, no GPU, no speculative deep model
ever earned its keep against this.

## 4. Parity-first is how you port without fear

The Rust port (fit-state and the entire recommend path) was done **parity-first**:
reproduce the existing engine *exactly* — the four reference NDCG numbers and
byte-level rec lists were the gate — before optimizing or redesigning anything.
Each phase had a differential harness (`bench/rust_parity.py`,
`bench/native_recommend_parity.py`). Because parity was the contract, the final
**native-only** step could *delete* ~2000 lines (the whole Python recommend path
+ an orphaned path-family package) with confidence: the gate caught any drift.

The corollary that makes parity-first worth the discipline: **the port preserved
accuracy exactly, so every prior accuracy result still holds.** The growth
curves measured against the Python engine are valid for the Rust engine
unchanged — the months of accuracy work were protected, and only the speed
changed (single recommend ~200 ms → sub-millisecond; batch parallel in Rust).

## 5. Determinism is a portability problem, not a numerics problem

The hardest parity bug was not arithmetic — it was **tie-breaking**. The
user-CF channel selects top-k neighbors by a similarity that is *highly
discrete* (small integer overlap counts), so many neighbors tie exactly at the
boundary. numpy's `argpartition` / `argsort` leave the order among ties
*unspecified*, and that order is not reproducible by any Rust selection. The
fix was to make the tie-break **deterministic on both sides** — a stable
secondary key (similarity desc, then ascending index) — which is NDCG-neutral
and byte-reproducible. Same lesson hit the final top-N and the cold-slot ranker.

What remains is genuinely irreducible: **floating-point summation order**. numpy
reduces with pairwise summation; a naive Rust loop reduces sequentially; the two
differ by ~1e-7, which is enough to flip the order of two items whose scores are
*exactly* tied after the boost. We accepted this — it is NDCG-identical and
ranking-stable — rather than chase numpy's summation internals. Knowing which
differences to fix (unspecified tie order) and which to accept (FP summation) is
the whole game.

## 6. Know your box

`amazon-book-chrono` (~18 GB working set) swap-thrashed for over an hour and
never cleanly finished on the development machine. The cooc-fused *code path*
that serves it was validated by forcing the cooc base on a small dataset
(byte-identical to the Python reference) — the algorithm is proven; the literal
at-scale number just needs more RAM. Validate the *path* on something that
fits; don't let a memory wall block a correctness claim.

## 7. Small meta-habits that compounded

- **A wheel that imports is a wheel that works.** numpy/pandas/scipy only; the
  linear algebra that matters runs on a pure-Rust core with no BLAS/system deps.
  This constraint killed a lot of otherwise-tempting dependencies.
- **One fact per commit, with the verdict in the message.** The git log is a
  usable lab notebook because each commit says what was tested and what
  happened — positive or negative.
- **Re-baseline deliberately, document the delta.** When the deterministic
  tie-break shifted steam 0.0660 → 0.0659, that one digit was re-baselined in
  `gates.toml` with a note explaining why — so the next person doesn't read it
  as a regression.

## 8. A held-out gate must mirror the benchmark's split protocol

The repeat module had to auto-decide which datasets it helps. The first gate
held out each user's most-recent items and confidently *kept* Steam — where the
module actually hurts. The benchmark splits **chronologically and globally**
(most-recent fraction by wall-clock time); the per-user-recent held-out
over-represented re-logged repeats and answered a different question than the one
being scored. Rebuilding the gate around the *same* global-time split — leak-free
(profile rebuilt on held-out history) and faithful (real timestamps, so the
timing logic behaves as at serve time) — made it decline Steam and keep the
genuine grocery logs. A held-out that doesn't match the eval doesn't approximate
the answer; it answers a different question. See [REPEAT-GATE.md](REPEAT-GATE.md).

## 9. "Fake repeat" — re-logging is not repurchase

Steam has a *higher* duplicate-interaction rate than Ta-Feng, yet recommending
repeats helps Ta-Feng (+47%) and hurts Steam (−4%) even when reorders are
credited. A duplicate (user, item) on a grocery log is intent to re-buy; on a
game log it's re-opening something you already own. The discriminator isn't *how
much* users repeat but *whether past repeats predict future ones* — which only a
held-out test (Lesson 8), not a count, can answer.

## 10. A non-universal win is better offered than imposed

EASE+ (EDLAE denoising) beats plain EASE on three datasets and loses on a fourth,
and a held-out search couldn't reliably pick the right δ per dataset. The
disciplined move was to ship it **opt-in** with the default unchanged, rather
than adopt a "win" that silently regresses one reference number. When an upgrade
isn't universal and you can't cheaply detect where it applies, a flag beats a
default.
