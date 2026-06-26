# kindling: a recommender that grows with your data — and what building it taught me

I'm open-sourcing **[kindling](https://github.com/rhoekstr/kindling)**, a hybrid
recommender with no training loop, no GPU, and a Rust core that serves
recommendations in well under a millisecond. This post is half announcement,
half field notes — because the most useful thing I can hand you alongside the
code is an honest account of how it got here.

## What it is

kindling builds **one fused score per (user, item)** from a closed-form base
(EASE for catalogs up to ~20k items, wilson-normalized co-occurrence above) plus
a handful of z-normalized counting-statistic channels — recent-trend, last-item,
sequential transitions, user–user CF — each of which **turns itself on only when
the data warrants it**. Timestamps activate trend; sparse histories activate
user-CF; true ratings activate rating-weighting. There is no config to tune your
way into a good model; the model configures itself from the data at `fit()`
time.

Across four public datasets it's the strongest *personalized* model — it beats
implicit ALS everywhere, and on cold-heavy catalogs it wins the cold-*user*
buckets outright. Here's how accuracy grows from cold to hot data against the
standard baselines:

![growth curves](../bench/reports/growth_curves_grid.png)

The pink line is kindling. Notice the left edge of the Steam and MovieLens NDCG
panels: at the very coldest data, plain popularity is competitive — and kindling
*knows* to lean on that prior, then pulls away as signal accumulates. That
crossover isn't an accident; it's the whole design thesis in one picture.

## ~Two months, and more dead ends than wins

I started this repo on April 20th and it's about 170 commits later. If you scan
the git log expecting a clean march toward the architecture above, you'll be
disappointed — and that's the honest part I want to share. The log is a lab
notebook, and most of the entries are experiments that **didn't work**:

- An **embedding-imputation program** to fill cold items from content — wired
  across three phases, with a passing synthetic control — that never beat the
  plain stack in any real regime. Closed, negative.
- A **force-directed projection** retrieval idea that landed *below the
  popularity floor* on top-K accuracy. Dead.
- **Metadata→co-occurrence grafting**, alive-but-marginal on rich catalogs and
  dead on thin ones — a niche tool, not the universal cold-start fix I'd hoped.

I count eight substantial experiment threads, thirteen benchmark scripts, and a
shelf of saved repros. The ratio of negative to positive results was not close.
And I've come to think that ratio is the *value*, not a tax on it: every closed
door is a door the next idea doesn't have to knock on. The discipline that made
it cheap was writing the repro, running it, and **recording the verdict** — so
"we lacked the data to test this" never quietly became "this doesn't work."

The thing that actually moved the needle wasn't a clever model. It was a
**diagnostic**: decomposing the error into *retrieval-bound* (the right item
never made the candidate pool) versus *ranking-bound* (it was there but
mis-ranked). That single lens showed raw co-occurrence collapsing toward a
popularity ranking — which is what pivoted the base to EASE, and told me, per
dataset, whether to spend effort on retrieval or on ranking. Diagnose where the
ceiling is *before* trying to raise it.

## Then I rewrote the hot path in Rust — without breaking anything

The last stretch was porting the entire recommend path to a Rust core. I did it
**parity-first**: the rule was to reproduce the existing engine *exactly* — the
four reference NDCG numbers and byte-for-byte recommendation lists were the gate
— before changing a single behavior. Every phase had a differential harness
comparing Rust against the Python reference.

That discipline bought something I didn't fully appreciate until the end:
**confidence to delete.** Once parity was the contract, I could rip out ~2,000
lines — the whole Python recommend path and an orphaned feature package — and
trust the gate to catch any drift. The engine is now native-only on the hot
path; recommendation is pure Rust end to end, single calls went from ~200 ms to
**sub-millisecond**, and a batch path runs in parallel with the GIL released.
And because the port was byte-faithful, *every accuracy result I'd already
earned still held* — the months of benchmarking didn't need redoing.

The most instructive bug of the whole project was, of all things,
**tie-breaking**. The user-CF channel ranks neighbors by a similarity that's
highly discrete, so dozens of them tie *exactly* at the cutoff. numpy leaves the
order among ties unspecified — and nothing in Rust can reproduce an unspecified
order. The fix was to make the tie-break deterministic on both sides (a stable
secondary key), which is reproducible and NDCG-neutral. What's left — a ~1e-7
disagreement from pairwise-vs-sequential floating-point summation that can flip
two exactly-tied items — I chose to *accept* rather than chase. Knowing which
differences to fix and which to live with turned out to be most of the work.

## Why it might be useful to you

If you want personalized recommendations without standing up a GPU, a feature
store, or a training pipeline — `pip install`, `fit`, `recommend`, and you have
a model that adapts its own behavior to your data's shape and serves in
microseconds. It persists to a self-contained artifact you can load in a serving
process with no re-fit; there's a `KindlingServer` class and a small FastAPI
example to make that a five-minute job.

The full benchmark record — including all the negative results, which really are
half the value — is in [`docs/EXPERIMENTS.md`](EXPERIMENTS.md), and the
synthesized takeaways are in [`docs/LESSONS.md`](LESSONS.md).

Kindling, in the end, is a small bet: that closed-form models, gated honestly
per dataset and made fast in Rust, can go a long way before you reach for
anything heavier. Two months of mostly-failed experiments is the evidence I have
that it's a bet worth making. The code is on GitHub — I'd love to hear where it
breaks for you.
