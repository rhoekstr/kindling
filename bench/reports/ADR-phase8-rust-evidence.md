# ADR addendum: Rust extension measured

**Status:** advisory — Rust is now an optional acceleration; v1 still ships Python-first with Rust on top where available

**Date:** 2026-04-21

## Context

The Phase 8 ADR decided "ship pure Python." A follow-up ask was to
validate that decision by actually building the Rust extension and
measuring. The ``native/`` crate now exists and ships five kernels:
``cooccurrence_signal``, ``tail_score_many`` / ``path_tree_score_many``,
``basket_score_many``, ``cosine_similarity_matrix``, and
``dedup_max_score``. Python call sites route through Rust when the
extension is present and fall back silently when it isn't.

## Measurements (same synthetic warm dataset)

| Metric           | Pure Python | Rust-accelerated | Speedup |
| ---------------- | ----------- | ---------------- | ------- |
| Fit              | 16.05 s     | 16.00 s          | 1.00×   |
| recommend p50    | 6.25 ms     | 5.66 ms          | 1.11×   |
| recommend p95    | 8.08 ms     | 7.25 ms          | 1.12×   |
| recommend p99    | 8.92 ms     | 7.53 ms          | 1.18×   |
| RSS              | 413 MB      | 414 MB           | -       |
| ``_cooccurrence_signal`` | 204 ms | 86 ms       | **2.37×** |

The kernel-level speedup is real (2.4× on the cooccurrence signal)
but the end-to-end recommend latency improves by only ~12% because:

1. Both regimes were already ~18× under the PRD target. Small absolute
   improvements on a 7-8ms budget are hard to see.
2. ``CoOccurrenceRetriever.retrieve`` (~430 ms in the profile) still
   does its own CSR row-sum + argsort in Python. Not yet ported.
3. The ``_compute_signal_features`` Python overhead (dict.get, list
   comprehensions) dominates for small candidate sets.

## Decision update

The Phase 8 decision ("ship pure Python") stands. What changes:

- **Rust extension is optional, not mandatory.** Python imports work
  without it. Wheels for Phase 11 can be shipped as sdist-only (pure
  Python) or with pre-built Rust wheels for common platforms - user
  chooses at install time.
- **Differential test suite locks the Python/Rust parity.** Five
  tests in ``tests/differential/`` verify bit-for-bit match on the
  five kernels.
- **Installing Rust is a soft requirement** for development. The
  ``maturin develop --release`` step is a one-liner but requires
  rustup. Documented in contributor guide.
- **Retained wisdom**: the 9-crate PRD plan is still overkill. One
  ``kindling-native`` crate with narrow hot-path ports is enough.

## Next actions

- Port ``CoOccurrenceRetriever.retrieve`` to Rust (cold-path argsort +
  row-sum). Expected additional ~15% on recommend p95.
- Investigate whether the ``_compute_signal_features`` overhead is
  dominated by the Python function-call structure vs the actual work;
  if yes, a single combined "score_all_signals" Rust kernel is the
  next port.
- Re-run real-data Phase 7 benchmarks with the Rust extension to see
  if the speedup compounds on larger-scale workloads.

## Honest framing

Rust was worth building as an exercise (differential tests earn their
keep; ``_cooccurrence_signal`` is now 2.4× faster) but the Phase 8
"ship pure Python" decision was correct in spirit. A v1.0 release can
ship without the Rust extension and still meet every PRD latency
target with >15× headroom. The Rust extension is a tuning knob for
users who want it, not a load-bearing component.
