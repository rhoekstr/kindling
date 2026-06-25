# Changelog

All notable changes to **kindling** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — first packaged release

The transition from an experiment harness to a packaged, installable, served
recommender. Earlier `0.0.x`/`0.1.x` work lived in the repository only.

### Added
- **Eval harness** (`kindling.harness`, `kindling bench`): the realistic-tier
  benchmark — chronological split, full-catalog ranking, sliced by user
  history length — packaged for reuse. Point it at a built-in dataset or your
  own interaction-log CSV (with column-alias normalization) and get
  per-warmth-bucket NDCG / Recall / MRR / HR for kindling alongside
  `popularity` / `item-kNN` / `ALS` / `BPR`. Trained baselines gate on the
  optional `implicit` library and degrade gracefully (skipped-with-reason).
- **Serving harness** (`kindling.serving`, `kindling serve`): `create_app()`
  turns a fitted or saved `Engine` into a FastAPI service — `/recommend`
  (known users, with int/str id resolution), `/recommend_for_items` (new /
  anonymous users with popularity fallback), `/recommend/batch`, `/health`.
  FastAPI + uvicorn are an optional `serve` extra, imported lazily.
- **`kindling` console command** with `bench` / `fit` / `serve` / `version`.
- **Persistence**: `Engine.save()` / `Engine.load()` (versioned header).
- **Release pipeline**: `release.yml` builds wheels (Linux x86_64/aarch64,
  macOS arm64/x86_64, Windows) + sdist and publishes to PyPI via Trusted
  Publishing on a version tag. Post-install end-to-end smoke (`scripts/smoke.py`).
- **Real-world validation** vs published GNNs (yelp2018) and trained models on
  production data (RetailRocket, H&M); see `docs/VALIDATION.md`.

### Changed
- **The engine is now a single validated stack**: wilson-normalized
  cooccurrence base + EASE + z-normalized auto-gated channels (trend,
  last-item, transitions, user-CF) + cold-slots / open-catalog + popularity
  fallback, governed by a deterministic `ActivationPlan`.
- Packaging is a maturin mixed Python/Rust build: `pip install kindling`
  ships the pure-Python package **and** the `kindling._core` Rust extension in
  one abi3-py311 wheel — no PyTorch, no system BLAS.
- Metadata moved to `Development Status :: 4 - Beta`; added keywords, audience
  classifiers, and Documentation / Source / Changelog URLs.

### Removed
- The retired **persona-clustering / ALS / graph-MF / per-fit calibration**
  subsystem (off-by-default, proven inert) — `engine.py` shed ~1,100 lines
  with no change to any reference metric.
- The vestigial `v2` naming (`engine_v2` → `engine`, `EngineV2` → `Engine`).
- The dead `personas` install extra (umap-learn / hdbscan / scikit-learn).

### Reference metrics (full-ranking NDCG@10, engine defaults)
movielens-1m **0.293** · amazon-beauty **0.033** · steam (realistic tier)
**0.066** · amazon-book-chrono **0.032**. Strongest personalized model on all
four; beats `implicit` ALS everywhere; wins cold-*user* buckets on cold-heavy
catalogs.

[Unreleased]: https://github.com/rhoekstr/kindling/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rhoekstr/kindling/releases/tag/v0.2.0
