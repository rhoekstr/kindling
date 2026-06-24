# Transition deletion map (Phase 3/5 working doc)

Generated from a pure-static import trace of the post-promotion production
surface (`engine_v2` + loaders + persist + explain + ingest + preprocess),
excluding the v1 `kindling/__init__` pull. Delete in Phase 5, through the
per-module gate (result captured ✓ · ADR retained ✓ · no production importer ✓).
Remove this file before the final PR.

## Caveats (handle in this order)
1. **Phase 3 first:** rewrite `kindling/__init__.py` to export EngineV2 as
   Engine; remove the `cold_impute` path from `engine_v2.py` (then
   `graph/cooc_impute.py` becomes deletable — it's a *current* lazy v2 dep).
2. **Keep for CI** (NOT in v2 runtime but needed by `bench/verify.py` /
   gates): `benchmarks/metrics.py`, `benchmarks/parity.py`, the
   `_load_dataset` half of `benchmarks/comparison.py`, `benchmarks/baselines.py`
   (ALS/kNN baselines, `implicit`-gated), `blend/layer_scoring.py` (holds
   `MetricReport`). Strip the `from kindling.engine import Engine` +
   `from kindling.personas import ...` module-level imports from
   `comparison.py` (move into the comparison-arm functions or delete them).
3. **Test triage:** delete the matching `tests/` files for each deleted
   module (incl. `test_temporal_interaction.py` — the 7 baseline failures).

## v2 true footprint: 57 modules KEEP.
## Deletion candidates (83):
=== DELETE CANDIDATES: 83 ===
  - kindling.benchmarks
  - kindling.benchmarks.als_ablation
  - kindling.benchmarks.baselines
  - kindling.benchmarks.clustering_coherence_sweep
  - kindling.benchmarks.coherence_percentile_sweep
  - kindling.benchmarks.comparison
  - kindling.benchmarks.cross_dataset
  - kindling.benchmarks.enrichment_probe
  - kindling.benchmarks.gap_decomposition
  - kindling.benchmarks.graph_mf_ablation
  - kindling.benchmarks.growth_curve
  - kindling.benchmarks.growth_curve_adaptive
  - kindling.benchmarks.harness
  - kindling.benchmarks.likelihood_suite
  - kindling.benchmarks.louvain_graph_variant_ablation
  - kindling.benchmarks.metrics
  - kindling.benchmarks.parity
  - kindling.benchmarks.perf
  - kindling.benchmarks.persona_diff
  - kindling.benchmarks.persona_method_ablation
  - kindling.benchmarks.prior_sensitivity
  - kindling.benchmarks.probe_engine_layered
  - kindling.benchmarks.probe_layered
  - kindling.benchmarks.probe_layered_adaptive
  - kindling.benchmarks.probe_persona_coldstart
  - kindling.benchmarks.probe_persona_cooc_stratified
  - kindling.benchmarks.probe_persona_cooc_sweep
  - kindling.benchmarks.probe_temporal_signals
  - kindling.benchmarks.profile_harness
  - kindling.benchmarks.retriever_matrix
  - kindling.benchmarks.retriever_standalone
  - kindling.benchmarks.retriever_union
  - kindling.benchmarks.scoring_architecture
  - kindling.benchmarks.signal_ablation
  - kindling.benchmarks.sweep_layered
  - kindling.benchmarks.temperature_suite
  - kindling.blend.bayesian
  - kindling.blend.diagnostics
  - kindling.blend.layer_scoring
  - kindling.blend.layered
  - kindling.blend.layered_calibrator
  - kindling.blend.normalize
  - kindling.blend.outcome_builder
  - kindling.blend.priors
  - kindling.dense_content
  - kindling.engine
  - kindling.gate
  - kindling.gate.config
  - kindling.gate.features
  - kindling.gate.fit
  - kindling.gate.model
  - kindling.graph.als_factors
  - kindling.graph.cost_graph
  - kindling.graph.item_cosine
  - kindling.graph.lightgcn
  - kindling.graph.persona_cooccurrence
  - kindling.graph.session_cooccurrence
  - kindling.graph.temporal_interaction
  - kindling.lifecycle.drift
  - kindling.llm_enrich
  - kindling.outcomes.replay
  - kindling.personas
  - kindling.personas.build
  - kindling.personas.clustering
  - kindling.personas.cold_start
  - kindling.personas.config
  - kindling.personas.index
  - kindling.personas.matching
  - kindling.profile
  - kindling.profile.plan
  - kindling.profile.profile
  - kindling.rank
  - kindling.rank.heuristic
  - kindling.rank.lightgbm_ranker
  - kindling.rank.protocol
  - kindling.rerank.calibration
  - kindling.rerank.constraints
  - kindling.rerank.lift
  - kindling.rerank.temperature
  - kindling.retrieve.interaction_neighborhood
  - kindling.retrieve.interaction_network
  - kindling.retrieve.policy
  - kindling.retrieve.signal_retrievers
