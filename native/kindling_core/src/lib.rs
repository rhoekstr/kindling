//! kindling_core — Rust core for kindling v2.
//!
//! Owns the algorithmic surface of the v2 architecture:
//!
//! - `signals/` — cooc, persona_cooc, session_cooc, temporal_cooc,
//!                path_tail, path_basket, interaction_network, ALS,
//!                cosine, lightgcn.
//! - `cluster/` — HDBSCAN over ALS factors (Phase 2).
//! - `persona/` — rate aggregation, z-filter, TF-IDF, L2, fit-percent gate.
//! - `score/`   — layered base + z-gated boosts + per-fit calibrator.
//! - `retrieve/`— cooc + path-endpoint retrievers, RRF fusion.
//! - `repeat/`  — period detection, 4-pattern classifier, multiplier.
//! - `loaders/` — Polars-backed dataset readers (Phase 4, feature-gated).
//!
//! See `/Users/rhoekstr/.claude/plans/read-this-prd-ponder-fluffy-turing.md`
//! for the full PRD that motivates this crate.

use pyo3::prelude::*;

pub mod cluster;
pub mod persona;
pub mod repeat;
pub mod retrieve;
pub mod score;
pub mod signals;

#[cfg(feature = "loaders")]
pub mod loaders;

/// Module entry point. Each submodule registers its functions here.
#[pymodule]
fn kindling_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase markers — every submodule's `register` is a no-op until the
    // corresponding phase lands. This keeps the crate buildable from
    // Phase 0 onward.
    signals::register(m)?;
    cluster::register(m)?;
    persona::register(m)?;
    score::register(m)?;
    retrieve::register(m)?;
    repeat::register(m)?;
    Ok(())
}
