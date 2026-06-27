//! kindling_core — Rust core for kindling.
//!
//! Owns the algorithmic surface of the shipped engine:
//!
//! - `signals/` — cooc (base + wilson transform), directional/session cooc
//!                boost layers, EASE, metadata-kNN.
//! - `engine/`  — native fit channels, EngineState, recommend (+ batch).
//! - `score/`   — layered base + z-gated boosts + per-fit calibrator.
//! - `repeat/`  — period detection + replenishment multiplier + profile.
//! - `loaders/` — Polars-backed dataset readers (feature-gated).

use pyo3::prelude::*;

pub mod engine;
pub mod repeat;
pub mod score;
pub mod signals;

#[cfg(feature = "loaders")]
pub mod loaders;

/// Module entry point. Each submodule registers its functions here.
/// Exposed to Python as `kindling._core` (packaged inside the wheel); the
/// Rust fn keeps its descriptive name via the pyo3 `name` override.
#[pymodule]
#[pyo3(name = "_core")]
fn kindling_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    signals::register(m)?;
    engine::register(m)?;
    score::register(m)?;
    repeat::register(m)?;
    Ok(())
}
