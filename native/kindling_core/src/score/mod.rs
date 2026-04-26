//! Layered scorer + per-fit calibrator. See PRD §"Boost layer specification".
//!
//! `score(c) = base(c) + Σ_layer boost · I[z_layer(c) > τ]`
//!
//! Sparse layers: z over the non-zero subset of the candidate pool.
//! Dense layers (ALS, cosine, LightGCN): z over the full candidate-pool
//! distribution.
//!
//! `boost = boost_multiplier × median(adjacent gaps in base.top_K)`.
//!
//! `(τ, boost_multiplier)` are calibrated per fit by held-out lift testing.
//! Hard defaults: τ=2.5, boost_multiplier=3.0.

use pyo3::prelude::*;

pub mod calibrator;
pub mod layered;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    layered::register(m)?;
    calibrator::register(m)?;
    Ok(())
}
