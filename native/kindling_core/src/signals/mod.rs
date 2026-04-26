//! Signal builders. See PRD §"Component-by-component spec → signals/".
//!
//! Each builder produces an item-space score vector for a candidate pool,
//! plus optional fit-time aggregate state used to compute scores at
//! recommend time.
//!
//! Phase 1 ports the eight boost-layer signals plus cooc and persona_cooc
//! into this module. Until then, `register` is a no-op so the crate
//! continues to build.

use pyo3::prelude::*;

pub mod cooccurrence;
pub mod path_family;
pub mod persona_cooccurrence;
// pub mod session_cooccurrence;  // Phase 1d
// pub mod temporal_cooccurrence; // Phase 1d
// pub mod interaction_network;   // Phase 1f
// pub mod als_factors;           // Phase 1f (linfa)
// pub mod cosine;                // Phase 1f (linfa)
// pub mod lightgcn;              // Phase 1f (hand-rolled)

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cooccurrence::register(m)?;
    path_family::register(m)?;
    persona_cooccurrence::register(m)?;
    Ok(())
}
