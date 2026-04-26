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

pub mod als;
pub mod cooccurrence;
pub mod cosine;
pub mod interaction_network;
pub mod lightgcn;
pub mod path_family;
pub mod persona_cooccurrence;
pub mod session_cooccurrence;
// temporal_cooccurrence is build_cooccurrence with kernel="hybrid_temporal".

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    als::register(m)?;
    cooccurrence::register(m)?;
    cosine::register(m)?;
    interaction_network::register(m)?;
    lightgcn::register(m)?;
    path_family::register(m)?;
    persona_cooccurrence::register(m)?;
    session_cooccurrence::register(m)?;
    Ok(())
}
