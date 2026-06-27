//! Signal builders.
//!
//! Each builder produces an item-space score vector for a candidate pool,
//! plus optional fit-time aggregate state used to compute scores at
//! recommend time. The shipped v2 stack uses the cooc family (base + wilson
//! transform), EASE, metadata-kNN, and the directional/session cooc boost
//! layers. Experimental dense signals (ALS, LightGCN, graph-MF, SVD, item
//! cosine) and the persona/path families were removed after losing to this
//! set in full-ranking evals.

use pyo3::prelude::*;

pub mod cooc_transform;
pub mod cooccurrence;
pub mod directional_cooc;
pub mod ease;
pub mod metadata_knn;
pub mod session_cooccurrence;
// temporal_cooccurrence is build_cooccurrence with kernel="hybrid_temporal".

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cooc_transform::register(m)?;
    cooccurrence::register(m)?;
    directional_cooc::register(m)?;
    ease::register(m)?;
    metadata_knn::register(m)?;
    session_cooccurrence::register(m)?;
    Ok(())
}
