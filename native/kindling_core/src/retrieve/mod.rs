//! Candidate generation. See PRD §"Component-by-component spec → retrieve/".
//!
//! v2 ships only two retrievers: cooc and path_endpoint. Both feed into
//! a reciprocal-rank fusion of candidate pools per recommend call.
//! No ALS / cosine / lightgcn / path_tail / path_full retrievers — those
//! signals contribute as boost layers only, not as candidate generators.

use pyo3::prelude::*;

// pub mod cooc;          // Phase 1 (port from kindling_native::cooccurrence)
// pub mod path_endpoint; // Phase 1
// pub mod rrf;           // Phase 1

pub(crate) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
