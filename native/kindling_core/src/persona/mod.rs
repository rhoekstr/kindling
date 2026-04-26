//! Persona pipeline. See PRD §"Component-by-component spec → persona/"
//! and §"Persona-fit gate spec".
//!
//! - `index` — rate aggregation (port of legacy `personas.rs`),
//!             z-filter for "distinctive items", TF-IDF, L2.
//! - `fit_gate` — secondary gate for adaptive base routing.
//!   `fit = |user.items ∩ distinctive_items[persona]| / |user.items|`.
//!   Use persona base iff `cluster != -1 && fit >= 0.70`.

use pyo3::prelude::*;

pub mod index;
pub mod fit_gate;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    index::register(m)?;
    fit_gate::register(m)?;
    Ok(())
}
