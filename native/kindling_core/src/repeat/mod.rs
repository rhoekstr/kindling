//! Repeat-consumption module (post-scoring multiplier).
//! See PRD §"Component-by-component spec → repeat/".
//!
//! - `period` — KDE-based period detection per (entity, category) pair.
//! - `pattern` — REPEAT / REPLENISH / SATIATION / ONE_SHOT classifier.
//! - `multiplier` — applies pattern-aware multiplier to top-K candidates
//!                  that match owned items.
//! - `profile` — fit-time aggregate over training data.

use pyo3::prelude::*;

// pub mod period;     // Phase 1
// pub mod pattern;    // Phase 1
// pub mod multiplier; // Phase 1
// pub mod profile;    // Phase 1

pub(crate) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
