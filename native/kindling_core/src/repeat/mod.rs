//! Repeat-consumption module (post-scoring multiplier).
//! See PRD §"Component-by-component spec → repeat/".
//!
//! - `period` — KDE-based period detection per (entity, category) pair.
//! - `pattern` — REPEAT / REPLENISH / SATIATION / ONE_SHOT classifier.
//! - `multiplier` — applies pattern-aware multiplier to top-K candidates
//!                  that match owned items.
//! - `profile` — fit-time aggregate over training data.

use pyo3::prelude::*;

pub mod multiplier;
pub mod period;
// pub mod pattern;    // Phase 1g.next — KS-distance shape classifier.
//                     // Currently fits in Python (uses scipy.stats.ks_2samp).

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    multiplier::register(m)?;
    period::register(m)?;
    Ok(())
}
