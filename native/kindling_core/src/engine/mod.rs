//! Engine orchestration ported to Rust (fit channels, state, recommend).
//!
//! Parity-first: each piece byte-matches the Python engine (`bench/rust_parity.py`).
//! The split that keeps parity cheap — Python retains ingest / preprocess /
//! activation-plan (resolve config); Rust takes the resolved arrays + config and
//! builds the fit state (and, later, scores). Base build (cooc / EASE / cosine /
//! metadata-kNN / cooc-transform) already lives in `signals`.

use pyo3::prelude::*;

pub mod channels;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    channels::register(m)?;
    Ok(())
}
