//! kindling-native: Rust implementations of the measured hot paths.
//!
//! Phase 8 profile data identified cooccurrence (~60%) and path-family
//! (~15%) as the warm-regime bottlenecks. This crate ports those paths
//! plus DPP kernel construction and dedup to Rust. The Python side
//! gracefully falls back to the pure-Python implementations when this
//! extension isn't built.

use pyo3::prelude::*;

mod cooccurrence;
mod dedup;
mod dpp_kernel;
mod path_family;
mod personas;

/// Module entry point. Each submodule registers its functions here.
#[pymodule]
fn kindling_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cooccurrence::register(m)?;
    path_family::register(m)?;
    dpp_kernel::register(m)?;
    dedup::register(m)?;
    personas::register(m)?;
    Ok(())
}
