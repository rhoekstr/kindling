//! Density-based clustering. See PRD §"Component-by-component spec → cluster/".
//!
//! Phase 2 wires `petal-clustering` HDBSCAN over ALS user factors. The
//! ALS factors *are* the dimensionality reduction — no UMAP detour, no
//! ABI break with numpy.
//!
//! Output of `hdbscan::fit`:
//! - `assignments: Vec<i64>` with -1 for noise points
//! - `probabilities: Vec<f64>` membership confidence
//! - `n_clusters: usize`
//!
//! Until Phase 2 lands, `register` is a no-op.

use pyo3::prelude::*;

pub mod hdbscan;
pub mod louvain;
pub mod user_user_graph;
pub mod dc_sbm;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    hdbscan::register(m)?;
    louvain::register(m)?;
    user_user_graph::register(m)?;
    dc_sbm::register(m)?;
    Ok(())
}
