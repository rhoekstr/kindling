//! Dataset loaders (feature-gated; Phase 4).
//! See PRD §"Data loaders — one-line dataset access".
//!
//! Each loader reads CSV / JSON / Parquet via Polars and returns the
//! canonical interaction schema:
//!   entity_id, item_id, [timestamp], [rating], [session_id], [category]
//!
//! Cache layer: parsed datasets persist as Arrow IPC under
//! `~/.kindling/datasets/<loader>/<version>/`. Re-loads are mmap, not
//! re-parse.
//!
//! Loaders to ship (PRD table):
//!   movielens (1m/10m/25m), amazon (beauty/books/...), gowalla, yelp,
//!   tafeng, dunnhumby, instacart, retailrocket, synthetic.

use pyo3::prelude::*;

// pub mod movielens;
// pub mod amazon;
// pub mod gowalla;
// pub mod yelp;
// pub mod tafeng;
// pub mod dunnhumby;
// pub mod instacart;
// pub mod retailrocket;
// pub mod synthetic;
// pub mod cache;

#[allow(dead_code)]
pub(crate) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
