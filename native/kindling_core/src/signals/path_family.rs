//! Path-family score_many kernels: tail + basket.
//!
//! The v2 PRD drops `path_full` (formerly the path-tree variant). Tail
//! and basket survive as boost layers — both are sparse signals that
//! fire only on a one-tailed z-gate above τ.
//!
//! These ports are mechanical from `kindling_native::path_family`. The
//! Python side keeps its index structure (anchors → counts dict, basket
//! posting list); these kernels handle the per-candidate gather with
//! FxHashMap'd inner loops.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Path-tail score_many. Returns `P(candidate | anchor)` for every candidate.
///
/// `row` is `counts[anchor]` as a flat `(candidate_id, weight)` vec;
/// `row_total` is the row sum (denominator for the conditional).
#[pyfunction]
fn tail_score_many<'py>(
    py: Python<'py>,
    row: Vec<(i64, f64)>,
    row_total: f64,
    candidates: Vec<i64>,
) -> Bound<'py, PyArray1<f64>> {
    let out = PyArray1::<f64>::zeros_bound(py, [candidates.len()], false);
    let out_slice = unsafe { out.as_slice_mut().unwrap() };
    if row.is_empty() || row_total <= 0.0 {
        return out;
    }
    let mut map: FxHashMap<i64, f64> =
        FxHashMap::with_capacity_and_hasher(row.len(), Default::default());
    for (cand, weight) in row {
        map.insert(cand, weight);
    }
    for (i, cid) in candidates.iter().enumerate() {
        if let Some(&w) = map.get(cid) {
            out_slice[i] = w / row_total;
        }
    }
    out
}

/// Basket score_many: weighted-coverage scoring over training observations.
///
/// CSR-like packing of baskets:
/// - `obs_start[i]`, `obs_len[i]` → slice into `obs_items` for observation i
/// - `obs_next_item[i]` → the item that followed this basket
/// - `obs_weight[i]` → observation weight (recency, frequency)
/// - `overlap_ids` → indices of observations whose basket shares ≥ 1 item
///                   with `query_items` (computed by Python posting list)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn basket_score_many<'py>(
    py: Python<'py>,
    obs_start: Vec<i64>,
    obs_len: Vec<i64>,
    obs_items: Vec<i64>,
    obs_next_item: Vec<i64>,
    obs_weight: Vec<f64>,
    overlap_ids: Vec<i64>,
    query_items: Vec<i64>,
    candidates: Vec<i64>,
) -> Bound<'py, PyArray1<f64>> {
    let out = PyArray1::<f64>::zeros_bound(py, [candidates.len()], false);
    let out_slice = unsafe { out.as_slice_mut().unwrap() };
    if overlap_ids.is_empty() || query_items.is_empty() || candidates.is_empty() {
        return out;
    }
    let query_size = query_items.len() as f64;
    let mut query_set: FxHashMap<i64, ()> =
        FxHashMap::with_capacity_and_hasher(query_items.len(), Default::default());
    for &q in &query_items {
        query_set.insert(q, ());
    }
    let mut candidate_slot: FxHashMap<i64, usize> =
        FxHashMap::with_capacity_and_hasher(candidates.len(), Default::default());
    for (slot, &cid) in candidates.iter().enumerate() {
        candidate_slot.insert(cid, slot);
    }

    let mut total_weight = 0.0_f64;
    for obs_idx in overlap_ids {
        let oi = obs_idx as usize;
        let start = obs_start[oi] as usize;
        let length = obs_len[oi] as usize;
        let basket = &obs_items[start..start + length];
        let mut overlap_count: usize = 0;
        for item in basket {
            if query_set.contains_key(item) {
                overlap_count += 1;
            }
        }
        if overlap_count == 0 {
            continue;
        }
        let sim = overlap_count as f64 / query_size;
        let weight = sim * obs_weight[oi];
        if weight <= 0.0 {
            continue;
        }
        total_weight += weight;
        let next_item = obs_next_item[oi];
        if let Some(&slot) = candidate_slot.get(&next_item) {
            out_slice[slot] += weight;
        }
    }

    if total_weight > 0.0 {
        for v in out_slice.iter_mut() {
            *v /= total_weight;
        }
    }
    out
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tail_score_many, m)?)?;
    m.add_function(wrap_pyfunction!(basket_score_many, m)?)?;
    Ok(())
}
