//! Path-family score_many kernels (tail, full, basket).
//!
//! The Python ``score_many`` methods do dict lookups in a Python loop -
//! the dict contents are small ints/str, so Python overhead dominates.
//! Rust's ``FxHashMap`` + stack-allocated inner loops eliminate that
//! overhead.
//!
//! The Python side converts the nested dict into ``(anchor, {inner
//! map})`` tuples + a row_totals dict before calling these functions.
//! For the transient candidate vector we accept owned Vec<i64> to
//! avoid PyArray overhead on the small (N_candidates) inputs.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Tail index: for a single anchor, return ``P(candidate | anchor)`` for
/// every candidate.
///
/// ``row`` is ``counts[anchor]`` as a flat (candidate_id, weight) vec.
/// ``row_total`` is ``row_totals[anchor]``.
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
    let mut map: FxHashMap<i64, f64> = FxHashMap::with_capacity_and_hasher(
        row.len(),
        Default::default(),
    );
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

/// Path tree: score against the single matching prefix's successor row.
/// Same structure as tail_score_many but called with the pre-resolved
/// (prefix-match) row so the Rust side doesn't have to walk the trie.
#[pyfunction]
fn path_tree_score_many<'py>(
    py: Python<'py>,
    row: Vec<(i64, f64)>,
    row_total: f64,
    candidates: Vec<i64>,
) -> Bound<'py, PyArray1<f64>> {
    // Same implementation as tail. Duplicated here for API clarity and
    // so the Python side can keep its back-off lookup in Python while
    // delegating only the per-candidate probability gather.
    tail_score_many(py, row, row_total, candidates)
}

/// Basket score_many: weighted-coverage scoring over training
/// observations that share any item with the query basket.
///
/// Inputs encode the basket index as "parallel arrays" to keep the
/// FFI simple:
/// - ``observations_basket_start`` / ``observations_basket_len`` /
///   ``observations_basket_items`` = CSR-like packing of baskets
/// - ``observations_next_item`` = one entry per observation
/// - ``observations_weight`` = one entry per observation
/// - ``overlap_ids`` = the set of observation indices that overlap the
///   query (already computed by the Python posting-list union)
/// - ``query_size`` = |query basket| (for coverage similarity)
/// - ``candidate_to_slot`` = mapping from item id to output slot
///   (candidates not tracked get slot -1)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn basket_score_many<'py>(
    py: Python<'py>,
    observations_basket_start: Vec<i64>,
    observations_basket_len: Vec<i64>,
    observations_basket_items: Vec<i64>,
    observations_next_item: Vec<i64>,
    observations_weight: Vec<f64>,
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
        let start = observations_basket_start[oi] as usize;
        let length = observations_basket_len[oi] as usize;
        let basket = &observations_basket_items[start..start + length];
        // Coverage similarity: |Q ∩ B| / |Q|.
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
        let weight = sim * observations_weight[oi];
        if weight <= 0.0 {
            continue;
        }
        total_weight += weight;
        let next_item = observations_next_item[oi];
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tail_score_many, m)?)?;
    m.add_function(wrap_pyfunction!(path_tree_score_many, m)?)?;
    m.add_function(wrap_pyfunction!(basket_score_many, m)?)?;
    Ok(())
}
