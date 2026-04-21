//! Cooccurrence signal: sum of adjacency rows across the owned set,
//! then gather candidate values.
//!
//! The Python implementation round-trips through scipy.sparse for the
//! row-sum and gather. The scipy ops are already C, but the per-
//! candidate dict lookup plus the intermediate np.asarray overhead
//! adds up to ~200ms in the warm regime. This version skips the
//! intermediate array and does the gather in one pass.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute ``adjacency[owned_indices].sum(axis=0)[candidate_indices]``
/// in one pass over the selected rows.
///
/// Inputs are the components of a scipy CSR matrix plus the list of
/// row indices (owned items, internal indices) and column indices
/// (candidate items, internal indices). Returns a float64 array of
/// length ``candidate_indices.len()``.
#[pyfunction]
fn cooccurrence_signal<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    owned_indices: Vec<usize>,
    candidate_indices: Vec<i64>,
) -> Bound<'py, PyArray1<f64>> {
    let data = data.as_slice().expect("data must be contiguous");
    let indices = indices.as_slice().expect("indices must be contiguous");
    let indptr = indptr.as_slice().expect("indptr must be contiguous");

    let n_cols = candidate_indices.len();
    let mut summed = vec![0.0_f64; indptr.len().saturating_sub(1)];
    // Sum adjacency rows for every owned index.
    for &row in &owned_indices {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for k in start..end {
            let col = indices[k] as usize;
            summed[col] += data[k] as f64;
        }
    }

    // Gather the candidate columns. candidate_indices < 0 means "not
    // in the item graph" - the caller already signals that, so just
    // emit 0 for those slots.
    let out = PyArray1::<f64>::zeros_bound(py, [n_cols], false);
    let out_slice = unsafe { out.as_slice_mut().unwrap() };
    for (i, &cid) in candidate_indices.iter().enumerate() {
        if cid >= 0 {
            let idx = cid as usize;
            if idx < summed.len() {
                out_slice[i] = summed[idx];
            }
        }
    }
    out
}

/// Full retriever kernel: row-sum + exclude-owned + partial top-k.
/// Returns two parallel vecs: (item_indices, scores) sorted descending
/// by score, length <= budget, with score > 0 and not in owned_indices.
#[pyfunction]
fn cooccurrence_retrieve(
    data: PyReadonlyArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, i32>,
    indptr: PyReadonlyArray1<'_, i32>,
    owned_indices: Vec<usize>,
    budget: usize,
) -> (Vec<i64>, Vec<f64>) {
    if budget == 0 {
        return (Vec::new(), Vec::new());
    }
    let data = data.as_slice().expect("data contiguous");
    let indices = indices.as_slice().expect("indices contiguous");
    let indptr = indptr.as_slice().expect("indptr contiguous");

    let n_items = indptr.len().saturating_sub(1);
    let mut summed = vec![0.0_f64; n_items];
    for &row in &owned_indices {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for k in start..end {
            let col = indices[k] as usize;
            summed[col] += data[k] as f64;
        }
    }
    // Zero out owned positions so the entity never recommends its own items.
    for &row in &owned_indices {
        summed[row] = 0.0;
    }

    // Collect (score, idx) for positives, then do a partial sort up
    // to ``budget``. For moderate budgets this is faster than a full
    // argsort over n_items. Tie-break on idx for determinism.
    let effective_budget = budget.min(n_items);
    let mut positives: Vec<(f64, usize)> = summed
        .iter()
        .enumerate()
        .filter_map(|(idx, &s)| if s > 0.0 { Some((s, idx)) } else { None })
        .collect();
    let take = effective_budget.min(positives.len());
    if take < positives.len() {
        positives.select_nth_unstable_by(take - 1, |a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        positives.truncate(take);
    }
    positives.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let ids: Vec<i64> = positives.iter().map(|(_, i)| *i as i64).collect();
    let scores: Vec<f64> = positives.into_iter().map(|(s, _)| s).collect();
    (ids, scores)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cooccurrence_signal, m)?)?;
    m.add_function(wrap_pyfunction!(cooccurrence_retrieve, m)?)?;
    Ok(())
}
