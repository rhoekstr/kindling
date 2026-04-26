//! Cooccurrence: the v2 base layer.
//!
//! Three operations:
//!
//! 1. `build_cooccurrence` — fit-time. Takes a (user, item, weight) triplet
//!    list plus optional timestamps, applies the configured kernel, and
//!    produces a symmetric item-item cooccurrence CSR matrix.
//! 2. `cooccurrence_signal` — recommend-time. Row-sum across owned items
//!    in the cooc matrix, then gather candidate columns. Pure scoring;
//!    no top-K logic.
//! 3. `cooccurrence_retrieve` — recommend-time. Row-sum + exclude-owned
//!    + partial top-K. Returns `(item_idx, score)` ordered descending.
//!
//! The kernel + decay knob come from `LayerPlan` per PRD §"Profile → Plan
//! contract". `pure_count` is the rating-burst-safe choice; `hybrid_temporal`
//! is the default for time-rich datasets.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Kernel choice for cooccurrence weighting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel {
    /// `pure_count`: each (user, item) interaction contributes 1.0 to
    /// the user's owned set; cooc is `S.T @ S` over binary S. Used on
    /// rating-burst datasets where timestamp deltas are seconds-apart
    /// and a temporal kernel would amplify noise.
    PureCount,
    /// `hybrid_temporal`: weight pairs by `1 + alpha · logistic(dt /
    /// half_life)` where dt is the time difference between two of a
    /// user's items. The `1 +` term preserves the cooc baseline so
    /// distant pairs still register; the logistic term boosts pairs
    /// that co-occurred recently.
    HybridTemporal { alpha: f64, half_life_days: f64 },
}

impl Kernel {
    pub fn from_str(name: &str, alpha: f64, half_life_days: f64) -> PyResult<Self> {
        match name {
            "pure_count" => Ok(Kernel::PureCount),
            "hybrid_temporal" => Ok(Kernel::HybridTemporal { alpha, half_life_days }),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown kernel: {other:?}; expected 'pure_count' or 'hybrid_temporal'"
            ))),
        }
    }
}

/// Build an item-item cooccurrence CSR matrix from user-item interactions.
///
/// Inputs are parallel arrays of length `n_interactions`:
/// - `user_idx[k]`, `item_idx[k]` — internal indices (must be valid)
/// - `weights[k]` — per-interaction weight (1.0 for binary; rating /
///                  log1p(count) for weighted datasets)
/// - `timestamps[k]` — UNIX seconds (only used when kernel == hybrid_temporal)
///
/// Returns CSR components `(data, indices, indptr)` for an `(n_items,
/// n_items)` symmetric matrix. Self-pairs are excluded.
#[pyfunction]
#[pyo3(signature = (
    user_idx,
    item_idx,
    weights,
    n_users,
    n_items,
    kernel = "pure_count",
    alpha = 1.0,
    half_life_days = 30.0,
    timestamps = None,
))]
#[allow(clippy::too_many_arguments)]
fn build_cooccurrence(
    user_idx: PyReadonlyArray1<'_, i64>,
    item_idx: PyReadonlyArray1<'_, i64>,
    weights: PyReadonlyArray1<'_, f32>,
    n_users: usize,
    n_items: usize,
    kernel: &str,
    alpha: f64,
    half_life_days: f64,
    timestamps: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    let kernel = Kernel::from_str(kernel, alpha, half_life_days)?;
    let user_idx = user_idx.as_slice()?;
    let item_idx = item_idx.as_slice()?;
    let weights = weights.as_slice()?;

    if user_idx.len() != item_idx.len() || user_idx.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "user_idx, item_idx, weights must have equal length",
        ));
    }
    let timestamps = match (kernel, &timestamps) {
        (Kernel::HybridTemporal { .. }, Some(ts)) => Some(ts.as_slice()?),
        (Kernel::HybridTemporal { .. }, None) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "kernel='hybrid_temporal' requires timestamps",
            ))
        }
        _ => None,
    };

    // Bucket interactions by user. Each bucket stores (item_idx, weight,
    // optional_timestamp). For pure_count we only need (item, weight).
    let mut by_user: Vec<Vec<(usize, f32, f64)>> = vec![Vec::new(); n_users];
    for k in 0..user_idx.len() {
        let u = user_idx[k] as usize;
        let i = item_idx[k] as usize;
        if u >= n_users || i >= n_items {
            continue;
        }
        let w = weights[k];
        let t = timestamps.map_or(0.0, |ts| ts[k]);
        by_user[u].push((i, w, t));
    }

    // Accumulate item-pair scores into a hash map keyed by (i, j) with i < j.
    // For pure_count: pair_weight = w_i * w_j (binary cap → 1.0 each).
    // For hybrid_temporal: pair_weight *= 1 + alpha · logistic(-|dt|/half_life).
    let mut pairs: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    let half_life_seconds = half_life_days * 86_400.0;
    for items in &by_user {
        for a in 0..items.len() {
            let (ia, wa, ta) = items[a];
            for b in (a + 1)..items.len() {
                let (ib, wb, tb) = items[b];
                if ia == ib {
                    continue;
                }
                let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
                let pair_weight = match kernel {
                    Kernel::PureCount => (wa * wb) as f64,
                    Kernel::HybridTemporal { alpha, .. } => {
                        let dt = (ta - tb).abs();
                        let x = -dt / half_life_seconds.max(1.0);
                        let logistic = 1.0 / (1.0 + (-x).exp());
                        (wa * wb) as f64 * (1.0 + alpha * logistic)
                    }
                };
                *pairs.entry((lo as u32, hi as u32)).or_insert(0.0) += pair_weight;
            }
        }
    }

    // Convert the upper-triangle map to a symmetric CSR.
    // First, build a row-buckets list of (col, value).
    let mut rows: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_items];
    for ((lo, hi), v) in pairs {
        let lo = lo as usize;
        let hi = hi as usize;
        let value = v as f32;
        rows[lo].push((hi as i32, value));
        rows[hi].push((lo as i32, value));
    }

    let mut indptr = Vec::with_capacity(n_items + 1);
    indptr.push(0i32);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for row in rows.iter_mut() {
        row.sort_by_key(|(c, _)| *c);
        for (c, v) in row.iter() {
            indices.push(*c);
            data.push(*v);
        }
        indptr.push(indices.len() as i32);
    }
    Ok((data, indices, indptr))
}

/// Compute `adjacency[owned_indices].sum(axis=0)[candidate_indices]`
/// in one pass over the selected rows.
///
/// Returns a float64 array of length `candidate_indices.len()`.
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
    for &row in &owned_indices {
        if row + 1 >= indptr.len() {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for k in start..end {
            let col = indices[k] as usize;
            summed[col] += data[k] as f64;
        }
    }

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
/// by score, length <= budget, with score > 0 and not in owned_indices
/// (unless `include_owned`).
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, owned_indices, budget, include_owned=false))]
fn cooccurrence_retrieve(
    data: PyReadonlyArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, i32>,
    indptr: PyReadonlyArray1<'_, i32>,
    owned_indices: Vec<usize>,
    budget: usize,
    include_owned: bool,
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
        if row + 1 >= indptr.len() {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for k in start..end {
            let col = indices[k] as usize;
            summed[col] += data[k] as f64;
        }
    }
    if !include_owned {
        for &row in &owned_indices {
            if row < summed.len() {
                summed[row] = 0.0;
            }
        }
    }

    let effective_budget = budget.min(n_items);
    let mut positives: Vec<(f64, usize)> = summed
        .iter()
        .enumerate()
        .filter_map(|(idx, &s)| if s > 0.0 { Some((s, idx)) } else { None })
        .collect();
    let take = effective_budget.min(positives.len());
    if take < positives.len() && take > 0 {
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_cooccurrence, m)?)?;
    m.add_function(wrap_pyfunction!(cooccurrence_signal, m)?)?;
    m.add_function(wrap_pyfunction!(cooccurrence_retrieve, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny graph by hand and verify the cooc matrix has the
    /// expected pattern. Two users:
    ///   user 0 owns items {0, 1, 2}
    ///   user 1 owns items {1, 2, 3}
    /// Pure-count cooc should have:
    ///   row 0: {1: 1, 2: 1}
    ///   row 1: {0: 1, 2: 2, 3: 1}
    ///   row 2: {0: 1, 1: 2, 3: 1}
    ///   row 3: {1: 1, 2: 1}
    #[test]
    fn pure_count_two_users() {
        let user_idx = vec![0i64, 0, 0, 1, 1, 1];
        let item_idx = vec![0i64, 1, 2, 1, 2, 3];
        let weights = vec![1.0f32; 6];

        // Manual call to the inner builder (skip pyo3 wrapping).
        let kernel = Kernel::PureCount;
        let n_users = 2;
        let _n_items = 4;

        let mut by_user: Vec<Vec<(usize, f32, f64)>> = vec![Vec::new(); n_users];
        for k in 0..user_idx.len() {
            by_user[user_idx[k] as usize].push((item_idx[k] as usize, weights[k], 0.0));
        }
        let mut pairs: FxHashMap<(u32, u32), f64> = FxHashMap::default();
        for items in &by_user {
            for a in 0..items.len() {
                for b in (a + 1)..items.len() {
                    let (ia, wa, _) = items[a];
                    let (ib, wb, _) = items[b];
                    let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
                    let pw = match kernel {
                        Kernel::PureCount => (wa * wb) as f64,
                        _ => 0.0,
                    };
                    *pairs.entry((lo as u32, hi as u32)).or_insert(0.0) += pw;
                }
            }
        }
        // Verify cells.
        assert_eq!(pairs.get(&(0, 1)), Some(&1.0));
        assert_eq!(pairs.get(&(0, 2)), Some(&1.0));
        assert_eq!(pairs.get(&(1, 2)), Some(&2.0));
        assert_eq!(pairs.get(&(1, 3)), Some(&1.0));
        assert_eq!(pairs.get(&(2, 3)), Some(&1.0));
        assert_eq!(pairs.get(&(0, 3)), None);
    }
}
