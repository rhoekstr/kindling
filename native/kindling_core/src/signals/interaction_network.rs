//! Personalized PageRank on the temporal interaction graph.
//!
//! v2 boost layer per the PRD: contributes only when its z-gate fires
//! confidently. Mechanism is the standard PPR power iteration:
//!
//! ```text
//! r_{t+1} = (1 - α) · P^T · r_t  +  α · s
//! ```
//!
//! Where:
//! - `P` is the row-normalized transition matrix (P[i,j] = adj[i,j] / Σ_k adj[i,k])
//! - `s` is the restart distribution: uniform over the user's seed items
//! - `α` is the restart probability (PinSage / Pixie default: 0.15)
//!
//! Convergence: stop when `|r_{t+1} - r_t|_1 < tol` or after `n_iters`
//! iterations. PPR converges quickly on real-world graphs (~30 iters).
//!
//! The transition matrix is computed at fit time (Python side, since the
//! row-normalization is a one-shot scipy.sparse.diags @ matmul). At
//! recommend time, this kernel takes the CSR triplet + seeds and returns
//! the stationary distribution.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// Personalized PageRank power iteration. Returns the per-item
/// stationary distribution, max-normalized so the result lives in [0, 1]
/// for parity with v1 score_many.
pub fn ppr_iterate(
    transition_data: &[f32],
    transition_indices: &[i32],
    transition_indptr: &[i32],
    seeds: &[usize],
    alpha: f64,
    n_iters: usize,
    tol: f64,
) -> Vec<f64> {
    let n = transition_indptr.len().saturating_sub(1);
    if n == 0 || seeds.is_empty() {
        return vec![0.0; n];
    }
    // Build restart distribution: uniform over distinct seeds.
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for &s in seeds {
        if s < n {
            seen.insert(s);
        }
    }
    if seen.is_empty() {
        return vec![0.0; n];
    }
    let s_value = 1.0 / seen.len() as f64;
    let mut s_dist = vec![0.0_f64; n];
    for idx in seen {
        s_dist[idx] = s_value;
    }

    let mut r = s_dist.clone();
    let mut r_next = vec![0.0_f64; n];
    let one_minus_alpha = 1.0 - alpha;

    for _ in 0..n_iters {
        // Reset r_next to α · s.
        for i in 0..n {
            r_next[i] = alpha * s_dist[i];
        }
        // Add (1 - α) · P^T · r. For each row i of P, for each non-zero
        // (col, val): r_next[col] += (1-α) · val · r[i].
        for i in 0..n {
            let r_i = r[i];
            if r_i == 0.0 {
                continue;
            }
            let start = transition_indptr[i] as usize;
            let end = transition_indptr[i + 1] as usize;
            for k in start..end {
                let col = transition_indices[k] as usize;
                r_next[col] += one_minus_alpha * (transition_data[k] as f64) * r_i;
            }
        }
        // Convergence check + swap.
        let delta: f64 = r.iter().zip(r_next.iter()).map(|(a, b)| (a - b).abs()).sum();
        std::mem::swap(&mut r, &mut r_next);
        if delta < tol {
            break;
        }
    }

    // Max-normalize for downstream signal-scale parity.
    let max_v = r.iter().copied().fold(0.0_f64, f64::max);
    if max_v > 0.0 {
        let inv = 1.0 / max_v;
        for v in r.iter_mut() {
            *v *= inv;
        }
    }
    r
}

/// PyO3: PPR over a CSR transition matrix + seed list. Returns
/// `(n_items,)` ndarray of normalized PPR scores.
#[pyfunction]
#[pyo3(signature = (
    transition_data,
    transition_indices,
    transition_indptr,
    seeds,
    alpha = 0.15,
    n_iters = 30,
    tol = 1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn ppr_iterate_py<'py>(
    py: Python<'py>,
    transition_data: PyReadonlyArray1<'py, f32>,
    transition_indices: PyReadonlyArray1<'py, i32>,
    transition_indptr: PyReadonlyArray1<'py, i32>,
    seeds: Vec<usize>,
    alpha: f64,
    n_iters: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = ppr_iterate(
        transition_data.as_slice()?,
        transition_indices.as_slice()?,
        transition_indptr.as_slice()?,
        &seeds,
        alpha,
        n_iters,
        tol,
    );
    Ok(PyArray1::<f64>::from_vec_bound(py, result))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ppr_iterate_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Disconnected-components graph: PPR mass should stay on the
    /// seed's component and not leak into the disconnected one.
    ///
    /// Component A (nodes 0,1,2) — fully connected triangle.
    /// Component B (nodes 3,4,5) — fully connected triangle.
    /// No edges between components.
    /// Seed = node 0. Expected: r[0..3] all positive, r[3..6] all zero.
    #[test]
    fn ppr_stays_in_seed_component() {
        // Row-normalized transitions for the two triangles.
        // Each node has 2 outgoing edges, weight 0.5 each.
        // Row 0: {1, 2} (cols 1, 2, vals 0.5, 0.5)
        // Row 1: {0, 2}
        // Row 2: {0, 1}
        // Row 3: {4, 5}
        // Row 4: {3, 5}
        // Row 5: {3, 4}
        let data: Vec<f32> = vec![0.5; 12];
        let indices = vec![1_i32, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4];
        let indptr = vec![0_i32, 2, 4, 6, 8, 10, 12];

        let r = ppr_iterate(&data, &indices, &indptr, &[0], 0.15, 50, 1e-9);
        // Component A (with seed) should hold all the mass; B should be 0.
        for i in 0..3 {
            assert!(r[i] > 0.0, "node {i} (seed component) should have positive PPR, got {}", r[i]);
        }
        for i in 3..6 {
            assert!(r[i] == 0.0, "node {i} (disconnected component) should be 0, got {}", r[i]);
        }
        // Among component A, the seed (node 0) should be the strongest.
        // (Node 0 is the only node receiving the α·s contribution every iter.)
        assert!(r[0] >= r[1], "seed {} not max of component: r[1]={}", r[0], r[1]);
        assert!(r[0] >= r[2], "seed {} not max of component: r[2]={}", r[0], r[2]);
    }

    #[test]
    fn empty_seeds_returns_zero_vector() {
        let data = vec![1.0_f32];
        let indices = vec![1_i32];
        let indptr = vec![0_i32, 1, 1];
        let r = ppr_iterate(&data, &indices, &indptr, &[], 0.15, 10, 1e-6);
        assert!(r.iter().all(|x| *x == 0.0));
    }
}
