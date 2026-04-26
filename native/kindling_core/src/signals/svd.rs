//! Truncated SVD via randomized projection (Halko, Martinsson, Tropp 2011).
//!
//! For the v2 persona pipeline we only need a low-dim user embedding
//! that preserves taste structure — HDBSCAN clusters on it. SVD via
//! randomized projection produces such an embedding much faster than
//! full ALS, with no LAPACK dep:
//!
//! 1. Random projection Ω of shape `(n_items, k+oversample)`.
//! 2. `Y = S · Ω` — sparse-times-dense matmul, gives `(n_users, k+os)`.
//! 3. Power iterations refine: `Y' = S · (S^T · Y)` re-orthogonalized.
//! 4. QR(Y) → Q. Q's columns are an orthonormal basis approximating
//!    the dominant left singular vectors.
//! 5. Output `Q · diag(approximate singular values)` truncated to k.
//!
//! No LAPACK; QR via modified Gram-Schmidt (~30 LOC); approximate
//! singular values via the column norms after a final projection.
//!
//! Use case: HDBSCAN clustering input. The output is a `(n_users, k)`
//! float64 matrix where users with similar taste vectors land near
//! each other in the embedding.

use ndarray::{s, Array2, Axis};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Build sparse user-item matrix in CSR (rows = users) and CSR^T
/// (rows = items, cols = users). Both transposes cached so the power
/// iterations have equal-cost matmuls in either direction.
fn build_user_item_csr(
    user_idx: &[i64],
    item_idx: &[i64],
    weights: &[f32],
    n_users: usize,
    n_items: usize,
) -> (
    (Vec<f32>, Vec<i32>, Vec<i32>),
    (Vec<f32>, Vec<i32>, Vec<i32>),
) {
    let n_obs = user_idx.len().min(item_idx.len()).min(weights.len());
    let mut by_user: Vec<FxHashMap<u32, f32>> =
        (0..n_users).map(|_| FxHashMap::default()).collect();
    for k in 0..n_obs {
        let u = user_idx[k];
        let i = item_idx[k];
        let w = weights[k];
        if u < 0 || i < 0 || w <= 0.0 {
            continue;
        }
        let u = u as usize;
        let i = i as usize;
        if u >= n_users || i >= n_items {
            continue;
        }
        let entry = by_user[u].entry(i as u32).or_insert(0.0);
        *entry = (*entry + w).min(1.0);
    }

    // Pack S as CSR (rows = users).
    let mut u_data: Vec<f32> = Vec::new();
    let mut u_indices: Vec<i32> = Vec::new();
    let mut u_indptr: Vec<i32> = Vec::with_capacity(n_users + 1);
    u_indptr.push(0);
    for u in 0..n_users {
        let mut row: Vec<(i32, f32)> =
            by_user[u].iter().map(|(c, w)| (*c as i32, *w)).collect();
        row.sort_by_key(|(c, _)| *c);
        for (c, w) in &row {
            u_data.push(*w);
            u_indices.push(*c);
        }
        u_indptr.push(u_indices.len() as i32);
    }

    // Pack S^T as CSR (rows = items).
    let mut by_item: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_items];
    for u in 0..n_users {
        let start = u_indptr[u] as usize;
        let end = u_indptr[u + 1] as usize;
        for k in start..end {
            let i = u_indices[k] as usize;
            let w = u_data[k];
            by_item[i].push((u as i32, w));
        }
    }
    let mut t_data: Vec<f32> = Vec::new();
    let mut t_indices: Vec<i32> = Vec::new();
    let mut t_indptr: Vec<i32> = Vec::with_capacity(n_items + 1);
    t_indptr.push(0);
    for i in 0..n_items {
        let mut row = std::mem::take(&mut by_item[i]);
        row.sort_by_key(|(c, _)| *c);
        for (c, w) in &row {
            t_data.push(*w);
            t_indices.push(*c);
        }
        t_indptr.push(t_indices.len() as i32);
    }
    ((u_data, u_indices, u_indptr), (t_data, t_indices, t_indptr))
}

/// Sparse-times-dense matmul. Reused logic but kept local to avoid
/// cross-module privacy juggling.
fn spmm(
    data: &[f32],
    indices: &[i32],
    indptr: &[i32],
    x: &Array2<f64>,
    n_rows: usize,
) -> Array2<f64> {
    let d = x.ncols();
    let mut out = Array2::<f64>::zeros((n_rows, d));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let start = indptr[i] as usize;
            let end = indptr[i + 1] as usize;
            for k in start..end {
                let j = indices[k] as usize;
                let w = data[k] as f64;
                let other = x.row(j);
                for kk in 0..d {
                    row[kk] += w * other[kk];
                }
            }
        });
    out
}

/// Modified Gram-Schmidt QR factorization, in place. After return,
/// `a` is column-orthonormal (an approximate Q). Numerically stable
/// enough for the (n_users, k+os) sizes we work with.
fn mgs_qr(a: &mut Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    for j in 0..n {
        // Orthogonalize column j against columns 0..j.
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..m {
                dot += a[[i, j]] * a[[i, k]];
            }
            for i in 0..m {
                a[[i, j]] -= dot * a[[i, k]];
            }
        }
        // Normalize.
        let mut norm_sq = 0.0;
        for i in 0..m {
            norm_sq += a[[i, j]] * a[[i, j]];
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for i in 0..m {
                a[[i, j]] *= inv;
            }
        }
    }
}

/// Truncated SVD via randomized projection. Returns the top-k user
/// factors as `Q · diag(σ)` — a `(n_users, k)` ndarray suitable for
/// HDBSCAN.
///
/// `n_oversample = 10` is the Halko et al. default; one or two
/// `n_power_iters` is enough on real data (3+ is overkill).
#[allow(clippy::too_many_arguments)]
pub fn truncated_svd(
    user_idx: &[i64],
    item_idx: &[i64],
    weights: &[f32],
    n_users: usize,
    n_items: usize,
    n_factors: usize,
    n_oversample: usize,
    n_power_iters: usize,
    seed: u64,
) -> Array2<f64> {
    let l = (n_factors + n_oversample).max(2);
    if n_users == 0 || n_items == 0 {
        return Array2::<f64>::zeros((n_users, n_factors));
    }

    let (s_csr, st_csr) =
        build_user_item_csr(user_idx, item_idx, weights, n_users, n_items);

    // Random projection Ω of shape (n_items, l) — Gaussian via Box-Muller-ish
    // approximation from a deterministic LCG.
    let mut omega = Array2::<f64>::zeros((n_items, l));
    let mut state = seed.max(1);
    for v in omega.iter_mut() {
        // central-limit approximation: sum 6 uniforms - 3 ~ N(0, ~1)
        let mut s = 0.0;
        for _ in 0..6 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s += ((state >> 40) as f64) / ((1u64 << 24) as f64);
        }
        *v = s - 3.0;
    }

    // Y = S · Ω → (n_users, l)
    let mut y = spmm(&s_csr.0, &s_csr.1, &s_csr.2, &omega, n_users);

    // Power iterations: Y ← S · (S^T · Y), with QR between for stability.
    for _ in 0..n_power_iters {
        mgs_qr(&mut y);
        let z = spmm(&st_csr.0, &st_csr.1, &st_csr.2, &y, n_items);
        y = spmm(&s_csr.0, &s_csr.1, &s_csr.2, &z, n_users);
    }
    mgs_qr(&mut y);

    // y is now Q ≈ orthonormal basis for the top-l left singular vectors.
    // Approximate the singular values: σ_i ≈ ||S^T q_i||_2 (one final
    // projection onto S^T's column space).
    let st_y = spmm(&st_csr.0, &st_csr.1, &st_csr.2, &y, n_items);
    let mut sigma = vec![0.0_f64; l];
    for j in 0..l {
        let mut norm_sq = 0.0;
        for i in 0..n_items {
            norm_sq += st_y[[i, j]] * st_y[[i, j]];
        }
        sigma[j] = norm_sq.sqrt();
    }

    // Sort columns of Q by descending sigma so the top-k factors are
    // the most informative.
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_by(|a, b| sigma[*b].partial_cmp(&sigma[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let take = n_factors.min(l);
    let mut out = Array2::<f64>::zeros((n_users, take));
    for (out_col, src_col) in order.iter().take(take).enumerate() {
        for i in 0..n_users {
            out[[i, out_col]] = y[[i, *src_col]] * sigma[*src_col];
        }
    }
    out
}

/// PyO3 wrapper.
#[pyfunction]
#[pyo3(signature = (
    user_idx,
    item_idx,
    weights,
    n_users,
    n_items,
    n_factors = 32,
    n_oversample = 10,
    n_power_iters = 1,
    seed = 0,
))]
#[allow(clippy::too_many_arguments)]
fn truncated_svd_py<'py>(
    py: Python<'py>,
    user_idx: PyReadonlyArray1<'py, i64>,
    item_idx: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f32>,
    n_users: usize,
    n_items: usize,
    n_factors: usize,
    n_oversample: usize,
    n_power_iters: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let factors = truncated_svd(
        user_idx.as_slice()?,
        item_idx.as_slice()?,
        weights.as_slice()?,
        n_users,
        n_items,
        n_factors,
        n_oversample,
        n_power_iters,
        seed,
    );
    Ok(PyArray2::<f64>::from_owned_array_bound(py, factors))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(truncated_svd_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-cluster recovery on a tiny synthetic. Same expectation as
    /// the ALS test: within-cluster cosine > across-cluster.
    #[test]
    fn two_cluster_separability() {
        let mut user_idx: Vec<i64> = Vec::new();
        let mut item_idx: Vec<i64> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        for u in 0..40 {
            let (lo, hi) = if u < 20 { (0, 20) } else { (20, 40) };
            for i in lo..hi {
                user_idx.push(u);
                item_idx.push(i);
                weights.push(1.0);
            }
        }
        let factors = truncated_svd(
            &user_idx, &item_idx, &weights,
            40, 40, 8, 10, 1, 42,
        );
        fn cos(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            if na <= 0.0 || nb <= 0.0 { 0.0 } else { dot / (na * nb) }
        }
        let within = cos(factors.row(0), factors.row(10));
        let across = cos(factors.row(0), factors.row(30));
        assert!(within > across,
            "within-cluster cos {within} should exceed across {across}");
    }
}
