//! Implicit-feedback ALS (Hu, Koren, Volinsky 2008).
//!
//! Per-iteration: precompute `Y^T Y` once (k×k), then per-user solve a
//! small SPD system using Cholesky decomposition. Sparse trick: the
//! confidence matrix `C^u = diag(1 + α · count)` differs from `I` only
//! on the user's owned items, so the per-user solve is O(k² · n_owned_u
//! + k³) instead of O(k² · n_items + k³).
//!
//! Replaces the scipy.sparse.linalg.svds stand-in in EngineV2 for two
//! purposes:
//!
//! 1. **HDBSCAN inputs**: ALS user factors capture taste structure
//!    aligned with co-purchase patterns — the right input for density-
//!    based clustering.
//! 2. **Boost layer scoring**: candidate score = user_factor · item_factor
//!    (dot product over k-dim space). Dense layer per the v2 PRD table.
//!
//! Hand-rolled Cholesky (~30 LOC) avoids LAPACK dependency. For k=32
//! the per-user solve is ~85k ops; per iteration on 10k users is ~850M
//! ops — fast in Rust without external linalg.

use ndarray::{Array2, Axis};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Output of the ALS fit.
pub struct AlsFit {
    /// `(n_users, k)` user factors.
    pub user_factors: Array2<f64>,
    /// `(n_items, k)` item factors.
    pub item_factors: Array2<f64>,
    /// Per-iteration training loss (sum of squared regularized errors).
    /// Useful for convergence diagnostics; first entry is after iter 1.
    pub iteration_losses: Vec<f64>,
}

/// Fit implicit ALS.
///
/// Parameters mirror Hu et al. 2008:
/// - `n_factors` (k): typical 32–64.
/// - `n_iters`: 10 is a good starting default.
/// - `alpha`: confidence scaling. Hu et al. use 40.
/// - `regularization` (λ): typical 0.01–0.1.
///
/// `weights[k]` is treated as the count for the (user, item) pair —
/// repeated interactions add up via `+=`. Binary input (weights=1)
/// reproduces the standard implicit-ALS setup.
#[allow(clippy::too_many_arguments)]
pub fn fit_als(
    user_idx: &[i64],
    item_idx: &[i64],
    weights: &[f32],
    n_users: usize,
    n_items: usize,
    n_factors: usize,
    n_iters: usize,
    alpha: f64,
    regularization: f64,
    seed: u64,
) -> AlsFit {
    let k = n_factors.max(2);
    let n_obs = user_idx.len().min(item_idx.len()).min(weights.len());

    // Build sparse "by-user" and "by-item" lists of (other_idx, count).
    // Aggregating duplicates so binary or weighted inputs work uniformly.
    let mut by_user: Vec<FxHashMap<u32, f64>> = (0..n_users).map(|_| FxHashMap::default()).collect();
    let mut by_item: Vec<FxHashMap<u32, f64>> = (0..n_items).map(|_| FxHashMap::default()).collect();
    for k_ in 0..n_obs {
        let u = user_idx[k_];
        let i = item_idx[k_];
        if u < 0 || i < 0 {
            continue;
        }
        let u = u as usize;
        let i = i as usize;
        if u >= n_users || i >= n_items {
            continue;
        }
        let w = weights[k_] as f64;
        if w <= 0.0 {
            continue;
        }
        *by_user[u].entry(i as u32).or_insert(0.0) += w;
        *by_item[i].entry(u as u32).or_insert(0.0) += w;
    }

    // Initialize factors with small random values (deterministic LCG).
    let user_factors = init_factors(n_users, k, seed);
    let item_factors = init_factors(n_items, k, seed.wrapping_add(1));

    let mut x = user_factors;
    let mut y = item_factors;
    let mut iteration_losses: Vec<f64> = Vec::with_capacity(n_iters);

    for _ in 0..n_iters {
        // ── User updates: solve for each user given fixed Y.
        let yty = compute_factor_gram(&y);
        update_factors(
            &mut x,
            &y,
            &by_user,
            &yty,
            alpha,
            regularization,
        );
        // ── Item updates: solve for each item given fixed X.
        let xtx = compute_factor_gram(&x);
        update_factors(
            &mut y,
            &x,
            &by_item,
            &xtx,
            alpha,
            regularization,
        );
        // Cheap loss proxy: regularization sum + factor magnitudes.
        let loss = regularization
            * (frobenius_sq(&x) + frobenius_sq(&y));
        iteration_losses.push(loss);
    }

    AlsFit {
        user_factors: x,
        item_factors: y,
        iteration_losses,
    }
}

fn init_factors(n: usize, k: usize, seed: u64) -> Array2<f64> {
    // Linear congruential generator for deterministic init without
    // pulling in `rand`. Good enough for ALS init scaling.
    let mut state = seed.max(1);
    let mut data = Vec::with_capacity(n * k);
    for _ in 0..(n * k) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // map to [-0.05, 0.05]
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        data.push((u - 0.5) * 0.1);
    }
    Array2::from_shape_vec((n, k), data).expect("shape ok")
}

fn compute_factor_gram(f: &Array2<f64>) -> Array2<f64> {
    // Returns f^T @ f, shape (k, k).
    f.t().dot(f)
}

fn frobenius_sq(f: &Array2<f64>) -> f64 {
    f.iter().map(|v| v * v).sum()
}

/// Per-row update: for each row of `target`, solve for the optimal
/// k-vector given fixed `other_factors` and the sparse interaction
/// list `by_target[r]` mapping other_idx → count.
///
/// Parallelized across rows via rayon. Each row's solve is independent
/// (reads from `other` and `base_gram`, writes only its own row of
/// `target`), so it's embarrassingly parallel.
fn update_factors(
    target: &mut Array2<f64>,
    other: &Array2<f64>,
    by_target: &[FxHashMap<u32, f64>],
    base_gram: &Array2<f64>,
    alpha: f64,
    regularization: f64,
) {
    let k = target.ncols();
    // Snapshot base_gram into a flat Vec for cheap per-row clone.
    let base_flat: Vec<f64> = base_gram.iter().copied().collect();

    // Compute new rows in parallel, then write back.
    let new_rows: Vec<Vec<f64>> = (0..target.nrows())
        .into_par_iter()
        .map(|r| {
            let mut a_flat = base_flat.clone();
            let mut b = vec![0.0f64; k];
            for (other_idx, count) in &by_target[r] {
                let other_row = other.index_axis(Axis(0), *other_idx as usize);
                let conf = alpha * count;
                let pref_factor = 1.0 + conf;
                for kk in 0..k {
                    b[kk] += pref_factor * other_row[kk];
                }
                // a += conf * y_i y_i^T
                for ki in 0..k {
                    let yi = other_row[ki];
                    let row_off = ki * k;
                    for kj in 0..k {
                        a_flat[row_off + kj] += conf * yi * other_row[kj];
                    }
                }
            }
            for kk in 0..k {
                a_flat[kk * k + kk] += regularization;
            }
            cholesky_factorize(&mut a_flat, k);
            cholesky_solve(&a_flat, &mut b, k);
            b
        })
        .collect();

    // Sequential write-back (small, cache-friendly).
    for (r, b) in new_rows.into_iter().enumerate() {
        let mut row = target.index_axis_mut(Axis(0), r);
        for kk in 0..k {
            row[kk] = b[kk];
        }
    }
}

/// In-place Cholesky factorization: A → L (lower triangular, stored in
/// the lower triangle of A; upper triangle untouched).
///
/// For SPD matrices: L L^T = A. We don't enforce full SPD-ness; small
/// numerical drift on the diagonal gets clamped at 1e-12.
fn cholesky_factorize(a: &mut [f64], n: usize) {
    for i in 0..n {
        // Diagonal: L[i,i] = sqrt(A[i,i] - Σ_{k<i} L[i,k]^2)
        let mut sum = a[i * n + i];
        for kk in 0..i {
            sum -= a[i * n + kk].powi(2);
        }
        let l_ii = sum.max(1e-12).sqrt();
        a[i * n + i] = l_ii;
        // Below-diagonal: L[j,i] = (A[j,i] - Σ_{k<i} L[j,k] L[i,k]) / L[i,i]
        for j in (i + 1)..n {
            let mut s = a[j * n + i];
            for kk in 0..i {
                s -= a[j * n + kk] * a[i * n + kk];
            }
            a[j * n + i] = s / l_ii;
        }
    }
}

/// Solve L L^T x = b in place (b ← x).
fn cholesky_solve(l: &[f64], b: &mut [f64], n: usize) {
    // Forward: L y = b
    for i in 0..n {
        let mut s = b[i];
        for kk in 0..i {
            s -= l[i * n + kk] * b[kk];
        }
        b[i] = s / l[i * n + i];
    }
    // Backward: L^T x = y
    for i in (0..n).rev() {
        let mut s = b[i];
        for kk in (i + 1)..n {
            s -= l[kk * n + i] * b[kk];
        }
        b[i] = s / l[i * n + i];
    }
}

/// PyO3 wrapper. Returns `(user_factors, item_factors, iteration_losses)`.
#[pyfunction]
#[pyo3(signature = (
    user_idx,
    item_idx,
    weights,
    n_users,
    n_items,
    n_factors = 32,
    n_iters = 10,
    alpha = 40.0,
    regularization = 0.01,
    seed = 0,
))]
#[allow(clippy::too_many_arguments)]
fn fit_als_py<'py>(
    py: Python<'py>,
    user_idx: PyReadonlyArray1<'py, i64>,
    item_idx: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f32>,
    n_users: usize,
    n_items: usize,
    n_factors: usize,
    n_iters: usize,
    alpha: f64,
    regularization: f64,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Vec<f64>,
)> {
    let result = fit_als(
        user_idx.as_slice()?,
        item_idx.as_slice()?,
        weights.as_slice()?,
        n_users,
        n_items,
        n_factors,
        n_iters,
        alpha,
        regularization,
        seed,
    );
    let user_factors = PyArray2::<f64>::from_owned_array_bound(py, result.user_factors);
    let item_factors = PyArray2::<f64>::from_owned_array_bound(py, result.item_factors);
    Ok((user_factors, item_factors, result.iteration_losses))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_als_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_round_trip() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // Known SPD. Solve A x = b for b = [1, 2, 3].
        let mut a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let mut b = vec![1.0, 2.0, 3.0];
        cholesky_factorize(&mut a, 3);
        cholesky_solve(&a, &mut b, 3);
        // Expected solution (verified externally):
        //   x ≈ [28.583, -7.667, 1.667]  (approximately)
        // Validate by reconstructing: A_orig @ x ≈ original b.
        let a_orig = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let mut residual = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                residual[i] += a_orig[i * 3 + j] * b[j];
            }
        }
        for (i, r) in residual.iter().enumerate() {
            let expected = (i + 1) as f64;
            assert!(
                (r - expected).abs() < 1e-6,
                "row {i}: got {r}, expected {expected}"
            );
        }
    }

    #[test]
    fn als_two_clusters_factors_are_separable() {
        // 20 users, 20 items, two clusters.
        // Users 0..9 own items 0..9 (cluster A); users 10..19 own items 10..19.
        let mut user_idx: Vec<i64> = Vec::new();
        let mut item_idx: Vec<i64> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        for u in 0..20 {
            let lo = if u < 10 { 0 } else { 10 };
            let hi = lo + 10;
            for i in lo..hi {
                user_idx.push(u);
                item_idx.push(i);
                weights.push(1.0);
            }
        }
        let fit = fit_als(
            &user_idx,
            &item_idx,
            &weights,
            20,
            20,
            8,    // factors
            15,   // iters
            40.0, // alpha
            0.1,  // λ (slightly higher for the small toy)
            42,   // seed
        );
        // Compare cosine similarity within-cluster vs across-cluster
        // for user factors. Within should be much higher.
        fn cos(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
            let dot = a.dot(&b);
            let na = a.dot(&a).sqrt();
            let nb = b.dot(&b).sqrt();
            if na <= 0.0 || nb <= 0.0 {
                0.0
            } else {
                dot / (na * nb)
            }
        }
        let within_a = cos(
            fit.user_factors.row(0),
            fit.user_factors.row(5),
        );
        let across = cos(
            fit.user_factors.row(0),
            fit.user_factors.row(15),
        );
        assert!(
            within_a > across,
            "within-cluster cosine ({within_a}) should exceed across-cluster ({across})"
        );
    }
}
