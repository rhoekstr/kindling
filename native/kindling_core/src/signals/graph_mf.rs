//! Graph-regularized matrix factorization (GR-MF).
//!
//! Adds a Laplacian penalty on the item-item graph to the standard ALS
//! objective:
//!
//! ```text
//!   L = Σᵤᵢ cᵤᵢ (pᵤᵢ - uᵤ·vᵢ)² + λ‖V‖² + α_data·tr(VᵀL_data·V)
//!                                       + α_hier·tr(VᵀL_hier·V)
//! ```
//!
//! Two graph layers (data-driven + asserted hierarchy) with independent
//! regularization coefficients. Per-item closed-form solve under the
//! coupling becomes:
//!
//! ```text
//!   (UᵀCⁱU + λI + α·D[i,i]·I)·vᵢ = UᵀCⁱpⁱ + α·Σⱼ W[i,j]·vⱼ
//! ```
//!
//! where W is the graph adjacency, D = row-sum diagonal, L = D − W is
//! the Laplacian. **Cold-start emergent property**: a new item with
//! empty `Cⁱpⁱ` solves to the weighted average of its graph neighbors'
//! factors, automatically filling in factor estimates for items that
//! lack interaction history.
//!
//! Solver: **Jacobi** parallel block coordinate descent. The off-diagonal
//! coupling reads from `V_prev` (a snapshot at the start of each
//! iteration); each thread writes to its own row of `V`. Trades faster
//! convergence (Gauss-Seidel) for trivial parallelism via rayon.
//! Convergence: typically 10-15 iters vs ~5 for plain ALS.
//!
//! User factors get the standard ALS update — no user-side graph (we'd
//! need a user-user graph, which kindling doesn't have today).

use ndarray::{Array2, Axis};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Configuration for `fit_graph_mf`.
#[derive(Clone, Copy)]
pub struct GraphMfConfig {
    pub dim: usize,
    pub n_iters: usize,
    pub alpha_data: f32,
    pub alpha_hierarchy: f32,
    pub regularization: f32,
    pub als_alpha: f64, // confidence scaling, like Hu et al.
    pub seed: u64,
    pub min_users: usize,
    pub min_items: usize,
}

impl Default for GraphMfConfig {
    fn default() -> Self {
        Self {
            dim: 32,
            n_iters: 15,
            alpha_data: 0.1,
            alpha_hierarchy: 0.5,
            regularization: 0.01,
            als_alpha: 40.0,
            seed: 0,
            min_users: 50,
            min_items: 50,
        }
    }
}

pub struct GraphMfFit {
    pub user_factors: Array2<f64>,
    pub item_factors: Array2<f64>,
    pub n_iters_run: usize,
    /// Per-iter ‖V_t − V_{t-1}‖² — convergence proxy.
    pub per_iter_delta: Vec<f64>,
}

/// Hand-rolled in-place Cholesky factorization (lower triangular).
fn cholesky_factorize(a: &mut [f64], n: usize) {
    for i in 0..n {
        let mut sum = a[i * n + i];
        for kk in 0..i {
            sum -= a[i * n + kk].powi(2);
        }
        let l_ii = sum.max(1e-12).sqrt();
        a[i * n + i] = l_ii;
        for j in (i + 1)..n {
            let mut s = a[j * n + i];
            for kk in 0..i {
                s -= a[j * n + kk] * a[i * n + kk];
            }
            a[j * n + i] = s / l_ii;
        }
    }
}

/// Solve `L L^T x = b` in place (b ← x).
fn cholesky_solve(l: &[f64], b: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = b[i];
        for kk in 0..i {
            s -= l[i * n + kk] * b[kk];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for kk in (i + 1)..n {
            s -= l[kk * n + i] * b[kk];
        }
        b[i] = s / l[i * n + i];
    }
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_gauss(&mut self) -> f64 {
        let mut s = 0.0;
        for _ in 0..6 {
            s += ((self.next_u64() >> 40) as f64) / ((1u64 << 24) as f64);
        }
        s - 3.0
    }
}

/// Pre-compute row-sum (degree diagonal) of a graph CSR.
fn graph_degree(data: &[f32], indptr: &[i32], n: usize) -> Vec<f32> {
    let mut d = vec![0.0_f32; n];
    for i in 0..n {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        let mut s = 0.0_f32;
        for k in start..end {
            s += data[k];
        }
        d[i] = s;
    }
    d
}

type GraphCsr<'a> = (&'a [f32], &'a [i32], &'a [i32]);

/// Fit graph-regularized matrix factorization. See module docstring.
#[allow(clippy::too_many_arguments)]
pub fn fit_graph_mf(
    user_idx: &[i64],
    item_idx: &[i64],
    weights: &[f32],
    n_users: usize,
    n_items: usize,
    data_graph: Option<GraphCsr>,
    hierarchy_graph: Option<GraphCsr>,
    cfg: GraphMfConfig,
) -> Option<GraphMfFit> {
    if n_users < cfg.min_users || n_items < cfg.min_items {
        return None;
    }
    let dim = cfg.dim.max(2);

    // Bucket interactions by user + by item — same as plain ALS.
    let mut by_user: Vec<FxHashMap<u32, f64>> =
        (0..n_users).map(|_| FxHashMap::default()).collect();
    let mut by_item: Vec<FxHashMap<u32, f64>> =
        (0..n_items).map(|_| FxHashMap::default()).collect();
    let n_obs = user_idx.len().min(item_idx.len()).min(weights.len());
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
        let w = w as f64;
        *by_user[u].entry(i as u32).or_insert(0.0) += w;
        *by_item[i].entry(u as u32).or_insert(0.0) += w;
    }

    // Pre-compute per-graph degree diagonals (for the diagonal reg term).
    let data_degree = data_graph.as_ref().map(|(d, _, p)| graph_degree(d, p, n_items));
    let hier_degree = hierarchy_graph
        .as_ref()
        .map(|(d, _, p)| graph_degree(d, p, n_items));

    // Initialize factors with small Gaussian noise.
    let mut rng = Lcg::new(cfg.seed);
    let mut x = Array2::<f64>::zeros((n_users, dim));
    let mut y = Array2::<f64>::zeros((n_items, dim));
    for v in x.iter_mut() {
        *v = rng.next_gauss() * 0.01;
    }
    for v in y.iter_mut() {
        *v = rng.next_gauss() * 0.01;
    }

    let mut per_iter_delta: Vec<f64> = Vec::with_capacity(cfg.n_iters);
    let alpha_d = cfg.alpha_data as f64;
    let alpha_h = cfg.alpha_hierarchy as f64;

    for _t in 0..cfg.n_iters {
        // ── 1. Update user factors (standard ALS, no graph term).
        let yty = y.t().dot(&y);
        update_factors_standard(&mut x, &y, &by_user, &yty, cfg.als_alpha, cfg.regularization as f64);

        // ── 2. Update item factors with Jacobi graph regularization.
        let xtx = x.t().dot(&x);
        let y_prev = y.clone();
        let delta = update_factors_with_graph_jacobi(
            &mut y,
            &y_prev,
            &x,
            &by_item,
            &xtx,
            cfg.als_alpha,
            cfg.regularization as f64,
            data_graph,
            data_degree.as_deref(),
            alpha_d,
            hierarchy_graph,
            hier_degree.as_deref(),
            alpha_h,
        );
        per_iter_delta.push(delta);
    }

    Some(GraphMfFit {
        user_factors: x,
        item_factors: y,
        n_iters_run: cfg.n_iters,
        per_iter_delta,
    })
}

/// Standard ALS row update — no graph regularization. Reads from
/// `other`, writes to `target`. Each row solved independently in
/// parallel via rayon.
fn update_factors_standard(
    target: &mut Array2<f64>,
    other: &Array2<f64>,
    by_target: &[FxHashMap<u32, f64>],
    base_gram: &Array2<f64>,
    als_alpha: f64,
    regularization: f64,
) {
    let k = target.ncols();
    let base_flat: Vec<f64> = base_gram.iter().copied().collect();
    let new_rows: Vec<Vec<f64>> = (0..target.nrows())
        .into_par_iter()
        .map(|r| {
            let mut a_flat = base_flat.clone();
            let mut b = vec![0.0_f64; k];
            for (other_idx, count) in &by_target[r] {
                let row = other.index_axis(Axis(0), *other_idx as usize);
                let conf = als_alpha * count;
                let pref = 1.0 + conf;
                for kk in 0..k {
                    b[kk] += pref * row[kk] as f64;
                }
                for ki in 0..k {
                    let yi = row[ki] as f64;
                    let off = ki * k;
                    for kj in 0..k {
                        a_flat[off + kj] += conf * yi * row[kj] as f64;
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
    for (r, b) in new_rows.into_iter().enumerate() {
        let mut row = target.index_axis_mut(Axis(0), r);
        for kk in 0..k {
            row[kk] = b[kk];
        }
    }
}

/// Item-side ALS update with Jacobi graph regularization.
///
/// Per item `i`:
///   A = UᵀCⁱU + λI + (α_d·D_d[i,i] + α_h·D_h[i,i]) · I
///   b = UᵀCⁱpⁱ + α_d·Σⱼ W_d[i,j]·V_prev[j] + α_h·Σⱼ W_h[i,j]·V_prev[j]
///   V[i] = solve(A, b)
///
/// Reads from `target_prev` (the snapshot at start of iteration);
/// writes to `target` (the new factors). Trivially parallel.
///
/// Returns ‖target − target_prev‖² for convergence diagnostics.
#[allow(clippy::too_many_arguments)]
fn update_factors_with_graph_jacobi(
    target: &mut Array2<f64>,
    target_prev: &Array2<f64>,
    other: &Array2<f64>,
    by_target: &[FxHashMap<u32, f64>],
    base_gram: &Array2<f64>,
    als_alpha: f64,
    regularization: f64,
    data_graph: Option<GraphCsr>,
    data_degree: Option<&[f32]>,
    alpha_data: f64,
    hierarchy_graph: Option<GraphCsr>,
    hier_degree: Option<&[f32]>,
    alpha_hier: f64,
) -> f64 {
    let k = target.ncols();
    let base_flat: Vec<f64> = base_gram.iter().copied().collect();
    let new_rows: Vec<(Vec<f64>, f64)> = (0..target.nrows())
        .into_par_iter()
        .map(|r| {
            let mut a_flat = base_flat.clone();
            let mut b = vec![0.0_f64; k];
            // Standard ALS contribution: confidence-weighted UᵀCⁱU + UᵀCⁱpⁱ.
            for (other_idx, count) in &by_target[r] {
                let row = other.index_axis(Axis(0), *other_idx as usize);
                let conf = als_alpha * count;
                let pref = 1.0 + conf;
                for kk in 0..k {
                    b[kk] += pref * row[kk] as f64;
                }
                for ki in 0..k {
                    let yi = row[ki] as f64;
                    let off = ki * k;
                    for kj in 0..k {
                        a_flat[off + kj] += conf * yi * row[kj] as f64;
                    }
                }
            }
            // Diagonal regularization: λI + (α_d·D_d[i,i] + α_h·D_h[i,i])·I
            let mut diag_add = regularization;
            if let Some(d) = data_degree {
                diag_add += alpha_data * d[r] as f64;
            }
            if let Some(d) = hier_degree {
                diag_add += alpha_hier * d[r] as f64;
            }
            for kk in 0..k {
                a_flat[kk * k + kk] += diag_add;
            }
            // Off-diagonal coupling: α·Σⱼ W[i,j]·V_prev[j].
            if let Some((wd, wi, wp)) = data_graph {
                let start = wp[r] as usize;
                let end = wp[r + 1] as usize;
                for kk_csr in start..end {
                    let j = wi[kk_csr] as usize;
                    let w = wd[kk_csr] as f64;
                    let prev = target_prev.index_axis(Axis(0), j);
                    let scale = alpha_data * w;
                    for kk in 0..k {
                        b[kk] += scale * prev[kk] as f64;
                    }
                }
            }
            if let Some((wd, wi, wp)) = hierarchy_graph {
                let start = wp[r] as usize;
                let end = wp[r + 1] as usize;
                for kk_csr in start..end {
                    let j = wi[kk_csr] as usize;
                    let w = wd[kk_csr] as f64;
                    let prev = target_prev.index_axis(Axis(0), j);
                    let scale = alpha_hier * w;
                    for kk in 0..k {
                        b[kk] += scale * prev[kk] as f64;
                    }
                }
            }
            // Solve.
            cholesky_factorize(&mut a_flat, k);
            cholesky_solve(&a_flat, &mut b, k);
            // Diff vs prev row, for convergence proxy.
            let prev_row = target_prev.index_axis(Axis(0), r);
            let mut row_delta = 0.0_f64;
            for kk in 0..k {
                let d = b[kk] - prev_row[kk] as f64;
                row_delta += d * d;
            }
            (b, row_delta)
        })
        .collect();
    let mut total_delta = 0.0_f64;
    for (r, (b, d)) in new_rows.into_iter().enumerate() {
        total_delta += d;
        let mut row = target.index_axis_mut(Axis(0), r);
        for kk in 0..k {
            row[kk] = b[kk];
        }
    }
    total_delta
}

/// PyO3 wrapper. Accepts optional graph CSRs as parallel arrays.
#[pyfunction]
#[pyo3(signature = (
    user_idx,
    item_idx,
    weights,
    n_users,
    n_items,
    dim = 32,
    n_iters = 15,
    alpha_data = 0.1,
    alpha_hierarchy = 0.5,
    regularization = 0.01,
    als_alpha = 40.0,
    seed = 0,
    min_users = 50,
    min_items = 50,
    data_graph_data = None,
    data_graph_indices = None,
    data_graph_indptr = None,
    hierarchy_graph_data = None,
    hierarchy_graph_indices = None,
    hierarchy_graph_indptr = None,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fit_graph_mf_py<'py>(
    py: Python<'py>,
    user_idx: PyReadonlyArray1<'py, i64>,
    item_idx: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f32>,
    n_users: usize,
    n_items: usize,
    dim: usize,
    n_iters: usize,
    alpha_data: f32,
    alpha_hierarchy: f32,
    regularization: f32,
    als_alpha: f64,
    seed: u64,
    min_users: usize,
    min_items: usize,
    data_graph_data: Option<PyReadonlyArray1<'py, f32>>,
    data_graph_indices: Option<PyReadonlyArray1<'py, i32>>,
    data_graph_indptr: Option<PyReadonlyArray1<'py, i32>>,
    hierarchy_graph_data: Option<PyReadonlyArray1<'py, f32>>,
    hierarchy_graph_indices: Option<PyReadonlyArray1<'py, i32>>,
    hierarchy_graph_indptr: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    usize,
    Vec<f64>,
)> {
    let cfg = GraphMfConfig {
        dim,
        n_iters,
        alpha_data,
        alpha_hierarchy,
        regularization,
        als_alpha,
        seed,
        min_users,
        min_items,
    };
    // Materialize optional graph slices. If any of the three CSR arrays
    // for a graph are None, the graph is treated as absent.
    let data_d = data_graph_data.as_ref().map(|x| x.as_slice().unwrap());
    let data_i = data_graph_indices.as_ref().map(|x| x.as_slice().unwrap());
    let data_p = data_graph_indptr.as_ref().map(|x| x.as_slice().unwrap());
    let hier_d = hierarchy_graph_data.as_ref().map(|x| x.as_slice().unwrap());
    let hier_i = hierarchy_graph_indices.as_ref().map(|x| x.as_slice().unwrap());
    let hier_p = hierarchy_graph_indptr.as_ref().map(|x| x.as_slice().unwrap());

    let data_graph: Option<GraphCsr> = match (data_d, data_i, data_p) {
        (Some(d), Some(i), Some(p)) => Some((d, i, p)),
        _ => None,
    };
    let hierarchy_graph: Option<GraphCsr> = match (hier_d, hier_i, hier_p) {
        (Some(d), Some(i), Some(p)) => Some((d, i, p)),
        _ => None,
    };

    let result = fit_graph_mf(
        user_idx.as_slice()?,
        item_idx.as_slice()?,
        weights.as_slice()?,
        n_users,
        n_items,
        data_graph,
        hierarchy_graph,
        cfg,
    );
    match result {
        Some(fit) => Ok((
            PyArray2::<f64>::from_owned_array_bound(py, fit.user_factors),
            PyArray2::<f64>::from_owned_array_bound(py, fit.item_factors),
            fit.n_iters_run,
            fit.per_iter_delta,
        )),
        None => Ok((
            PyArray2::<f64>::zeros_bound(py, [0, dim], false),
            PyArray2::<f64>::zeros_bound(py, [0, dim], false),
            0,
            Vec::new(),
        )),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_graph_mf_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: with no graphs supplied, GR-MF reduces to standard ALS.
    /// Verify it produces taste-coherent factors on a 2-cluster synthetic.
    #[test]
    fn no_graphs_reduces_to_als() {
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
        let cfg = GraphMfConfig {
            dim: 8,
            n_iters: 10,
            alpha_data: 0.0,
            alpha_hierarchy: 0.0,
            regularization: 0.01,
            als_alpha: 40.0,
            seed: 42,
            min_users: 10,
            min_items: 10,
        };
        let fit = fit_graph_mf(&user_idx, &item_idx, &weights, 40, 40, None, None, cfg).unwrap();
        // Within-cluster cosine should exceed across-cluster.
        fn cos(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            if na <= 0.0 || nb <= 0.0 {
                0.0
            } else {
                dot / (na * nb)
            }
        }
        let within = cos(fit.user_factors.row(0), fit.user_factors.row(10));
        let across = cos(fit.user_factors.row(0), fit.user_factors.row(30));
        assert!(within > across, "within {within} not > across {across}");
    }

    /// Cold-start property: an item with NO interactions should inherit
    /// factors from its graph neighbors.
    ///
    /// Setup: two clusters of 10 users × 10 items. Item 20 has no users;
    /// graph connects it to items 0 and 1 (cluster A) with weight 1.0
    /// each. Compare graph-on vs graph-off factor magnitudes: with
    /// graph regularization the cold item should have a *non-trivial*
    /// factor (driven entirely by the graph term); without graph it
    /// should be near-zero (only λ regularization on the diagonal,
    /// and zero RHS).
    #[test]
    fn cold_start_inherits_neighbor_factors() {
        let mut user_idx: Vec<i64> = Vec::new();
        let mut item_idx: Vec<i64> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        for u in 0..20 {
            let (lo, hi) = if u < 10 { (0, 10) } else { (10, 20) };
            for i in lo..hi {
                user_idx.push(u);
                item_idx.push(i);
                weights.push(1.0);
            }
        }
        // Sparse symmetric graph: only edges 20↔0 and 20↔1.
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut indptr: Vec<i32> = Vec::with_capacity(22);
        indptr.push(0);
        for i in 0..21 {
            if i == 0 || i == 1 {
                data.push(1.0);
                indices.push(20);
            } else if i == 20 {
                data.push(1.0);
                indices.push(0);
                data.push(1.0);
                indices.push(1);
            }
            indptr.push(indices.len() as i32);
        }
        let graph_csr: GraphCsr = (&data[..], &indices[..], &indptr[..]);

        // Plain ALS (no graph): cold item factor magnitude should be
        // near-zero — it has zero RHS, only λ regularization fights.
        let cfg_no_graph = GraphMfConfig {
            dim: 8,
            n_iters: 20,
            alpha_data: 0.0,
            alpha_hierarchy: 0.0,
            regularization: 0.01,
            als_alpha: 1.0, // gentler confidence so magnitudes are stable
            seed: 42,
            min_users: 10,
            min_items: 10,
        };
        let fit_no_graph = fit_graph_mf(
            &user_idx, &item_idx, &weights, 20, 21, None, None, cfg_no_graph,
        ).unwrap();

        // Graph regularization on: cold item factor should be lifted by
        // graph-coupled neighbor evidence.
        let cfg_with_graph = GraphMfConfig {
            alpha_data: 1.0,
            ..cfg_no_graph
        };
        let fit_with_graph = fit_graph_mf(
            &user_idx, &item_idx, &weights, 20, 21,
            Some(graph_csr), None, cfg_with_graph,
        ).unwrap();

        let mag_cold_no_graph: f64 = fit_no_graph
            .item_factors
            .row(20)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let mag_cold_with_graph: f64 = fit_with_graph
            .item_factors
            .row(20)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            mag_cold_with_graph > mag_cold_no_graph * 5.0,
            "graph regularization should lift the cold item's factor magnitude. \
             without graph: {mag_cold_no_graph}, with graph: {mag_cold_with_graph}"
        );
    }

    /// Convergence: per-iter delta should monotonically decrease (allowing
    /// for small fluctuations near convergence).
    #[test]
    fn convergence_delta_decreases() {
        let mut user_idx: Vec<i64> = Vec::new();
        let mut item_idx: Vec<i64> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        for u in 0..50 {
            for i in 0..30 {
                if (u + i) % 3 == 0 {
                    user_idx.push(u);
                    item_idx.push(i);
                    weights.push(1.0);
                }
            }
        }
        let cfg = GraphMfConfig {
            dim: 8,
            n_iters: 15,
            alpha_data: 0.0,
            alpha_hierarchy: 0.0,
            regularization: 0.01,
            als_alpha: 40.0,
            seed: 42,
            min_users: 10,
            min_items: 10,
        };
        let fit = fit_graph_mf(&user_idx, &item_idx, &weights, 50, 30, None, None, cfg).unwrap();
        // Last delta should be smaller than the first.
        let first = fit.per_iter_delta[0];
        let last = *fit.per_iter_delta.last().unwrap();
        assert!(
            last < first,
            "expected delta to decrease: first={first}, last={last}"
        );
    }
}
