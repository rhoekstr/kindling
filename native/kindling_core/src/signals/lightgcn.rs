//! LightGCN (He et al., SIGIR 2020) — pure-Rust hand-rolled implementation.
//!
//! K-layer embedding propagation over the bipartite user-item graph
//! with symmetric degree normalization, no nonlinearity, no feature
//! transformation. Final user/item embeddings are an unweighted layer
//! mean of E^(0..K). Trained via end-to-end BPR with mini-batch SGD.
//!
//! No PyTorch, no LAPACK. Forward propagation is sparse-times-dense
//! matmul; backward exploits the symmetry `A_hat^T = A_hat` for the
//! bipartite block adjacency, so the backward pass has the same
//! shape (and cost) as the forward pass.
//!
//! The bipartite block adjacency is `[[0, U], [U.T, 0]]` so we never
//! materialize the (n_u + n_i, n_u + n_i) matrix — we keep the upper-
//! right block U_hat as CSR and synthesize `A_hat @ E` by alternating
//! `U_hat @ E_i` and `U_hat.T @ E_u`.
//!
//! See PRD §"Component-by-component spec → signals/lightgcn.rs".

use ndarray::{Array2, Axis};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashSet;

/// Configuration parallel to v1's `LightGCNConfig`.
#[derive(Clone, Copy)]
pub struct LightGcnConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub min_users: usize,
    pub min_items: usize,
}

impl Default for LightGcnConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            n_layers: 3,
            learning_rate: 0.005,
            weight_decay: 1e-4,
            n_epochs: 30,
            batch_size: 8192,
            seed: 0,
            min_users: 50,
            min_items: 50,
        }
    }
}

/// Output of a fit.
pub struct LightGcnFit {
    pub user_factors: Array2<f32>,
    pub item_factors: Array2<f32>,
    pub n_epochs_trained: usize,
}

/// LCG random for deterministic init / sampling without `rand`.
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
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn next_range(&mut self, hi: usize) -> usize {
        if hi == 0 {
            return 0;
        }
        (self.next_u64() as usize) % hi
    }
    /// Approximate Gaussian via central-limit (sum of 6 uniforms - 3).
    fn next_gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..6 {
            s += self.next_f32();
        }
        s - 3.0
    }
}

/// Sparse-times-dense matmul: out[i, :] = Σ_k data[k] * x[indices[k], :]
/// for k in indptr[i]..indptr[i+1]. Parallel over rows.
fn spmm(
    data: &[f32],
    indices: &[i32],
    indptr: &[i32],
    x: &Array2<f32>,
    n_rows: usize,
) -> Array2<f32> {
    let d = x.ncols();
    let mut out = Array2::<f32>::zeros((n_rows, d));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let start = indptr[i] as usize;
            let end = indptr[i + 1] as usize;
            for k in start..end {
                let j = indices[k] as usize;
                let w = data[k];
                let other_row = x.row(j);
                for kk in 0..d {
                    row[kk] += w * other_row[kk];
                }
            }
        });
    out
}

/// Build symmetrically-normalized bipartite adjacency from user-item
/// triplets. Returns CSR for U_hat (n_u × n_i) and U_hat^T (n_i × n_u).
/// Both transposes are stored explicitly so propagation has equal-cost
/// access in both directions.
fn build_normalized_bipartite(
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
    // Aggregate (user, item) → weight (cap at 1.0 for binary semantics).
    let mut by_user: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_users];
    let mut by_item: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_items];
    let mut user_deg = vec![0.0_f64; n_users];
    let mut item_deg = vec![0.0_f64; n_items];
    {
        // Aggregate first via a hashmap-per-user to dedupe (user, item) pairs.
        let mut pair_weights: Vec<rustc_hash::FxHashMap<u32, f32>> =
            (0..n_users).map(|_| rustc_hash::FxHashMap::default()).collect();
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
            let entry = pair_weights[u].entry(i as u32).or_insert(0.0);
            *entry = (*entry + w).min(1.0); // cap at 1.0
        }
        for (u, m) in pair_weights.iter().enumerate() {
            for (i, w) in m.iter() {
                let i_ = *i as usize;
                user_deg[u] += *w as f64;
                item_deg[i_] += *w as f64;
                by_user[u].push((*i as i32, *w));
                by_item[i_].push((u as i32, *w));
            }
        }
    }
    // Inverse-sqrt degrees.
    let inv_u: Vec<f32> = user_deg
        .iter()
        .map(|d| if *d > 0.0 { (1.0 / d.sqrt()) as f32 } else { 0.0 })
        .collect();
    let inv_i: Vec<f32> = item_deg
        .iter()
        .map(|d| if *d > 0.0 { (1.0 / d.sqrt()) as f32 } else { 0.0 })
        .collect();

    // Pack U_hat (n_u rows): U_hat[u, i] = w * inv_u[u] * inv_i[i].
    let mut u_data: Vec<f32> = Vec::new();
    let mut u_indices: Vec<i32> = Vec::new();
    let mut u_indptr: Vec<i32> = Vec::with_capacity(n_users + 1);
    u_indptr.push(0);
    for u in 0..n_users {
        let mut row = std::mem::take(&mut by_user[u]);
        row.sort_by_key(|(c, _)| *c);
        for (c, w) in &row {
            u_data.push(*w * inv_u[u] * inv_i[*c as usize]);
            u_indices.push(*c);
        }
        u_indptr.push(u_indices.len() as i32);
    }
    // Pack U_hat^T (n_i rows).
    let mut t_data: Vec<f32> = Vec::new();
    let mut t_indices: Vec<i32> = Vec::new();
    let mut t_indptr: Vec<i32> = Vec::with_capacity(n_items + 1);
    t_indptr.push(0);
    for i in 0..n_items {
        let mut row = std::mem::take(&mut by_item[i]);
        row.sort_by_key(|(c, _)| *c);
        for (c, w) in &row {
            t_data.push(*w * inv_u[*c as usize] * inv_i[i]);
            t_indices.push(*c);
        }
        t_indptr.push(t_indices.len() as i32);
    }
    ((u_data, u_indices, u_indptr), (t_data, t_indices, t_indptr))
}

/// Forward propagation + layer combine. Returns (E_u^final, E_i^final).
fn propagate_and_combine(
    u_hat: &(Vec<f32>, Vec<i32>, Vec<i32>),
    u_hat_t: &(Vec<f32>, Vec<i32>, Vec<i32>),
    e_u: &Array2<f32>,
    e_i: &Array2<f32>,
    n_layers: usize,
) -> (Array2<f32>, Array2<f32>) {
    let n_u = e_u.nrows();
    let n_i = e_i.nrows();
    let mut acc_u = e_u.clone();
    let mut acc_i = e_i.clone();
    let mut cur_u = e_u.clone();
    let mut cur_i = e_i.clone();
    for _ in 0..n_layers {
        // E_u^(k+1) = U_hat @ E_i^(k)
        let new_u = spmm(&u_hat.0, &u_hat.1, &u_hat.2, &cur_i, n_u);
        // E_i^(k+1) = U_hat^T @ E_u^(k)
        let new_i = spmm(&u_hat_t.0, &u_hat_t.1, &u_hat_t.2, &cur_u, n_i);
        cur_u = new_u;
        cur_i = new_i;
        acc_u += &cur_u;
        acc_i += &cur_i;
    }
    let scale = 1.0_f32 / (n_layers as f32 + 1.0);
    acc_u *= scale;
    acc_i *= scale;
    (acc_u, acc_i)
}

fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// End-to-end BPR training with gradient through K-layer propagation.
///
/// Returns final user + item factors + actual epochs run.
#[allow(clippy::too_many_arguments)]
pub fn fit_lightgcn(
    user_idx: &[i64],
    item_idx: &[i64],
    weights: &[f32],
    n_users: usize,
    n_items: usize,
    cfg: LightGcnConfig,
) -> Option<LightGcnFit> {
    if n_users < cfg.min_users || n_items < cfg.min_items {
        return None;
    }
    let dim = cfg.dim.max(1);

    // Build normalized adjacency (both directions).
    let (u_hat, u_hat_t) = build_normalized_bipartite(user_idx, item_idx, weights, n_users, n_items);

    // Owned-set indexed for negative sampling rejection.
    let owned_sets: Vec<FxHashSet<u32>> = (0..n_users)
        .map(|u| {
            let start = u_hat.2[u] as usize;
            let end = u_hat.2[u + 1] as usize;
            u_hat.1[start..end].iter().map(|c| *c as u32).collect()
        })
        .collect();

    // Flat positive pool (rows + cols of U_hat, weights ≥ 0 by construction).
    let mut rows_flat: Vec<i32> = Vec::with_capacity(u_hat.0.len());
    let mut cols_flat: Vec<i32> = Vec::with_capacity(u_hat.0.len());
    for u in 0..n_users {
        let start = u_hat.2[u] as usize;
        let end = u_hat.2[u + 1] as usize;
        for k in start..end {
            if u_hat.0[k] > 0.0 {
                rows_flat.push(u as i32);
                cols_flat.push(u_hat.1[k]);
            }
        }
    }
    let n_pos = rows_flat.len();
    if n_pos == 0 {
        return None;
    }

    // Initialize embeddings with small Gaussian.
    let mut rng = Lcg::new(cfg.seed);
    let mut e_u = Array2::<f32>::zeros((n_users, dim));
    let mut e_i = Array2::<f32>::zeros((n_items, dim));
    for v in e_u.iter_mut() {
        *v = rng.next_gauss() * 0.01;
    }
    for v in e_i.iter_mut() {
        *v = rng.next_gauss() * 0.01;
    }

    let big_b = cfg.batch_size.min(n_pos);
    let steps_per_epoch = (n_pos / big_b).max(1);
    let layer_scale = 1.0_f32 / (cfg.n_layers as f32 + 1.0);
    let lr = cfg.learning_rate;
    let decay_mul = 1.0 - lr * cfg.weight_decay;
    let mut n_trained = 0;

    for _epoch in 0..cfg.n_epochs {
        for _step in 0..steps_per_epoch {
            // Sample BPR triples (uniform over n_pos).
            let mut u_batch = Vec::with_capacity(big_b);
            let mut i_pos = Vec::with_capacity(big_b);
            let mut i_neg = Vec::with_capacity(big_b);
            for _ in 0..big_b {
                let pidx = rng.next_range(n_pos);
                let u = rows_flat[pidx] as usize;
                u_batch.push(u);
                i_pos.push(cols_flat[pidx] as usize);
                // Negative sampling with rejection.
                let mut neg = rng.next_range(n_items);
                let mut tries = 0;
                while owned_sets[u].contains(&(neg as u32)) && tries < 20 {
                    neg = rng.next_range(n_items);
                    tries += 1;
                }
                i_neg.push(neg);
            }

            // Forward: build E_final via K-layer propagation + layer mean.
            let (ef_u, ef_i) =
                propagate_and_combine(&u_hat, &u_hat_t, &e_u, &e_i, cfg.n_layers);

            // Compute BPR diffs and the sparse dL/dE_final scatter.
            let mut dl_u = Array2::<f32>::zeros((n_users, dim));
            let mut dl_i = Array2::<f32>::zeros((n_items, dim));
            for b in 0..big_b {
                let u = u_batch[b];
                let ip = i_pos[b];
                let ineg = i_neg[b];
                let eu = ef_u.row(u);
                let eip = ef_i.row(ip);
                let ein = ef_i.row(ineg);
                let mut diff = 0.0_f32;
                for kk in 0..dim {
                    diff += eu[kk] * (eip[kk] - ein[kk]);
                }
                let s_neg_d = sigmoid_stable(-diff);
                let scale = -s_neg_d;
                // Scatter into dl_u, dl_i.
                let mut dl_u_row = dl_u.row_mut(u);
                for kk in 0..dim {
                    dl_u_row[kk] += scale * (eip[kk] - ein[kk]);
                }
                let mut dl_ip_row = dl_i.row_mut(ip);
                for kk in 0..dim {
                    dl_ip_row[kk] += scale * eu[kk];
                }
                let mut dl_ineg_row = dl_i.row_mut(ineg);
                for kk in 0..dim {
                    dl_ineg_row[kk] += -scale * eu[kk];
                }
            }

            // Backward through propagation (A_hat symmetric).
            let mut acc_gu = dl_u.clone();
            let mut acc_gi = dl_i.clone();
            let mut cur_gu = dl_u.clone();
            let mut cur_gi = dl_i.clone();
            for _ in 0..cfg.n_layers {
                let new_gu = spmm(&u_hat.0, &u_hat.1, &u_hat.2, &cur_gi, n_users);
                let new_gi = spmm(&u_hat_t.0, &u_hat_t.1, &u_hat_t.2, &cur_gu, n_items);
                cur_gu = new_gu;
                cur_gi = new_gi;
                acc_gu += &cur_gu;
                acc_gi += &cur_gi;
            }
            // Scale to E^(0) gradient.
            let de_u = acc_gu * layer_scale;
            let de_i = acc_gi * layer_scale;

            // SGD step.
            e_u.scaled_add(-lr, &de_u);
            e_i.scaled_add(-lr, &de_i);

            // Sparse L2 reg on the BPR-triple base rows.
            for b in 0..big_b {
                let u = u_batch[b];
                let ip = i_pos[b];
                let ineg = i_neg[b];
                e_u.row_mut(u).iter_mut().for_each(|v| *v *= decay_mul);
                e_i.row_mut(ip).iter_mut().for_each(|v| *v *= decay_mul);
                e_i.row_mut(ineg).iter_mut().for_each(|v| *v *= decay_mul);
            }

            // NaN guard.
            if !e_u.iter().all(|v| v.is_finite()) || !e_i.iter().all(|v| v.is_finite()) {
                return Some(LightGcnFit {
                    user_factors: e_u,
                    item_factors: e_i,
                    n_epochs_trained: n_trained,
                });
            }
        }
        n_trained += 1;
    }

    // Final propagation + layer combine (the served embeddings).
    let (e_u_final, e_i_final) =
        propagate_and_combine(&u_hat, &u_hat_t, &e_u, &e_i, cfg.n_layers);

    Some(LightGcnFit {
        user_factors: e_u_final,
        item_factors: e_i_final,
        n_epochs_trained: n_trained,
    })
}

/// PyO3 wrapper. Returns `(user_factors, item_factors, n_epochs_trained)`
/// or `None` (mapped to `(empty, empty, 0)`) when below min_users/items.
#[pyfunction]
#[pyo3(signature = (
    user_idx,
    item_idx,
    weights,
    n_users,
    n_items,
    dim = 64,
    n_layers = 3,
    learning_rate = 0.005,
    weight_decay = 1e-4,
    n_epochs = 30,
    batch_size = 8192,
    seed = 0,
    min_users = 50,
    min_items = 50,
))]
#[allow(clippy::too_many_arguments)]
fn fit_lightgcn_py<'py>(
    py: Python<'py>,
    user_idx: PyReadonlyArray1<'py, i64>,
    item_idx: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f32>,
    n_users: usize,
    n_items: usize,
    dim: usize,
    n_layers: usize,
    learning_rate: f32,
    weight_decay: f32,
    n_epochs: usize,
    batch_size: usize,
    seed: u64,
    min_users: usize,
    min_items: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, usize)> {
    let cfg = LightGcnConfig {
        dim,
        n_layers,
        learning_rate,
        weight_decay,
        n_epochs,
        batch_size,
        seed,
        min_users,
        min_items,
    };
    let result = fit_lightgcn(
        user_idx.as_slice()?,
        item_idx.as_slice()?,
        weights.as_slice()?,
        n_users,
        n_items,
        cfg,
    );
    match result {
        Some(fit) => Ok((
            PyArray2::<f32>::from_owned_array_bound(py, fit.user_factors),
            PyArray2::<f32>::from_owned_array_bound(py, fit.item_factors),
            fit.n_epochs_trained,
        )),
        None => Ok((
            PyArray2::<f32>::zeros_bound(py, [0, dim], false),
            PyArray2::<f32>::zeros_bound(py, [0, dim], false),
            0,
        )),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_lightgcn_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny bipartite + verify A_hat row sums match the symmetric
    /// normalization formula.
    #[test]
    fn normalization_is_symmetric_degree_corrected() {
        // 2 users, 2 items, fully connected (each user owns each item).
        let user_idx = vec![0i64, 0, 1, 1];
        let item_idx = vec![0i64, 1, 0, 1];
        let weights = vec![1.0_f32; 4];
        let (u_hat, _) = build_normalized_bipartite(&user_idx, &item_idx, &weights, 2, 2);
        // Each user has degree 2; each item has degree 2. So
        // A_hat[u, i] = 1 / sqrt(2 * 2) = 0.5 for all 4 cells.
        for v in &u_hat.0 {
            assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");
        }
    }

    /// Tiny end-to-end fit: 2 well-separated user clusters; verify that
    /// after training, users in the same cluster have higher cosine
    /// similarity than across clusters.
    #[test]
    fn two_cluster_recovery() {
        // Cluster A: users 0..49 own items 0..49.
        // Cluster B: users 50..99 own items 50..99.
        let mut user_idx: Vec<i64> = Vec::new();
        let mut item_idx: Vec<i64> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        for u in 0..100 {
            let (lo, hi) = if u < 50 { (0, 50) } else { (50, 100) };
            for i in lo..hi {
                user_idx.push(u);
                item_idx.push(i);
                weights.push(1.0);
            }
        }
        let cfg = LightGcnConfig {
            dim: 16,
            n_layers: 2,
            learning_rate: 0.01,
            weight_decay: 1e-4,
            n_epochs: 5,
            batch_size: 256,
            seed: 42,
            min_users: 10,
            min_items: 10,
        };
        let fit = fit_lightgcn(
            &user_idx, &item_idx, &weights,
            100, 100, cfg,
        ).expect("fit");
        // Check: user 0 ~ user 25 (same cluster) > user 0 ~ user 75 (different).
        fn cos(a: ndarray::ArrayView1<f32>, b: ndarray::ArrayView1<f32>) -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
            if na <= 0.0 || nb <= 0.0 { 0.0 } else { dot / (na * nb) }
        }
        let within = cos(fit.user_factors.row(0), fit.user_factors.row(25));
        let across = cos(fit.user_factors.row(0), fit.user_factors.row(75));
        assert!(
            within > across,
            "within-cluster cos {within} not > across {across}"
        );
    }
}
