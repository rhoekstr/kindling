//! EASE — Embarrassingly Shallow Autoencoder (Steck, WWW 2019).
//!
//! Closed-form linear item-item model. Given the binary user-item
//! matrix X (n_users × n_items):
//!
//! ```text
//!   G = XᵀX + λI                 (regularized item-item Gram)
//!   P = G⁻¹
//!   B[i,j] = −P[i,j] / P[j,j],   B[j,j] = 0
//!   score(u, j) = Σ_{i ∈ history(u)} B[i, j]
//! ```
//!
//! Why this beats raw co-occurrence: the inverse-Gram subtracts the
//! redundant/popularity structure out of the co-occurrence counts —
//! B learns which co-occurrences are *informative* rather than just
//! frequent. Raw-count cooc scoring degenerates toward popularity
//! ranking (gap-decomposition diagnostic, 2026-06); EASE is the
//! closed-form fix, consistently matching or beating neural models on
//! full-ranking evals of exactly the datasets we benchmark.
//!
//! Cost: one dense Cholesky inversion, O(n_items³). The Python plan
//! layer gates EASE by catalog size (default ≤ 20k items). f64 math
//! (the ALS f32 instability lesson), f32 storage for the returned B.
//!
//! X is binarized: duplicate (user, item) interactions count once.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use faer::{Mat, Side};
use faer::prelude::*;

/// Fit EASE. Returns the dense item-item weight matrix B (row-major,
/// n_items × n_items, f32, zero diagonal).
pub fn fit_ease(
    user_idx: &[i64],
    item_idx: &[i64],
    weights: Option<&[f32]>,
    n_users: usize,
    n_items: usize,
    lambda: f64,
    delta: f64,
) -> Result<Vec<f32>, String> {
    if n_items == 0 {
        return Ok(Vec::new());
    }
    let n_obs = user_idx.len().min(item_idx.len());

    // ── 1. Per-user item sets. Without weights, X is binarized
    // (duplicates count once). With weights, X_ui = the user's MAX
    // weight for the item (a re-rated item keeps its strongest signal).
    let mut by_user: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n_users];
    {
        let mut seen: Vec<FxHashMap<u32, usize>> =
            vec![FxHashMap::default(); n_users];
        for k in 0..n_obs {
            let u = user_idx[k];
            let i = item_idx[k];
            if u < 0 || i < 0 {
                continue;
            }
            let (u, i) = (u as usize, i as usize);
            if u >= n_users || i >= n_items {
                continue;
            }
            let w = match weights {
                Some(ws) => ws.get(k).copied().unwrap_or(1.0) as f64,
                None => 1.0,
            };
            if w <= 0.0 {
                continue;
            }
            match seen[u].entry(i as u32) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    let pos = *e.get();
                    if w > by_user[u][pos].1 {
                        by_user[u][pos].1 = w;
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(by_user[u].len());
                    by_user[u].push((i as u32, w));
                }
            }
        }
    }

    // ── 2. Dense Gram G = XᵀX, accumulated per-user outer products.
    // Symmetric; fill both triangles directly (simpler than mirroring).
    let mut g = Mat::<f64>::zeros(n_items, n_items);
    for items in &by_user {
        for (a_pos, &(a, wa)) in items.iter().enumerate() {
            let a = a as usize;
            g[(a, a)] += wa * wa;
            for &(b, wb) in &items[a_pos + 1..] {
                let b = b as usize;
                let v = wa * wb;
                g[(a, b)] += v;
                g[(b, a)] += v;
            }
        }
    }

    // ── 3. Regularize + invert via Cholesky (G + λI is SPD).
    // EDLAE denoising (δ>0): add δ·diag(G) to the ridge, i.e. an extra
    // popularity-proportional penalty on each item's self-term — corrects the
    // train/serve mismatch of the dropout-free autoencoder. δ=0 ⇒ canonical EASE.
    for d in 0..n_items {
        let gdd = g[(d, d)];
        g[(d, d)] += lambda + delta * gdd;
    }
    let llt = g
        .cholesky(Side::Lower)
        .map_err(|_| "Cholesky failed: Gram + λI not positive definite (λ too small?)".to_string())?;
    drop(g);
    let p = llt.inverse();

    // ── 4. B = −P / diag(P) (columnwise), zero diagonal. f32 storage.
    let mut b = vec![0.0_f32; n_items * n_items];
    for j in 0..n_items {
        let pjj = p[(j, j)];
        if pjj == 0.0 {
            return Err(format!("diag(P)[{j}] == 0; cannot form B"));
        }
        let inv_pjj = 1.0 / pjj;
        for i in 0..n_items {
            if i != j {
                b[i * n_items + j] = (-p[(i, j)] * inv_pjj) as f32;
            }
        }
    }
    Ok(b)
}

/// PyO3 wrapper. Returns B as an (n_items, n_items) float32 ndarray.
#[pyfunction]
#[pyo3(signature = (user_idx, item_idx, n_users, n_items, lambda_ = 250.0, weights = None, delta = 0.0))]
fn fit_ease_py<'py>(
    py: Python<'py>,
    user_idx: PyReadonlyArray1<'py, i64>,
    item_idx: PyReadonlyArray1<'py, i64>,
    n_users: usize,
    n_items: usize,
    lambda_: f64,
    weights: Option<PyReadonlyArray1<'py, f32>>,
    delta: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Own the index arrays so the GIL can be released during the
    // O(n_items³) inversion.
    let users: Vec<i64> = user_idx.as_slice()?.to_vec();
    let items: Vec<i64> = item_idx.as_slice()?.to_vec();
    let w_vec: Option<Vec<f32>> = match &weights {
        Some(w) => Some(w.as_slice()?.to_vec()),
        None => None,
    };
    let b = py
        .allow_threads(|| {
            fit_ease(&users, &items, w_vec.as_deref(), n_users, n_items, lambda_, delta)
        })
        .map_err(PyValueError::new_err)?;
    let arr = Array2::from_shape_vec((n_items, n_items), b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::<f32>::from_owned_array_bound(py, arr))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_ease_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2-item closed form check.
    /// Users: u0 owns {0, 1}, u1 owns {0}, u2 owns {1}.
    /// X = [[1,1],[1,0],[0,1]] → G = [[2,1],[1,2]].
    /// With λ: G' = [[2+λ, 1],[1, 2+λ]].
    /// P = G'⁻¹ = 1/det [[2+λ, −1],[−1, 2+λ]], det = (2+λ)² − 1.
    /// B01 = −P01/P11 = (1/det)/((2+λ)/det) = 1/(2+λ).
    #[test]
    fn two_item_closed_form() {
        let user_idx = vec![0_i64, 0, 1, 2];
        let item_idx = vec![0_i64, 1, 0, 1];
        let lambda = 0.5;
        let b = fit_ease(&user_idx, &item_idx, None, 3, 2, lambda, 0.0).unwrap();
        let expect = 1.0 / (2.0 + lambda);
        assert!((b[0 * 2 + 1] as f64 - expect).abs() < 1e-6, "B01 = {}, want {}", b[1], expect);
        assert!((b[1 * 2 + 0] as f64 - expect).abs() < 1e-6, "B10 = {}, want {}", b[2], expect);
        assert_eq!(b[0], 0.0, "diagonal must be zero");
        assert_eq!(b[3], 0.0, "diagonal must be zero");
    }

    /// Duplicate interactions are binarized: a user rating the same
    /// item twice must not change G.
    #[test]
    fn duplicates_binarized() {
        let once = fit_ease(&[0, 0, 1], &[0, 1, 0], None, 2, 2, 1.0, 0.0).unwrap();
        let dup = fit_ease(&[0, 0, 0, 1, 1], &[0, 1, 1, 0, 0], None, 2, 2, 1.0, 0.0).unwrap();
        for (a, b) in once.iter().zip(dup.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    /// EASE should rank the truly-associated item above the merely-
    /// popular one. Construct: item 0 is popular (everyone owns it);
    /// items 1 and 2 co-occur exclusively with each other beyond that.
    #[test]
    fn association_beats_popularity() {
        // 6 users. Item 0: owned by all (popular).
        // Items 1+2: owned together by users 0-2 only.
        // Item 3: owned by users 3-5 only (with item 0).
        let mut users = Vec::new();
        let mut items = Vec::new();
        for u in 0..6_i64 {
            users.push(u);
            items.push(0); // everyone owns item 0
        }
        for u in 0..3_i64 {
            users.push(u);
            items.push(1);
            users.push(u);
            items.push(2);
        }
        for u in 3..6_i64 {
            users.push(u);
            items.push(3);
        }
        let b = fit_ease(&users, &items, None, 6, 4, 0.5, 0.0).unwrap();
        let n = 4;
        // For a user who owns item 1: B[1, 2] (true association) should
        // exceed B[1, 0] (popularity).
        let assoc = b[1 * n + 2];
        let pop = b[1 * n + 0];
        assert!(
            assoc > pop,
            "EASE should favor association over popularity: B[1,2]={assoc} vs B[1,0]={pop}"
        );
    }

    #[test]
    fn empty_inputs_safe() {
        let b = fit_ease(&[], &[], None, 0, 0, 1.0, 0.0).unwrap();
        assert!(b.is_empty());
    }
}
