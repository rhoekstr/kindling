//! Directional cooccurrence: D[i, j] = count of (i → j) transitions
//! within sessions, where ordering is determined by within-session
//! timestamps (or session-row ordering when timestamps are absent).
//!
//! Adjacent-pair semantics: only consecutive items in a session contribute
//! a transition. Non-adjacent within-session pairs contribute nothing.
//! This is the cleaner topical-affinity signal vs. all-ordered-pairs and
//! avoids quadratic blow-up on long sessions.
//!
//! Companion to `signals/cooccurrence.rs`. The two are not equivalent:
//! - Cooccurrence (existing): user-level co-ownership, time-blind.
//! - Directional (this module): session-level adjacency, asymmetric.
//!
//! Used by graph-regularized matrix factorization (`signals/graph_mf.rs`)
//! to produce a session-coherent item-item graph. The symmetric variant
//! used by the (undirected) Laplacian is built via
//! `symmetrize_via_transpose`: `W = D + D^T`. Each cell of W counts the
//! total adjacent-pair transitions involving (i, j) regardless of order.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Build directional cooccurrence CSR from session-bucketed interactions.
///
/// `session_idx[k]` and `item_idx[k]` are parallel arrays of length
/// `n_obs`; `timestamps[k]` is the within-session ordering signal.
/// When `timestamps` is `None`, items are processed in array order
/// (caller is responsible for sorting).
///
/// `weights[k]` is per-interaction weight; the contribution of an
/// adjacent pair `(i, j)` is `w_i * w_j`.
///
/// Returns the CSR triple `(data, indices, indptr)` of shape
/// `(n_items, n_items)`. Non-symmetric.
#[pyfunction]
#[pyo3(signature = (
    session_idx,
    item_idx,
    weights,
    n_sessions,
    n_items,
    timestamps = None,
))]
#[allow(clippy::too_many_arguments)]
fn build_directional_cooc(
    session_idx: PyReadonlyArray1<'_, i64>,
    item_idx: PyReadonlyArray1<'_, i64>,
    weights: PyReadonlyArray1<'_, f32>,
    n_sessions: usize,
    n_items: usize,
    timestamps: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    let session_idx = session_idx.as_slice()?;
    let item_idx = item_idx.as_slice()?;
    let weights = weights.as_slice()?;
    let n_obs = session_idx.len().min(item_idx.len()).min(weights.len());
    if n_obs == 0 || n_items == 0 {
        return Ok((Vec::new(), Vec::new(), vec![0i32; n_items + 1]));
    }
    let timestamps = match &timestamps {
        Some(t) => Some(t.as_slice()?),
        None => None,
    };

    // Bucket by session: (timestamp, item, weight) tuples per session.
    let mut by_session: Vec<Vec<(f64, usize, f32)>> = vec![Vec::new(); n_sessions];
    for k in 0..n_obs {
        let s = session_idx[k];
        let i = item_idx[k];
        let w = weights[k];
        if s < 0 || i < 0 || w <= 0.0 {
            continue;
        }
        let s = s as usize;
        let i = i as usize;
        if s >= n_sessions || i >= n_items {
            continue;
        }
        let t = timestamps.map_or(k as f64, |ts| ts[k]);
        by_session[s].push((t, i, w));
    }

    // Sort each session by timestamp; accumulate adjacent-pair weights.
    let mut pairs: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    for sess in &mut by_session {
        if sess.len() < 2 {
            continue;
        }
        sess.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for w in sess.windows(2) {
            let (_, i, wi) = w[0];
            let (_, j, wj) = w[1];
            if i == j {
                // Self-loops are noise; drop.
                continue;
            }
            let pair = (i as u32, j as u32);
            *pairs.entry(pair).or_insert(0.0) += (wi * wj) as f64;
        }
    }

    // Pack as CSR (rows = i in transition i → j).
    let mut by_row: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_items];
    for ((i, j), v) in pairs {
        by_row[i as usize].push((j as i32, v as f32));
    }
    let mut data: Vec<f32> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = Vec::with_capacity(n_items + 1);
    indptr.push(0);
    for row in by_row.iter_mut() {
        row.sort_by_key(|(c, _)| *c);
        for (c, v) in row.iter() {
            indices.push(*c);
            data.push(*v);
        }
        indptr.push(indices.len() as i32);
    }
    Ok((data, indices, indptr))
}

/// Compute `W = D + D^T` for an arbitrary square CSR `D`. Used to
/// derive the symmetric/undirected version of a directional cooc graph.
///
/// Output is a symmetric CSR. `W[i, j] = D[i, j] + D[j, i]`. When `D`
/// has a non-zero on one side only, the symmetric form copies it.
#[pyfunction]
fn symmetrize_via_transpose(
    data: PyReadonlyArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, i32>,
    indptr: PyReadonlyArray1<'_, i32>,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    let data = data.as_slice()?;
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let n = indptr.len().saturating_sub(1);
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), vec![0]));
    }
    // Accumulate (row, col) → sum into a hash, then pack.
    let mut acc: FxHashMap<(u32, u32), f32> = FxHashMap::default();
    for i in 0..n {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        for k in start..end {
            let j = indices[k] as usize;
            let v = data[k];
            // D[i, j] contributes to W[i, j] and W[j, i].
            *acc.entry((i as u32, j as u32)).or_insert(0.0) += v;
            *acc.entry((j as u32, i as u32)).or_insert(0.0) += v;
        }
    }
    // Pack symmetric CSR.
    let mut by_row: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n];
    for ((i, j), v) in acc {
        by_row[i as usize].push((j as i32, v));
    }
    let mut out_data: Vec<f32> = Vec::new();
    let mut out_indices: Vec<i32> = Vec::new();
    let mut out_indptr: Vec<i32> = Vec::with_capacity(n + 1);
    out_indptr.push(0);
    for row in by_row.iter_mut() {
        row.sort_by_key(|(c, _)| *c);
        for (c, v) in row.iter() {
            out_indices.push(*c);
            out_data.push(*v);
        }
        out_indptr.push(out_indices.len() as i32);
    }
    Ok((out_data, out_indices, out_indptr))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_directional_cooc, m)?)?;
    m.add_function(wrap_pyfunction!(symmetrize_via_transpose, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sessions, each with three items in known order. Verify
    /// adjacent-pair counts.
    ///
    /// Session 0: [A, B, C] (timestamps 0, 1, 2)
    /// Session 1: [B, A, C] (timestamps 0, 1, 2)
    /// Expected D (item indices A=0, B=1, C=2):
    ///   D[A,B]=1 (A→B in s0)
    ///   D[B,C]=1 (B→C in s0)
    ///   D[B,A]=1 (B→A in s1)
    ///   D[A,C]=1 (A→C in s1)
    ///   D[A,A]=D[B,B]=D[C,C]=0 (no self-loops)
    #[test]
    fn directional_two_sessions_adjacent_pairs() {
        // Inline build (skip pyo3 layer).
        let session_idx: [i64; 6] = [0, 0, 0, 1, 1, 1];
        let item_idx: [i64; 6] = [0, 1, 2, 1, 0, 2];
        let weights: [f32; 6] = [1.0; 6];
        let timestamps: [f64; 6] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0];

        let n_sessions = 2;
        let n_items = 3;
        let mut by_session: Vec<Vec<(f64, usize, f32)>> = vec![Vec::new(); n_sessions];
        for k in 0..6 {
            by_session[session_idx[k] as usize].push((
                timestamps[k],
                item_idx[k] as usize,
                weights[k],
            ));
        }
        let mut pairs: FxHashMap<(u32, u32), f64> = FxHashMap::default();
        for sess in &mut by_session {
            sess.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for w in sess.windows(2) {
                if w[0].1 != w[1].1 {
                    *pairs
                        .entry((w[0].1 as u32, w[1].1 as u32))
                        .or_insert(0.0) += (w[0].2 * w[1].2) as f64;
                }
            }
        }
        // Verify expected cells.
        assert_eq!(pairs.get(&(0, 1)), Some(&1.0)); // A→B
        assert_eq!(pairs.get(&(1, 2)), Some(&1.0)); // B→C
        assert_eq!(pairs.get(&(1, 0)), Some(&1.0)); // B→A
        assert_eq!(pairs.get(&(0, 2)), Some(&1.0)); // A→C
        // Reverse directions should NOT exist (no B→A in s0, etc.).
        assert!(!pairs.contains_key(&(2, 1))); // C→B never occurred
        assert!(!pairs.contains_key(&(2, 0))); // C→A never occurred
    }

    /// Symmetrize a small triangular CSR and verify W = D + D^T.
    /// D = [[0, 2, 0],
    ///      [0, 0, 3],
    ///      [1, 0, 0]]
    /// W expected:
    ///      [[0, 2, 1],
    ///       [2, 0, 3],
    ///       [1, 3, 0]]
    #[test]
    fn symmetrize_d_plus_d_transpose() {
        // CSR for D.
        let data: Vec<f32> = vec![2.0, 3.0, 1.0];
        let indices: Vec<i32> = vec![1, 2, 0];
        let indptr: Vec<i32> = vec![0, 1, 2, 3];
        // Inline the algorithm.
        let n = 3;
        let mut acc: FxHashMap<(u32, u32), f32> = FxHashMap::default();
        for i in 0..n {
            let start = indptr[i] as usize;
            let end = indptr[i + 1] as usize;
            for k in start..end {
                let j = indices[k] as usize;
                let v = data[k];
                *acc.entry((i as u32, j as u32)).or_insert(0.0) += v;
                *acc.entry((j as u32, i as u32)).or_insert(0.0) += v;
            }
        }
        // Verify expected cells.
        assert_eq!(acc.get(&(0, 1)), Some(&2.0));
        assert_eq!(acc.get(&(1, 0)), Some(&2.0));
        assert_eq!(acc.get(&(1, 2)), Some(&3.0));
        assert_eq!(acc.get(&(2, 1)), Some(&3.0));
        assert_eq!(acc.get(&(0, 2)), Some(&1.0));
        assert_eq!(acc.get(&(2, 0)), Some(&1.0));
        assert!(!acc.contains_key(&(0, 0)));
        assert!(!acc.contains_key(&(1, 1)));
        assert!(!acc.contains_key(&(2, 2)));
    }
}
