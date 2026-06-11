//! Project the user-item bipartite into a user-user similarity graph.
//!
//! For each item, the users who touched it form a clique. Materializing
//! all cliques would be O(Σ_item n_users_per_item²) — fine for sparse
//! datasets but blows up on popular items (e.g., movielens, where some
//! movies have 1000+ ratings).
//!
//! Solution: cap users-per-item at `max_users_per_item`. When a popular
//! item exceeds the cap, take a deterministic sample (LCG-derived
//! sampling for reproducibility). Edge weights are accumulated across
//! all items the pair shares.
//!
//! Output is a symmetric CSR `W[u, v] = sum over items both u and v
//! touched of (w_u · w_v)`. Self-loops dropped. Suitable as input to
//! Louvain community detection.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Build symmetric user-user CSR by projecting the bipartite. Per-item
/// user-cap bounds memory; deterministic sampling makes results stable.
///
/// Optional post-processing controls the edge-weight distribution that
/// Louvain (or any downstream community detector) sees:
///
/// * `weight_transform = "raw"` — leave the accumulated `Σ w_u · w_v`
///   counts as-is. Heavy-tailed distribution; popular-item-sharing
///   pairs dominate modularity.
/// * `weight_transform = "log"` — apply `ln(1 + w)` to each edge.
///   Compresses dynamic range so a 100-shared pair no longer dwarfs a
///   5-shared pair by 20×.
///
/// `min_edge_percentile ∈ [0, 1)` drops edges whose weight is below
/// the given percentile across all non-zero edges. `0.05` removes the
/// bottom 5% of edges (typically the long tail of single-shared-item
/// pairs that contribute mostly noise to community detection).
/// Pruning is applied **after** the weight transform so percentile
/// thresholds are stable across transform choices.
#[pyfunction]
#[pyo3(signature = (
    user_idx,
    item_idx,
    weights,
    n_users,
    n_items,
    max_users_per_item = 100,
    seed = 0,
    weight_transform = "raw",
    min_edge_percentile = 0.0,
))]
#[allow(clippy::too_many_arguments)]
fn build_user_user_graph(
    user_idx: PyReadonlyArray1<'_, i64>,
    item_idx: PyReadonlyArray1<'_, i64>,
    weights: PyReadonlyArray1<'_, f32>,
    n_users: usize,
    n_items: usize,
    max_users_per_item: usize,
    seed: u64,
    weight_transform: &str,
    min_edge_percentile: f64,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    let user_idx = user_idx.as_slice()?;
    let item_idx = item_idx.as_slice()?;
    let weights = weights.as_slice()?;
    let n_obs = user_idx.len().min(item_idx.len()).min(weights.len());
    if n_obs == 0 || n_users == 0 || n_items == 0 {
        return Ok((Vec::new(), Vec::new(), vec![0i32; n_users + 1]));
    }

    // Bucket interactions by item.
    let mut by_item: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_items];
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
        by_item[i].push((u as u32, w));
    }

    // Accumulate user-user edge weights via per-item cliques.
    // Use a hash map keyed by (u_lo, u_hi) to dedupe across items.
    let mut pair_weights: FxHashMap<(u32, u32), f32> = FxHashMap::default();
    let mut state = seed.max(1);
    for users in &mut by_item {
        if users.len() < 2 {
            continue;
        }
        // Cap by deterministic LCG-shuffle if oversized.
        let cap = max_users_per_item.min(users.len());
        if users.len() > max_users_per_item {
            // Fisher-Yates with LCG: produces a random permutation; take first cap.
            for i in 0..cap {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = i + ((state as usize) % (users.len() - i));
                users.swap(i, j);
            }
            users.truncate(cap);
        }
        // All pairs within the (possibly capped) clique.
        for a in 0..users.len() {
            let (ua, wa) = users[a];
            for b in (a + 1)..users.len() {
                let (ub, wb) = users[b];
                let (lo, hi) = if ua < ub { (ua, ub) } else { (ub, ua) };
                if lo == hi {
                    continue;
                }
                *pair_weights.entry((lo, hi)).or_insert(0.0) += wa * wb;
            }
        }
    }

    // ── Optional weight transform.
    match weight_transform {
        "raw" => { /* no-op */ }
        "log" => {
            for v in pair_weights.values_mut() {
                *v = (1.0 + *v).ln();
            }
        }
        "cosine" => {
            // Otsuka-Ochiai cosine: W'[u,v] = W[u,v] / sqrt(deg_u · deg_v)
            // where deg_u = sum of W[u, ·]. Bounds edge weights to [0, 1]
            // and discounts the inflation that very-active users get from
            // sharing many items with everyone. Standard discount used in
            // bipartite-projection community detection.
            let mut deg = vec![0.0_f32; n_users];
            for ((u, v), w) in pair_weights.iter() {
                deg[*u as usize] += *w;
                deg[*v as usize] += *w;
            }
            for ((u, v), w) in pair_weights.iter_mut() {
                let du = deg[*u as usize];
                let dv = deg[*v as usize];
                let denom = (du * dv).sqrt();
                if denom > 0.0 {
                    *w /= denom;
                }
            }
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weight_transform must be 'raw' | 'log' | 'cosine'; got {other:?}"
            )));
        }
    }

    // ── Optional bottom-percentile edge prune. Computes the threshold
    // across all (transformed) edge weights, then drops edges below it.
    if min_edge_percentile > 0.0 && min_edge_percentile < 1.0 && !pair_weights.is_empty() {
        let mut sorted_w: Vec<f32> = pair_weights.values().copied().collect();
        sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cut_idx = ((sorted_w.len() as f64) * min_edge_percentile).floor() as usize;
        let cut_idx = cut_idx.min(sorted_w.len().saturating_sub(1));
        let threshold = sorted_w[cut_idx];
        pair_weights.retain(|_, v| *v >= threshold);
    }

    // Pack symmetric CSR.
    let mut by_row: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_users];
    for ((lo, hi), v) in pair_weights {
        by_row[lo as usize].push((hi as i32, v));
        by_row[hi as usize].push((lo as i32, v));
    }
    let mut data: Vec<f32> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = Vec::with_capacity(n_users + 1);
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_user_user_graph, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three users forming two cliques via shared items:
    /// item 0: users 0, 1, 2 (full clique)
    /// item 1: users 0, 1
    ///
    /// Expected:
    ///   W[0,1] = w(item 0)·1 + w(item 1)·1 = 1+1 = 2
    ///   W[0,2] = w(item 0)·1 = 1
    ///   W[1,2] = w(item 0)·1 = 1
    #[test]
    fn cliques_accumulate_per_item_weights() {
        let user_idx: Vec<i64> = vec![0, 1, 2, 0, 1];
        let item_idx: Vec<i64> = vec![0, 0, 0, 1, 1];
        let weights: Vec<f32> = vec![1.0; 5];
        // Inline build.
        let mut by_item: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 2];
        for k in 0..5 {
            by_item[item_idx[k] as usize].push((user_idx[k] as u32, weights[k]));
        }
        let mut pairs: FxHashMap<(u32, u32), f32> = FxHashMap::default();
        for users in &by_item {
            for a in 0..users.len() {
                for b in (a + 1)..users.len() {
                    let (ua, wa) = users[a];
                    let (ub, wb) = users[b];
                    let (lo, hi) = if ua < ub { (ua, ub) } else { (ub, ua) };
                    *pairs.entry((lo, hi)).or_insert(0.0) += wa * wb;
                }
            }
        }
        assert_eq!(pairs.get(&(0, 1)), Some(&2.0));
        assert_eq!(pairs.get(&(0, 2)), Some(&1.0));
        assert_eq!(pairs.get(&(1, 2)), Some(&1.0));
    }
}
