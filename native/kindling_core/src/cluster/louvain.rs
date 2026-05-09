//! Louvain community detection (Blondel et al. 2008).
//!
//! Two-phase iterative algorithm on a weighted undirected graph:
//!
//! Phase 1 — local modularity optimization:
//!   While any node moves, for each node:
//!     - Compute the modularity gain of moving it to each neighbor's community.
//!     - Move it to the community with the largest positive gain.
//!
//! Phase 2 — community aggregation:
//!   Collapse each community into a super-node; rebuild edge weights.
//!
//! Repeat phases 1 and 2 until modularity stops improving across passes.
//!
//! Modularity (Newman):
//!   Q = (1/2m) Σ_{i,j} (A_ij - k_i·k_j / 2m) δ(c_i, c_j)
//!
//! Modularity gain when moving node i to community C:
//!   ΔQ = k_i,C / m  −  k_i · Σ_tot(C) / (2m²)
//!   where k_i,C is the sum of edge weights from i to nodes in C
//!   (excluding the loss from i's current community — handled by
//!   removing i first).
//!
//! Output: per-node community label as `i64` (dense ids `0..n_communities`).
//!
//! Parameters:
//!   - `min_community_size`: communities below this are reassigned to -1
//!     (noise) — analogous to HDBSCAN's noise label, gives the persona
//!     fit-gate the same routing semantics.
//!   - `max_passes`: hard cap on outer iterations (phase-1 + phase-2)
//!     to bound runtime on degenerate inputs.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Output of Louvain clustering.
pub struct LouvainResult {
    pub assignments: Vec<i64>,
    pub n_communities: usize,
    pub final_modularity: f64,
    pub n_passes_run: usize,
    pub noise_fraction: f64,
}

/// Internal phase-1 + phase-2 state. The graph is held as a CSR.
struct Graph {
    n: usize,
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f64>,
    /// Per-node degree (sum of edge weights, including self-loops).
    degree: Vec<f64>,
    /// Total weight m = sum_e w_e (each undirected edge counted once,
    /// implemented as Σ_i Σ_j A_ij / 2). For Louvain math we use 2m.
    two_m: f64,
}

impl Graph {
    fn from_csr(data: &[f32], indices: &[i32], indptr: &[i32]) -> Self {
        let n = indptr.len().saturating_sub(1);
        let data_f64: Vec<f64> = data.iter().map(|x| *x as f64).collect();
        let mut degree = vec![0.0_f64; n];
        for i in 0..n {
            let start = indptr[i] as usize;
            let end = indptr[i + 1] as usize;
            for k in start..end {
                degree[i] += data_f64[k];
            }
        }
        let two_m: f64 = degree.iter().sum();
        Self {
            n,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
            data: data_f64,
            degree,
            two_m,
        }
    }

    /// Iterate (neighbor, weight) for node i. Skips self-loops; the
    /// Louvain step needs them separately for the diagonal term.
    fn neighbors(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        (start..end).map(move |k| (self.indices[k] as usize, self.data[k]))
    }
}

/// Run Louvain on a weighted undirected CSR graph. Returns per-node
/// community labels (dense `0..n_communities-1`, or `-1` if filtered as
/// noise).
pub fn fit_louvain(
    data: &[f32],
    indices: &[i32],
    indptr: &[i32],
    min_community_size: usize,
    max_passes: usize,
    modularity_tol: f64,
) -> LouvainResult {
    let n = indptr.len().saturating_sub(1);
    if n == 0 {
        return LouvainResult {
            assignments: Vec::new(),
            n_communities: 0,
            final_modularity: 0.0,
            n_passes_run: 0,
            noise_fraction: 0.0,
        };
    }

    let graph = Graph::from_csr(data, indices, indptr);
    if graph.two_m <= 0.0 {
        // Empty / weightless graph — every node is its own community
        // (which we then collapse to noise via the size filter).
        let assignments = vec![-1_i64; n];
        return LouvainResult {
            assignments,
            n_communities: 0,
            final_modularity: 0.0,
            n_passes_run: 0,
            noise_fraction: 1.0,
        };
    }

    // Each node starts in its own community; track aggregation across
    // passes via `super_to_inner`: each "super" community in the
    // aggregated graph maps back to a set of original nodes.
    let mut super_to_originals: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut current_graph = graph;
    let mut last_modularity = compute_modularity(&current_graph, &(0..current_graph.n).map(|i| i as i32).collect::<Vec<_>>());
    let mut passes = 0;
    while passes < max_passes {
        passes += 1;
        let community = phase1_local_optimize(&current_graph);
        let new_modularity = compute_modularity(&current_graph, &community);
        if new_modularity - last_modularity < modularity_tol {
            // Converged. Map current super-communities back to originals.
            let assignments = expand_to_originals(&community, &super_to_originals, n);
            return finalize(assignments, new_modularity, passes, min_community_size);
        }
        last_modularity = new_modularity;
        // Phase 2: aggregate.
        let (new_graph, new_super_to_originals) = phase2_aggregate(&current_graph, &community, &super_to_originals);
        current_graph = new_graph;
        super_to_originals = new_super_to_originals;
    }
    // Hit max_passes — return current best.
    let community = (0..current_graph.n).map(|i| i as i32).collect::<Vec<_>>();
    let assignments = expand_to_originals(&community, &super_to_originals, n);
    finalize(assignments, last_modularity, passes, min_community_size)
}

/// Phase 1: greedy local modularity optimization. Returns per-node
/// community label (in 0..k where k <= n).
fn phase1_local_optimize(graph: &Graph) -> Vec<i32> {
    let n = graph.n;
    let mut community: Vec<i32> = (0..n).map(|i| i as i32).collect();
    // tot[c] = sum of degrees of nodes in community c (= 2 × internal weight + boundary weight)
    let mut tot: Vec<f64> = graph.degree.clone();
    let two_m = graph.two_m;
    if two_m <= 0.0 {
        return community;
    }
    let inv_two_m = 1.0 / two_m;
    let inv_two_m_sq = inv_two_m * inv_two_m;

    let mut moved_in_pass = true;
    let mut iter = 0;
    let max_iters = 50; // safety cap on phase-1 inner loops
    while moved_in_pass && iter < max_iters {
        moved_in_pass = false;
        iter += 1;
        for i in 0..n {
            // Sum weights from i to each candidate community (across i's neighbors).
            let mut weight_to_c: FxHashMap<i32, f64> = FxHashMap::default();
            let mut self_loop = 0.0_f64;
            for (j, w) in graph.neighbors(i) {
                if j == i {
                    self_loop = w;
                    continue;
                }
                *weight_to_c.entry(community[j]).or_insert(0.0) += w;
            }
            let current_c = community[i];
            // Remove i from its current community.
            tot[current_c as usize] -= graph.degree[i];
            // The current "k_i,current_c" is whatever weight_to_c[current_c] holds (or 0).
            let stay_weight = weight_to_c.get(&current_c).copied().unwrap_or(0.0);
            let mut best_c = current_c;
            // Gain of staying (i.e., joining current_c again).
            let mut best_gain =
                stay_weight * inv_two_m - graph.degree[i] * tot[current_c as usize] * inv_two_m_sq * 0.5;
            for (c, k_i_c) in &weight_to_c {
                if *c == current_c {
                    continue;
                }
                let gain = *k_i_c * inv_two_m - graph.degree[i] * tot[*c as usize] * inv_two_m_sq * 0.5;
                if gain > best_gain {
                    best_gain = gain;
                    best_c = *c;
                }
            }
            // Add i to best_c.
            tot[best_c as usize] += graph.degree[i];
            if best_c != current_c {
                community[i] = best_c;
                moved_in_pass = true;
            }
            // Self-loops: we ignore in the gain calc (they don't affect
            // the choice between communities), but they do affect modularity.
            let _ = self_loop;
        }
    }
    community
}

/// Phase 2: collapse communities into super-nodes; rebuild edge weights.
fn phase2_aggregate(
    graph: &Graph,
    community: &[i32],
    super_to_originals: &[Vec<usize>],
) -> (Graph, Vec<Vec<usize>>) {
    // Re-index communities to dense 0..k.
    let mut remap: FxHashMap<i32, i32> = FxHashMap::default();
    let mut next_id: i32 = 0;
    for c in community {
        remap.entry(*c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
    }
    let k = next_id as usize;
    // Aggregate edge weights between super-nodes.
    let mut agg_edges: FxHashMap<(i32, i32), f64> = FxHashMap::default();
    for i in 0..graph.n {
        let c_i = remap[&community[i]];
        for (j, w) in graph.neighbors(i) {
            let c_j = remap[&community[j]];
            // Each undirected edge counted once: the CSR stores both
            // (i,j) and (j,i), so summing all yields 2·sum_e. We don't
            // halve here because compute_modularity expects the same
            // CSR convention.
            *agg_edges.entry((c_i, c_j)).or_insert(0.0) += w;
        }
    }
    // Build the aggregated CSR.
    let mut by_row: Vec<Vec<(i32, f64)>> = vec![Vec::new(); k];
    for ((a, b), w) in agg_edges {
        by_row[a as usize].push((b, w));
    }
    let mut data: Vec<f64> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = Vec::with_capacity(k + 1);
    indptr.push(0);
    for row in by_row.iter_mut() {
        row.sort_by_key(|(c, _)| *c);
        for (c, w) in row.iter() {
            indices.push(*c);
            data.push(*w);
        }
        indptr.push(indices.len() as i32);
    }
    let degree: Vec<f64> = (0..k)
        .map(|i| {
            let s = indptr[i] as usize;
            let e = indptr[i + 1] as usize;
            data[s..e].iter().sum()
        })
        .collect();
    let two_m: f64 = degree.iter().sum();
    let new_graph = Graph {
        n: k,
        indptr,
        indices,
        data,
        degree,
        two_m,
    };
    // Rebuild super_to_originals.
    let mut new_super_to_originals: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    for (super_old, originals) in super_to_originals.iter().enumerate() {
        let new_id = remap[&community[super_old]] as usize;
        new_super_to_originals[new_id].extend(originals.iter().copied());
    }
    (new_graph, new_super_to_originals)
}

/// Expand super-community labels back to per-original-node labels.
fn expand_to_originals(
    community: &[i32],
    super_to_originals: &[Vec<usize>],
    n_original: usize,
) -> Vec<i64> {
    // Re-index communities dense.
    let mut remap: FxHashMap<i32, i64> = FxHashMap::default();
    let mut next_id: i64 = 0;
    let mut out = vec![0_i64; n_original];
    for (super_idx, originals) in super_to_originals.iter().enumerate() {
        let raw_c = community[super_idx];
        let dense_c = *remap.entry(raw_c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        for o in originals {
            out[*o] = dense_c;
        }
    }
    out
}

/// Compute modularity Q = (1/2m) Σ_{i,j} (A_ij - k_i·k_j / 2m) δ(c_i, c_j)
fn compute_modularity(graph: &Graph, community: &[i32]) -> f64 {
    if graph.two_m <= 0.0 {
        return 0.0;
    }
    // Aggregate per-community internal weight + total degree.
    let mut in_weight: FxHashMap<i32, f64> = FxHashMap::default();
    let mut tot: FxHashMap<i32, f64> = FxHashMap::default();
    for i in 0..graph.n {
        let c = community[i];
        *tot.entry(c).or_insert(0.0) += graph.degree[i];
        for (j, w) in graph.neighbors(i) {
            if community[j] == c {
                *in_weight.entry(c).or_insert(0.0) += w;
            }
        }
    }
    let inv_two_m = 1.0 / graph.two_m;
    let mut q = 0.0;
    for (c, &iw) in in_weight.iter() {
        let t = *tot.get(c).unwrap_or(&0.0);
        q += iw * inv_two_m - (t * inv_two_m) * (t * inv_two_m);
    }
    q
}

/// Apply min_community_size filter; return final result.
fn finalize(
    mut assignments: Vec<i64>,
    modularity: f64,
    passes: usize,
    min_community_size: usize,
) -> LouvainResult {
    let n = assignments.len();
    if min_community_size > 0 {
        let mut counts: FxHashMap<i64, usize> = FxHashMap::default();
        for c in &assignments {
            *counts.entry(*c).or_insert(0) += 1;
        }
        // Communities below the threshold → -1 (noise).
        for c in assignments.iter_mut() {
            if counts.get(c).copied().unwrap_or(0) < min_community_size {
                *c = -1;
            }
        }
    }
    // Re-densify the kept community ids to 0..k.
    let mut remap: FxHashMap<i64, i64> = FxHashMap::default();
    let mut next_id: i64 = 0;
    for c in assignments.iter_mut() {
        if *c < 0 {
            continue;
        }
        let dense = *remap.entry(*c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *c = dense;
    }
    let n_communities = remap.len();
    let n_noise = assignments.iter().filter(|c| **c < 0).count();
    let noise_fraction = if n > 0 { n_noise as f64 / n as f64 } else { 0.0 };
    LouvainResult {
        assignments,
        n_communities,
        final_modularity: modularity,
        n_passes_run: passes,
        noise_fraction,
    }
}

/// PyO3 wrapper. Returns `(assignments, n_communities, modularity, n_passes, noise_fraction)`.
#[pyfunction]
#[pyo3(signature = (
    data,
    indices,
    indptr,
    min_community_size = 30,
    max_passes = 30,
    modularity_tol = 1e-6,
))]
fn fit_louvain_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    min_community_size: usize,
    max_passes: usize,
    modularity_tol: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, usize, f64, usize, f64)> {
    let result = fit_louvain(
        data.as_slice()?,
        indices.as_slice()?,
        indptr.as_slice()?,
        min_community_size,
        max_passes,
        modularity_tol,
    );
    let assignments = PyArray1::<i64>::from_vec_bound(py, result.assignments);
    Ok((
        assignments,
        result.n_communities,
        result.final_modularity,
        result.n_passes_run,
        result.noise_fraction,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_louvain_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-cluster graph: nodes 0-4 fully connected (community A),
    /// nodes 5-9 fully connected (community B), one weak edge between.
    /// Louvain should find these two communities.
    #[test]
    fn finds_two_clear_communities() {
        // Build CSR for the graph.
        // Cluster A: complete graph on 0..5 with weight 1.
        // Cluster B: complete graph on 5..10 with weight 1.
        // Bridge: weight 0.1 between 4 and 5.
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((i, j, 1.0));
            }
        }
        for i in 5..10 {
            for j in (i + 1)..10 {
                edges.push((i, j, 1.0));
            }
        }
        edges.push((4, 5, 0.1));

        let n = 10;
        let mut by_row: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n];
        for (a, b, w) in edges {
            by_row[a].push((b as i32, w));
            by_row[b].push((a as i32, w));
        }
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut indptr: Vec<i32> = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in by_row.iter_mut() {
            row.sort_by_key(|(c, _)| *c);
            for (c, w) in row.iter() {
                indices.push(*c);
                data.push(*w);
            }
            indptr.push(indices.len() as i32);
        }
        let result = fit_louvain(&data, &indices, &indptr, 1, 30, 1e-6);
        assert!(result.n_communities >= 2, "expected ≥2 communities, got {}", result.n_communities);
        assert!(result.final_modularity > 0.3, "expected high modularity for two-cluster graph, got {}", result.final_modularity);
        // Nodes 0-4 should be in the same community; same for 5-9.
        let c0 = result.assignments[0];
        for i in 0..5 {
            assert_eq!(result.assignments[i], c0, "node {i} not in same community as node 0");
        }
        let c5 = result.assignments[5];
        for i in 5..10 {
            assert_eq!(result.assignments[i], c5, "node {i} not in same community as node 5");
        }
        assert_ne!(c0, c5, "the two clusters should be in different communities");
    }

    /// Empty graph: every node should end up as noise.
    #[test]
    fn empty_graph_all_noise() {
        let data: Vec<f32> = Vec::new();
        let indices: Vec<i32> = Vec::new();
        let indptr: Vec<i32> = vec![0; 11];
        let result = fit_louvain(&data, &indices, &indptr, 5, 30, 1e-6);
        assert_eq!(result.assignments.len(), 10);
        // Every node is noise (since each was its own community of size 1, below 5).
        for c in &result.assignments {
            assert_eq!(*c, -1);
        }
    }

    /// Min-community-size filter: in a graph where Louvain finds a
    /// 1-node community, that node should be filtered to noise.
    #[test]
    fn small_communities_filtered_to_noise() {
        // 6 nodes: clique on 0-4 (community A), node 5 isolated (its
        // own community of size 1). With min_community_size=2, node 5
        // → -1.
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((i, j, 1.0));
            }
        }
        let n = 6;
        let mut by_row: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n];
        for (a, b, w) in edges {
            by_row[a].push((b as i32, w));
            by_row[b].push((a as i32, w));
        }
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut indptr: Vec<i32> = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in by_row.iter_mut() {
            row.sort_by_key(|(c, _)| *c);
            for (c, w) in row.iter() {
                indices.push(*c);
                data.push(*w);
            }
            indptr.push(indices.len() as i32);
        }
        let result = fit_louvain(&data, &indices, &indptr, 2, 30, 1e-6);
        // Node 5 is isolated → its own community of size 1 → noise.
        assert_eq!(result.assignments[5], -1, "isolated node should be noise");
        // Nodes 0-4 should be in one community.
        let c0 = result.assignments[0];
        assert!(c0 >= 0, "node 0 should not be noise, got {c0}");
    }
}
