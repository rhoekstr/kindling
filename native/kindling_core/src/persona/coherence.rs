//! Per-persona coherence score from the item-item cooc matrix.
//!
//! For each persona `p`, given its `distinctive_items[p]` (the z-filter
//! survivors), coherence is the mean `cooc[i, j]` over **all** ordered
//! pairs `(i, j)` with `i, j ∈ distinctive(p)`, `i != j`:
//!
//! ```text
//!   coherence(p) = (Σ cooc[i, j] for i,j ∈ distinctive(p), i≠j)
//!                  ──────────────────────────────────────────────
//!                            |distinctive(p)| · (|distinctive(p)|-1)
//! ```
//!
//! The denominator includes pairs whose cooc[i, j] is zero (not stored
//! in CSR) — that's what we want, since "items that don't co-occur" is
//! evidence of incoherence, not a missing data point.
//!
//! This score is post-hoc and **algorithm-agnostic**: any clustering
//! method (Louvain, HDBSCAN, Leiden, spectral, …) produces a partition
//! whose persona quality can be ranked by this metric.
//!
//! At fit time the engine uses coherence to filter low-quality personas
//! — users assigned to dropped personas get reassigned to `-1`, which
//! the fit-gate routes to global cooc base (same as HDBSCAN noise).
//!
//! Cost: O(Σ_p |distinctive(p)| · max_row_nnz). Cheap; the cooc CSR is
//! already in memory and distinctive-item lists are typically tens of
//! items.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// Compute per-persona coherence from cooc + distinctive lists.
///
/// `cooc_*` is a symmetric item-item CSR. `distinctive_items[p]` is the
/// list of item indices the persona over-indexes on.
///
/// Returns `coherence[p] ∈ [0, max_cooc_weight]`, with 0.0 for personas
/// with fewer than 2 distinctive items.
pub fn compute_persona_coherence(
    cooc_data: &[f32],
    cooc_indices: &[i32],
    cooc_indptr: &[i32],
    distinctive_items: &[Vec<i32>],
) -> Vec<f64> {
    let n_personas = distinctive_items.len();
    let n_items = indptr_n_rows(cooc_indptr);
    let mut coherence = vec![0.0_f64; n_personas];
    for (p, items) in distinctive_items.iter().enumerate() {
        let k = items.len();
        if k < 2 {
            continue;
        }
        // Membership set for O(1) "is j in distinctive(p)" lookup.
        let item_set: FxHashSet<i32> = items.iter().copied().collect();
        let mut sum = 0.0_f64;
        for &i in items.iter() {
            let i_us = i as usize;
            if i_us >= n_items {
                continue;
            }
            let start = cooc_indptr[i_us] as usize;
            let end = cooc_indptr[i_us + 1] as usize;
            for slot in start..end {
                let j = cooc_indices[slot];
                if j != i && item_set.contains(&j) {
                    sum += cooc_data[slot] as f64;
                }
            }
        }
        // Denominator is total ordered pairs (k*(k-1)). The CSR is
        // symmetric so each unordered pair is visited twice — sum over
        // ordered pairs / (k*(k-1)) gives the correct mean.
        let denom = (k * (k - 1)) as f64;
        coherence[p] = sum / denom;
    }
    coherence
}

fn indptr_n_rows(indptr: &[i32]) -> usize {
    indptr.len().saturating_sub(1)
}

/// PyO3 wrapper. Returns `Vec<f64>` of length `n_personas`.
#[pyfunction]
#[pyo3(signature = (cooc_data, cooc_indices, cooc_indptr, distinctive_items))]
fn compute_persona_coherence_py(
    cooc_data: PyReadonlyArray1<'_, f32>,
    cooc_indices: PyReadonlyArray1<'_, i32>,
    cooc_indptr: PyReadonlyArray1<'_, i32>,
    distinctive_items: Vec<Vec<i32>>,
) -> PyResult<Vec<f64>> {
    Ok(compute_persona_coherence(
        cooc_data.as_slice()?,
        cooc_indices.as_slice()?,
        cooc_indptr.as_slice()?,
        &distinctive_items,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_persona_coherence_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny 4-item cooc matrix:
    ///   0 — 1: weight 2
    ///   0 — 2: weight 0 (no edge)
    ///   1 — 2: weight 1
    ///   2 — 3: weight 4
    /// CSR (symmetric):
    fn small_cooc() -> (Vec<f32>, Vec<i32>, Vec<i32>) {
        // Per row:
        //   0: [(1, 2.0)]
        //   1: [(0, 2.0), (2, 1.0)]
        //   2: [(1, 1.0), (3, 4.0)]
        //   3: [(2, 4.0)]
        let data: Vec<f32> = vec![2.0, 2.0, 1.0, 1.0, 4.0, 4.0];
        let indices: Vec<i32> = vec![1, 0, 2, 1, 3, 2];
        let indptr: Vec<i32> = vec![0, 1, 3, 5, 6];
        (data, indices, indptr)
    }

    #[test]
    fn coherence_high_for_connected_set() {
        // distinctive = {0, 1}: only pair (0,1) has weight 2.
        // Mean = (2 + 2) / (2 * 1) = 2.0  (sum counts both directions, denom is 2*1=2)
        let (data, indices, indptr) = small_cooc();
        let distinctive = vec![vec![0_i32, 1]];
        let c = compute_persona_coherence(&data, &indices, &indptr, &distinctive);
        assert_eq!(c.len(), 1);
        assert!((c[0] - 2.0).abs() < 1e-9, "got {}", c[0]);
    }

    #[test]
    fn coherence_zero_for_disconnected_set() {
        // distinctive = {0, 3}: no edge between them.
        let (data, indices, indptr) = small_cooc();
        let distinctive = vec![vec![0_i32, 3]];
        let c = compute_persona_coherence(&data, &indices, &indptr, &distinctive);
        assert_eq!(c[0], 0.0);
    }

    #[test]
    fn coherence_lower_for_partial_set() {
        // distinctive = {0, 1, 2}: edges (0,1)=2, (1,2)=1, (0,2)=0
        // Sum (both directions) = 2+2 + 1+1 + 0 = 6
        // Denom = 3*2 = 6 → mean 1.0
        let (data, indices, indptr) = small_cooc();
        let distinctive = vec![vec![0_i32, 1, 2]];
        let c = compute_persona_coherence(&data, &indices, &indptr, &distinctive);
        assert!((c[0] - 1.0).abs() < 1e-9, "got {}", c[0]);
    }

    #[test]
    fn singleton_or_empty_distinctive_returns_zero() {
        let (data, indices, indptr) = small_cooc();
        let distinctive = vec![vec![], vec![0_i32]];
        let c = compute_persona_coherence(&data, &indices, &indptr, &distinctive);
        assert_eq!(c, vec![0.0, 0.0]);
    }

    /// Coherence ranks a tight cluster above a loose one.
    #[test]
    fn coherence_ordering_across_personas() {
        // Build a 6-item cooc: items {0,1,2} form a tight triangle (all
        // pairwise edges weight 5), items {3,4,5} form a loose triangle
        // (all pairwise edges weight 1).
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut indptr: Vec<i32> = vec![0];
        // Row 0: edges to 1, 2 weight 5
        data.extend([5.0, 5.0]); indices.extend([1, 2]); indptr.push(2);
        // Row 1: edges to 0, 2 weight 5
        data.extend([5.0, 5.0]); indices.extend([0, 2]); indptr.push(4);
        // Row 2: edges to 0, 1 weight 5
        data.extend([5.0, 5.0]); indices.extend([0, 1]); indptr.push(6);
        // Row 3: edges to 4, 5 weight 1
        data.extend([1.0, 1.0]); indices.extend([4, 5]); indptr.push(8);
        // Row 4: edges to 3, 5 weight 1
        data.extend([1.0, 1.0]); indices.extend([3, 5]); indptr.push(10);
        // Row 5: edges to 3, 4 weight 1
        data.extend([1.0, 1.0]); indices.extend([3, 4]); indptr.push(12);

        let distinctive = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let c = compute_persona_coherence(&data, &indices, &indptr, &distinctive);
        assert!((c[0] - 5.0).abs() < 1e-9, "tight cluster coherence {}", c[0]);
        assert!((c[1] - 1.0).abs() < 1e-9, "loose cluster coherence {}", c[1]);
        assert!(c[0] > c[1]);
    }
}
