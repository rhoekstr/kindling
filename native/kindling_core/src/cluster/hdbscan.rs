//! HDBSCAN over ALS user factors via the `petal-clustering` crate.
//!
//! Inputs: ALS user factors as a `(n_users, n_factors)` ndarray. Output:
//! per-user assignment with `-1` for noise, plus a membership-confidence
//! array. The persona pipeline consumes both: assignments drive the
//! cluster-membership gate, and membership probabilities are stored on
//! the persona index for downstream diagnostics.
//!
//! The PRD calls out HDBSCAN as the production-default clusterer that
//! has been functionally unavailable in Python because of the
//! umap-learn × numpy ABI segfault. This module is the resolution:
//! a pure-Rust HDBSCAN that takes already-reduced inputs (ALS factors)
//! and never touches UMAP. Build determinism is bounded by petal-
//! clustering's algorithm + the input data; no JIT toolchain
//! involvement.

use ndarray::Array2;
use numpy::{PyArray1, PyReadonlyArray2};
use petal_clustering::{Fit, HDbscan};
use pyo3::prelude::*;

/// Output of an HDBSCAN fit.
pub struct ClusterResult {
    /// Length `n_points`. `-1` = noise, `0..n_clusters` = cluster id.
    pub assignments: Vec<i64>,
    /// Length `n_points`. `1.0` for clustered points, `0.0` for noise.
    /// petal-clustering doesn't expose per-point density; this is a
    /// best-effort proxy aligned with the v1 KMeansWithNoise placeholder
    /// behavior. A future enhancement can compute proper density-
    /// confidence using the cluster's persistence value.
    pub probabilities: Vec<f64>,
    /// Number of distinct clusters (excluding noise).
    pub n_clusters: usize,
    /// Fraction of points labeled as noise.
    pub noise_fraction: f64,
}

/// Run HDBSCAN over a `(n_points, n_factors)` matrix.
pub fn fit_hdbscan(
    factors: &Array2<f64>,
    min_cluster_size: usize,
    min_samples: usize,
) -> ClusterResult {
    let n_points = factors.nrows();
    // Guard tiny inputs. Below the parameter floor petal-clustering
    // panics when its internal k-NN slice doesn't fit min_samples.
    // Below the floor we treat the entire population as noise (no
    // clusters) — consistent with the plan's "personas off when
    // n_users is small" expectation.
    if n_points < 3 || n_points < min_samples {
        return ClusterResult {
            assignments: vec![-1; n_points],
            probabilities: vec![0.0; n_points],
            n_clusters: 0,
            noise_fraction: if n_points > 0 { 1.0 } else { 0.0 },
        };
    }

    // Clamp parameters to feasible values for the input size.
    let safe_min_samples = min_samples.max(1).min(n_points.saturating_sub(1).max(1));
    let safe_min_cluster_size = min_cluster_size.max(2).min(n_points);

    let mut clusterer: HDbscan<f64, _> = HDbscan::default();
    clusterer.min_cluster_size = safe_min_cluster_size;
    clusterer.min_samples = safe_min_samples;

    let (clusters, outliers) = clusterer.fit(factors);

    // Flatten the (cluster_id → members) map into per-point assignments.
    // petal-clustering's cluster ids are arbitrary usize keys; we
    // remap them to dense 0..n_clusters in insertion order so downstream
    // code can use them as array indices.
    let mut assignments = vec![-1i64; n_points];
    let mut probabilities = vec![0.0f64; n_points];

    let mut sorted_keys: Vec<usize> = clusters.keys().copied().collect();
    sorted_keys.sort_unstable();
    for (dense_id, original_id) in sorted_keys.iter().enumerate() {
        if let Some(members) = clusters.get(original_id) {
            for &m in members {
                if m < n_points {
                    assignments[m] = dense_id as i64;
                    probabilities[m] = 1.0;
                }
            }
        }
    }
    // Outliers stay -1 / 0.0; mark explicitly for safety.
    for &o in &outliers {
        if o < n_points {
            assignments[o] = -1;
            probabilities[o] = 0.0;
        }
    }

    let n_clusters = sorted_keys.len();
    let noise_count = assignments.iter().filter(|&&a| a < 0).count();
    let noise_fraction = noise_count as f64 / n_points as f64;

    ClusterResult {
        assignments,
        probabilities,
        n_clusters,
        noise_fraction,
    }
}

/// PyO3 wrapper. Accepts a 2D float64 numpy array (n_points, n_factors).
/// Returns `(assignments, probabilities, n_clusters, noise_fraction)`.
#[pyfunction]
#[pyo3(signature = (factors, min_cluster_size = 15, min_samples = 15))]
fn fit_hdbscan_py<'py>(
    py: Python<'py>,
    factors: PyReadonlyArray2<'py, f64>,
    min_cluster_size: usize,
    min_samples: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    f64,
)> {
    let factors_view = factors.as_array();
    let factors_owned = factors_view.to_owned();
    let result = fit_hdbscan(&factors_owned, min_cluster_size, min_samples);
    let assignments = PyArray1::<i64>::from_vec_bound(py, result.assignments);
    let probabilities = PyArray1::<f64>::from_vec_bound(py, result.probabilities);
    Ok((
        assignments,
        probabilities,
        result.n_clusters,
        result.noise_fraction,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_hdbscan_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_dense_blobs() {
        // Two well-separated 2D blobs of 10 points each, plus 2 outliers.
        // Construct manually.
        let mut data: Vec<Vec<f64>> = Vec::new();
        for i in 0..10 {
            data.push(vec![0.0 + (i as f64) * 0.05, 0.0 + (i as f64) * 0.05]);
        }
        for i in 0..10 {
            data.push(vec![10.0 + (i as f64) * 0.05, 10.0 + (i as f64) * 0.05]);
        }
        // Far outliers
        data.push(vec![100.0, -100.0]);
        data.push(vec![-100.0, 100.0]);

        let n = data.len();
        let mut flat: Vec<f64> = Vec::with_capacity(n * 2);
        for row in &data {
            flat.extend_from_slice(row);
        }
        let factors = Array2::from_shape_vec((n, 2), flat).unwrap();

        let result = fit_hdbscan(&factors, 5, 5);
        // Expect 2 clusters, ≥ 2 noise points (the two far outliers).
        assert_eq!(result.assignments.len(), n);
        assert!(
            result.n_clusters >= 2,
            "expected at least 2 clusters, got {}",
            result.n_clusters
        );
        // The two outliers should be -1.
        let n_noise = result.assignments.iter().filter(|&&a| a < 0).count();
        assert!(
            n_noise >= 2,
            "expected at least 2 noise points, got {n_noise}"
        );
    }

    #[test]
    fn empty_input_returns_empty() {
        let factors: Array2<f64> = Array2::zeros((0, 4));
        let result = fit_hdbscan(&factors, 15, 15);
        assert_eq!(result.n_clusters, 0);
        assert!(result.assignments.is_empty());
    }

    #[test]
    fn tiny_input_no_crash() {
        let factors = array![[0.0, 0.0], [1.0, 1.0]];
        let _ = fit_hdbscan(&factors, 15, 15);
    }
}
