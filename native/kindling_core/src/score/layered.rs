//! Layered scorer: base + cumulative one-tailed z-gated boosts.
//! See PRD §"Boost layer specification".
//!
//! Per candidate `c`:
//!
//! ```text
//!   score(c) = base(c) + Σ_layer  boost · I[ z_layer(c) > τ ]
//!     where  boost = boost_multiplier × median(adjacent_gaps(base.top_K))
//! ```
//!
//! Z modes:
//! - `NonzeroSubset` (sparse layers): mean / std over the layer's
//!   non-zero candidates only. Zero-valued candidates never fire.
//! - `CandidatePool` (dense layers like ALS / cosine / LightGCN): mean
//!   / std over the entire candidate-pool distribution.

use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// How a boost layer's z-score is computed for the candidate pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZMode {
    /// Sparse layers: z over the non-zero subset.
    NonzeroSubset,
    /// Dense layers: z over the full candidate-pool distribution.
    CandidatePool,
}

impl ZMode {
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "nonzero" | "nonzero_subset" | "sparse" => Ok(ZMode::NonzeroSubset),
            "pool" | "candidate_pool" | "dense" => Ok(ZMode::CandidatePool),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown z_mode: {other:?}; expected 'nonzero' or 'pool'"
            ))),
        }
    }
}

/// Calibrate the boost magnitude.
///
/// `boost = boost_multiplier × median(adjacent gaps in base.top_K)`.
/// Adjacent gaps are computed on the K largest scores after sorting
/// descending. Zero gaps (ties) are excluded so we measure the typical
/// separation between distinct ranks. Returns 0 when too few non-zero
/// scores to estimate.
pub fn calibrate_boost(base: ArrayView1<f64>, top_k: usize, boost_multiplier: f64) -> f64 {
    if base.len() < 2 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = base.iter().copied().collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(sorted.len());
    let top = &sorted[..k];
    let mut gaps: Vec<f64> = Vec::with_capacity(k.saturating_sub(1));
    for w in top.windows(2) {
        let d = w[0] - w[1];
        if d > 0.0 {
            gaps.push(d);
        }
    }
    if gaps.is_empty() {
        return 0.0;
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if gaps.len() % 2 == 0 {
        (gaps[gaps.len() / 2 - 1] + gaps[gaps.len() / 2]) * 0.5
    } else {
        gaps[gaps.len() / 2]
    };
    boost_multiplier * median
}

/// Apply the layered score formula.
///
/// `base` and every entry of `layers` must have equal length (the
/// candidate-pool size). Mismatched shapes are skipped silently.
pub fn layered_score(
    base: ArrayView1<f64>,
    layers: &[(Vec<f64>, ZMode)],
    z_threshold: f64,
    boost_multiplier: f64,
    top_k_for_calibration: usize,
    min_nonzero_for_zscore: usize,
) -> Array1<f64> {
    let mut out = base.to_owned();
    let boost = calibrate_boost(base, top_k_for_calibration, boost_multiplier);
    if boost <= 0.0 || layers.is_empty() {
        return out;
    }
    let n = base.len();
    for (scores, mode) in layers {
        if scores.len() != n {
            continue;
        }
        // Compute mean + std for the requested z-mode.
        let (mu, sigma) = match mode {
            ZMode::NonzeroSubset => {
                let nz: Vec<f64> = scores.iter().copied().filter(|x| *x > 0.0).collect();
                if nz.len() < min_nonzero_for_zscore {
                    continue;
                }
                let mu = nz.iter().copied().sum::<f64>() / nz.len() as f64;
                let var = nz.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / nz.len() as f64;
                let sigma = var.sqrt().max(1e-9);
                (mu, sigma)
            }
            ZMode::CandidatePool => {
                if scores.len() < min_nonzero_for_zscore {
                    continue;
                }
                let mu = scores.iter().copied().sum::<f64>() / scores.len() as f64;
                let var =
                    scores.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / scores.len() as f64;
                let sigma = var.sqrt().max(1e-9);
                (mu, sigma)
            }
        };

        // Apply gate per candidate. Zero-valued sparse candidates skip
        // automatically (their value is below mu by ≥ 0.5σ in the typical
        // case, so they almost never reach z>τ; we *also* short-circuit
        // them explicitly to mirror the v1 semantics where zeros are
        // never gated for firing).
        for i in 0..n {
            let v = scores[i];
            let candidate_in_subset = match mode {
                ZMode::NonzeroSubset => v > 0.0,
                ZMode::CandidatePool => true,
            };
            if !candidate_in_subset {
                continue;
            }
            let z = (v - mu) / sigma;
            if z > z_threshold {
                out[i] += boost;
            }
        }
    }
    out
}

/// PyO3: layered scoring. `layers_specs` is a list of `(scores, z_mode)`
/// tuples, one per refinement layer.
#[pyfunction]
#[pyo3(signature = (
    base,
    layer_specs,
    z_threshold = 2.5,
    boost_multiplier = 3.0,
    top_k_for_calibration = 20,
    min_nonzero_for_zscore = 3,
))]
fn layered_score_py<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, f64>,
    layer_specs: &Bound<'py, PyList>,
    z_threshold: f64,
    boost_multiplier: f64,
    top_k_for_calibration: usize,
    min_nonzero_for_zscore: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Decode list of (scores: numpy array, z_mode: str) tuples.
    let mut layers: Vec<(Vec<f64>, ZMode)> = Vec::with_capacity(layer_specs.len());
    for spec in layer_specs.iter() {
        let tup = spec.downcast::<pyo3::types::PyTuple>()?;
        let scores: PyReadonlyArray1<'py, f64> = tup.get_item(0)?.extract()?;
        let mode_str: &str = &tup.get_item(1)?.extract::<String>()?;
        layers.push((scores.as_slice()?.to_vec(), ZMode::from_str(mode_str)?));
    }
    let base_view = base.as_array();
    let composite = layered_score(
        base_view,
        &layers,
        z_threshold,
        boost_multiplier,
        top_k_for_calibration,
        min_nonzero_for_zscore,
    );
    Ok(PyArray1::<f64>::from_vec_bound(py, composite.to_vec()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(layered_score_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn calibrate_boost_basic() {
        // sorted desc: [10, 7, 4, 2, 1]; gaps: [3, 3, 2, 1]; median=2.5
        let base = array![1.0, 4.0, 10.0, 2.0, 7.0];
        let b = calibrate_boost(base.view(), 5, 1.0);
        assert!((b - 2.5).abs() < 1e-9, "expected 2.5, got {b}");
        // Multiplier scales the result.
        let b3 = calibrate_boost(base.view(), 5, 3.0);
        assert!((b3 - 7.5).abs() < 1e-9);
    }

    #[test]
    fn calibrate_boost_all_ties_returns_zero() {
        let base = array![5.0, 5.0, 5.0, 5.0];
        assert_eq!(calibrate_boost(base.view(), 4, 1.0), 0.0);
    }

    #[test]
    fn nonzero_z_gate_fires_above_threshold() {
        // Base: 10 candidates with linear scores 1..=10.
        let base: Array1<f64> = (1..=10).map(|i| i as f64).collect();
        // Layer (sparse): [0, 0, 0, 0, 0, 0, 0, 0, 0, 100].
        // Non-zero subset: only candidate 9 has value 100, but
        // min_nonzero_for_zscore=3 → layer skipped entirely.
        let layers_skipped = vec![(
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            ZMode::NonzeroSubset,
        )];
        let s_skipped = layered_score(base.view(), &layers_skipped, 2.5, 3.0, 20, 3);
        // Layer skipped → identical to base.
        assert_eq!(s_skipped.to_vec(), base.to_vec());

        // Now a layer with enough non-zero entries, one of which is far above.
        // values [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]; nz subset has 10 vals,
        // mean ~10.9, std large; the 100 has z ≈ very high.
        let layers_fires = vec![(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0],
            ZMode::NonzeroSubset,
        )];
        let s_fires = layered_score(base.view(), &layers_fires, 2.5, 1.0, 20, 3);
        // boost = 1.0 * median_gap_of_top20(base) = 1.0 * 1.0 = 1.0
        // So candidate 9 (last, base=10) should be boosted to 11.
        assert!(s_fires[9] > base[9], "candidate 9 should be boosted");
    }

    #[test]
    fn dense_z_gate_uses_full_pool() {
        let base: Array1<f64> = (1..=10).map(|i| i as f64).collect();
        // Dense layer: most values clustered at 0.5, one outlier at 5.0.
        let layer: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0];
        // pool mean = (4.5+5.0)/10 = 0.95, std ≈ 1.42, z(5.0) ≈ 2.85 > 2.5
        let layers = vec![(layer, ZMode::CandidatePool)];
        let s = layered_score(base.view(), &layers, 2.5, 1.0, 20, 3);
        assert!(s[9] > base[9], "outlier should fire under candidate-pool z-mode");
    }
}
