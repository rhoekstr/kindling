//! Cooc weight transforms (wilson / cosine / jaccard) — popularity-normalize
//! the raw co-counts so the base stops degenerating toward a popularity ranker.
//!
//! Pure function over the cooc CSR `data` (shape/sparsity preserved); mirrors
//! `kindling.graph.cooc_transform.apply_cooc_transform` exactly so the
//! Rust-native fit produces a byte-identical base. `item_counts[i]` is item i's
//! marginal (distinct users), floored at 1.

use pyo3::prelude::*;

#[inline]
fn wilson_lb(phat: f64, n: f64, z: f64, z2: f64) -> f64 {
    // phat is a conditional probability; weighted/multi-session co-counts can
    // push c/d past 1, so clip before the variance sqrt.
    let p = phat.clamp(0.0, 1.0);
    (p + z2 / (2.0 * n) - z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / (1.0 + z2 / n)
}

/// Rescale cooc CSR `data` by `transform`. Returns the new data array.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, item_counts, transform, wilson_z = 1.96))]
fn cooc_transform(
    data: numpy::PyReadonlyArray1<'_, f32>,
    indices: numpy::PyReadonlyArray1<'_, i32>,
    indptr: numpy::PyReadonlyArray1<'_, i32>,
    item_counts: numpy::PyReadonlyArray1<'_, f64>,
    transform: &str,
    wilson_z: f64,
) -> PyResult<Vec<f32>> {
    let d = data.as_slice()?;
    let ind = indices.as_slice()?;
    let ip = indptr.as_slice()?;
    let cnt = item_counts.as_slice()?;
    if transform == "raw" {
        return Ok(d.to_vec());
    }
    let n_items = ip.len().saturating_sub(1);
    let z = wilson_z;
    let z2 = z * z;
    let mut out = vec![0f32; d.len()];
    for i in 0..n_items {
        let di = cnt.get(i).copied().unwrap_or(1.0).max(1.0);
        for k in (ip[i] as usize)..(ip[i + 1] as usize) {
            let j = ind[k] as usize;
            let dj = cnt.get(j).copied().unwrap_or(1.0).max(1.0);
            let c = d[k] as f64;
            let v = match transform {
                "cosine" => c / (di * dj).sqrt(),
                "jaccard" => c / (di + dj - c).max(1.0),
                "wilson" => wilson_lb(c / di, di, z, z2).min(wilson_lb(c / dj, dj, z, z2)),
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown cooc transform: {other:?}"
                    )))
                }
            };
            out[k] = v as f32;
        }
    }
    Ok(out)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cooc_transform, m)?)?;
    Ok(())
}
