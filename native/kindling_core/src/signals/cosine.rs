//! Item-item cosine similarity from a cooccurrence matrix.
//!
//! Sparse cosine kNN (Sarwar et al. 2001). On session-poor datasets it's
//! the strongest single-signal baseline, and as a v2 boost layer it
//! contributes wherever the cooc base alone over-rewards popularity.
//!
//! Build:
//! - Input: cooc CSR (n_items × n_items) + per-item user counts.
//! - `cosine[i, j] = cooc[i, j] / sqrt(count[i] · count[j])`
//! - Strip diagonal, drop entries below `min_cosine`, top-K-per-row
//!   prune for memory.
//!
//! Score: same shape as cooc — at recommend time, `score(c) = Σ_o
//! cosine[c, o]`. Reuses the v2 `cooccurrence_signal` PyO3 function.

use pyo3::prelude::*;

/// Build cosine CSR from cooc CSR + item counts.
///
/// `top_k = 0` disables the top-K prune (dense cosine; only sane for
/// small catalogs).
#[pyfunction]
#[pyo3(signature = (
    cooc_data,
    cooc_indices,
    cooc_indptr,
    item_counts,
    top_k = 200,
    min_cosine = 0.01,
))]
#[allow(clippy::too_many_arguments)]
fn build_item_cosine(
    cooc_data: numpy::PyReadonlyArray1<'_, f32>,
    cooc_indices: numpy::PyReadonlyArray1<'_, i32>,
    cooc_indptr: numpy::PyReadonlyArray1<'_, i32>,
    item_counts: numpy::PyReadonlyArray1<'_, i64>,
    top_k: usize,
    min_cosine: f64,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    let data = cooc_data.as_slice()?;
    let indices = cooc_indices.as_slice()?;
    let indptr = cooc_indptr.as_slice()?;
    let counts = item_counts.as_slice()?;
    let n_items = indptr.len().saturating_sub(1);
    if n_items == 0 {
        return Ok((Vec::new(), Vec::new(), vec![0i32]));
    }
    // Pre-compute 1 / sqrt(count) per item (clamp count to 1 to avoid
    // divide-by-zero on items the caller hasn't filtered).
    let inv_sqrt_count: Vec<f64> = counts
        .iter()
        .map(|&c| 1.0 / ((c.max(1) as f64).sqrt()))
        .collect();

    // Build per-row (col, cos) lists, applying min_cosine + top-K.
    let mut out_data: Vec<f32> = Vec::new();
    let mut out_indices: Vec<i32> = Vec::new();
    let mut out_indptr: Vec<i32> = Vec::with_capacity(n_items + 1);
    out_indptr.push(0i32);

    for i in 0..n_items {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        let mut row: Vec<(i32, f32)> = Vec::with_capacity(end - start);
        let inv_i = inv_sqrt_count[i];
        for k in start..end {
            let j = indices[k] as usize;
            if j == i {
                continue; // Strip diagonal.
            }
            let cos_val = (data[k] as f64) * inv_i * inv_sqrt_count[j];
            if cos_val < min_cosine {
                continue;
            }
            row.push((j as i32, cos_val as f32));
        }
        // Top-K prune per row.
        if top_k > 0 && row.len() > top_k {
            // Partial sort: nth_element by descending value.
            row.select_nth_unstable_by(top_k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            row.truncate(top_k);
        }
        // Sort by column for canonical CSR layout.
        row.sort_by_key(|(c, _)| *c);
        for (c, v) in &row {
            out_indices.push(*c);
            out_data.push(*v);
        }
        out_indptr.push(out_indices.len() as i32);
    }
    Ok((out_data, out_indices, out_indptr))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_item_cosine, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build cosine from a tiny cooc + counts and verify cells.
    /// cooc:
    ///   row 0: {1: 2}
    ///   row 1: {0: 2, 2: 1}
    ///   row 2: {1: 1}
    /// counts: [2, 2, 1]
    /// cosine(0, 1) = 2 / sqrt(2 * 2) = 1.0
    /// cosine(1, 2) = 1 / sqrt(2 * 1) ≈ 0.707
    #[test]
    fn cosine_cells_have_expected_values() {
        let data = vec![2.0f32, 2.0, 1.0, 1.0];
        let indices = vec![1i32, 0, 2, 1];
        let indptr = vec![0i32, 1, 3, 4];
        let counts = vec![2i64, 2, 1];

        // Inline the algorithm so we don't have to wrestle with PyArrays in tests.
        let inv_sqrt_count: Vec<f64> =
            counts.iter().map(|&c| 1.0 / ((c.max(1) as f64).sqrt())).collect();
        // cell (0, 1)
        let lo = indptr[0] as usize;
        let _hi = indptr[1] as usize;
        let cos_0_1 = (data[lo] as f64) * inv_sqrt_count[0] * inv_sqrt_count[1];
        assert!((cos_0_1 - 1.0).abs() < 1e-9, "cos(0,1) = {cos_0_1}");
        // cell (1, 2): in row 1, col 2 is at index 2.
        let cos_1_2 = (data[2] as f64) * inv_sqrt_count[1] * inv_sqrt_count[2];
        assert!(
            (cos_1_2 - 1.0 / 2.0_f64.sqrt()).abs() < 1e-9,
            "cos(1,2) = {cos_1_2}"
        );
    }
}
