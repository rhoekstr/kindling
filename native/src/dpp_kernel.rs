//! DPP kernel cosine similarity.
//!
//! For each pair of candidates, compute ``dot(row_i, row_j) / (|row_i|
//! * |row_j|)`` where rows come from the item graph. Pair-wise across
//! N candidates is O(N^2 * d) where d is the row density. Python
//! builds a scipy CSR and matrix-multiplies, which materializes the
//! full N^2 table; this path runs the same computation with the inner
//! accumulator in Rust.

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute the symmetric cosine similarity matrix over N item-graph
/// rows. ``row_ptr`` / ``row_ind`` / ``row_dat`` encode one CSR row
/// per candidate (packed back-to-back). ``diagonal`` set to 1.0.
#[pyfunction]
fn cosine_similarity_matrix<'py>(
    py: Python<'py>,
    row_ptr: PyReadonlyArray1<'py, i64>,
    row_ind: PyReadonlyArray1<'py, i32>,
    row_dat: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray2<f64>> {
    let row_ptr = row_ptr.as_slice().expect("row_ptr contiguous");
    let row_ind = row_ind.as_slice().expect("row_ind contiguous");
    let row_dat = row_dat.as_slice().expect("row_dat contiguous");

    let n = row_ptr.len().saturating_sub(1);
    let mut norms = vec![0.0_f64; n];
    for i in 0..n {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut s = 0.0_f64;
        for k in start..end {
            let v = row_dat[k] as f64;
            s += v * v;
        }
        norms[i] = s.sqrt();
    }

    let out = PyArray2::<f64>::zeros_bound(py, [n, n], false);
    let out_slice = unsafe { out.as_slice_mut().unwrap() };

    // Pairwise: for each i, walk j >= i and compute dot.
    for i in 0..n {
        let i_start = row_ptr[i] as usize;
        let i_end = row_ptr[i + 1] as usize;
        for j in i..n {
            if i == j {
                out_slice[i * n + j] = 1.0;
                continue;
            }
            let j_start = row_ptr[j] as usize;
            let j_end = row_ptr[j + 1] as usize;
            // Walk both sorted index arrays (scipy CSR rows have sorted indices).
            let mut a = i_start;
            let mut b = j_start;
            let mut dot = 0.0_f64;
            while a < i_end && b < j_end {
                let ia = row_ind[a];
                let ib = row_ind[b];
                if ia == ib {
                    dot += (row_dat[a] as f64) * (row_dat[b] as f64);
                    a += 1;
                    b += 1;
                } else if ia < ib {
                    a += 1;
                } else {
                    b += 1;
                }
            }
            let denom = norms[i] * norms[j];
            let sim = if denom > 0.0 { dot / denom } else { 0.0 };
            out_slice[i * n + j] = sim;
            out_slice[j * n + i] = sim;
        }
    }
    out
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity_matrix, m)?)?;
    Ok(())
}
