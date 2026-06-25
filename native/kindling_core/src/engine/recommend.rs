//! Engine recommend — ported for parity with the Python engine's
//! `_recommend_core` / `_blend_channels` (`engine.py`).
//!
//! Phase 3a: the EASE base + trend + last-item blend (the ml1m path — no boost
//! layers, no user-CF / content / cold-slots, so `layered_score` is identity
//! and top-N is the top of the blended scores). Subsequent commits add the
//! transitions / user-CF channels, the temporal boost layer, the cooc base, and
//! cold-slots.

use ndarray::ArrayView2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

#[inline]
fn z_in_place(v: &mut [f64], alpha: f64, src: &[f64]) {
    // scores += alpha * z(src), population mean/std; no-op if std == 0.
    let n = src.len() as f64;
    let mean = src.iter().sum::<f64>() / n;
    let std = (src.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
    if std > 0.0 {
        for (s, &x) in v.iter_mut().zip(src) {
            *s += alpha * (x - mean) / std;
        }
    }
}

/// Top-n recommendations for the EASE base + trend + last-item blend.
///
/// `ease_b` is the dense n_items×n_items EASE matrix; `trend_z` is the
/// fit-time z-normalized trend (len n_items, or empty). Returns
/// `(item_indices, scores)` of length ≤ n, owned excluded, ordered by score
/// descending with ties broken by ascending index (matches the Python stable
/// retrieval sort; top-N is identity over the descending pool here).
#[pyfunction]
#[pyo3(signature = (ease_b, trend_z, owned, trend_alpha, last_item_alpha, n))]
fn recommend_ease_blend(
    ease_b: PyReadonlyArray2<'_, f32>,
    trend_z: numpy::PyReadonlyArray1<'_, f64>,
    owned: numpy::PyReadonlyArray1<'_, i64>,
    trend_alpha: f64,
    last_item_alpha: f64,
    n: usize,
) -> PyResult<(Vec<i64>, Vec<f64>)> {
    let b: ArrayView2<'_, f32> = ease_b.as_array();
    let n_items = b.ncols();
    let owned = owned.as_slice()?;
    let tz = trend_z.as_slice()?;

    // base_vec = Σ_{o∈owned} ease_b[o, :]
    let mut score = vec![0f64; n_items];
    for &o in owned {
        let row = b.row(o as usize);
        for (s, &x) in score.iter_mut().zip(row.iter()) {
            *s += x as f64;
        }
    }
    // z-normalize the base (population mean/std) before adding channels —
    // matches `_blend_channels` (only when ≥1 channel is active, which it is
    // for the trend + last-item path here).
    {
        let nn = n_items as f64;
        let mean = score.iter().sum::<f64>() / nn;
        let std = (score.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
        if std > 0.0 {
            for s in score.iter_mut() {
                *s = (*s - mean) / std;
            }
        }
    }
    // trend channel (trend_z already z-normed at fit).
    if trend_alpha > 0.0 && tz.len() == n_items {
        for (s, &x) in score.iter_mut().zip(tz) {
            *s += trend_alpha * x;
        }
    }
    // last-item channel: alpha * z(ease_b[owned[-1], :]).
    if last_item_alpha > 0.0 && !owned.is_empty() {
        let last = *owned.last().unwrap() as usize;
        let row: Vec<f64> = b.row(last).iter().map(|&x| x as f64).collect();
        z_in_place(&mut score, last_item_alpha, &row);
    }
    // Exclude owned.
    for &o in owned {
        score[o as usize] = f64::NEG_INFINITY;
    }
    // Top-n by score desc, ties by ascending index (stable).
    let mut idx: Vec<usize> = (0..n_items).filter(|&i| score[i].is_finite()).collect();
    let k = n.min(idx.len());
    let pivot = k.saturating_sub(1).min(idx.len().saturating_sub(1));
    idx.select_nth_unstable_by(pivot, |&a, &b2| {
        score[b2].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b2))
    });
    let mut top: Vec<usize> = idx.into_iter().take(k).collect();
    top.sort_by(|&a, &b2| {
        score[b2].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b2))
    });
    let items: Vec<i64> = top.iter().map(|&i| i as i64).collect();
    let scores: Vec<f64> = top.iter().map(|&i| score[i]).collect();
    Ok((items, scores))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recommend_ease_blend, m)?)?;
    Ok(())
}
