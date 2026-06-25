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

/// Full channel blend — port of `_blend_channels` (engine.py). Base-agnostic:
/// `base_vec` is the raw (pre-z-norm) full-catalog base (EASE sum or cooc
/// accumulation); returns the blended full-catalog score vector. Covers the
/// trend / user-CF / last-item / transitions channels (content is off in every
/// reference dataset and is not ported here — guarded by alpha == 0).
///
/// Channel state is passed as numpy arrays; an inactive channel is signalled by
/// alpha == 0 and/or an empty array. `user_row_items` is a CSR
/// (`uri_data`/`uri_indptr`, indexed by user row) of each user's owned items,
/// for neighbor voting.
#[pyfunction]
#[pyo3(signature = (
    base_vec, owned, n_items,
    trend_z, trend_alpha,
    last_row, last_item_alpha,
    trans_data, trans_indices, trans_indptr, transition_alpha, transition_last_k, transition_decay,
    uu_data, uu_indptr, uu_deg, uri_data, uri_indptr, user_cf_alpha, user_cf_k, user_row, n_users,
))]
#[allow(clippy::too_many_arguments)]
fn blend_channels(
    base_vec: numpy::PyReadonlyArray1<'_, f64>,
    owned: numpy::PyReadonlyArray1<'_, i64>,
    n_items: usize,
    trend_z: numpy::PyReadonlyArray1<'_, f64>,
    trend_alpha: f64,
    last_row: numpy::PyReadonlyArray1<'_, f64>,
    last_item_alpha: f64,
    trans_data: numpy::PyReadonlyArray1<'_, f64>,
    trans_indices: numpy::PyReadonlyArray1<'_, i32>,
    trans_indptr: numpy::PyReadonlyArray1<'_, i64>,
    transition_alpha: f64,
    transition_last_k: usize,
    transition_decay: f64,
    uu_data: numpy::PyReadonlyArray1<'_, i64>,
    uu_indptr: numpy::PyReadonlyArray1<'_, i64>,
    uu_deg: numpy::PyReadonlyArray1<'_, f64>,
    uri_data: numpy::PyReadonlyArray1<'_, i64>,
    uri_indptr: numpy::PyReadonlyArray1<'_, i64>,
    user_cf_alpha: f64,
    user_cf_k: usize,
    user_row: i64,
    n_users: usize,
) -> PyResult<Vec<f64>> {
    let owned = owned.as_slice()?;
    let tz = trend_z.as_slice()?;
    let lr = last_row.as_slice()?;
    let mut score: Vec<f64> = base_vec.as_slice()?.to_vec();

    let trend_on = trend_alpha > 0.0 && tz.len() == n_items;
    let last_on = last_item_alpha > 0.0 && lr.len() == n_items && !owned.is_empty();
    let uu_ip = uu_indptr.as_slice()?;
    let uu_on = user_cf_alpha > 0.0 && !uu_ip.is_empty() && !owned.is_empty();
    let tr_ip = trans_indptr.as_slice()?;
    let trans_on = transition_alpha > 0.0 && !tr_ip.is_empty();
    if !(trend_on || last_on || uu_on || trans_on) {
        return Ok(score);
    }

    // z-normalize base (population).
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
    // trend (pre-z-normed at fit).
    if trend_on {
        for (s, &x) in score.iter_mut().zip(tz) {
            *s += trend_alpha * x;
        }
    }
    // user-CF: Otsuka-Ochiai k-NN over the inverted index.
    if uu_on {
        let uu_d = uu_data.as_slice()?;
        let deg = uu_deg.as_slice()?;
        let uri_d = uri_data.as_slice()?;
        let uri_ip = uri_indptr.as_slice()?;
        let mut counts = vec![0f64; n_users];
        for &i in owned {
            let i = i as usize;
            for &u in &uu_d[uu_ip[i] as usize..uu_ip[i + 1] as usize] {
                counts[u as usize] += 1.0;
            }
        }
        if user_row >= 0 && (user_row as usize) < n_users {
            counts[user_row as usize] = 0.0;
        }
        let denom = (owned.len().max(1) as f64).sqrt();
        let nz: Vec<usize> = (0..n_users).filter(|&u| counts[u] != 0.0).collect();
        if !nz.is_empty() {
            let sims: Vec<f64> = nz.iter().map(|&u| counts[u] / (deg[u].sqrt() * denom)).collect();
            // Deterministic top-k: similarity desc, ties broken by ascending
            // position (== ascending user row, since nz is ascending). Matches
            // the Python `argsort(-sims, kind="stable")[:k]` byte-for-byte.
            let mut order: Vec<usize> = (0..nz.len()).collect();
            if nz.len() > user_cf_k {
                order.select_nth_unstable_by(user_cf_k, |&a, &b| {
                    sims[b]
                        .partial_cmp(&sims[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                order.truncate(user_cf_k);
            }
            let mut uu_vec = vec![0f64; n_items];
            for &o in &order {
                let v = nz[o];
                let sim = sims[o];
                for &it in &uri_d[uri_ip[v] as usize..uri_ip[v + 1] as usize] {
                    uu_vec[it as usize] += sim;
                }
            }
            z_in_place(&mut score, user_cf_alpha, &uu_vec);
        }
    }
    // last-item.
    if last_on {
        z_in_place(&mut score, last_item_alpha, lr);
    }
    // transitions: decay-weighted over the most-recent items.
    if trans_on {
        let td = trans_data.as_slice()?;
        let ti = trans_indices.as_slice()?;
        let mut trans = vec![0f64; n_items];
        for (j, &item) in owned.iter().rev().take(transition_last_k).enumerate() {
            let item = item as usize;
            // float pow (not powi) to byte-match numpy's `decay ** j`.
            let w = transition_decay.powf(j as f64);
            for k in tr_ip[item] as usize..tr_ip[item + 1] as usize {
                trans[ti[k] as usize] += w * td[k];
            }
        }
        z_in_place(&mut score, transition_alpha, &trans);
    }
    Ok(score)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recommend_ease_blend, m)?)?;
    m.add_function(wrap_pyfunction!(blend_channels, m)?)?;
    Ok(())
}
