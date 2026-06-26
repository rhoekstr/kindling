//! Engine recommend — ported for parity with the Python engine's
//! `_recommend_core` / `_blend_channels` (`engine.py`).
//!
//! Phase 3a: the EASE base + trend + last-item blend (the ml1m path — no boost
//! layers, no user-CF / content / cold-slots, so `layered_score` is identity
//! and top-N is the top of the blended scores). Subsequent commits add the
//! transitions / user-CF channels, the temporal boost layer, the cooc base, and
//! cold-slots.

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


/// Channel state for [`blend_full`] — slice views, so both the `blend_channels`
/// pyfunction and the native engine can share the blend without re-marshaling.
pub(crate) struct BlendState<'a> {
    pub trend_z: &'a [f64],
    pub trend_alpha: f64,
    pub last_row: &'a [f64],
    pub last_item_alpha: f64,
    pub trans_data: &'a [f64],
    pub trans_indices: &'a [i32],
    pub trans_indptr: &'a [i64],
    pub transition_alpha: f64,
    pub transition_last_k: usize,
    pub transition_decay: f64,
    pub uu_data: &'a [i64],
    pub uu_indptr: &'a [i64],
    pub uu_deg: &'a [f64],
    pub uri_data: &'a [i64],
    pub uri_indptr: &'a [i64],
    pub user_cf_alpha: f64,
    pub user_cf_k: usize,
    pub user_row: i64,
    pub n_users: usize,
    /// Content channel: precomputed `coldness · z(content_scores)` (len n_items
    /// or empty), added as `content_alpha · content_contrib`.
    pub content_alpha: f64,
    pub content_contrib: &'a [f64],
}

/// Port of `_blend_channels`. `score` enters as the raw (pre-z-norm) base
/// vector of length `n_items` and is returned blended. Inactive channels are
/// signalled by alpha == 0 and/or empty arrays. `last_row` (when active) is the
/// last owned item's EASE row, padded to `n_items`.
pub(crate) fn blend_full(mut score: Vec<f64>, n_items: usize, owned: &[i64], b: &BlendState) -> Vec<f64> {
    let trend_on = b.trend_alpha > 0.0 && b.trend_z.len() == n_items;
    let last_on = b.last_item_alpha > 0.0 && b.last_row.len() == n_items && !owned.is_empty();
    let uu_on = b.user_cf_alpha > 0.0 && !b.uu_indptr.is_empty() && !owned.is_empty();
    let trans_on = b.transition_alpha > 0.0 && !b.trans_indptr.is_empty();
    let content_on = b.content_alpha > 0.0 && b.content_contrib.len() == n_items;
    if !(trend_on || last_on || uu_on || trans_on || content_on) {
        return score;
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
        for (s, &x) in score.iter_mut().zip(b.trend_z) {
            *s += b.trend_alpha * x;
        }
    }
    // user-CF: Otsuka-Ochiai k-NN over the inverted index.
    if uu_on {
        let mut counts = vec![0f64; b.n_users];
        for &i in owned {
            let i = i as usize;
            for &u in &b.uu_data[b.uu_indptr[i] as usize..b.uu_indptr[i + 1] as usize] {
                counts[u as usize] += 1.0;
            }
        }
        if b.user_row >= 0 && (b.user_row as usize) < b.n_users {
            counts[b.user_row as usize] = 0.0;
        }
        let denom = (owned.len().max(1) as f64).sqrt();
        let nz: Vec<usize> = (0..b.n_users).filter(|&u| counts[u] != 0.0).collect();
        if !nz.is_empty() {
            let sims: Vec<f64> = nz.iter().map(|&u| counts[u] / (b.uu_deg[u].sqrt() * denom)).collect();
            // Deterministic top-k: similarity desc, ties broken by ascending
            // position (== ascending user row, since nz is ascending). Matches
            // the Python `argsort(-sims, kind="stable")[:k]` byte-for-byte.
            let mut order: Vec<usize> = (0..nz.len()).collect();
            if nz.len() > b.user_cf_k {
                order.select_nth_unstable_by(b.user_cf_k, |&a, &c| {
                    sims[c]
                        .partial_cmp(&sims[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&c))
                });
                order.truncate(b.user_cf_k);
            }
            let mut uu_vec = vec![0f64; n_items];
            for &o in &order {
                let v = nz[o];
                let sim = sims[o];
                for &it in &b.uri_data[b.uri_indptr[v] as usize..b.uri_indptr[v + 1] as usize] {
                    uu_vec[it as usize] += sim;
                }
            }
            z_in_place(&mut score, b.user_cf_alpha, &uu_vec);
        }
    }
    // content channel (precomputed coldness · z(content); direct add).
    if content_on {
        for (s, &c) in score.iter_mut().zip(b.content_contrib) {
            *s += b.content_alpha * c;
        }
    }
    // last-item.
    if last_on {
        z_in_place(&mut score, b.last_item_alpha, b.last_row);
    }
    // transitions: decay-weighted over the most-recent items.
    if trans_on {
        let mut trans = vec![0f64; n_items];
        for (j, &item) in owned.iter().rev().take(b.transition_last_k).enumerate() {
            let item = item as usize;
            // float pow (not powi) to byte-match numpy's `decay ** j`.
            let w = b.transition_decay.powf(j as f64);
            for k in b.trans_indptr[item] as usize..b.trans_indptr[item + 1] as usize {
                trans[b.trans_indices[k] as usize] += w * b.trans_data[k];
            }
        }
        z_in_place(&mut score, b.transition_alpha, &trans);
    }
    score
}

pub(crate) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // No pyfunctions here now — the blend is internal (`blend_full`), exposed
    // only through the native `EngineState`.
    Ok(())
}
