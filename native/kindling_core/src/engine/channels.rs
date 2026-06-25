//! Engine fit — channel state, ported for byte-exact parity with the Python
//! engine's channel builds (`engine.py`).
//!
//! Produces, from the preprocessed interaction arrays + resolved config:
//!   * `item_popularity` — per-item interaction counts (over n_items_ext).
//!   * `trend_z` — recency-window item counts, z-normalized across the catalog
//!     (empty if not built / degenerate).
//!   * user-CF inverted CSR — `(uu_users_data, uu_users_indptr, uu_user_deg)`:
//!     item → sorted unique user ids, per-item offsets, and per-user degree.
//!
//! Transitions are the existing `build_directional_cooc` kernel; last-item is
//! computed at recommend time from EASE — neither is fit-state here.

use pyo3::prelude::*;

type ChannelState = (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>, Vec<f64>);

#[pyfunction]
#[pyo3(signature = (
    user_idx, item_idx, timestamps, n_users, n_items, n_items_ext,
    trend_window_fraction, want_trend, want_user_cf,
))]
#[allow(clippy::too_many_arguments)]
fn fit_channels(
    user_idx: numpy::PyReadonlyArray1<'_, i64>,
    item_idx: numpy::PyReadonlyArray1<'_, i64>,
    timestamps: Option<numpy::PyReadonlyArray1<'_, f64>>,
    n_users: usize,
    n_items: usize,
    n_items_ext: usize,
    trend_window_fraction: f64,
    want_trend: bool,
    want_user_cf: bool,
) -> PyResult<ChannelState> {
    let u = user_idx.as_slice()?;
    let it = item_idx.as_slice()?;
    let len = it.len();

    // item_popularity = bincount(item_idx, minlength=n_items_ext)
    let mut pop = vec![0f64; n_items_ext];
    for &x in it {
        pop[x as usize] += 1.0;
    }

    // trend_z = z(recency-window counts) over the full catalog (population std).
    let mut trend: Vec<f64> = Vec::new();
    if want_trend {
        if let Some(ts) = timestamps.as_ref() {
            let ts = ts.as_slice()?;
            if !ts.is_empty() {
                let t_hi = ts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let t_lo = ts.iter().copied().fold(f64::INFINITY, f64::min);
                if t_hi > t_lo {
                    let cut = t_hi - (t_hi - t_lo) * trend_window_fraction;
                    let mut counts = vec![0f64; n_items_ext];
                    for k in 0..len {
                        if ts[k] >= cut {
                            counts[it[k] as usize] += 1.0;
                        }
                    }
                    let nn = n_items_ext as f64;
                    let mean = counts.iter().sum::<f64>() / nn;
                    let std = (counts.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / nn).sqrt();
                    if std > 0.0 {
                        trend = counts.iter().map(|c| (c - mean) / std).collect();
                    }
                }
            }
        }
    }

    // user-CF: item → sorted unique users CSR, from binarized (user,item) pairs.
    let mut uu_data: Vec<i64> = Vec::new();
    let mut uu_indptr: Vec<i64> = Vec::new();
    let mut uu_deg: Vec<f64> = Vec::new();
    if want_user_cf {
        let ni = n_items as i64;
        let mut keys: Vec<i64> = (0..len).map(|k| u[k] * ni + it[k]).collect();
        keys.sort_unstable();
        keys.dedup();
        // (item, user) ascending == Python's stable sort-by-item on the
        // user-sorted unique keys (within an item, users ascending).
        let mut pairs: Vec<(i64, i64)> = keys.iter().map(|&kk| (kk % ni, kk / ni)).collect();
        pairs.sort_unstable();
        uu_data = pairs.iter().map(|&(_, uu)| uu).collect();
        let mut indptr = vec![0i64; n_items + 1];
        for &(i, _) in &pairs {
            indptr[i as usize + 1] += 1;
        }
        for k in 1..=n_items {
            indptr[k] += indptr[k - 1];
        }
        uu_indptr = indptr;
        let mut deg = vec![0f64; n_users];
        for &(_, uu) in &pairs {
            deg[uu as usize] += 1.0;
        }
        uu_deg = deg;
    }

    Ok((pop, trend, uu_data, uu_indptr, uu_deg))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_channels, m)?)?;
    Ok(())
}
