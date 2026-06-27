//! Fit-time repeat profile.
//!
//! For each (user, item) pair seen `>= min_count` times, records the reorder
//! count, the last interaction timestamp, and — when timestamps are present —
//! the characteristic repurchase period + fit quality (`detect_period`). Output
//! is a per-user CSR consumed by the recommend-time repeat multiplier
//! (`multiplier_one`): on a repeat dataset, these items are exempted from the
//! owned-mask and re-surfaced when due.
//!
//! Timestamp-less logs (e.g. Instacart) get period = NaN → the multiplier
//! defaults to 1.0, so reorders surface on base affinity alone.

use pyo3::prelude::*;

use super::period::detect_period;

// (indptr[n_users+1], items, counts, last_ts, periods, quality) — per-user CSR.
type RepeatProfile = (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Build the per-user repeat profile from the preprocessed interaction arrays.
#[pyfunction]
#[pyo3(signature = (user_idx, item_idx, timestamps, n_users, min_count = 2))]
fn fit_repeat_profile(
    user_idx: numpy::PyReadonlyArray1<'_, i64>,
    item_idx: numpy::PyReadonlyArray1<'_, i64>,
    timestamps: Option<numpy::PyReadonlyArray1<'_, f64>>,
    n_users: usize,
    min_count: usize,
) -> PyResult<RepeatProfile> {
    let u = user_idx.as_slice()?;
    let it = item_idx.as_slice()?;
    let ts_owned = match &timestamps {
        Some(t) => Some(t.as_slice()?),
        None => None,
    };
    let n = u.len();

    // Sort indices by (user, item, ts) so each (user, item) group is contiguous
    // and chronologically ordered.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        u[a]
            .cmp(&u[b])
            .then(it[a].cmp(&it[b]))
            .then_with(|| match ts_owned {
                Some(ts) => ts[a].partial_cmp(&ts[b]).unwrap_or(std::cmp::Ordering::Equal),
                None => std::cmp::Ordering::Equal,
            })
    });

    let mut indptr = vec![0i64; n_users + 1];
    let mut items: Vec<i64> = Vec::new();
    let mut counts: Vec<f64> = Vec::new();
    let mut last_ts: Vec<f64> = Vec::new();
    let mut periods: Vec<f64> = Vec::new();
    let mut quality: Vec<f64> = Vec::new();

    let mut i = 0usize;
    while i < n {
        let gu = u[order[i]];
        let gi = it[order[i]];
        let mut j = i;
        while j < n && u[order[j]] == gu && it[order[j]] == gi {
            j += 1;
        }
        let count = j - i;
        if count >= min_count {
            let (period, qual, lts) = if let Some(ts) = ts_owned {
                // intervals between consecutive (sorted) interactions
                let mut iv = Vec::with_capacity(count - 1);
                for k in (i + 1)..j {
                    iv.push(ts[order[k]] - ts[order[k - 1]]);
                }
                let (p, q) = detect_period(&iv);
                (p, q, ts[order[j - 1]])
            } else {
                (f64::NAN, 0.0, f64::NAN)
            };
            items.push(gi);
            counts.push(count as f64);
            last_ts.push(lts);
            periods.push(period);
            quality.push(qual);
            indptr[gu as usize + 1] += 1;
        }
        i = j;
    }
    for k in 1..=n_users {
        indptr[k] += indptr[k - 1];
    }
    Ok((indptr, items, counts, last_ts, periods, quality))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_repeat_profile, m)?)?;
    Ok(())
}
