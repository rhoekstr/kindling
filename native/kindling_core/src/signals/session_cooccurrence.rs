//! Session-row cooccurrence: `S.T @ S` over (session_id, item_id) instead
//! of (entity_id, item_id). See PRD §"Boost layer specification" — sparse
//! signal, plan-gated by `deep_session_check`.
//!
//! Algorithmically identical to global cooc; only the bipartite row
//! dimension changes (sessions vs users). The Rust impl thin-wraps the
//! cooccurrence builder by treating session_idx as the "user" axis.
//!
//! Note: `temporal_cooccurrence` is **not a separate module**. It is
//! `cooccurrence::build_cooccurrence(kernel="hybrid_temporal")` with the
//! `time_decay_half_life_days` knob from `LayerPlan`. The v1 GMM-based
//! midpoint/steepness fitting is replaced by the single profile-driven
//! decay knob per PRD §"Profile → Plan contract".

use pyo3::prelude::*;

use super::cooccurrence::Kernel;

/// Build session-row cooccurrence. Identical mechanics to
/// `build_cooccurrence`, but the bipartite row axis is `session_idx`
/// rather than `user_idx`.
///
/// `min_deep_session_fraction` is enforced *outside* this function — the
/// caller (Python or Rust orchestrator) decides whether to build at all
/// based on the profile's deep-session signal.
#[pyfunction]
#[pyo3(signature = (
    session_idx,
    item_idx,
    weights,
    n_sessions,
    n_items,
    kernel = "pure_count",
    alpha = 1.0,
    half_life_days = 30.0,
    timestamps = None,
))]
#[allow(clippy::too_many_arguments)]
fn build_session_cooccurrence(
    session_idx: numpy::PyReadonlyArray1<'_, i64>,
    item_idx: numpy::PyReadonlyArray1<'_, i64>,
    weights: numpy::PyReadonlyArray1<'_, f32>,
    n_sessions: usize,
    n_items: usize,
    kernel: &str,
    alpha: f64,
    half_life_days: f64,
    timestamps: Option<numpy::PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    // Delegate to the same algorithm by reusing the inner builder.
    // We re-implement the build inline so we don't need to expose the
    // private helpers in cooccurrence.rs — and because the only thing
    // that differs is the docstring.
    use rustc_hash::FxHashMap;

    let kernel = Kernel::from_str(kernel, alpha, half_life_days)?;
    let session_idx = session_idx.as_slice()?;
    let item_idx = item_idx.as_slice()?;
    let weights = weights.as_slice()?;
    if session_idx.len() != item_idx.len() || session_idx.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "session_idx, item_idx, weights must have equal length",
        ));
    }
    let timestamps = match (kernel, &timestamps) {
        (Kernel::HybridTemporal { .. }, Some(ts)) => Some(ts.as_slice()?),
        (Kernel::HybridTemporal { .. }, None) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "kernel='hybrid_temporal' requires timestamps",
            ))
        }
        _ => None,
    };

    let mut by_session: Vec<Vec<(usize, f32, f64)>> = vec![Vec::new(); n_sessions];
    for k in 0..session_idx.len() {
        let s = session_idx[k] as usize;
        let i = item_idx[k] as usize;
        if s >= n_sessions || i >= n_items {
            continue;
        }
        let w = weights[k];
        let t = timestamps.map_or(0.0, |ts| ts[k]);
        by_session[s].push((i, w, t));
    }

    let mut pairs: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    let half_life_seconds = match kernel {
        Kernel::HybridTemporal { half_life_days, .. } => half_life_days * 86_400.0,
        _ => 0.0,
    };
    let alpha_v = match kernel {
        Kernel::HybridTemporal { alpha, .. } => alpha,
        _ => 0.0,
    };
    for items in &by_session {
        for a in 0..items.len() {
            let (ia, wa, ta) = items[a];
            for b in (a + 1)..items.len() {
                let (ib, wb, tb) = items[b];
                if ia == ib {
                    continue;
                }
                let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
                let pair_weight = match kernel {
                    Kernel::PureCount => (wa * wb) as f64,
                    Kernel::HybridTemporal { .. } => {
                        let dt = (ta - tb).abs();
                        let x = -dt / half_life_seconds.max(1.0);
                        let logistic = 1.0 / (1.0 + (-x).exp());
                        (wa * wb) as f64 * (1.0 + alpha_v * logistic)
                    }
                };
                *pairs.entry((lo as u32, hi as u32)).or_insert(0.0) += pair_weight;
            }
        }
    }

    let mut rows: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_items];
    for ((lo, hi), v) in pairs {
        let lo = lo as usize;
        let hi = hi as usize;
        let value = v as f32;
        rows[lo].push((hi as i32, value));
        rows[hi].push((lo as i32, value));
    }
    let mut indptr = Vec::with_capacity(n_items + 1);
    indptr.push(0i32);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for row in rows.iter_mut() {
        row.sort_by_key(|(c, _)| *c);
        for (c, v) in row.iter() {
            indices.push(*c);
            data.push(*v);
        }
        indptr.push(indices.len() as i32);
    }
    Ok((data, indices, indptr))
}

/// Compute the deep-session fraction: fraction of sessions with ≥ 2
/// distinct items. The plan's deep-session gate uses this against
/// `min_deep_session_fraction` (default 0.3). Below the threshold,
/// session_cooccurrence collapses to a degenerate sparse copy of global
/// cooc and adds noise; the plan should drop it from
/// `enabled_boost_layers`.
#[pyfunction]
fn deep_session_fraction(session_idx: Vec<i64>, item_idx: Vec<i64>) -> f64 {
    use rustc_hash::FxHashMap;
    if session_idx.is_empty() {
        return 0.0;
    }
    let mut sets: FxHashMap<i64, std::collections::HashSet<i64>> = FxHashMap::default();
    for k in 0..session_idx.len().min(item_idx.len()) {
        sets.entry(session_idx[k])
            .or_default()
            .insert(item_idx[k]);
    }
    let total = sets.len();
    if total == 0 {
        return 0.0;
    }
    let deep = sets.values().filter(|s| s.len() >= 2).count();
    deep as f64 / total as f64
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_session_cooccurrence, m)?)?;
    m.add_function(wrap_pyfunction!(deep_session_fraction, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn deep_session_fraction_basic() {
        // 3 sessions: session 0 has {a, b}, session 1 has {c}, session 2 has {d, e}
        // → deep = 2 / 3 ≈ 0.667
        let session_idx = vec![0i64, 0, 1, 2, 2];
        let item_idx = vec![10i64, 11, 20, 30, 31];
        let f = super::deep_session_fraction(session_idx, item_idx);
        assert!((f - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn deep_session_fraction_singletons() {
        // All sessions are singletons → fraction = 0.
        let session_idx = vec![0i64, 1, 2];
        let item_idx = vec![10i64, 20, 30];
        assert_eq!(super::deep_session_fraction(session_idx, item_idx), 0.0);
    }
}
