//! Per-persona cooccurrence (the v2 persona-base scorer).
//!
//! Builds N item-item cooc matrices (one per persona) using the same
//! kernel + decay knob as global cooc (PRD §"Profile → Plan contract").
//! At recommend time, the engine selects the user's `cluster_id`'s
//! matrix and runs `cooccurrence_signal` against it — the persona-fit
//! gate has already decided this is the right base.
//!
//! Hard one-hot at scoring time per the PRD ("USE_PERSONA_BASE iff
//! cluster_id != -1 AND fit >= 0.7" then `base = persona_cooc(user,
//! persona=cluster)`). No soft cosine across all personas in v2.

use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use super::cooccurrence::Kernel;

/// Build per-persona cooccurrence matrices.
///
/// Returns one CSR triplet `(data, indices, indptr)` per persona,
/// packed into parallel `Vec`s. Personas with fewer than
/// `min_persona_users` distinct users get empty matrices (`indptr =
/// vec![0; n_items + 1]`) and contribute nothing at score time.
///
/// Filtering: noise users (`user_to_persona[u] == -1`) are skipped.
/// Items with `item_idx >= n_items` and users with `user_idx >= n_users`
/// are skipped.
#[allow(clippy::too_many_arguments)]
fn build_one_persona_cooc(
    persona_user_items: &[(usize, usize, f32, f64)], // (user, item, weight, ts)
    n_items: usize,
    kernel: Kernel,
) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    if persona_user_items.is_empty() {
        return (Vec::new(), Vec::new(), vec![0i32; n_items + 1]);
    }
    // Bucket by user.
    let mut by_user: FxHashMap<usize, Vec<(usize, f32, f64)>> = FxHashMap::default();
    for &(u, i, w, t) in persona_user_items {
        by_user.entry(u).or_default().push((i, w, t));
    }

    let half_life_seconds = match kernel {
        Kernel::HybridTemporal { half_life_days, .. } => half_life_days * 86_400.0,
        _ => 0.0,
    };
    let alpha = match kernel {
        Kernel::HybridTemporal { alpha, .. } => alpha,
        _ => 0.0,
    };

    let mut pairs: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    for items in by_user.values() {
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
                        (wa * wb) as f64 * (1.0 + alpha * logistic)
                    }
                };
                *pairs.entry((lo as u32, hi as u32)).or_insert(0.0) += pair_weight;
            }
        }
    }

    // Pack symmetric CSR.
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
    (data, indices, indptr)
}

/// PyO3 wrapper. Returns three parallel Vec<Vec<...>>: each outer index
/// is a persona; the inner vectors are the CSR components for that
/// persona's adjacency. Personas below `min_persona_users` get empty
/// CSRs.
///
/// Plus `persona_sizes: Vec<i64>` — distinct user counts per persona.
#[pyfunction]
#[pyo3(signature = (
    user_idx,
    item_idx,
    weights,
    user_to_persona,
    n_users,
    n_items,
    n_personas,
    kernel = "pure_count",
    alpha = 1.0,
    half_life_days = 30.0,
    timestamps = None,
    min_persona_users = 5,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_persona_cooccurrence(
    user_idx: numpy::PyReadonlyArray1<'_, i64>,
    item_idx: numpy::PyReadonlyArray1<'_, i64>,
    weights: numpy::PyReadonlyArray1<'_, f32>,
    user_to_persona: Vec<i64>,
    n_users: usize,
    n_items: usize,
    n_personas: usize,
    kernel: &str,
    alpha: f64,
    half_life_days: f64,
    timestamps: Option<numpy::PyReadonlyArray1<'_, f64>>,
    min_persona_users: usize,
) -> PyResult<(
    Vec<Vec<f32>>,                      // per-persona CSR data
    Vec<Vec<i32>>,                      // per-persona CSR indices
    Vec<Vec<i32>>,                      // per-persona CSR indptr
    Vec<i64>,                           // persona_sizes
)> {
    let kernel = Kernel::from_str(kernel, alpha, half_life_days)?;
    let user_idx = user_idx.as_slice()?;
    let item_idx = item_idx.as_slice()?;
    let weights = weights.as_slice()?;
    if user_idx.len() != item_idx.len() || user_idx.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "user_idx, item_idx, weights must have equal length",
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

    // Partition (user, item, weight, timestamp) tuples by persona.
    let mut by_persona: Vec<Vec<(usize, usize, f32, f64)>> = vec![Vec::new(); n_personas];
    let mut unique_user_in_persona: Vec<std::collections::HashSet<u32>> =
        (0..n_personas).map(|_| std::collections::HashSet::new()).collect();

    for k in 0..user_idx.len() {
        let u = user_idx[k] as usize;
        let i = item_idx[k] as usize;
        if u >= n_users || i >= n_items {
            continue;
        }
        if u >= user_to_persona.len() {
            continue;
        }
        let p = user_to_persona[u];
        if p < 0 || (p as usize) >= n_personas {
            continue;
        }
        let p = p as usize;
        let w = weights[k];
        let t = timestamps.map_or(0.0, |ts| ts[k]);
        by_persona[p].push((u, i, w, t));
        unique_user_in_persona[p].insert(u as u32);
    }

    let persona_sizes: Vec<i64> = unique_user_in_persona
        .iter()
        .map(|s| s.len() as i64)
        .collect();

    // Build per-persona CSR.
    let mut all_data: Vec<Vec<f32>> = Vec::with_capacity(n_personas);
    let mut all_indices: Vec<Vec<i32>> = Vec::with_capacity(n_personas);
    let mut all_indptr: Vec<Vec<i32>> = Vec::with_capacity(n_personas);
    for (p, items) in by_persona.into_iter().enumerate() {
        let n_p_users = persona_sizes[p];
        if (n_p_users as usize) < min_persona_users || items.is_empty() {
            // Empty CSR: indptr of length n_items+1, all zeros.
            all_data.push(Vec::new());
            all_indices.push(Vec::new());
            all_indptr.push(vec![0i32; n_items + 1]);
            continue;
        }
        let (d, idx, ptr) = build_one_persona_cooc(&items, n_items, kernel);
        all_data.push(d);
        all_indices.push(idx);
        all_indptr.push(ptr);
    }
    Ok((all_data, all_indices, all_indptr, persona_sizes))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_persona_cooccurrence, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_in_persona_only_present_in_that_persona() {
        // 4 users:
        //   user 0, 1 → persona 0; both own {10, 11}
        //   user 2, 3 → persona 1; both own {20, 21}
        // Expected: persona 0 cooc has cell (10,11)=2, others empty.
        //           persona 1 cooc has cell (20,21)=2, others empty.
        let user_to_persona = vec![0i64, 0, 1, 1];
        // Inline build of the items list.
        let mut p0_items: Vec<(usize, usize, f32, f64)> = Vec::new();
        for &u in &[0usize, 1] {
            for &i in &[10usize, 11] {
                p0_items.push((u, i, 1.0, 0.0));
            }
        }
        let (d, idx, ptr) = build_one_persona_cooc(&p0_items, 30, Kernel::PureCount);
        // Cell (10, 11) should have value 2.
        let row10_lo = ptr[10] as usize;
        let row10_hi = ptr[11] as usize;
        let row10_cells: Vec<(i32, f32)> = (row10_lo..row10_hi)
            .map(|k| (idx[k], d[k]))
            .collect();
        let cell_10_11 = row10_cells.iter().find(|(c, _)| *c == 11);
        assert_eq!(cell_10_11.map(|(_, v)| *v), Some(2.0));
        // Verify there's no spurious entry for item 20.
        assert!(row10_cells.iter().all(|(c, _)| *c != 20));
        let _ = user_to_persona; // unused but kept for documentation
    }
}
