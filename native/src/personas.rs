//! Persona signal kernels (PRD supplement: persona_signal §2.3).
//!
//! The Rust-first kernel here is rate aggregation: given user→persona
//! assignments and the raw interactions, compute per-persona item rates
//! (fraction of persona members who interacted with each item). This is
//! a scatter-add over interactions that the Python equivalent would do
//! as a sparse indicator-matrix multiply - Rust is tighter for the same
//! work and integrates cleanly with the existing native kernels.
//!
//! Downstream operations (z-score filter, TF-IDF, L2 normalization,
//! online matching) are linear algebra over sparse matrices where
//! numpy/scipy already calls out to C - Python handles them natively.

use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

/// Compute per-persona item rates from interactions.
///
/// Inputs:
/// - ``user_to_persona``: length = n_users. Maps user index → persona index,
///   or -1 for unassigned (noise) users.
/// - ``interaction_users`` / ``interaction_items``: parallel arrays of length
///   n_interactions. One row per (user, item) pair.
/// - ``n_personas``: total persona count.
/// - ``n_items``: total catalog size.
///
/// Returns ``(persona_sizes, rate_rows, rate_cols, rate_values)``:
/// - ``persona_sizes[p]`` = number of unique users in persona p.
/// - ``(rate_rows[k], rate_cols[k], rate_values[k])`` is the k-th
///   non-zero entry of the (n_personas × n_items) rate matrix, where
///   ``rate(p, i) = n_unique_users_in_p_who_saw_i / persona_sizes[p]``.
///
/// Noise users (persona = -1) contribute nothing. Personas with zero
/// size produce no output rows.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn persona_rates(
    user_to_persona: Vec<i64>,
    interaction_users: Vec<i64>,
    interaction_items: Vec<i64>,
    n_personas: i64,
    n_items: i64,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<f64>)> {
    if n_personas <= 0 || n_items <= 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    let n_personas_u = n_personas as usize;

    // First pass: count unique users per persona (ignoring noise).
    let mut persona_sizes = vec![0i64; n_personas_u];
    let mut seen_users: Vec<HashSet<i64>> = (0..n_personas_u).map(|_| HashSet::new()).collect();
    for (u_idx, &p_idx) in user_to_persona.iter().enumerate() {
        if p_idx < 0 || p_idx >= n_personas {
            continue;
        }
        let u = u_idx as i64;
        let p = p_idx as usize;
        if seen_users[p].insert(u) {
            persona_sizes[p] += 1;
        }
    }

    // Second pass: aggregate (persona, item) → count of UNIQUE users.
    // FxHashMap keyed by (persona_idx, item_idx) for memory locality.
    let mut counts: FxHashMap<(i32, i32), FxHashMap<i64, ()>> = FxHashMap::default();
    if interaction_users.len() != interaction_items.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "interaction_users and interaction_items must have equal length",
        ));
    }
    for i in 0..interaction_users.len() {
        let u = interaction_users[i];
        let item = interaction_items[i];
        if u < 0 || item < 0 || item >= n_items {
            continue;
        }
        let u_idx = u as usize;
        if u_idx >= user_to_persona.len() {
            continue;
        }
        let p_idx = user_to_persona[u_idx];
        if p_idx < 0 || p_idx >= n_personas {
            continue;
        }
        let key = (p_idx as i32, item as i32);
        counts.entry(key).or_default().insert(u, ());
    }

    let mut rate_rows: Vec<i64> = Vec::with_capacity(counts.len());
    let mut rate_cols: Vec<i64> = Vec::with_capacity(counts.len());
    let mut rate_values: Vec<f64> = Vec::with_capacity(counts.len());
    for ((p, i), users) in counts.into_iter() {
        let size = persona_sizes[p as usize];
        if size == 0 {
            continue;
        }
        let rate = users.len() as f64 / size as f64;
        if rate > 0.0 {
            rate_rows.push(p as i64);
            rate_cols.push(i as i64);
            rate_values.push(rate);
        }
    }
    Ok((persona_sizes, rate_rows, rate_cols, rate_values))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persona_rates, m)?)?;
    Ok(())
}
