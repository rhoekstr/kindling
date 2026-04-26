//! Persona-fit gate (PRD §"Persona-fit gate spec").
//!
//! Given a user's owned items and a persona's distinctive-items list,
//! compute:
//!
//! ```text
//! fit = |user.owned_items ∩ distinctive_items[persona]| / |user.owned_items|
//! ```
//!
//! The base-routing decision is then:
//!
//! ```text
//! USE_PERSONA_BASE iff cluster_id != -1 AND fit >= 0.70
//! ```
//!
//! Why `distinctive_items` (z-filter survivors) and not the full persona
//! vector: items below the z-filter cutoff are noise the cluster doesn't
//! over-index on. Counting them in the denominator would tautologically
//! push fit toward 1.0 for any cluster member. Distinctive items are
//! items the cluster's signature actually depends on.

use pyo3::prelude::*;
use std::collections::HashSet;

/// Compute the persona-fit fraction for a single user against a single
/// persona's distinctive-items list.
///
/// Returns 0.0 if the user has no items, the persona has no distinctive
/// items, or the persona id is invalid.
pub fn persona_fit(user_owned_items: &[i32], distinctive_items: &[i32]) -> f64 {
    if user_owned_items.is_empty() || distinctive_items.is_empty() {
        return 0.0;
    }
    let distinctive: HashSet<i32> = distinctive_items.iter().copied().collect();
    let hits = user_owned_items
        .iter()
        .filter(|&&i| distinctive.contains(&i))
        .count();
    hits as f64 / user_owned_items.len() as f64
}

/// PyO3 wrapper for `persona_fit`.
#[pyfunction]
fn persona_fit_py(user_owned_items: Vec<i32>, distinctive_items: Vec<i32>) -> f64 {
    persona_fit(&user_owned_items, &distinctive_items)
}

/// Should this user use the persona base (vs falling back to global cooc)?
///
/// Implements the two-gate routing logic from the PRD:
/// - HDBSCAN noise (`cluster_id == -1`) → false (use cooc)
/// - In-cluster + fit >= threshold → true (use persona_cooc)
/// - In-cluster + fit < threshold → false (use cooc; persona doesn't enrich)
#[pyfunction]
#[pyo3(signature = (cluster_id, user_owned_items, distinctive_items, threshold = 0.70))]
fn should_use_persona_base(
    cluster_id: i64,
    user_owned_items: Vec<i32>,
    distinctive_items: Vec<i32>,
    threshold: f64,
) -> bool {
    if cluster_id < 0 {
        return false;
    }
    persona_fit(&user_owned_items, &distinctive_items) >= threshold
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persona_fit_py, m)?)?;
    m.add_function(wrap_pyfunction!(should_use_persona_base, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_inputs_zero() {
        assert_eq!(persona_fit(&[], &[1, 2, 3]), 0.0);
        assert_eq!(persona_fit(&[1, 2], &[]), 0.0);
        assert_eq!(persona_fit(&[], &[]), 0.0);
    }

    #[test]
    fn full_overlap_unit() {
        assert_eq!(persona_fit(&[1, 2, 3], &[1, 2, 3, 4, 5]), 1.0);
    }

    #[test]
    fn partial_overlap_fraction() {
        // user owns {1, 2, 3, 4}; distinctive = {2, 3}; fit = 2/4 = 0.5
        let f = persona_fit(&[1, 2, 3, 4], &[2, 3]);
        assert!((f - 0.5).abs() < 1e-9);
    }

    #[test]
    fn duplicate_user_items_count_each() {
        // user has {1, 1, 2}; distinctive = {1}; fit = 2/3.
        let f = persona_fit(&[1, 1, 2], &[1]);
        assert!((f - 2.0 / 3.0).abs() < 1e-9);
    }
}
