//! Dedup candidates by ``(item_id, max_score)`` across retrievers,
//! preserving the winning candidate's source.
//!
//! Accepts flat (item_id, score, source) arrays and returns the
//! deduped indices sorted descending by score. Ties break on the
//! original order (first occurrence wins).

use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[pyfunction]
fn dedup_max_score(
    item_ids: Vec<i64>,
    scores: Vec<f64>,
    budget: usize,
) -> Vec<usize> {
    let mut best: FxHashMap<i64, (f64, usize)> =
        FxHashMap::with_capacity_and_hasher(item_ids.len(), Default::default());
    for (i, (&item, &score)) in item_ids.iter().zip(scores.iter()).enumerate() {
        match best.get(&item) {
            Some(&(existing_score, _existing_idx)) if existing_score >= score => {}
            _ => {
                best.insert(item, (score, i));
            }
        }
    }
    let mut entries: Vec<(f64, usize)> = best.values().copied().collect();
    // Descending by score, ties break on original index (stable-ish).
    entries.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    entries.truncate(budget);
    entries.into_iter().map(|(_, idx)| idx).collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dedup_max_score, m)?)?;
    Ok(())
}
