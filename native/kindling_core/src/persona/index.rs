//! Persona index: full v2 pipeline.
//!
//! Pipeline (PRD supplement §2.3):
//! 1. **Rate aggregation** — per (persona, item), fraction of persona
//!    members who interacted with that item.
//! 2. **Z-score filter** — drop items whose rate is more than `z_filter`
//!    std-devs *below* the persona's mean rate. Survivors are the
//!    persona's "distinctive items": items the cluster over-indexes on
//!    relative to its noise floor.
//! 3. **TF-IDF weighting** — `weight(p, i) = log1p(rate(p, i)) ·
//!    log(n_personas / (1 + Σ_p rate(p, i)))`.
//! 4. **L2 normalization** — each persona row is unit-length so cosine
//!    similarity reduces to a dot product at query time.
//!
//! The `distinctive_items` field exposed to the caller is exactly the
//! set the persona-fit gate (PRD §"Persona-fit gate spec") consumes.

use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

/// Output of the v2 persona-index build.
///
/// Returned to Python as a tuple to keep the PyO3 surface flat. The
/// Python wrapper packages it into a dataclass.
pub struct PersonaIndexBuild {
    /// `[p] -> n_unique_users_in_persona_p`, length `n_personas`.
    pub persona_sizes: Vec<i64>,
    /// CSR of the rate matrix `(n_personas, n_items)`. Used for diagnostics
    /// + downstream lookup. Filtered = false (raw rates).
    pub rate_data: Vec<f64>,
    pub rate_indices: Vec<i32>,
    pub rate_indptr: Vec<i32>,
    /// CSR of the TF-IDF + L2-normalized persona vectors `(n_personas, n_items)`.
    /// This is what `match_user` operates on at query time.
    pub tfidf_data: Vec<f64>,
    pub tfidf_indices: Vec<i32>,
    pub tfidf_indptr: Vec<i32>,
    /// IDF vector, length `n_items`.
    pub idf: Vec<f64>,
    /// Distinctive-items lists: `[p] -> sorted Vec<i32>` of item indices
    /// that survived the z-filter for persona `p`. Used by the fit gate.
    pub distinctive_items: Vec<Vec<i32>>,
}

/// Build the v2 persona index.
///
/// Inputs:
/// - `user_to_persona[u]` ∈ {0..n_personas-1, -1=noise}
/// - `interaction_users[k]`, `interaction_items[k]` parallel arrays
/// - `n_personas`, `n_items` catalog sizes
/// - `z_filter` — z-score threshold for the noise filter (default 1.5,
///   single-tailed: keep items with rate > mean - z·std)
#[allow(clippy::too_many_arguments)]
pub fn build_persona_index(
    user_to_persona: &[i64],
    interaction_users: &[i64],
    interaction_items: &[i64],
    n_personas: usize,
    n_items: usize,
    z_filter: f64,
) -> PersonaIndexBuild {
    if n_personas == 0 || n_items == 0 {
        return PersonaIndexBuild {
            persona_sizes: Vec::new(),
            rate_data: Vec::new(),
            rate_indices: Vec::new(),
            rate_indptr: vec![0],
            tfidf_data: Vec::new(),
            tfidf_indices: Vec::new(),
            tfidf_indptr: vec![0],
            idf: vec![0.0; n_items],
            distinctive_items: Vec::new(),
        };
    }

    // ── Stage 1: per-persona unique-user counts (sizes).
    let mut persona_sizes = vec![0i64; n_personas];
    let mut seen_in_persona: Vec<HashSet<i64>> =
        (0..n_personas).map(|_| HashSet::new()).collect();
    for (u_idx, &p) in user_to_persona.iter().enumerate() {
        if p < 0 || (p as usize) >= n_personas {
            continue;
        }
        let p = p as usize;
        if seen_in_persona[p].insert(u_idx as i64) {
            persona_sizes[p] += 1;
        }
    }

    // ── Stage 2: per-persona unique-user count per item (rates numerator).
    // Use FxHashMap keyed by (persona, item) → set of users for dedup.
    let mut counts: FxHashMap<(u32, u32), FxHashMap<i64, ()>> = FxHashMap::default();
    let nrec = interaction_users.len().min(interaction_items.len());
    for k in 0..nrec {
        let u = interaction_users[k];
        let i = interaction_items[k];
        if u < 0 || i < 0 || (i as usize) >= n_items {
            continue;
        }
        let u_idx = u as usize;
        if u_idx >= user_to_persona.len() {
            continue;
        }
        let p = user_to_persona[u_idx];
        if p < 0 || (p as usize) >= n_personas {
            continue;
        }
        counts
            .entry((p as u32, i as u32))
            .or_default()
            .insert(u, ());
    }

    // Group by persona row to build CSR.
    let mut by_persona: Vec<Vec<(i32, f64)>> = vec![Vec::new(); n_personas];
    for ((p, i), users) in counts.into_iter() {
        let p = p as usize;
        let size = persona_sizes[p];
        if size == 0 {
            continue;
        }
        let rate = users.len() as f64 / size as f64;
        by_persona[p].push((i as i32, rate));
    }
    for row in by_persona.iter_mut() {
        row.sort_by_key(|(c, _)| *c);
    }

    // ── Stage 3: z-filter per persona. Survivors = distinctive_items[p].
    // Compute mean + std of the persona's non-zero rates and keep items
    // with rate > mean - z_filter*std.
    let mut filtered_by_persona: Vec<Vec<(i32, f64)>> = vec![Vec::new(); n_personas];
    let mut distinctive_items: Vec<Vec<i32>> = vec![Vec::new(); n_personas];
    for (p, row) in by_persona.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let n = row.len() as f64;
        let mean = row.iter().map(|(_, r)| *r).sum::<f64>() / n;
        let var = row.iter().map(|(_, r)| (*r - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let cutoff = if std <= 0.0 {
            f64::NEG_INFINITY
        } else {
            mean - z_filter * std
        };
        for (col, rate) in row.iter() {
            if *rate > cutoff {
                filtered_by_persona[p].push((*col, *rate));
                distinctive_items[p].push(*col);
            }
        }
    }

    // ── Stage 4: IDF. idf[i] = log(n_personas / (1 + Σ_p rate(p, i))).
    let mut rate_sum_per_item = vec![0.0f64; n_items];
    for row in filtered_by_persona.iter() {
        for (col, rate) in row.iter() {
            rate_sum_per_item[*col as usize] += *rate;
        }
    }
    let n_personas_f = n_personas as f64;
    let idf: Vec<f64> = rate_sum_per_item
        .iter()
        .map(|s| (n_personas_f / (1.0 + s)).ln().max(0.0))
        .collect();

    // ── Stage 5: TF-IDF + L2 row normalization.
    // weight(p, i) = log1p(rate) * idf[i]
    let mut tfidf_data: Vec<f64> = Vec::new();
    let mut tfidf_indices: Vec<i32> = Vec::new();
    let mut tfidf_indptr: Vec<i32> = Vec::with_capacity(n_personas + 1);
    tfidf_indptr.push(0);
    for row in filtered_by_persona.iter() {
        // Compute weighted vector.
        let weighted: Vec<(i32, f64)> = row
            .iter()
            .map(|(col, rate)| (*col, rate.ln_1p() * idf[*col as usize]))
            .collect();
        // L2 normalize.
        let norm_sq: f64 = weighted.iter().map(|(_, w)| w * w).sum();
        let inv = if norm_sq > 0.0 {
            1.0 / norm_sq.sqrt()
        } else {
            0.0
        };
        for (col, w) in weighted.iter() {
            let v = w * inv;
            if v != 0.0 {
                tfidf_indices.push(*col);
                tfidf_data.push(v);
            }
        }
        tfidf_indptr.push(tfidf_indices.len() as i32);
    }

    // Pack the raw-rate CSR (filtered) for diagnostics.
    let mut rate_data: Vec<f64> = Vec::new();
    let mut rate_indices: Vec<i32> = Vec::new();
    let mut rate_indptr: Vec<i32> = Vec::with_capacity(n_personas + 1);
    rate_indptr.push(0);
    for row in filtered_by_persona.iter() {
        for (col, rate) in row.iter() {
            rate_indices.push(*col);
            rate_data.push(*rate);
        }
        rate_indptr.push(rate_indices.len() as i32);
    }

    PersonaIndexBuild {
        persona_sizes,
        rate_data,
        rate_indices,
        rate_indptr,
        tfidf_data,
        tfidf_indices,
        tfidf_indptr,
        idf,
        distinctive_items,
    }
}

/// PyO3 wrapper. Returns a flat tuple (so callers don't need a type stub):
/// (persona_sizes, rate_csr, tfidf_csr, idf, distinctive_items)
/// where each `*_csr = (data, indices, indptr)`.
#[pyfunction]
#[pyo3(signature = (
    user_to_persona,
    interaction_users,
    interaction_items,
    n_personas,
    n_items,
    z_filter = 1.5,
))]
#[allow(clippy::type_complexity)]
fn build_persona_index_py(
    user_to_persona: Vec<i64>,
    interaction_users: Vec<i64>,
    interaction_items: Vec<i64>,
    n_personas: usize,
    n_items: usize,
    z_filter: f64,
) -> PyResult<(
    Vec<i64>,                          // persona_sizes
    (Vec<f64>, Vec<i32>, Vec<i32>),    // rate CSR
    (Vec<f64>, Vec<i32>, Vec<i32>),    // tfidf CSR
    Vec<f64>,                          // idf
    Vec<Vec<i32>>,                     // distinctive_items
)> {
    let b = build_persona_index(
        &user_to_persona,
        &interaction_users,
        &interaction_items,
        n_personas,
        n_items,
        z_filter,
    );
    Ok((
        b.persona_sizes,
        (b.rate_data, b.rate_indices, b.rate_indptr),
        (b.tfidf_data, b.tfidf_indices, b.tfidf_indptr),
        b.idf,
        b.distinctive_items,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_persona_index_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two personas, each with two distinct users:
    ///   persona 0: users {0, 1}, both own item 10. user 0 also owns item 11.
    ///   persona 1: users {2, 3}, both own item 20.
    /// All other users assigned to persona but with no overlap.
    #[test]
    fn rate_then_zfilter_then_tfidf() {
        let user_to_persona = vec![0i64, 0, 1, 1];
        let interaction_users = vec![0i64, 0, 1, 2, 3];
        let interaction_items = vec![10i64, 11, 10, 20, 20];
        let b = build_persona_index(
            &user_to_persona,
            &interaction_users,
            &interaction_items,
            2, // n_personas
            30, // n_items
            1.5, // z_filter
        );
        assert_eq!(b.persona_sizes, vec![2, 2]);
        // Persona 0 has rates: {10: 1.0, 11: 0.5}.  Z-filter on small N
        // shouldn't drop either (std is small but cutoff is mean-1.5*std).
        // Persona 1 has rates: {20: 1.0}.
        // Distinctive items should reflect the filtered survivors.
        assert!(b.distinctive_items[0].contains(&10));
        assert!(b.distinctive_items[1].contains(&20));
        // TF-IDF rows should be L2 unit when non-empty.
        for p in 0..2 {
            let lo = b.tfidf_indptr[p] as usize;
            let hi = b.tfidf_indptr[p + 1] as usize;
            if hi > lo {
                let norm: f64 = b.tfidf_data[lo..hi].iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!((norm - 1.0).abs() < 1e-9, "persona {p} not L2-normalized: {norm}");
            }
        }
    }

    #[test]
    fn empty_inputs_dont_crash() {
        let b = build_persona_index(&[], &[], &[], 0, 0, 1.5);
        assert_eq!(b.persona_sizes, Vec::<i64>::new());
        assert_eq!(b.distinctive_items, Vec::<Vec<i32>>::new());
    }
}
