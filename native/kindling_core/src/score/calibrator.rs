//! Per-fit (z, boost) auto-calibrator.
//!
//! The full Python calibrator (`blend/layered_calibrator.py`) does a lot
//! of orchestration: sample held-out users, run the cooc retriever per
//! user, compute per-layer scores, etc. Most of that depends on signal
//! builders that are still in the Python shell during the v1→v2 transition.
//!
//! This Rust primitive isolates the bit that's worth porting first:
//! the grid sweep itself. The Python wrapper supplies the pre-computed
//! per-user (`base`, `layers`, `held_out_indicators`) tuples; this kernel
//! sweeps the (z, boost) grid and picks the cell that maximizes
//! NDCG@k on the held-out items, with the same default-preference and
//! sparse-data-cap heuristics as the v1 calibrator.

use ndarray::{Array1, ArrayView1};
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::layered::{layered_score, ZMode};

/// One held-out user's data for the calibrator.
pub struct UserSlice {
    pub base: Array1<f64>,
    pub layers: Vec<(Vec<f64>, ZMode)>,
    /// `held_out_indicators[c]` = 1.0 if candidate `c` is a held-out
    /// item we want the model to rank highly, 0.0 otherwise.
    pub held_out_indicators: Vec<f64>,
}

/// Result of a single grid cell.
#[derive(Clone, Debug)]
pub struct GridCell {
    pub z: f64,
    pub boost: f64,
    pub ndcg_at_k: f64,
    pub n_users: usize,
}

fn dcg_at_k(rels: &[f64], k: usize) -> f64 {
    let mut s = 0.0;
    for (i, r) in rels.iter().take(k).enumerate() {
        let denom = (i as f64 + 2.0).log2();
        s += (2.0_f64.powf(*r) - 1.0) / denom;
    }
    s
}

fn ndcg_at_k(scores: &[f64], rels: &[f64], k: usize) -> f64 {
    if scores.is_empty() || scores.len() != rels.len() {
        return 0.0;
    }
    // Order rels by score desc, with stable tie-break on index.
    let mut indexed: Vec<(usize, f64)> =
        scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let ranked: Vec<f64> = indexed.iter().map(|(i, _)| rels[*i]).collect();
    let dcg = dcg_at_k(&ranked, k);
    let mut ideal: Vec<f64> = rels.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg_at_k(&ideal, k);
    if idcg <= 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Sweep the (z, boost) grid. For each cell, compute NDCG@k averaged
/// across users. Returns one `GridCell` per (z, boost) tuple.
#[allow(clippy::too_many_arguments)]
pub fn sweep_grid(
    users: &[UserSlice],
    z_grid: &[f64],
    boost_grid: &[f64],
    k: usize,
    top_k_for_calibration: usize,
    min_nonzero_for_zscore: usize,
) -> Vec<GridCell> {
    let mut out = Vec::with_capacity(z_grid.len() * boost_grid.len());
    for &z in z_grid {
        for &b in boost_grid {
            let mut sum = 0.0;
            let mut n = 0;
            for u in users {
                let composite = layered_score(
                    u.base.view(),
                    &u.layers,
                    z,
                    b,
                    top_k_for_calibration,
                    min_nonzero_for_zscore,
                );
                let s = composite.to_vec();
                let nd = ndcg_at_k(&s, &u.held_out_indicators, k);
                sum += nd;
                n += 1;
            }
            let avg = if n > 0 { sum / n as f64 } else { 0.0 };
            out.push(GridCell {
                z,
                boost: b,
                ndcg_at_k: avg,
                n_users: n,
            });
        }
    }
    out
}

/// Cooc-only baseline NDCG: just sort by `base` and evaluate held-out
/// recovery. Used to gate the layered config: if no grid cell beats this
/// by `min_lift`, layered isn't helping and the calibrator should
/// disable boosting (return boost_multiplier=0).
fn cooc_only_ndcg(users: &[UserSlice], k: usize) -> f64 {
    if users.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut n = 0;
    for u in users {
        let s = u.base.to_vec();
        sum += ndcg_at_k(&s, &u.held_out_indicators, k);
        n += 1;
    }
    sum / n.max(1) as f64
}

/// Pick the winning grid cell with default-preference + tie-break logic.
///
/// Returns `(z, boost, ndcg, fallback_to_default)`.
/// - `fallback_to_default = true` if the grid is fully tied or no cell
///   beats cooc-only by `min_lift_over_cooc_only` (boost set to 0 in
///   that case to disable layering).
pub fn pick_winner(
    grid: &[GridCell],
    cooc_baseline_ndcg: f64,
    default_z: f64,
    default_boost: f64,
    tie_tolerance: f64,
    min_lift_over_cooc_only: f64,
) -> (f64, f64, f64, bool) {
    if grid.is_empty() {
        return (default_z, default_boost, 0.0, true);
    }
    // Best NDCG.
    let best = grid
        .iter()
        .max_by(|a, b| {
            a.ndcg_at_k
                .partial_cmp(&b.ndcg_at_k)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
        .ndcg_at_k;
    if best - cooc_baseline_ndcg < min_lift_over_cooc_only {
        // Layered isn't beating cooc-alone — disable boosting.
        return (default_z, 0.0, best, true);
    }
    let tied: Vec<&GridCell> = grid
        .iter()
        .filter(|c| best - c.ndcg_at_k <= tie_tolerance)
        .collect();
    if tied.len() == grid.len() {
        return (default_z, default_boost, best, true);
    }
    // Default-preference: if (default_z, default_boost) is in the tied
    // set, prefer it.
    if let Some(d) = tied.iter().find(|c| {
        (c.z - default_z).abs() < 1e-9 && (c.boost - default_boost).abs() < 1e-9
    }) {
        if tied.len() > 1 {
            return (d.z, d.boost, d.ndcg_at_k, false);
        }
    }
    // Otherwise: highest z first, then lowest boost.
    let mut sorted: Vec<&&GridCell> = tied.iter().collect();
    sorted.sort_by(|a, b| {
        b.z.partial_cmp(&a.z)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.boost.partial_cmp(&b.boost).unwrap_or(std::cmp::Ordering::Equal))
    });
    let chosen = sorted[0];
    (chosen.z, chosen.boost, chosen.ndcg_at_k, false)
}

/// PyO3 wrapper: takes a Python list of user-slice tuples and returns
/// a list of `(z, boost, ndcg, n_users)` plus `(best_z, best_boost,
/// best_ndcg, fallback_to_default)`.
#[pyfunction]
#[pyo3(signature = (
    users,
    z_grid,
    boost_grid,
    k = 10,
    top_k_for_calibration = 20,
    min_nonzero_for_zscore = 3,
    default_z = 2.5,
    default_boost = 3.0,
    tie_tolerance = 0.003,
    min_lift_over_cooc_only = 0.0,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn calibrate_layered_py(
    users: &Bound<'_, PyList>,
    z_grid: Vec<f64>,
    boost_grid: Vec<f64>,
    k: usize,
    top_k_for_calibration: usize,
    min_nonzero_for_zscore: usize,
    default_z: f64,
    default_boost: f64,
    tie_tolerance: f64,
    min_lift_over_cooc_only: f64,
) -> PyResult<(Vec<(f64, f64, f64, usize)>, (f64, f64, f64, bool))> {
    // Decode each user as (base, layer_specs, held_out_indicators).
    let mut decoded: Vec<UserSlice> = Vec::with_capacity(users.len());
    for u in users.iter() {
        let tup = u.downcast::<pyo3::types::PyTuple>()?;
        let base: numpy::PyReadonlyArray1<'_, f64> = tup.get_item(0)?.extract()?;
        let layer_specs = tup.get_item(1)?;
        let layer_specs = layer_specs.downcast::<PyList>()?;
        let held_out: numpy::PyReadonlyArray1<'_, f64> = tup.get_item(2)?.extract()?;

        let mut layers: Vec<(Vec<f64>, ZMode)> = Vec::with_capacity(layer_specs.len());
        for spec in layer_specs.iter() {
            let lt = spec.downcast::<pyo3::types::PyTuple>()?;
            let s: numpy::PyReadonlyArray1<'_, f64> = lt.get_item(0)?.extract()?;
            let mode_str: String = lt.get_item(1)?.extract()?;
            layers.push((s.as_slice()?.to_vec(), ZMode::from_str(&mode_str)?));
        }
        decoded.push(UserSlice {
            base: base.as_array().to_owned(),
            layers,
            held_out_indicators: held_out.as_slice()?.to_vec(),
        });
    }

    let grid = sweep_grid(
        &decoded,
        &z_grid,
        &boost_grid,
        k,
        top_k_for_calibration,
        min_nonzero_for_zscore,
    );
    let baseline = cooc_only_ndcg(&decoded, k);
    let winner = pick_winner(
        &grid,
        baseline,
        default_z,
        default_boost,
        tie_tolerance,
        min_lift_over_cooc_only,
    );
    let grid_out: Vec<(f64, f64, f64, usize)> =
        grid.iter().map(|c| (c.z, c.boost, c.ndcg_at_k, c.n_users)).collect();
    Ok((grid_out, winner))
}

/// Helper for callers: pure NDCG@k for a (scores, relevance) pair.
#[pyfunction]
fn ndcg_at_k_py(scores: Vec<f64>, relevance: Vec<f64>, k: usize) -> f64 {
    ndcg_at_k(&scores, &relevance, k)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calibrate_layered_py, m)?)?;
    m.add_function(wrap_pyfunction!(ndcg_at_k_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        // Scores rank held-out item first.
        let scores = vec![10.0, 1.0, 0.5];
        let rels = vec![1.0, 0.0, 0.0];
        let nd = ndcg_at_k(&scores, &rels, 3);
        assert!((nd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_worst_ranking_below_perfect() {
        // 5 candidates; held-out at rank 5 (worst).
        let scores = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let rels = vec![0.0, 0.0, 0.0, 0.0, 1.0];
        let nd = ndcg_at_k(&scores, &rels, 5);
        // Held-out at last rank: dcg = 1/log2(6); idcg = 1; ratio < 0.5.
        assert!(nd < 0.5, "expected NDCG < 0.5 for held-out at rank 5, got {nd}");
    }

    #[test]
    fn pick_winner_falls_back_when_no_lift() {
        let grid = vec![
            GridCell { z: 2.5, boost: 3.0, ndcg_at_k: 0.50, n_users: 100 },
            GridCell { z: 2.0, boost: 5.0, ndcg_at_k: 0.49, n_users: 100 },
        ];
        let cooc_baseline = 0.51; // best layered cell loses by 0.01
        let (z, b, _, fallback) = pick_winner(&grid, cooc_baseline, 2.5, 3.0, 0.003, 0.005);
        assert!(fallback);
        assert_eq!(b, 0.0, "boost should be zeroed when no lift over cooc");
        assert_eq!(z, 2.5);
    }

    #[test]
    fn pick_winner_chooses_default_in_tie() {
        let grid = vec![
            GridCell { z: 2.5, boost: 3.0, ndcg_at_k: 0.50, n_users: 100 },
            GridCell { z: 3.0, boost: 5.0, ndcg_at_k: 0.501, n_users: 100 },
            // Clearly worse cell so the "all cells tied" fallback doesn't fire.
            GridCell { z: 2.0, boost: 1.0, ndcg_at_k: 0.30, n_users: 100 },
        ];
        let (z, b, _, fallback) = pick_winner(&grid, 0.0, 2.5, 3.0, 0.005, 0.0);
        // Top two within 0.005 → tied; default preference picks (2.5, 3.0).
        assert!(!fallback, "should not fall back");
        assert_eq!(z, 2.5);
        assert_eq!(b, 3.0);
    }

    #[test]
    fn pick_winner_falls_back_when_all_tied() {
        // All cells tied within tolerance → fall back to default with
        // fallback_to_default=true (v1 semantics).
        let grid = vec![
            GridCell { z: 2.5, boost: 3.0, ndcg_at_k: 0.50, n_users: 100 },
            GridCell { z: 3.0, boost: 5.0, ndcg_at_k: 0.501, n_users: 100 },
        ];
        let (_, _, _, fallback) = pick_winner(&grid, 0.0, 2.5, 3.0, 0.005, 0.0);
        assert!(fallback);
    }

    #[test]
    fn sweep_grid_basic() {
        // 10 candidates: base scores favor candidate 0 (descending); held-out
        // = candidate 9 (the worst by base). Layer is sparse with nine
        // small values and one large outlier on candidate 9 — outlier has
        // z ≈ 3, so it should fire at z_threshold=2.5 and lift the
        // held-out item.
        let base: Array1<f64> = (0..10).map(|i| (10 - i) as f64).collect();
        let layer: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
        let mut held = vec![0.0; 10];
        held[9] = 1.0;
        let user = UserSlice {
            base,
            layers: vec![(layer, ZMode::NonzeroSubset)],
            held_out_indicators: held,
        };
        let grid = sweep_grid(
            &[user],
            &[2.5],
            &[0.0, 5.0],
            10,
            20,
            3,
        );
        let cell_no_boost = grid
            .iter()
            .find(|c| (c.z - 2.5).abs() < 1e-9 && c.boost == 0.0)
            .unwrap();
        let cell_with_boost = grid
            .iter()
            .find(|c| (c.z - 2.5).abs() < 1e-9 && c.boost == 5.0)
            .unwrap();
        assert!(
            cell_with_boost.ndcg_at_k > cell_no_boost.ndcg_at_k,
            "boosted cell ({}) should outrank no-boost ({})",
            cell_with_boost.ndcg_at_k,
            cell_no_boost.ndcg_at_k
        );
    }
}
