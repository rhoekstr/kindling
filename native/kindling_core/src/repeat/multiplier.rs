//! Recommend-time repeat multiplier.
//!
//! Implements the four functional forms (PRD §"Component-by-component
//! spec → repeat/" and the v1 ADR):
//!
//! - Pattern 1 (REPEAT):     m = 1.0     (no adjustment)
//! - Pattern 2 (REPLENISH):  m = sigmoid(6 · (r - 0.7))
//! - Pattern 3 (SATIATION):  m = 1 - exp(-(r / refractory_r)^2)
//! - Pattern 4 (ONE_SHOT):   m = epsilon (default 1e-3)
//!
//! Where `r = time_since_last / period_seconds`. Final multiplier is
//! a probability-weighted mixture across the four patterns plus a
//! confidence-dampened blend toward 1.0.
//!
//! The hot path is the batch wrapper: many candidates → many multipliers
//! per recommend call. Pure scalar math; no allocations beyond the
//! output vector.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

const PATTERN_REPEAT: usize = 0;
const PATTERN_REPLENISH: usize = 1;
const PATTERN_SATIATION: usize = 2;
const PATTERN_ONE_SHOT: usize = 3;

/// Pattern-specific multiplier given normalized time-ratio `r`.
fn pattern_multiplier(
    pattern_idx: usize,
    r: f64,
    period_seconds: f64,
    refractory_seconds: f64,
    one_shot_epsilon: f64,
) -> f64 {
    match pattern_idx {
        PATTERN_REPEAT => 1.0,
        PATTERN_REPLENISH => {
            // sigmoid(6 · (r - 0.7))
            let x = 6.0 * (r - 0.7);
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                x.exp() / (1.0 + x.exp())
            }
        }
        PATTERN_SATIATION => {
            let refractory_r = refractory_seconds / period_seconds.max(1e-9);
            let ratio = r / refractory_r.max(1e-9);
            1.0 - (-(ratio * ratio)).exp()
        }
        PATTERN_ONE_SHOT => one_shot_epsilon,
        _ => 1.0,
    }
}

/// Compute the multiplier for one candidate.
///
/// `pattern_probs` is `[REPEAT, REPLENISH, SATIATION, ONE_SHOT]` —
/// must sum to 1.0 (or thereabouts). `time_since_last_seconds = None`
/// (passed as f64::NAN) means the entity has never interacted with the
/// candidate; return 1.0 unconditionally.
pub fn multiplier_one(
    pattern_probs: [f64; 4],
    period_seconds: f64,
    refractory_seconds: f64,
    confidence: f64,
    time_since_last_seconds: f64,
    one_shot_epsilon: f64,
) -> f64 {
    if !time_since_last_seconds.is_finite() {
        return 1.0;
    }
    let period = period_seconds.max(1e-9);
    let r = time_since_last_seconds.max(0.0) / period;

    let mut weighted = 0.0;
    for (idx, prob) in pattern_probs.iter().enumerate() {
        if *prob <= 0.0 {
            continue;
        }
        weighted += prob
            * pattern_multiplier(idx, r, period_seconds, refractory_seconds, one_shot_epsilon);
    }
    let conf = confidence.clamp(0.0, 1.0);
    conf * weighted + (1.0 - conf) * 1.0
}

/// Batch multiplier: one call per recommend, processes all candidates.
///
/// Inputs are parallel arrays of length `n_candidates`:
/// - `pattern_probs[i, 0..4]` — four patterns for candidate `i`
/// - `periods[i]`, `refractories[i]`, `confidences[i]` — profile params
/// - `times_since_last[i]` — NaN if entity never interacted with candidate
///
/// Output: `(n_candidates,)` multipliers in `[0, 1]` (typically; one_shot
/// epsilon is < 1 by definition; SATIATION saturates to 1).
#[pyfunction]
#[pyo3(signature = (
    pattern_probs,
    periods,
    refractories,
    confidences,
    times_since_last,
    one_shot_epsilon = 1e-3,
))]
fn multipliers_batch<'py>(
    py: Python<'py>,
    pattern_probs: PyReadonlyArray2<'py, f64>,
    periods: PyReadonlyArray1<'py, f64>,
    refractories: PyReadonlyArray1<'py, f64>,
    confidences: PyReadonlyArray1<'py, f64>,
    times_since_last: PyReadonlyArray1<'py, f64>,
    one_shot_epsilon: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let probs = pattern_probs.as_array();
    let periods = periods.as_slice()?;
    let refrs = refractories.as_slice()?;
    let confs = confidences.as_slice()?;
    let times = times_since_last.as_slice()?;
    let n = periods.len();
    if probs.shape() != [n, 4] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "pattern_probs must be (n, 4); got {:?} with n={}",
            probs.shape(),
            n
        )));
    }
    let out = PyArray1::<f64>::zeros_bound(py, [n], false);
    let out_slice = unsafe { out.as_slice_mut().unwrap() };
    for i in 0..n {
        let pp = [
            probs[[i, 0]],
            probs[[i, 1]],
            probs[[i, 2]],
            probs[[i, 3]],
        ];
        out_slice[i] = multiplier_one(
            pp,
            periods[i],
            refrs[i],
            confs[i],
            times[i],
            one_shot_epsilon,
        );
    }
    Ok(out)
}

/// Single-candidate convenience wrapper for callers that don't have
/// arrays handy (rare; the batch form is the hot path).
#[pyfunction]
#[pyo3(signature = (
    pattern_probs,
    period_seconds,
    refractory_seconds,
    confidence,
    time_since_last_seconds,
    one_shot_epsilon = 1e-3,
))]
fn multiplier_one_py(
    pattern_probs: Vec<f64>,
    period_seconds: f64,
    refractory_seconds: f64,
    confidence: f64,
    time_since_last_seconds: f64,
    one_shot_epsilon: f64,
) -> PyResult<f64> {
    if pattern_probs.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pattern_probs must have length 4 [REPEAT, REPLENISH, SATIATION, ONE_SHOT]",
        ));
    }
    let pp = [
        pattern_probs[0],
        pattern_probs[1],
        pattern_probs[2],
        pattern_probs[3],
    ];
    Ok(multiplier_one(
        pp,
        period_seconds,
        refractory_seconds,
        confidence,
        time_since_last_seconds,
        one_shot_epsilon,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multipliers_batch, m)?)?;
    m.add_function(wrap_pyfunction!(multiplier_one_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_interacted_returns_one() {
        let m = multiplier_one(
            [0.25, 0.25, 0.25, 0.25],
            86400.0,
            86400.0 * 90.0,
            1.0,
            f64::NAN,
            1e-3,
        );
        assert_eq!(m, 1.0);
    }

    #[test]
    fn pure_repeat_pattern_always_one() {
        // 100% REPEAT pattern → multiplier = 1.0 regardless of r.
        let m = multiplier_one(
            [1.0, 0.0, 0.0, 0.0],
            86400.0,
            86400.0,
            1.0,
            86400.0 * 5.0, // 5 periods later
            1e-3,
        );
        assert!((m - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pure_one_shot_returns_epsilon() {
        // 100% ONE_SHOT → multiplier = epsilon.
        let m = multiplier_one(
            [0.0, 0.0, 0.0, 1.0],
            86400.0,
            86400.0,
            1.0,
            86400.0 * 5.0,
            1e-3,
        );
        assert!((m - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn replenish_pattern_low_pre_period_high_post() {
        // 100% REPLENISH. Half-period: r=0.5, sigmoid(6*(0.5-0.7)) = sigmoid(-1.2) ≈ 0.231.
        // 1.5 period: r=1.5, sigmoid(6*(1.5-0.7)) = sigmoid(4.8) ≈ 0.992.
        let pp = [0.0, 1.0, 0.0, 0.0];
        let m_pre = multiplier_one(pp, 86400.0, 86400.0, 1.0, 86400.0 * 0.5, 1e-3);
        let m_post = multiplier_one(pp, 86400.0, 86400.0, 1.0, 86400.0 * 1.5, 1e-3);
        assert!(m_pre < 0.3, "pre-period should be low: {m_pre}");
        assert!(m_post > 0.9, "post-period should be high: {m_post}");
    }

    #[test]
    fn confidence_dampens_toward_one() {
        // Pure ONE_SHOT (would give 1e-3) but confidence=0 should
        // dampen all the way to 1.0.
        let pp = [0.0, 0.0, 0.0, 1.0];
        let m = multiplier_one(pp, 86400.0, 86400.0, 0.0, 86400.0, 1e-3);
        assert_eq!(m, 1.0);
    }

    #[test]
    fn satiation_grows_past_refractory() {
        // SATIATION at r=1.0 with refractory=period (refractory_r=1.0):
        // m = 1 - exp(-1) ≈ 0.632
        // r=3.0: m = 1 - exp(-9) ≈ 0.9999
        let pp = [0.0, 0.0, 1.0, 0.0];
        let m_one = multiplier_one(pp, 86400.0, 86400.0, 1.0, 86400.0 * 1.0, 1e-3);
        let m_three = multiplier_one(pp, 86400.0, 86400.0, 1.0, 86400.0 * 3.0, 1e-3);
        assert!((m_one - (1.0 - (-1.0_f64).exp())).abs() < 1e-9);
        assert!(m_three > 0.999);
    }
}
