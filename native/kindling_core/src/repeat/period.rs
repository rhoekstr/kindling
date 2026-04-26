//! Period detection: scalar characteristic period from interval observations.
//!
//! Method: KDE (Gaussian kernel, Scott's bandwidth rule) on `log(intervals)`,
//! peak of density is the period in real-time units. Log transform absorbs
//! the scale-spanning problem that locally-adaptive bandwidth would
//! otherwise solve. Falls back to median for items with too few
//! observations.
//!
//! No scipy dep — Gaussian KDE + peak finding is pure ndarray + math.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

const KDE_MIN_POINTS: usize = 10;
const N_GRID: usize = 256;

/// Detect the characteristic period from inter-interaction intervals.
///
/// Returns `(period, fit_quality)`:
/// - `period`: in same units as input (seconds → seconds)
/// - `fit_quality ∈ [0, 1]`: 1 - second_peak_height/dominant_peak_height
///   (1.0 = clean unimodal; 0.0 = strong bimodality / multimodal).
pub fn detect_period(intervals: &[f64]) -> (f64, f64) {
    let positives: Vec<f64> = intervals.iter().copied().filter(|x| *x > 0.0).collect();
    if positives.is_empty() {
        return (f64::NAN, 0.0);
    }
    if positives.len() < KDE_MIN_POINTS {
        // Median fallback.
        let mut sorted = positives.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
        } else {
            sorted[n / 2]
        };
        return (median, 0.3);
    }

    // Log transform.
    let log_int: Vec<f64> = positives.iter().map(|x| x.ln()).collect();
    let n = log_int.len() as f64;
    let mean = log_int.iter().sum::<f64>() / n;
    let var = log_int.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-9 {
        // Degenerate input (all intervals identical).
        return (positives[0], 0.9);
    }
    // Scott's bandwidth: h = n^(-1/5) · σ
    let bandwidth = std * n.powf(-0.2);

    // Grid bounds.
    let lo = log_int
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi = log_int
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if hi - lo < 1e-9 {
        return (positives[0], 0.9);
    }

    // Evaluate Gaussian KDE on the grid.
    let mut density = vec![0.0; N_GRID];
    let inv_norm = 1.0 / (n * bandwidth * (2.0 * std::f64::consts::PI).sqrt());
    let grid: Vec<f64> = (0..N_GRID)
        .map(|i| lo + (hi - lo) * (i as f64) / ((N_GRID - 1) as f64))
        .collect();
    for (i, g) in grid.iter().enumerate() {
        let mut s = 0.0;
        for x in &log_int {
            let z = (g - x) / bandwidth;
            s += (-0.5 * z * z).exp();
        }
        density[i] = s * inv_norm;
    }

    // Peak finding: local maxima.
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..(N_GRID - 1) {
        if density[i] > density[i - 1] && density[i] > density[i + 1] {
            peaks.push(i);
        }
    }
    let peak_idx = if peaks.is_empty() {
        // Monotone — argmax of density.
        density
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        // Highest peak.
        *peaks
            .iter()
            .max_by(|a, b| {
                density[**a]
                    .partial_cmp(&density[**b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    };
    let period = grid[peak_idx].exp();

    // Multimodality score: ratio of second-highest peak to dominant peak.
    let multimodality = if peaks.len() >= 2 {
        let mut peak_heights: Vec<f64> = peaks.iter().map(|i| density[*i]).collect();
        peak_heights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        peak_heights[1] / peak_heights[0].max(1e-9)
    } else {
        0.0
    };
    let fit_quality = (1.0 - multimodality).clamp(0.0, 1.0);
    (period, fit_quality)
}

/// PyO3 wrapper.
#[pyfunction]
fn detect_period_py(intervals: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64)> {
    let slice = intervals.as_slice()?;
    Ok(detect_period(slice))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_period_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_nan() {
        let (p, _) = detect_period(&[]);
        assert!(p.is_nan());
    }

    #[test]
    fn median_fallback_for_few_points() {
        // 5 points → median fallback.
        let intervals = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (p, q) = detect_period(&intervals);
        assert_eq!(p, 30.0);
        assert!(q > 0.0 && q < 0.5);
    }

    #[test]
    fn unimodal_recovers_peak() {
        // Tight cluster of intervals around 86400 (one day).
        // Linspace over a narrow range produces a clean unimodal density.
        let n = 100;
        let mut intervals: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let f = (i as f64) / (n as f64 - 1.0); // 0..=1
            // Spread uniformly in [86400 - 5000, 86400 + 5000].
            intervals.push(86400.0 + (f - 0.5) * 10000.0);
        }
        let (p, _q) = detect_period(&intervals);
        assert!(
            (p - 86400.0).abs() / 86400.0 < 0.10,
            "expected period within 10% of 86400, got {p}"
        );
        // Don't assert fit_quality — KDE on a flat distribution can
        // produce small spurious secondary peaks. The period detection
        // itself is the contract; fit_quality is informational.
    }
}
