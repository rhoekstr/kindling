"""Period detection: scalar characteristic period from interval observations.

Method: KDE (Gaussian kernel, Scott's bandwidth rule) on ``log(intervals)``,
peak of density is the period. Log transform absorbs the scale-spanning
problem that locally-adaptive bandwidth would otherwise solve. Falls
back to ``median(intervals)`` for items with fewer than ~10 observations.

Returns period in the same time units as the input (seconds if the
input is seconds).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

# Below this count, use median instead of KDE - KDE with <10 points
# produces unreliable density estimates.
_KDE_MIN_POINTS = 10


def detect_period(intervals: np.ndarray) -> tuple[float, float]:
    """Return ``(period, fit_quality)`` where fit_quality is in [0, 1].

    - ``period`` is NaN if too few observations.
    - ``fit_quality`` reflects the dominant peak's height relative to
      the next-highest peak (higher = more unimodal = higher quality).
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    intervals = intervals[intervals > 0.0]
    if intervals.size == 0:
        return float("nan"), 0.0
    if intervals.size < _KDE_MIN_POINTS:
        # Median fallback: robust to outliers, defensible on sparse data.
        return float(np.median(intervals)), 0.3

    log_int = np.log(intervals)
    try:
        kde = gaussian_kde(log_int)
    except (ValueError, np.linalg.LinAlgError):
        # gaussian_kde can fail on degenerate input (all same value).
        return float(np.median(intervals)), 0.3

    # Evaluate on a dense grid in log space.
    n_grid = 256
    lo, hi = float(log_int.min()), float(log_int.max())
    if hi - lo < 1e-9:
        # All intervals identical.
        return float(np.exp((lo + hi) / 2.0)), 0.9
    grid = np.linspace(lo, hi, n_grid)
    density = kde(grid)

    # Find local maxima.
    peak_mask = np.r_[False, (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:]), False]
    peaks = np.where(peak_mask)[0]
    if peaks.size == 0:
        # Monotone - take argmax.
        peak_idx = int(np.argmax(density))
    else:
        peak_idx = int(peaks[np.argmax(density[peaks])])

    period = float(np.exp(grid[peak_idx]))

    # Multimodality score: ratio of second-highest peak to dominant peak.
    # High ratio -> messy / multimodal -> lower fit quality.
    if peaks.size >= 2:
        peak_heights = np.sort(density[peaks])[::-1]
        multimodality = peak_heights[1] / peak_heights[0]
    else:
        multimodality = 0.0
    fit_quality = max(0.0, min(1.0, 1.0 - multimodality))
    return period, fit_quality
