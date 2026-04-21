"""Likelihood models for the Bayesian blend (PRD §6.2).

Four likelihoods ship in v1: listwise calibration (default), pairwise
Bradley-Terry, multinomial, binary independent. The default selection is
validated empirically in the Phase 3 critical-path benchmarks.

All likelihoods share the same ``OutcomeBatch`` input and return a scalar
log-likelihood given blend weights ``w``. The variational inference loop
calls ``log_prob`` once per Monte Carlo sample.

Position-bias correction (PRD §6.2) is included in the listwise calibration
likelihood when ``positions`` is provided. The Position-Based Model (PBM)
with ``examination(p) = (1/p)**eta`` and default ``eta=1.0`` separates
relevance from display-position effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import gammaln

DEFAULT_ETA = 1.0


@dataclass(frozen=True)
class OutcomeBatch:
    """A batch of outcome observations for posterior update.

    One row per (shown item in a recommendation list). Items belonging to
    the same recommendation share a ``list_ids`` value.

    Attributes
    ----------
    signal_matrix:
        Shape ``(N_outcomes, K_signals)``. Each row is the signal vector
        for one shown item.
    selected:
        Shape ``(N_outcomes,)``. Boolean / 0-1 indicator of whether the
        item was selected by the entity.
    positions:
        Shape ``(N_outcomes,)``. 1-indexed display position within the
        recommendation list. Used for PBM position-bias correction.
    list_ids:
        Shape ``(N_outcomes,)``. Identifier grouping items from the same
        recommendation list (used by pairwise/multinomial likelihoods).
    """

    signal_matrix: np.ndarray
    selected: np.ndarray
    positions: np.ndarray
    list_ids: np.ndarray

    @property
    def n_outcomes(self) -> int:
        return int(self.signal_matrix.shape[0])

    @property
    def n_signals(self) -> int:
        return int(self.signal_matrix.shape[1])


@runtime_checkable
class LikelihoodProtocol(Protocol):
    """A likelihood computes log P(outcomes | weights)."""

    name: str

    def log_prob(self, weights: np.ndarray, batch: OutcomeBatch) -> float:
        """Scalar log-likelihood. Sums over the whole batch."""
        ...


# ---------------------------------------------------------------------------
# Binary independent: simplest likelihood. Each shown item is an independent
# Bernoulli trial with p = sigmoid(<w, signal_i>). No list structure, no
# position correction. Included as a baseline and as the simplest smoke
# test for the VI machinery.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryIndependent:
    """Bernoulli per shown item. P(selected_i) = sigmoid(<w, signal_i>)."""

    name: str = "binary_independent"

    def log_prob(self, weights: np.ndarray, batch: OutcomeBatch) -> float:
        logits = batch.signal_matrix @ weights
        # Clip to avoid log(0) from extreme logits.
        logits = np.clip(logits, -30.0, 30.0)
        # log sigmoid(z) = -log(1+exp(-z)); log(1-sigmoid(z)) = -log(1+exp(z))
        # Use a numerically stable formulation.
        pos = batch.selected.astype(np.float64)
        ll = -np.logaddexp(0.0, -logits) * pos - np.logaddexp(0.0, logits) * (1.0 - pos)
        return float(ll.sum())


# ---------------------------------------------------------------------------
# Listwise calibration: the PRD default. Bins shown items by score, compares
# observed selection rate per bin to the rate predicted under the current
# weights. Requires enough observations to populate bins; bins with < 10
# displayed items downweight to avoid spurious estimates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ListwiseCalibration:
    """Calibration-focused likelihood (PRD §6.2).

    Treats the score as a calibrated probability and compares observed
    selection counts per score bin to what the bin midpoint would predict
    under the current weights, with position-based examination correction.

    Parameters
    ----------
    n_bins:
        Number of score bins. 10 is a sensible default - balances bin
        resolution against per-bin observation count.
    eta:
        Position-bias exponent for the PBM. ``1.0`` matches typical
        sequential web/mobile contexts (PRD §6.2). Overridable per engine.
    min_observations_per_bin:
        Bins with fewer than this many shown items contribute zero
        observations rather than a noisy binomial likelihood on tiny data.
    """

    n_bins: int = 10
    eta: float = DEFAULT_ETA
    min_observations_per_bin: int = 10
    name: str = "listwise_calibration"

    def log_prob(self, weights: np.ndarray, batch: OutcomeBatch) -> float:
        if batch.n_outcomes == 0:
            return 0.0

        # Score each shown item under current w.
        raw_scores = batch.signal_matrix @ weights

        # Squash to [0, 1] so score is interpretable as a "raw probability"
        # before PBM correction. sigmoid is the standard choice.
        scores_01 = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -30.0, 30.0)))

        # Bin edges: linear in [0, 1]. Items with score exactly 1 land in
        # the last bin.
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_idx = np.clip(np.digitize(scores_01, edges[1:-1]), 0, self.n_bins - 1)

        # Examination probability per shown item.
        pos = np.maximum(batch.positions.astype(np.float64), 1.0)
        exam = np.power(pos, -self.eta)

        # Per bin, aggregate:
        #   n_shown_b = sum exam_i (effective shown count)
        #   n_selected_b = sum (selected_i * exam_i)  (effective selected)
        # Predicted rate under listwise calibration: the bin midpoint.
        total_ll = 0.0
        selected = batch.selected.astype(np.float64)
        for b in range(self.n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            n_shown = float(exam[mask].sum())
            if n_shown < self.min_observations_per_bin:
                continue
            n_sel = float((selected[mask] * exam[mask]).sum())
            predicted_rate = 0.5 * (edges[b] + edges[b + 1])
            # Binomial log-pmf with effective sample sizes.
            total_ll += _binomial_logpmf(n_sel, n_shown, predicted_rate)
        return total_ll


def _binomial_logpmf(k: float, n: float, p: float) -> float:
    """Effective-sample-size Binomial log-pmf. n and k may be fractional
    (weighted by PBM examination probabilities)."""
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    if n <= 0:
        return 0.0
    k = float(np.clip(k, 0.0, n))
    return float(
        gammaln(n + 1.0)
        - gammaln(k + 1.0)
        - gammaln(n - k + 1.0)
        + k * np.log(p)
        + (n - k) * np.log(1.0 - p)
    )


# ---------------------------------------------------------------------------
# Pairwise Bradley-Terry: classic learning-to-rank. For each pair
# (selected, non-selected) within the same list, P(selected preferred) =
# sigmoid(score_selected - score_nonselected).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseBradleyTerry:
    """Pairwise (selected vs. non-selected) within each list."""

    name: str = "pairwise_bradley_terry"

    def log_prob(self, weights: np.ndarray, batch: OutcomeBatch) -> float:
        if batch.n_outcomes == 0:
            return 0.0
        scores = batch.signal_matrix @ weights
        total_ll = 0.0
        unique_lists = np.unique(batch.list_ids)
        for list_id in unique_lists:
            mask = batch.list_ids == list_id
            if not mask.any():
                continue
            list_scores = scores[mask]
            list_selected = batch.selected[mask]
            pos = np.where(list_selected)[0]
            neg = np.where(~list_selected.astype(bool))[0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            # All pair differences (pos vs neg); sigmoid(diff).
            diffs = np.clip(list_scores[pos][:, None] - list_scores[neg][None, :], -30, 30)
            total_ll += float(-np.logaddexp(0.0, -diffs).sum())
        return total_ll


# ---------------------------------------------------------------------------
# Multinomial (softmax): within each list, the selected item is the
# argmax of the softmax distribution over shown items.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultinomialSoftmax:
    """Softmax per list: P(selected = i | list) = exp(score_i) / sum exp(score_j)."""

    name: str = "multinomial_softmax"

    def log_prob(self, weights: np.ndarray, batch: OutcomeBatch) -> float:
        if batch.n_outcomes == 0:
            return 0.0
        scores = batch.signal_matrix @ weights
        total_ll = 0.0
        unique_lists = np.unique(batch.list_ids)
        for list_id in unique_lists:
            mask = batch.list_ids == list_id
            if not mask.any():
                continue
            list_scores = scores[mask]
            list_selected = batch.selected[mask].astype(bool)
            if not list_selected.any():
                # No selection - this is a non-event, contributes 0 to
                # likelihood. Multinomial is silent when the user chose
                # nothing (which is a known limitation; listwise calibration
                # handles this more gracefully).
                continue
            # Stable softmax + gather selected.
            max_s = list_scores.max()
            log_norm = max_s + np.log(np.exp(list_scores - max_s).sum())
            sel_scores = list_scores[list_selected]
            total_ll += float(sel_scores.sum() - log_norm * list_selected.sum())
        return total_ll
