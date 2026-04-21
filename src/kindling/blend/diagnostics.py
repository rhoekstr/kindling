"""VI convergence diagnostics (PRD §6.7).

Three checks run after ``BayesianBlend.fit_posterior`` completes. All
three must pass for the posterior to be declared valid; failures surface
as warnings through ``Engine.posterior_summary()`` so users who run in
production see them in their monitoring dashboards.

1. ELBO trajectory monotonicity: ELBO should trend up, modulo the noise
   of MC sampling. Non-monotonic trajectories beyond the tolerance
   indicate a bad optimizer init or a broken gradient.
2. Posterior predictive check: Brier score of the simulated outcomes
   should match the actual Brier within 10%. Large deviation = poor fit.
3. Effective sample size of the variational approximation: ESS/S < 0.1
   indicates a poor variational family (user should try a richer one).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

from kindling.blend.likelihoods import OutcomeBatch

if TYPE_CHECKING:
    from kindling.blend.bayesian import BayesianBlend
    from kindling.blend.likelihoods import LikelihoodProtocol


@dataclass(frozen=True)
class DiagnosticsReport:
    """Output of the three convergence diagnostics."""

    elbo_monotonic: bool
    elbo_final: float
    elbo_peak: float
    ppc_brier_actual: float
    ppc_brier_simulated: float
    ppc_deviation: float  # relative
    ppc_passes: bool
    ess_ratio: float  # ESS / S, in [0, 1]
    ess_passes: bool

    @property
    def all_pass(self) -> bool:
        return self.elbo_monotonic and self.ppc_passes and self.ess_passes

    def warnings(self) -> list[str]:
        out: list[str] = []
        if not self.elbo_monotonic:
            out.append(
                f"ELBO trajectory non-monotonic (final={self.elbo_final:.3f}, "
                f"peak={self.elbo_peak:.3f}). Consider a different random seed "
                "or a lower learning rate."
            )
        if not self.ppc_passes:
            out.append(
                f"Posterior predictive check failed (actual Brier "
                f"{self.ppc_brier_actual:.4f} vs simulated "
                f"{self.ppc_brier_simulated:.4f}, deviation "
                f"{self.ppc_deviation:.1%}). The fitted posterior does not "
                "match the data well."
            )
        if not self.ess_passes:
            out.append(
                f"Variational effective sample size ratio {self.ess_ratio:.2%} "
                "below 10%. Consider a richer variational family."
            )
        return out


def run_diagnostics(
    blend: BayesianBlend,
    batch: OutcomeBatch,
    likelihood: LikelihoodProtocol,
    rng: np.random.Generator,
    n_ppc_samples: int = 128,
) -> DiagnosticsReport:
    """Compute all three diagnostics against a fitted blend."""
    elbo_monotonic, elbo_final, elbo_peak = _check_elbo(blend.elbo_trace)
    ppc_actual, ppc_sim, ppc_dev, ppc_pass = _check_ppc(
        blend=blend, batch=batch, likelihood=likelihood, rng=rng, n_samples=n_ppc_samples
    )
    ess, ess_pass = _check_ess(
        blend=blend, batch=batch, likelihood=likelihood, rng=rng, n_samples=n_ppc_samples
    )
    return DiagnosticsReport(
        elbo_monotonic=elbo_monotonic,
        elbo_final=elbo_final,
        elbo_peak=elbo_peak,
        ppc_brier_actual=ppc_actual,
        ppc_brier_simulated=ppc_sim,
        ppc_deviation=ppc_dev,
        ppc_passes=ppc_pass,
        ess_ratio=ess,
        ess_passes=ess_pass,
    )


def _check_elbo(trace: list[float]) -> tuple[bool, float, float]:
    """A trajectory is 'monotonic enough' if the final ELBO is within
    Monte Carlo noise tolerance of the peak - i.e., the optimizer has
    plateaued rather than regressed."""
    if not trace:
        return False, float("nan"), float("nan")
    arr = np.asarray(trace, dtype=np.float64)
    final = float(arr[-1])
    peak = float(arr.max())
    # Tolerance scales with trajectory range - a tiny bit of noise in a
    # flat tail is fine; a big drop from the peak is not.
    mag = max(abs(peak), 1.0)
    monotonic = (peak - final) / mag < 0.05
    return bool(monotonic), final, peak


def _check_ppc(
    blend: BayesianBlend,
    batch: OutcomeBatch,
    likelihood: LikelihoodProtocol,
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[float, float, float, bool]:
    """Brier score of observed outcomes vs. a predictive simulation under
    the current posterior. Lower Brier = better calibration. The test
    passes when the simulated and actual Brier are within 10% of each
    other, indicating the posterior's predicted selection rates track
    reality."""
    # Predictive mean for each shown item: score under posterior mean.
    scores = batch.signal_matrix @ blend.posterior_mean
    probs_actual = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
    actual = float(np.mean((batch.selected.astype(np.float64) - probs_actual) ** 2))

    # Simulated: draw weights from posterior, compute predictive probs,
    # simulate bernoulli outcomes, score Brier.
    samples = rng.dirichlet(blend.posterior_beta, size=n_samples)
    sim_scores = batch.signal_matrix @ samples.T  # (N_outcomes, S)
    sim_probs = 1.0 / (1.0 + np.exp(-np.clip(sim_scores, -30, 30)))
    sim_outcomes = (rng.uniform(size=sim_probs.shape) < sim_probs).astype(np.float64)
    # Brier scoring sim outcomes against the same predictive probs.
    sim = float(np.mean((sim_outcomes - sim_probs) ** 2))
    dev = abs(sim - actual) / max(actual, 1e-9)
    return actual, sim, dev, dev < 0.10


def _check_ess(
    blend: BayesianBlend,
    batch: OutcomeBatch,
    likelihood: LikelihoodProtocol,
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[float, bool]:
    """Variational effective sample size: ESS = (sum w)^2 / sum w^2 where
    w = exp(log_true_posterior - log_variational). For a matched variational
    family ESS/S is near 1; for a poor match it tends to 0."""
    samples = rng.dirichlet(blend.posterior_beta, size=n_samples)
    # Log true posterior density up to a constant: log lik + log prior.
    # Log variational density: Dirichlet(beta) pdf.
    log_lik = np.array([likelihood.log_prob(s, batch) for s in samples])
    log_prior = np.array([_dirichlet_logpdf(s, blend.prior_alpha) for s in samples])
    log_q = np.array([_dirichlet_logpdf(s, blend.posterior_beta) for s in samples])
    log_w = log_lik + log_prior - log_q
    log_w -= log_w.max()  # for numerical stability
    w = np.exp(log_w)
    ess = float((w.sum() ** 2) / (w**2).sum())
    ratio = ess / n_samples
    return ratio, ratio >= 0.10


def _dirichlet_logpdf(x: np.ndarray, alpha: np.ndarray) -> float:
    """Dirichlet log-density evaluated at x."""
    x = np.clip(x, 1e-30, 1.0)
    return float(gammaln(alpha.sum()) - gammaln(alpha).sum() + ((alpha - 1.0) * np.log(x)).sum())
