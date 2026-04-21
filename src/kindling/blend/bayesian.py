"""Bayesian blend (PRD §3.2, §6.2, §6.7).

Mean-field Dirichlet variational posterior over blend weights. Prior and
variational posterior are both Dirichlet, giving closed-form KL. The log-
likelihood expectation is estimated by Monte Carlo. Gradients flow via the
REINFORCE / score-function estimator with a mean baseline for variance
reduction. Optimization uses an Adam-lite step rule.

Why not a heavier tool (NumPyro / JAX / Pyro): K is small (typically <= 20
signals), the prior and variational family are both Dirichlet (conjugate
structure), and the KL is closed-form. A hand-rolled loop is cleaner for
debugging, keeps dependency surface minimal, and makes the reproducibility
story (fixed ``np.random.Generator``) trivial. Phase 3 is about getting the
math right; heavy autodiff can come in Phase 9 if needed.

Plan gap closed: all randomness flows through an explicit generator passed
at engine construction. Single-threaded by default, so the VI result is
bitwise reproducible for a fixed seed, dataset, and iteration count. This
is the reproducibility claim the PRD §6.7 makes checkable.

The variational parameters are ``beta`` (shape ``(K,)``), tracked on an
unconstrained scale via ``beta = softplus(unconstrained) + eps`` so the
optimizer can run on unbounded R^K while maintaining the
positivity constraint that Dirichlet concentration requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from kindling.blend.decorrelate import DecorrelationBasis
from kindling.blend.heuristic import SignalFeatures
from kindling.blend.likelihoods import LikelihoodProtocol, OutcomeBatch

if TYPE_CHECKING:
    from kindling.blend.diagnostics import DiagnosticsReport

DEFAULT_N_MC_SAMPLES = 32
DEFAULT_MAX_ITER = 500
DEFAULT_LEARNING_RATE = 0.02
ELBO_CONVERGENCE_WINDOW = 10
ELBO_CONVERGENCE_REL_TOL = 1e-3
MIN_BETA = 0.5  # Clip per PRD §6.7 - keep prior from going pathologically sharp.
MAX_BETA = 100.0


@dataclass
class BayesianBlend:
    """Signal combination with Dirichlet posterior over weights.

    Attributes
    ----------
    signal_names:
        Column names matching ``SignalFeatures.signal_names`` order.
    prior_alpha:
        Shape ``(K,)``. Prior Dirichlet concentration parameters. Always
        clipped to ``[MIN_BETA, MAX_BETA]`` per PRD §6.7.
    posterior_beta:
        Shape ``(K,)``. Variational Dirichlet parameters. Initialized from
        the prior at construction; updated by ``fit_posterior``.
    path_basis:
        Optional path-family decorrelation basis (PRD §6.2). Applied to
        signals before they enter the likelihood so the learned weights are
        interpretable.
    elbo_trace:
        ELBO trajectory from the last call to ``fit_posterior``. Empty
        until a fit is run.
    """

    signal_names: tuple[str, ...]
    prior_alpha: np.ndarray
    posterior_beta: np.ndarray
    path_basis: DecorrelationBasis | None = None
    elbo_trace: list[float] = field(default_factory=list)
    diagnostics: DiagnosticsReport | None = None

    @classmethod
    def from_prior(
        cls,
        signal_names: tuple[str, ...],
        prior_alpha: np.ndarray,
        path_basis: DecorrelationBasis | None = None,
    ) -> BayesianBlend:
        alpha = _clip_concentration(np.asarray(prior_alpha, dtype=np.float64))
        return cls(
            signal_names=tuple(signal_names),
            prior_alpha=alpha,
            posterior_beta=alpha.copy(),
            path_basis=path_basis,
        )

    # ---- posterior moments ------------------------------------------------

    @property
    def posterior_mean(self) -> np.ndarray:
        out = self.posterior_beta / self.posterior_beta.sum()
        return np.asarray(out, dtype=np.float64)

    @property
    def posterior_variance(self) -> np.ndarray:
        total = self.posterior_beta.sum()
        mean = self.posterior_beta / total
        out = mean * (1.0 - mean) / (total + 1.0)
        return np.asarray(out, dtype=np.float64)

    def credible_interval(self, coverage: float = 0.9) -> np.ndarray:
        """Per-signal marginal credible interval using Beta marginals.

        Each weight ``w_k`` marginally follows ``Beta(beta_k, sum(beta) -
        beta_k)``. Returns shape ``(K, 2)`` with columns ``[lower, upper]``.
        """
        alpha_margin = self.posterior_beta
        beta_margin = self.posterior_beta.sum() - self.posterior_beta
        lo = 0.5 * (1.0 - coverage)
        hi = 1.0 - lo
        lower = beta_dist.ppf(lo, alpha_margin, beta_margin)
        upper = beta_dist.ppf(hi, alpha_margin, beta_margin)
        return np.asarray(np.stack([lower, upper], axis=1), dtype=np.float64)

    # ---- scoring ----------------------------------------------------------

    def score(self, features: SignalFeatures) -> np.ndarray:
        """Posterior-mean blended score per candidate row."""
        processed = self._apply_decorrelation(features)
        return np.asarray(processed @ self.posterior_mean, dtype=np.float64)

    def score_with_uncertainty(
        self,
        features: SignalFeatures,
        coverage: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mean, lower, upper)`` per candidate.

        The per-candidate variance uses the closed-form Dirichlet
        covariance:

            Var[score_i] = sum_k Var[w_k] * s_k(i)^2
                         + 2 sum_{j<k} Cov[w_j, w_k] * s_j(i) * s_k(i)

        Then a symmetric Gaussian interval at the target coverage. This is
        an approximation to the true Beta-mixture credible interval on the
        score; the Gaussian approximation is tight when K is small and the
        posterior is not near the simplex boundary.
        """
        processed = self._apply_decorrelation(features)
        mean = processed @ self.posterior_mean

        cov = _dirichlet_covariance(self.posterior_beta)
        # Per-row variance = s_i^T Sigma s_i
        var = np.einsum("ij,jk,ik->i", processed, cov, processed)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        z = float(norm.ppf(0.5 + 0.5 * coverage))
        return mean, mean - z * std, mean + z * std

    # ---- fitting ----------------------------------------------------------

    def fit_posterior(
        self,
        batch: OutcomeBatch,
        likelihood: LikelihoodProtocol,
        rng: np.random.Generator,
        n_mc_samples: int = DEFAULT_N_MC_SAMPLES,
        max_iter: int = DEFAULT_MAX_ITER,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> BayesianBlend:
        """Run MC-VI on the posterior.

        Mutates ``self.posterior_beta`` and records ``self.elbo_trace``.
        ``rng`` is the authoritative source of randomness - the VI result
        is a deterministic function of ``(rng seed, batch, likelihood,
        hyperparameters)``.
        """
        # Optimize on unconstrained scale: beta = MIN_BETA + softplus(x).
        x = _inv_softplus(self.posterior_beta - MIN_BETA)

        # Adam-lite moments.
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        elbo_trace: list[float] = []
        best_beta = self.posterior_beta.copy()
        best_elbo = -np.inf

        for step in range(1, max_iter + 1):
            beta = _clip_concentration(MIN_BETA + _softplus(x))

            # Monte Carlo draw.
            samples = rng.dirichlet(beta, size=n_mc_samples)
            log_liks = np.array([likelihood.log_prob(s, batch) for s in samples], dtype=np.float64)

            # ELBO = E[log_lik] - KL(q || prior). Mean ELBO for this step.
            expected_ll = float(log_liks.mean())
            kl = _dirichlet_kl(beta, self.prior_alpha)
            elbo = expected_ll - kl
            elbo_trace.append(elbo)
            if elbo > best_elbo:
                best_elbo = elbo
                best_beta = beta.copy()

            # REINFORCE gradient of E[log_lik] w.r.t. beta via score function,
            # then chain through the softplus to x.
            baseline = float(log_liks.mean())
            digamma_sum = digamma(beta.sum())
            score_contrib = (
                digamma_sum - digamma(beta[None, :]) + np.log(np.clip(samples, 1e-12, 1.0))
            )  # shape (S, K)
            # Variance-reduced REINFORCE.
            grad_expected_ll_beta = ((log_liks[:, None] - baseline) * score_contrib).mean(axis=0)

            # Analytic gradient of -KL w.r.t. beta:
            # KL(q(beta) || p(alpha)) =
            #   gammaln(sum(beta)) - gammaln(sum(alpha))
            #   + sum(gammaln(alpha_k) - gammaln(beta_k))
            #   + sum((beta_k - alpha_k) * (digamma(beta_k) - digamma(sum(beta))))
            # dKL/dbeta_k = (beta_k - alpha_k) * (trigamma(beta_k) - trigamma(sum(beta)))
            # derivation: the lgamma derivatives on sum(beta) cancel with
            # digamma(sum(beta)); the per-element digamma(beta_k) difference
            # contribution collapses to the above. We approximate trigamma
            # numerically to avoid an extra scipy import for this one-off.
            trig_sum = _trigamma(beta.sum())
            grad_kl_beta = (beta - self.prior_alpha) * (_trigamma(beta) - trig_sum)
            grad_elbo_beta = grad_expected_ll_beta - grad_kl_beta

            # Chain through x -> beta: d beta / d x = sigmoid(x) elementwise.
            grad_elbo_x = grad_elbo_beta * _sigmoid(x)

            # Maximize ELBO => take a step in +grad_elbo_x direction.
            m = beta1 * m + (1 - beta1) * grad_elbo_x
            v = beta2 * v + (1 - beta2) * (grad_elbo_x**2)
            m_hat = m / (1 - beta1**step)
            v_hat = v / (1 - beta2**step)
            x = x + learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # Convergence test: rolling-window relative ELBO improvement.
            if step > ELBO_CONVERGENCE_WINDOW:
                window = elbo_trace[-ELBO_CONVERGENCE_WINDOW:]
                if _relative_plateau(window):
                    break

        # Use the best-ELBO beta seen during the run, not the last step's
        # (Adam can overshoot on the noisy MC estimate).
        self.posterior_beta = _clip_concentration(best_beta)
        self.elbo_trace = elbo_trace
        return self

    # ---- helpers ----------------------------------------------------------

    def _apply_decorrelation(self, features: SignalFeatures) -> np.ndarray:
        matrix = features.matrix.astype(np.float64, copy=True)
        if self.path_basis is None:
            return matrix
        path_cols = [
            features.signal_names.index(n)
            for n in self.path_basis.signal_names
            if n in features.signal_names
        ]
        if len(path_cols) != len(self.path_basis.signal_names):
            return matrix
        matrix[:, path_cols] = self.path_basis.apply(matrix[:, path_cols])
        return matrix


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    return np.asarray(np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20)))), dtype=np.float64)


def _inv_softplus(y: np.ndarray) -> np.ndarray:
    """Inverse softplus for y > 0. Stable for large y."""
    y = np.maximum(y, 1e-12)
    return np.asarray(np.where(y > 20, y, np.log(np.expm1(np.clip(y, 0, 20)))), dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))), dtype=np.float64)


def _clip_concentration(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(x, MIN_BETA, MAX_BETA), dtype=np.float64)


def _trigamma(x: np.ndarray | float) -> np.ndarray | float:
    """Numerical trigamma via scipy.special.polygamma(1, x)."""
    result = polygamma(1, x)
    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=np.float64)
    return float(result)


def _dirichlet_kl(beta: np.ndarray, alpha: np.ndarray) -> float:
    """Closed-form KL(Dirichlet(beta) || Dirichlet(alpha))."""
    sum_beta = float(beta.sum())
    sum_alpha = float(alpha.sum())
    kl = (
        gammaln(sum_beta)
        - gammaln(sum_alpha)
        + (gammaln(alpha) - gammaln(beta)).sum()
        + ((beta - alpha) * (digamma(beta) - digamma(sum_beta))).sum()
    )
    return float(kl)


def _dirichlet_covariance(beta: np.ndarray) -> np.ndarray:
    """Closed-form covariance matrix of a Dirichlet posterior."""
    total = beta.sum()
    mean = beta / total
    cov = -np.outer(mean, mean) / (total + 1.0)
    np.fill_diagonal(cov, mean * (1.0 - mean) / (total + 1.0))
    return np.asarray(cov, dtype=np.float64)


def _relative_plateau(window: list[float]) -> bool:
    """ELBO has plateaued if the window's relative range is below tol."""
    w = np.asarray(window)
    span = float(w.max() - w.min())
    mag = max(abs(float(w.mean())), 1e-6)
    return span / mag < ELBO_CONVERGENCE_REL_TOL
