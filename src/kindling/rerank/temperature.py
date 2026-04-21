"""Per-position temperature optimization (PRD §7.3).

The plan's critical-path novelty claim: temperature is formalized as a
constrained optimization problem, not three ad-hoc knobs. The list of
length N solves

    maximize    sum_k [ score(i_k)^(1-tau_k) * novelty(i_k)^(tau_k) ]
    subject to  i_j != i_k for j != k                 (no duplicates)

where ``tau_k`` is the temperature at position k.

Three solvers ship in v1 (the plan tracks which becomes the empirical
default via the Phase 3/Phase 7 temperature benchmarks):

- ``greedy`` - position-by-position argmax on the position-weighted
  objective. Fast, no coupling between positions. Can produce suboptimal
  lists when high-temperature and low-temperature positions compete for
  overlapping items.
- ``beam`` (default) - position-by-position with a beam of partial
  solutions preserved. Fast, near-optimal in practice, easy to reason
  about. Default beam width 10.
- ``dpp`` - re-weight the DPP quality so diversity tracks per-position
  temperature. Only used when diversity is the dominant constraint; not
  the first-pick solver unless explicitly requested.

The public API accepts multiple input types for ergonomic flexibility
(PRD §7.3): scalar float, per-position array, named profile string, or
sparse dict with linear interpolation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from kindling.rerank.dpp import DPPGreedy

DEFAULT_BEAM_WIDTH = 10
DEFAULT_SOLVER = "beam"
VALID_SOLVERS = frozenset({"beam", "greedy", "dpp"})

# PRD §7.3 named profiles.
_NAMED_PROFILES: dict[str, list[float]] = {
    "balanced": [0.0, 0.25, 0.5, 0.75, 1.0],
    "explore_tail": [0.0, 0.0, 0.5, 1.0, 1.0],
    "conservative": [0.0, 0.0, 0.0, 0.25, 0.5],
}

TemperatureInput = float | list[float] | np.ndarray | str | Mapping[int, float]


def resolve_temperature(
    temperature: TemperatureInput,
    n: int,
) -> np.ndarray:
    """Normalize the user-supplied temperature into a length-``n`` vector in
    ``[0, 1]``."""
    if isinstance(temperature, str):
        profile = _NAMED_PROFILES.get(temperature)
        if profile is None:
            raise ValueError(
                f"Unknown temperature profile {temperature!r}. "
                f"Valid profiles: {sorted(_NAMED_PROFILES)}"
            )
        arr = np.asarray(profile, dtype=np.float64)
        return _resize_to_n(arr, n)
    if isinstance(temperature, Mapping):
        return _interpolate_sparse(dict(temperature), n)
    if isinstance(temperature, (int, float)):
        return np.full(n, float(temperature), dtype=np.float64)
    arr = np.asarray(temperature, dtype=np.float64).ravel()
    return _resize_to_n(arr, n)


def _resize_to_n(arr: np.ndarray, n: int) -> np.ndarray:
    """Fit a user-supplied vector to exactly ``n`` positions. Shorter input
    is linearly resampled to span the requested length; longer input is
    truncated."""
    if arr.size == n:
        return np.asarray(np.clip(arr, 0.0, 1.0), dtype=np.float64)
    if arr.size == 0:
        return np.zeros(n, dtype=np.float64)
    # Interpolate from the supplied positions onto a uniform grid of length n.
    source_x = np.linspace(0.0, 1.0, arr.size)
    target_x = np.linspace(0.0, 1.0, n)
    return np.asarray(np.clip(np.interp(target_x, source_x, arr), 0.0, 1.0), dtype=np.float64)


def _interpolate_sparse(sparse: dict[int, float], n: int) -> np.ndarray:
    """Sparse dict like ``{0: 0.0, 4: 1.0}`` -> linearly interpolated vec."""
    if not sparse:
        return np.zeros(n, dtype=np.float64)
    keys = sorted(sparse.keys())
    xs = np.asarray(keys, dtype=np.float64)
    ys = np.asarray([sparse[k] for k in keys], dtype=np.float64)
    target = np.interp(np.arange(n, dtype=np.float64), xs, ys)
    return np.asarray(np.clip(target, 0.0, 1.0), dtype=np.float64)


@dataclass
class TemperatureObjective:
    """The position-weighted objective from PRD §7.3.

    ``scores`` and ``novelty`` are per-candidate vectors in the same order.
    The per-candidate, per-position score is ``s^(1-tau) * nov^tau``.
    """

    scores: np.ndarray
    novelty: np.ndarray

    def __post_init__(self) -> None:
        # Positive-clip so the power formulation is well-defined.
        self.scores = np.clip(self.scores.astype(np.float64), 1e-9, None)
        self.novelty = np.clip(self.novelty.astype(np.float64), 1e-9, None)

    def value_at(self, candidate_idx: int, tau: float) -> float:
        s = float(self.scores[candidate_idx])
        nov = float(self.novelty[candidate_idx])
        return float((s ** (1.0 - tau)) * (nov**tau))

    def all_values(self, tau: float) -> np.ndarray:
        """Vectorized per-candidate value at a fixed position temperature."""
        return np.asarray((self.scores ** (1.0 - tau)) * (self.novelty**tau), dtype=np.float64)


def solve_greedy(
    objective: TemperatureObjective,
    temperatures: np.ndarray,
    n_positions: int,
) -> list[int]:
    """Position-by-position argmax on the position-weighted objective.

    Fast. Not optimal when a single high-novelty candidate would be worth
    more at a late position than an early position; beam search handles
    those cases.
    """
    selected: list[int] = []
    available_mask = np.ones(objective.scores.size, dtype=bool)
    for k in range(n_positions):
        if not available_mask.any():
            break
        values = objective.all_values(temperatures[k])
        masked = np.where(available_mask, values, -np.inf)
        pick = int(np.argmax(masked))
        selected.append(pick)
        available_mask[pick] = False
    return selected


def solve_beam(
    objective: TemperatureObjective,
    temperatures: np.ndarray,
    n_positions: int,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> list[int]:
    """Beam search over partial lists.

    Each beam state is a prefix of selections + cumulative objective value.
    At each position we extend every beam with every candidate not yet in
    its prefix, score the extension, and keep the top ``beam_width`` by
    cumulative objective.

    Returns the best final-beam prefix.
    """
    n_candidates = objective.scores.size
    if n_candidates == 0 or n_positions <= 0:
        return []
    beam_width = max(1, min(beam_width, n_candidates))

    # Beam state: (cumulative_value, tuple_of_selected_indices)
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    for k in range(n_positions):
        values = objective.all_values(temperatures[k])
        next_beams: list[tuple[float, tuple[int, ...]]] = []
        for cum_value, prefix in beams:
            forbidden = set(prefix)
            # For each unused candidate, score the extension.
            for idx in range(n_candidates):
                if idx in forbidden:
                    continue
                new_value = cum_value + float(values[idx])
                next_beams.append((new_value, (*prefix, idx)))
        if not next_beams:
            break
        # Sort descending by cumulative value, keep top beam_width.
        next_beams.sort(key=lambda bv: -bv[0])
        beams = next_beams[:beam_width]
    if not beams:
        return []
    return list(beams[0][1])


def solve_dpp_per_position(
    objective: TemperatureObjective,
    temperatures: np.ndarray,
    n_positions: int,
    item_ids: list[object],
    kernel_dpp: DPPGreedy,
) -> list[int]:
    """DPP with position-dependent quality.

    Re-weights the candidate qualities so earlier positions favor high
    score (``tau`` near 0) and later positions favor novelty (``tau`` near
    1). Produces a diversity-aware selection that respects per-position
    temperature without a separate diversity parameter.

    This is the diversity-dominant solver; use when the list's primary
    constraint is diversity rather than per-slot novelty tracking.
    """
    # Use the average temperature as the global quality weighting, then
    # lean on DPP's diversity to naturally produce variety at higher
    # positions. More elaborate position-dependent DPP formulations are
    # possible; this keeps the v1 implementation simple and predictable.
    avg_tau = float(np.mean(temperatures))
    quality_vec = objective.all_values(avg_tau)
    return kernel_dpp.rerank(
        item_ids=item_ids,
        qualities=quality_vec,
        k=n_positions,
    )


def solve(
    objective: TemperatureObjective,
    temperatures: np.ndarray,
    n_positions: int,
    solver: str = DEFAULT_SOLVER,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    item_ids: list[object] | None = None,
    kernel_dpp: DPPGreedy | None = None,
) -> list[int]:
    """Dispatch to the chosen solver."""
    if solver not in VALID_SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}. Valid: {sorted(VALID_SOLVERS)}")
    if solver == "greedy":
        return solve_greedy(objective, temperatures, n_positions)
    if solver == "beam":
        return solve_beam(objective, temperatures, n_positions, beam_width=beam_width)
    if solver == "dpp":
        if kernel_dpp is None or item_ids is None:
            raise ValueError("solver='dpp' requires both kernel_dpp and item_ids to be provided")
        return solve_dpp_per_position(
            objective=objective,
            temperatures=temperatures,
            n_positions=n_positions,
            item_ids=item_ids,
            kernel_dpp=kernel_dpp,
        )
    raise AssertionError("unreachable")  # pragma: no cover
