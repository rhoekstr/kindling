"""Prior-sensitivity study for the Bayesian blend.

For each prior coefficient in priors.toml, vary it by +/-50% and refit
the Engine. Measure the resulting shift in posterior means per signal.
Small shifts on high-data signals indicate the prior is dominated by
the likelihood (good); large shifts reveal signals where the prior is
doing most of the work (candidate for Phase 7 retuning, or candidate
for removal per ADR growth-curves item G).

CLI:
    python -m kindling.benchmarks.prior_sensitivity \\
        --dataset synthetic-grocery-deep \\
        --fractions 0.1,0.5,1.0 \\
        --output bench/reports/prior_sensitivity.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kindling import Engine, __version__
from kindling.benchmarks.comparison import _load_dataset
from kindling.blend.priors import load_prior_coefficients


# The set of coefficient paths we'll perturb. (section, subkey) pairs; the
# subkey is None when the whole section has a single scalar coefficient.
_PERTURBABLE: list[tuple[str, str]] = [
    ("graph_density", "coefficient"),
    ("clustering_coefficient", "coefficient"),
    ("session_density_full", "coefficient"),
    ("session_density_tail", "coefficient"),
    ("session_density_basket", "coefficient"),
]


@dataclass(frozen=True)
class PerturbationResult:
    section: str
    direction: str  # "+50%" or "-50%"
    fraction: float
    signal_shifts: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "section": self.section,
            "direction": self.direction,
            "fraction": self.fraction,
            "signal_shifts": self.signal_shifts,
        }


def _fit_and_get_posterior(
    train_df,
    coefs_override: dict | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Fit an engine with the given coefficient override and return
    (signal_names, posterior_mean)."""
    # Monkey-patch the load_prior_coefficients function just for this fit.
    import kindling.blend.priors as priors_mod

    original_loader = priors_mod.load_prior_coefficients
    if coefs_override is not None:
        priors_mod.load_prior_coefficients = lambda: coefs_override  # type: ignore[assignment]
    try:
        engine = Engine(vi_max_iter=100)
        engine.fit(train_df)
        blend = engine._bayesian_blend
        assert blend is not None, "Bayesian blend not active"
        return tuple(blend.signal_names), blend.posterior_mean.copy()
    finally:
        priors_mod.load_prior_coefficients = original_loader  # type: ignore[assignment]


def _perturb(
    coefs: dict, section: str, subkey: str, multiplier: float
) -> dict:
    out = copy.deepcopy(coefs)
    out[section][subkey] = float(out[section][subkey]) * multiplier
    return out


def run_prior_sensitivity(
    dataset: str,
    fractions: list[float],
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction=0.1)
    base_coefs = load_prior_coefficients()

    results: list[PerturbationResult] = []
    for frac in fractions:
        subset = split.train.iloc[: int(len(split.train) * frac)].reset_index(drop=True)
        print(f"\n=== fraction={frac:.2f} ({len(subset):,} interactions) ===", flush=True)

        # Baseline posterior with default priors.
        signal_names, baseline_mean = _fit_and_get_posterior(subset, None)
        name_idx = {n: i for i, n in enumerate(signal_names)}
        print("  baseline posterior:", {n: round(float(baseline_mean[i]), 3) for n, i in name_idx.items()})

        for section, subkey in _PERTURBABLE:
            if section not in base_coefs or subkey not in base_coefs[section]:
                continue
            for multiplier, label in [(1.5, "+50%"), (0.5, "-50%")]:
                perturbed = _perturb(base_coefs, section, subkey, multiplier)
                _, mean = _fit_and_get_posterior(subset, perturbed)
                shifts = {
                    n: float(mean[i] - baseline_mean[i]) for n, i in name_idx.items()
                }
                results.append(
                    PerturbationResult(
                        section=section,
                        direction=label,
                        fraction=frac,
                        signal_shifts=shifts,
                    )
                )
            max_abs_shift = max(
                abs(s) for r in results[-2:] for s in r.signal_shifts.values()
            )
            print(f"    {section:<28} max |shift| = {max_abs_shift:.3f}")

    return {
        "dataset": dataset,
        "kindling_version": __version__,
        "perturbations": [r.as_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prior-sensitivity study.")
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=["movielens-1m", "synthetic-grocery", "synthetic-grocery-deep"],
    )
    parser.add_argument("--fractions", default="0.1,0.5,1.0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    report = run_prior_sensitivity(args.dataset, fractions)
    pretty = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"\nWrote {args.output}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
