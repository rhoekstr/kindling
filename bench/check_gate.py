"""CI regression gate: fail if NDCG@10 drops below the gates.toml baseline.

Runs the default-config verification (``verify.evaluate``) for the given
datasets and compares to the frozen per-dataset baseline in
``bench/gates.toml``. A drop greater than ``ndcg_at_k_regression_limit``
(relative) exits non-zero so CI fails.

Usage:
    python bench/check_gate.py                # CI-feasible datasets (ml1m)
    python bench/check_gate.py amazon-beauty  # explicit dataset(s)

Only movielens-1m is run in CI (cached, ~10s); steam/book need large
local caches and are validated manually.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import evaluate

GATES = Path(__file__).resolve().parent / "gates.toml"
CI_DATASETS = ["movielens-1m"]


def main(argv: list[str]) -> int:
    cfg = tomllib.loads(GATES.read_text())
    limit = float(cfg["gates"]["ndcg_at_k_regression_limit"])
    baselines = cfg["baseline"]["ndcg_at_10"]

    datasets = argv[1:] or CI_DATASETS
    failures = 0
    for ds in datasets:
        if ds not in baselines:
            print(f"SKIP {ds}: no baseline in gates.toml")
            continue
        base = float(baselines[ds])
        got = float(evaluate(ds, quiet=True)["ndcg@10"])
        drop = (base - got) / base if base else 0.0
        status = "PASS"
        if drop > limit:
            status = "FAIL"
            failures += 1
        print(
            f"[{status}] {ds}: NDCG@10 {got:.4f} vs baseline {base:.4f} "
            f"({drop:+.1%}; limit {limit:.0%})"
        )

    if failures:
        print(f"\n{failures} dataset(s) regressed beyond the gate.")
        return 1
    print("\nAll datasets within the regression gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
