"""Render the new-user onboarding curve: NDCG@10 vs number of seed interactions,
kindling (recommend_for_items) vs popularity. One panel per dataset.

Run: .venv/bin/python bench/plot_onboarding.py amazon-beauty steam movielens-1m
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path(__file__).parent / "reports"


def main() -> None:
    datasets = sys.argv[1:] or ["amazon-beauty", "steam", "movielens-1m"]
    data = {d: json.loads((REPORTS / f"onboarding_{d}.json").read_text()) for d in datasets}
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.0), squeeze=False)
    for c, d in enumerate(datasets):
        ax = axes[0][c]
        rows = sorted(data[d]["rows"], key=lambda r: r["seeds"])
        xs = [r["seeds"] for r in rows]
        ax.plot(xs, [r["kindling_ndcg"] for r in rows], marker="o", ms=5,
                color="#d62728", lw=2.6, label="kindling (recommend_for_items)")
        ax.plot(xs, [r["popularity_ndcg"] for r in rows], marker="s", ms=4,
                color="#7f7f7f", lw=1.6, label="popularity (new-user fallback)")
        ax.set_xlabel("# seed interactions from the new user")
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"{d} — new-user onboarding", fontsize=11)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_xticks(xs)
        if c == 0:
            ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = REPORTS / "onboarding_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
