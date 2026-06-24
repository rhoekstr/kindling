"""Render per-user-warmth segmentation: NDCG@10 by user history-length bucket,
grouped bars per algorithm. The cold-user proof — within each bucket, who wins?

Run: .venv/bin/python bench/plot_user_warmth.py amazon-beauty movielens-1m
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORTS = Path(__file__).parent / "reports"
ORDER = ["kindling", "implicit_als", "item_item_knn", "popularity"]
STYLE = {"kindling": ("kindling (ours)", "#d62728"),
         "implicit_als": ("implicit ALS", "#1f77b4"),
         "item_item_knn": ("item-item kNN", "#2ca02c"),
         "popularity": ("popularity", "#7f7f7f")}


def main() -> None:
    datasets = sys.argv[1:] or ["amazon-beauty", "movielens-1m"]
    data = {d: json.loads((REPORTS / f"user_warmth_{d}.json").read_text()) for d in datasets}
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.2), squeeze=False)
    for c, d in enumerate(datasets):
        ax = axes[0][c]
        res = data[d]["results"]
        buckets = list(next(iter(res.values())).keys())
        counts = data[d]["bucket_counts"]
        xlabels = [f"{b}\n(n={counts[b]})" for b in buckets]
        x = np.arange(len(buckets))
        w = 0.2
        for i, model in enumerate(ORDER):
            if model not in res:
                continue
            ys = [res[model][b]["ndcg"] or 0.0 for b in buckets]
            lab, col = STYLE[model]
            ax.bar(x + (i - 1.5) * w, ys, w, label=lab, color=col,
                   edgecolor="black" if model == "kindling" else "none", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlabel("user train-history length (cold → warm)")
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"{d} — accuracy by user warmth", fontsize=11)
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
        if c == 0:
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = REPORTS / "user_warmth_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
