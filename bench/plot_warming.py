"""Render the warming-curve benchmark: kindling vs standard algorithms as data
warms. One row per dataset, columns = NDCG@10, Recall@10, fit-time (speed).

Run: .venv/bin/python bench/plot_warming.py movielens-1m amazon-beauty
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path(__file__).parent / "reports"

# model -> (label, color, linewidth, zorder)
STYLE = {
    "kindling":      ("kindling (ours)", "#d62728", 2.6, 5),
    "implicit_als":  ("implicit ALS",    "#1f77b4", 1.6, 3),
    "item_item_knn": ("item-item kNN",   "#2ca02c", 1.6, 2),
    "popularity":    ("popularity",      "#7f7f7f", 1.6, 1),
}
PANELS = [("ndcg@k", "NDCG@10  (accuracy)", False),
          ("recall@k", "Recall@10", False),
          ("fit_seconds", "fit time (s)  — lower better", True)]


def main() -> None:
    datasets = sys.argv[1:] or ["movielens-1m", "amazon-beauty"]
    data = {d: json.loads((REPORTS / f"warming_{d}.json").read_text()) for d in datasets}
    nrow, ncol = len(datasets), len(PANELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)

    for r, d in enumerate(datasets):
        rows = data[d]["rows"]
        by_model: dict[str, list] = {}
        for row in rows:
            by_model.setdefault(row["model"], []).append(row)
        for c, (key, ylab, logy) in enumerate(PANELS):
            ax = axes[r][c]
            for model, recs in by_model.items():
                if model not in STYLE:
                    continue
                recs = sorted(recs, key=lambda x: x["fraction"])
                xs = [x["fraction"] for x in recs]
                ys = [x[key] for x in recs]
                lab, col, lw, z = STYLE[model]
                ax.plot(xs, ys, marker="o", ms=3.5, color=col, lw=lw, zorder=z,
                        label=lab if (r == 0 and c == 0) else None)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("fraction of interactions (data warmth)")
            ax.grid(True, which="both", alpha=0.25, lw=0.5)
            ax.set_title(f"{d} — {ylab}", fontsize=10)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center",
               ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPORTS / "warming_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
