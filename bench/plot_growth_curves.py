"""Growth-curve grid: kindling vs baseline algorithms across datasets.

Renders a grid — one row per dataset, three columns (NDCG@10, Recall@10, Fit
time) — where each cell is a cold→hot growth curve (x = fraction of training
data) with one colored line per algorithm. Reads the cached warming-curve
benchmark (``bench/reports/warming_<dataset>.json``).

The kindling accuracy curves carry over to the native-only engine unchanged
(the Rust port is NDCG-identical to the Python engine it replaced; fit stays in
Python). The native recommend-latency win is reported separately in
``bench/reports/final_state_perf.json`` (see bench/final_state_perf.py).

Run:  python bench/plot_growth_curves.py
Out:  bench/reports/growth_curves_grid.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

REPORTS = Path(__file__).resolve().parent / "reports"

# Dataset rows (file stem → display label). Only those with cached data render.
DATASETS = [
    ("movielens-1m", "MovieLens-1M"),
    ("amazon-beauty", "Amazon Beauty"),
    ("steam", "Steam"),
    ("amazon-book-academic", "Amazon Books"),
]

# Algorithm → (display label, color, linewidth, z-order). kindling stands out.
MODELS = {
    "kindling": ("kindling", "#d6336c", 2.6, 6),
    "ease": ("EASE (base only)", "#f08c00", 1.8, 5),
    "lightgcn": ("LightGCN", "#7048e8", 1.8, 4),
    "implicit_als": ("implicit ALS", "#1c7ed6", 1.6, 3),
    "item_item_knn": ("item-kNN", "#2f9e44", 1.6, 3),
    "popularity": ("popularity", "#868e96", 1.6, 2),
}

COLS = [("ndcg@k", "NDCG@10"), ("recall@k", "Recall@10"), ("fit_seconds", "Fit time (s)")]


def _load(stem: str) -> dict | None:
    path = REPORTS / f"warming_{stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    available = [(s, label, d) for s, label in DATASETS if (d := _load(s)) is not None]
    if not available:
        print("no warming_*.json data found")
        return 1
    nrows, ncols = len(available), len(COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows), squeeze=False)
    fig.suptitle(
        "kindling vs. baseline recommenders — cold→hot growth curves",
        fontsize=15, fontweight="bold", y=0.997,
    )

    for r, (stem, label, data) in enumerate(available):
        rows = data["rows"]
        models_here = [m for m in MODELS if any(x["model"] == m for x in rows)]
        for c, (metric, col_label) in enumerate(COLS):
            ax = axes[r][c]
            for model in models_here:
                disp, color, lw, z = MODELS[model]
                pts = sorted(
                    ((x["fraction"], x.get(metric)) for x in rows if x["model"] == model),
                    key=lambda t: t[0],
                )
                xs = [p[0] for p in pts if p[1] is not None]
                ys = [p[1] for p in pts if p[1] is not None]
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", ms=3.5, color=color, lw=lw, zorder=z, label=disp)
            ax.set_xscale("log")
            if metric == "fit_seconds":
                # Log scale (fit spans ~0.1s→250s) but labelled in plain seconds
                # — "0.1s / 1s / 10s / 100s", not matplotlib's 10⁰ powers.
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}s"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", ls=":", alpha=0.4)
            if r == 0:
                ax.set_title(col_label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            if r == nrows - 1:
                ax.set_xlabel("fraction of training data (cold → hot)", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels),
        frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    out = REPORTS / "growth_curves_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({nrows} datasets × {ncols} metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
