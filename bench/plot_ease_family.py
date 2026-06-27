"""EASE-family performance chart: EASE vs EDLAE vs RLAE vs ADMM-SLIM (+ cooc
reference) at full data — NDCG@10 and fit time, per dataset. Reads the warming
JSONs (full-data rows). Only the EASE-feasible catalogs render a bar.

Run: python bench/plot_ease_family.py
Out: bench/reports/ease_family_chart.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"

# EASE-feasible datasets that completed the dense variant solve on this box,
# small → large. tafeng (24k, ~9GB / 145s) is empirically the largest that fits;
# 38k+ (yelp/hm/instacart) exceed this box's ~18GB Gram wall — see the doc.
DATASETS = [
    ("movielens-1m", "ML-1M\n3.7k"),
    ("amazon-beauty", "Beauty\n12k"),
    ("tafeng", "Ta-Feng\n24k"),
]
# model key → (label, color)
MODELS = [
    ("ease_full", "EASE", "#1c7ed6"),
    ("edlae", "EDLAE", "#d6336c"),
    ("rlae", "RLAE", "#f59f00"),
    ("admm_slim", "ADMM-SLIM", "#2f9e44"),
    ("cooc", "cooc (base)", "#868e96"),
]


def _full(data, model, field):
    rows = [r for r in data["rows"] if r["model"] == model]
    if not rows:
        return None
    r = max(rows, key=lambda r: r["fraction"])  # full-data point
    return r.get(field)


def main() -> int:
    loaded = []
    for stem, label in DATASETS:
        p = REPORTS / f"warming_{stem}.json"
        if p.exists():
            loaded.append((label, json.loads(p.read_text())))
    if not loaded:
        print("no data")
        return 1

    labels = [lbl for lbl, _ in loaded]
    x = np.arange(len(labels))
    nm = len(MODELS)
    w = 0.8 / nm

    fig, (ax_n, ax_f) = plt.subplots(2, 1, figsize=(1.3 * len(labels) + 3, 7))
    for mi, (key, mlabel, color) in enumerate(MODELS):
        nd = [(_full(d, key, "ndcg@k") or np.nan) for _, d in loaded]
        ft = [(_full(d, key, "fit_seconds") or np.nan) for _, d in loaded]
        off = (mi - (nm - 1) / 2) * w
        ax_n.bar(x + off, nd, w, label=mlabel, color=color)
        ax_f.bar(x + off, ft, w, label=mlabel, color=color)

    ax_n.set_title("EASE family — NDCG@10 at full data", fontweight="bold")
    ax_n.set_ylabel("NDCG@10")
    ax_n.set_xticks(x)
    ax_n.set_xticklabels(labels)
    ax_n.legend(ncol=nm, fontsize=8, loc="upper right")
    ax_n.grid(axis="y", ls=":", alpha=0.4)

    ax_f.set_title("Fit time (s, log) — dense item-item solve", fontweight="bold")
    ax_f.set_ylabel("fit seconds")
    ax_f.set_yscale("log")
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(labels)
    ax_f.grid(axis="y", which="both", ls=":", alpha=0.4)

    fig.suptitle("EASE-family variants across catalog sizes", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = REPORTS / "ease_family_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({len(labels)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
