"""Make the cooc base line explicit on every dataset.

On catalogs > 20k items kindling's auto base resolves to wilson-cooc, so the
`ease` (base-only) warming line on those datasets IS the cooc base already —
this copies those rows to explicit `cooc` rows so the grid plots one consistent
cooc line everywhere. The ≤20k datasets get real `cooc` rows from a forced-cooc
run (run_warming_curve MODELS=cooc); this only fills the large-catalog gap.

Run: PYTHONPATH=src .venv/bin/python bench/relabel_base_cooc.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "reports"


def main() -> int:
    for p in sorted(REPORTS.glob("warming_*.json")):
        d = json.loads(p.read_text())
        models = {r["model"] for r in d["rows"]}
        if "cooc" in models:
            continue
        if d.get("catalog", 0) <= 20000:
            continue  # ≤20k: ease != cooc; needs a real forced-cooc run
        copied = [{**r, "model": "cooc"} for r in d["rows"] if r["model"] == "ease"]
        if not copied:
            continue
        d["rows"].extend(copied)
        d["rows"].sort(key=lambda r: (r["fraction"], r["model"]))
        p.write_text(json.dumps(d, indent=2) + "\n")
        print(f"{p.name}: +{len(copied)} cooc rows (copied from ease, catalog {d['catalog']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
