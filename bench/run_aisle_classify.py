"""Constrained-vocabulary 2-pass store-aisle enrichment (Robert's prompt-design
test, take 2). The free-form prompt degenerated on the 4-bit model (action-
fixation, 25% parse-fail); classification into a FIXED shared taxonomy plays to
the small model's strength and yields the shared-aisle property that makes the
signal useful.

  pass 1 — derive a fixed numbered shelf menu ("Aisle > Section") for the topic
  pass 2 — classify each item into ONE shelf number + confidence (batched)

Books are the one catalog with headroom: 87% have native category == ['Books']
(useless), but real titles. Does a clean LLM shelf taxonomy reconstruct cooc
structure where native metadata cannot? (Payoff metric: run_meta_cooc_map.py
meta_mode "aisle".)

  cache row: {item_id, aisle, section, conf}

Run: DATASET=amazon-book-chrono TOPIC="a large bookstore" N=6000 \
     .venv/bin/python bench/run_aisle_classify.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kindling.llm_enrich import _extract_json
from run_meta_cooc_map import load as load_ds  # same loader the probe uses

MODEL = "mlx-community/Phi-4-mini-instruct-4bit"
CACHE_DIR = Path(__file__).parent / "cache"
BATCH = int(os.environ.get("BATCH", "10"))
DESCRIBE = {"amazon-book-chrono": ["title"], "movielens-1m": ["title", "genres"]}

TAXONOMY_PROMPT = """List ~35 shelves of {topic}. Each line is a broad genre
followed by " > " then a specific section within it. Use real, distinct
bookstore genres (no duplicates). Example lines:
Fiction > Mystery & Thriller
Fiction > Science Fiction & Fantasy
Young Adult > Dystopian & Fantasy
Romance > Contemporary Romance
Non-Fiction > Biography & Memoir
Children > Picture Books
Output ONLY the lines, one shelf per line, no numbering."""

CLASSIFY_PROMPT = """{topic} is organized into these numbered shelves:
{menu}

For EACH item below, pick the ONE shelf number where it best belongs and a
confidence 0-1. Output ONLY a JSON object mapping each item's ID to a 2-element
array [shelf_number, confidence]. Example: {{"12": [7, 0.8]}}

Items:
{items_block}

JSON:"""


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def warm_items(train: pd.DataFrame, n: int) -> pd.Index:
    ids = pd.Index(train["item_id"].unique())
    idx = train["item_id"].map({it: i for i, it in enumerate(ids)}).to_numpy()
    deg = np.bincount(idx, minlength=len(ids))
    keep = np.where(deg >= 5)[0]
    keep = keep[np.argsort(-deg[keep])][:n]
    return ids[keep]


def main() -> None:
    dataset = os.environ.get("DATASET", "amazon-book-chrono")
    topic = os.environ.get("TOPIC", "a large bookstore")
    n = int(os.environ.get("N", "6000"))
    cache_path = CACHE_DIR / f"{dataset}_aisle.jsonl"
    menu_path = CACHE_DIR / f"{dataset}_aisle_menu.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train, _test, items = load_ds(dataset)
    items = items.set_index("item_id")
    cols = [c for c in DESCRIBE.get(dataset, ["title"]) if c in items.columns]
    warm = warm_items(train, n)
    log(f"{dataset}: classifying {len(warm)} warm items, cols={cols}, topic={topic!r}")

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(MODEL)
    sampler = make_sampler(temp=0.0)

    def gen(prompt: str, mx: int) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        return generate(model, tok, prompt=text, max_tokens=mx, sampler=sampler, verbose=False)

    # ── pass 1: shelf menu ────────────────────────────────────────────
    if menu_path.exists():
        menu = json.loads(menu_path.read_text())
    else:
        raw = gen(TAXONOMY_PROMPT.format(topic=topic.capitalize()), 900)
        seen, menu = set(), []
        for ln in raw.splitlines():
            ln = ln.strip(" -*\t0123456789.")
            key = ln.lower()
            if ">" in ln and len(ln) > 3 and key not in seen:
                seen.add(key); menu.append(ln)
        menu = menu[:48]
        menu_path.write_text(json.dumps(menu, indent=2) + "\n")
    log(f"pass1 menu: {len(menu)} shelves, e.g. {menu[:3]}")
    menu_block = "\n".join(f"{i}: {s}" for i, s in enumerate(menu))

    # ── pass 2: classify (resumable) ──────────────────────────────────
    done = set()
    if cache_path.exists():
        done = {json.loads(line)["item_id"]
                for line in cache_path.read_text().splitlines() if line.strip()}
    todo = [it for it in warm if it not in done]
    log(f"{len(done)} cached, {len(todo)} to classify")
    n_ok = n_fail = 0
    with open(cache_path, "a") as fh:
        n_batches = (len(todo) + BATCH - 1) // BATCH
        for bi in range(n_batches):
            batch = todo[bi * BATCH:(bi + 1) * BATCH]
            block = "\n".join(
                f'- ID "{it}": ' +
                "; ".join(f"{c}: {items.loc[it, c]}" for c in cols
                          if it in items.index and pd.notna(items.loc[it, c]))
                for it in batch)
            try:
                parsed = _extract_json(gen(
                    CLASSIFY_PROMPT.format(topic=topic.capitalize(), menu=menu_block,
                                           items_block=block), 600))
            except Exception:  # noqa: BLE001
                parsed = {}
            for it in batch:
                v = parsed.get(str(it), parsed.get(it))
                rec = {"item_id": it, "aisle": "", "section": "", "conf": 0.0}
                try:
                    shelf = int(v[0])
                    if 0 <= shelf < len(menu):
                        a, _, s = menu[shelf].partition(">")
                        rec.update(aisle=a.strip().lower(), section=s.strip().lower(),
                                   conf=float(v[1]))
                        n_ok += 1
                    else:
                        n_fail += 1
                except (TypeError, ValueError, IndexError):
                    n_fail += 1
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (bi + 1) % 25 == 0:
                log(f"batch {bi + 1}/{n_batches} ok={n_ok} fail={n_fail} "
                    f"({n_fail / max(n_ok + n_fail, 1):.0%} fail)")
    log(f"DONE ok={n_ok} fail={n_fail} ({n_fail / max(n_ok + n_fail, 1):.0%} fail) → {cache_path}")


if __name__ == "__main__":
    main()
