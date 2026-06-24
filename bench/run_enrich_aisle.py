"""LLM enrichment via STORE-AISLE placement (Robert's prompt-design test).

Hypothesis: free-keyword prompts capture *topical* similarity, but asking the
model where an item would be *shelved* — aisle + section, with confidence —
captures *shopping-context* similarity (what a shopper considers together),
which may align with co-purchase (cooc) better than keywords.

This script only GENERATES the metadata (resumable JSONL cache). The payoff
metric — does it reconstruct cooc structure better than keywords/native? — is
the mapping-R² / neighbor-recovery probe in run_meta_cooc_map.py (meta_mode
"aisle"), run after generation.

  cache row: {item_id, aisle, aisle_conf, section, section_conf}

Run: DATASET=movielens-1m TOPIC=movies .venv/bin/python bench/run_enrich_aisle.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from kindling.benchmarks.comparison import _load_dataset
from kindling.llm_enrich import _extract_json

MODEL = "mlx-community/Phi-4-mini-instruct-4bit"
CACHE_DIR = Path(__file__).parent / "cache"
BATCH = int(os.environ.get("BATCH", "6"))
# Title only for ml1m: feeding the native genres makes the model echo them
# (circular vs the native-genre baseline); title-only forces world-knowledge
# placement — an independent signal to test against native genres.
DESCRIBE = {"movielens-1m": ["title", "genres"], "amazon-beauty": ["store", "category"],
            "steam": ["title", "tags"], "amazon-book-chrono": ["title", "categories"]}


def _clean(s: str) -> str:
    s = str(s).strip().lower()
    for p in ("aisle:", "section:", "aisle ", "section "):
        if s.startswith(p):
            s = s[len(p):].strip()
    return s

PROMPT = """You are shelving catalog items in a store that sells {topic}.
For EACH item below, decide which AISLE it belongs in and which SECTION within
that aisle, with a confidence 0-1 for each. Use broad, reusable aisle and
section names (like a real store) so many items naturally share them. Output
ONLY a JSON object mapping each item's ID to a 4-element array
[aisle, aisle_confidence, section, section_confidence].
Example: {{"12": ["skincare", 0.9, "facial moisturizers", 0.7]}}

Items:
{items_block}

JSON:"""


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    dataset = os.environ.get("DATASET", "movielens-1m")
    topic = os.environ.get("TOPIC", "movies")
    limit = int(os.environ.get("LIMIT", "0")) or None
    cache_path = CACHE_DIR / f"{dataset}_aisle.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    items = _load_dataset(dataset, test_fraction=0.1).items
    cols = [c for c in DESCRIBE.get(dataset, ["title"]) if c in items.columns]
    log(f"{dataset}: {len(items)} items, describe cols {cols}, topic={topic!r}")

    done = set()
    if cache_path.exists():
        done = {json.loads(line)["item_id"]
                for line in cache_path.read_text().splitlines() if line.strip()}
    todo = items[~items["item_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    rows = todo[["item_id", *cols]].to_dict("records")
    log(f"{len(done)} cached, {len(rows)} to generate")
    if not rows:
        log("nothing to do"); return

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(MODEL)
    sampler = make_sampler(temp=0.0)

    def gen(prompt: str) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        return generate(model, tok, prompt=text, max_tokens=700, sampler=sampler, verbose=False)

    n_batches = (len(rows) + BATCH - 1) // BATCH
    n_ok = n_fail = 0
    with open(cache_path, "a") as fh:
        for bi in range(n_batches):
            batch = rows[bi * BATCH:(bi + 1) * BATCH]
            block = "\n".join(
                f'- ID "{r["item_id"]}": ' +
                "; ".join(f"{c}: {r[c]}" for c in cols if pd.notna(r.get(c)))
                for r in batch)
            try:
                parsed = _extract_json(gen(PROMPT.format(topic=topic, items_block=block)))
            except Exception:  # noqa: BLE001
                parsed = {}
            for r in batch:
                iid = r["item_id"]
                v = parsed.get(str(iid), parsed.get(iid))
                rec = {"item_id": iid, "aisle": "", "aisle_conf": 0.0,
                       "section": "", "section_conf": 0.0}
                if isinstance(v, list) and len(v) >= 4:
                    try:
                        rec.update(aisle=_clean(v[0]), aisle_conf=float(v[1]),
                                   section=_clean(v[2]), section_conf=float(v[3]))
                        n_ok += 1
                    except (ValueError, TypeError):
                        n_fail += 1
                else:
                    n_fail += 1
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (bi + 1) % 20 == 0:
                log(f"batch {bi + 1}/{n_batches}  ok={n_ok} fail={n_fail} "
                    f"({n_fail / max(n_ok + n_fail, 1):.0%} parse-fail)")
    log(f"DONE ok={n_ok} fail={n_fail} ({n_fail / max(n_ok + n_fail, 1):.0%} parse-fail) → {cache_path}")


if __name__ == "__main__":
    main()
