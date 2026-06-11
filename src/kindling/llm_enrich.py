"""LLM metadata enrichment.

Generates descriptive keywords per item from a small local model (MLX)
or any callable backend, producing a list-valued column that feeds
straight into ``ItemFeatureExtractor`` (which ingests Python lists as
multi-hot features — no special handling needed downstream).

Design points:
  - **Batched**: N items per prompt; the model returns one JSON object
    mapping each item key to its keyword list.
  - **Resumable**: results append to a JSONL cache keyed by item_id;
    re-runs skip cached items. Interrupting costs nothing.
  - **Backend-pluggable**: default backend is mlx_lm with whatever
    model path is given; pass any ``generate_fn(prompt) -> str`` to use
    an API instead.
  - Parse failures degrade to empty keyword lists (items keep their
    curated features; the channel just gets nothing extra for them).

Usage:
    enr = LLMEnricher(cache_path="bench/cache/ml1m_keywords.jsonl")
    kw = enr.enrich(items, describe_cols=["title", "genres"])  # {item_id: [kw, ...]}
    items["llm_keywords"] = items["item_id"].map(kw)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_DEFAULT_MODEL = "mlx-community/Phi-4-mini-instruct-4bit"

_PROMPT_TEMPLATE = """You are labeling catalog items for a recommender system.
For EACH item below, give 8-12 short lowercase keywords capturing its genre,
style, themes, era, tone, and unique features. Be specific (prefer
"slow-burn-thriller" over "movie"). Output ONLY a JSON object mapping each
item's ID to its keyword array, nothing else.

Items:
{items_block}

JSON:"""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model output (tolerates fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Truncated output: trim to the last complete entry.
        s = m.group(0)
        last = s.rfind("]")
        if last > 0:
            try:
                return json.loads(s[: last + 1] + "}")
            except json.JSONDecodeError:
                return {}
        return {}


class LLMEnricher:
    def __init__(
        self,
        cache_path: str | Path,
        model_path: str = _DEFAULT_MODEL,
        batch_size: int = 8,
        max_tokens: int = 800,
        generate_fn: Callable[[str], str] | None = None,
        prompt_template: str = _PROMPT_TEMPLATE,
    ):
        self.cache_path = Path(cache_path)
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template
        self._generate_fn = generate_fn
        self._model = None
        self._tokenizer = None

    # ── backend ────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(prompt)
        if self._model is None:
            from mlx_lm import load
            self._model, self._tokenizer = load(self.model_path)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self._model, self._tokenizer, prompt=text,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )

    # ── cache ──────────────────────────────────────────────────────

    def _load_cache(self) -> dict[Any, list[str]]:
        out: dict[Any, list[str]] = {}
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        out[d["item_id"]] = d["keywords"]
                    except Exception:
                        continue
        return out

    def _append_cache(self, item_id: Any, keywords: list[str]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"item_id": item_id, "keywords": keywords}) + "\n")

    # ── enrichment ─────────────────────────────────────────────────

    def enrich(
        self,
        items: pd.DataFrame,
        describe_cols: list[str],
        item_id_column: str = "item_id",
        limit: int | None = None,
        progress_every: int = 25,
    ) -> dict[Any, list[str]]:
        """Return {item_id: [keyword, ...]} for every item, generating
        only the ones missing from the cache."""
        cache = self._load_cache()
        todo = items[~items[item_id_column].isin(cache.keys())]
        if limit is not None:
            todo = todo.head(limit)
        rows = todo[[item_id_column, *describe_cols]].to_dict("records")
        n_batches = (len(rows) + self.batch_size - 1) // max(self.batch_size, 1)
        for bi in range(n_batches):
            batch = rows[bi * self.batch_size : (bi + 1) * self.batch_size]
            block_lines = []
            for r in batch:
                desc = "; ".join(
                    f"{c}: {r[c]}" for c in describe_cols if pd.notna(r.get(c))
                )
                block_lines.append(f'- ID "{r[item_id_column]}": {desc}')
            prompt = self.prompt_template.format(items_block="\n".join(block_lines))
            try:
                parsed = _extract_json(self._generate(prompt))
            except Exception:
                parsed = {}
            for r in batch:
                iid = r[item_id_column]
                kws = parsed.get(str(iid), parsed.get(iid, []))
                if not isinstance(kws, list):
                    kws = []
                kws = [str(k).strip().lower() for k in kws if str(k).strip()][:12]
                cache[iid] = kws
                self._append_cache(iid, kws)
            if progress_every and (bi + 1) % progress_every == 0:
                done = sum(1 for v in cache.values() if v)
                print(f"batch {bi + 1}/{n_batches} ({done} items keyworded)", flush=True)
        return cache
