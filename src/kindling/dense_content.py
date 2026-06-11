"""Dense content embeddings from niche-positioning LLM text.

Upgrades the content channel along two axes, per the validated
weaknesses of keyword bags:

1. **Representation**: dense sentence embeddings (MiniLM, 384-d)
   instead of multi-hot keyword features. Synonyms land near each
   other ("whimsical" ≈ "lighthearted") instead of sharing no feature.
2. **Prompting**: niche-POSITIONING text instead of generic keywords.
   The instruction bans catalog-wide category words ("movie",
   "product"), filler adjectives ("unique", "great", "classic"), and
   meta-language ("this item stands out") — all of which inflate
   pairwise similarity without discriminating. What remains is the
   item's coordinates within its niches: micro-genre, style, era,
   audience, mood.

Also provides USER profile text generation (the user's taste niches,
inferred from their history) so user-side embeddings can replace or
augment the mean-of-owned-items profile.

Embedding backend: sentence-transformers all-MiniLM-L6-v2 (local,
~80MB, already cached). Embeddings are L2-normalized so dot = cosine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kindling.llm_enrich import LLMEnricher

NICHE_ITEM_PROMPT = """You are positioning catalog items for a recommender system.
For EACH item below, output 4-6 short phrases that place it within its
specific niche(s): micro-genre, style, era, audience, mood. Every phrase
must discriminate this item from the broad catalog — name what it IS,
specifically.

RULES:
- BANNED: category words that apply to the whole catalog ("movie", "film",
  "product", "item"), filler adjectives ("unique", "great", "classic",
  "popular", "well-known", "iconic"), and meta-language ("this item",
  "stands out", "appeals to").
- Prefer compound niche terms: "psychological courtroom thriller",
  "french new wave romance", "stop-motion dark fantasy",
  "synth-heavy 80s nostalgia".

Output ONLY a JSON object mapping each item's ID to its phrase array,
nothing else.

Items:
{items_block}

JSON:"""

USER_PROFILE_PROMPT = """You are profiling users of a recommender system by their history.
For EACH user below, output 8-12 short lowercase niche descriptors naming
the taste patterns their choices reveal: recurring micro-genres, styles,
eras, moods. Name the pattern across their items, not the items themselves.

RULES:
- BANNED: category words ("movies", "films", "products"), filler
  ("varied", "diverse", "eclectic", "enjoys", "likes"), and meta-language
  ("this user", "their taste").
- Prefer compound niche terms like "crime-epic", "practical-effects-horror",
  "screwball-romance", "90s-indie-drama".

Output ONLY a JSON object mapping each user's ID to its descriptor array,
nothing else.

Users:
{items_block}

JSON:"""

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """L2-normalized MiniLM embeddings, shape (n, 384). Empty strings
    embed to zero vectors (cosine 0 with everything)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_EMBED_MODEL)
    nonempty = [i for i, t in enumerate(texts) if t and t.strip()]
    out = np.zeros((len(texts), 384), dtype=np.float32)
    if nonempty:
        vecs = _embedder.encode(
            [texts[i] for i in nonempty],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out[nonempty] = np.asarray(vecs, dtype=np.float32)
    return out


def phrases_to_text(phrases: list[str]) -> str:
    """Join niche phrases for embedding. Comma-joined keeps MiniLM's
    attention on the content words rather than sentence scaffolding."""
    return ", ".join(p.strip() for p in phrases if p and p.strip())


def gen_item_niches(
    items: pd.DataFrame,
    describe_cols: list[str],
    cache_path: str | Path,
    limit: int | None = None,
    batch_size: int = 8,
) -> dict[Any, list[str]]:
    """Generate niche-positioning phrases per item (cached/resumable)."""
    enr = LLMEnricher(
        cache_path=cache_path,
        prompt_template=NICHE_ITEM_PROMPT,
        batch_size=batch_size,
        max_tokens=900,
    )
    return enr.enrich(items, describe_cols=describe_cols, limit=limit)


def gen_user_profiles(
    histories: dict[Any, list[str]],
    cache_path: str | Path,
    max_history_items: int = 25,
    limit: int | None = None,
    batch_size: int = 4,
) -> dict[Any, list[str]]:
    """Generate taste-niche phrases per user from item names.

    `histories` maps user id → chronological item display names (most
    recent last; the most recent `max_history_items` are shown).
    """
    rows = []
    for uid, names in histories.items():
        recent = names[-max_history_items:]
        rows.append({"item_id": uid, "history": "; ".join(str(n) for n in recent)})
    frame = pd.DataFrame(rows)
    if limit is not None:
        frame = frame.head(limit)
    enr = LLMEnricher(
        cache_path=cache_path,
        prompt_template=USER_PROFILE_PROMPT,
        batch_size=batch_size,
        max_tokens=900,
    )
    return enr.enrich(frame, describe_cols=["history"])


def embed_phrase_map(
    phrases: dict[Any, list[str]],
) -> tuple[list[Any], np.ndarray]:
    """Embed a {key: phrases} map. Returns (keys, matrix) with rows
    aligned to keys; keys with empty phrase lists get zero rows."""
    keys = list(phrases.keys())
    texts = [phrases_to_text(phrases[k]) for k in keys]
    return keys, embed_texts(texts)
