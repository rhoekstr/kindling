"""Prompt-search for cooc-predictive metadata enrichment.

Description-framed enrichment LOWERED ml1m mapping-R^2 (genres 0.084 -> keywords
0.037): richer description moves AWAY from the co-consumption driver. Hypothesis:
prompts framed for AUDIENCE / co-consumption raise it. Loss = transfer mapping-R^2
(the validated metadata-value metric). Each framing emits keyword tags (identical
downstream pipeline), so only what the prompt ELICITS varies.

Transfer guardrail: enrichment must be item-derivable (content + world knowledge)
so it generalizes to cold items; "people in THIS dataset who watched X..." is
cooc leakage and is banned by framing.

Run one framing (resumable Phi-4 generation), then it reports all available reps:
  FRAMING=audience SAMPLE_N=1200 .venv/bin/python bench/run_enrich_search.py
Framings: description / audience / co_audience / use_context.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kindling.item_features import ItemFeatureExtractor
from kindling.llm_enrich import LLMEnricher
from kindling.loaders import movielens
from run_meta_cooc_map import DIM, ppmi, ridge_transfer, svd_emb

CACHE = Path(__file__).parent / "cache"
WARM_MIN = 5

FRAMINGS = {
    "description": (
        "You are labeling catalog items for a recommender system.\n"
        "For EACH item below, give 8-12 short lowercase keywords capturing its genre,\n"
        "style, themes, era, tone, and unique features. Be specific. Output ONLY a JSON\n"
        "object mapping each item's ID to its keyword array.\n\nItems:\n{items_block}\n\nJSON:"
    ),
    "audience": (
        "You are profiling the AUDIENCE of catalog items for a recommender system.\n"
        "For EACH item, give 8-12 short lowercase tags describing the TYPE OF VIEWER who\n"
        "seeks it out and the taste-cluster it belongs to -- the audience and their\n"
        "sensibility, NOT the plot. e.g. 'film-school-crowd', 'date-night-mainstream',\n"
        "'cult-horror-fans', 'prestige-tv-viewers'. Output ONLY a JSON object mapping each\n"
        "item's ID to its tag array.\n\nItems:\n{items_block}\n\nJSON:"
    ),
    "co_audience": (
        "You are mapping SHARED-AUDIENCE affinities for a recommender system.\n"
        "For EACH item, give 8-12 short lowercase tags naming the kinds of OTHER films its\n"
        "core audience also gravitates to -- shared-audience affinities as attributes, NOT\n"
        "this film's own plot. Use your general knowledge of who watches what; do NOT\n"
        "reference any specific user data. e.g. 'overlaps-with-noir-fans',\n"
        "'wes-anderson-crowd', 'arthouse-scifi-adjacent', 'blockbuster-franchise-loyalists'.\n"
        "Output ONLY a JSON object mapping each item's ID to its tag array.\n\nItems:\n{items_block}\n\nJSON:"
    ),
    "series": (
        "You are identifying the FRANCHISE/SERIES of catalog items for a recommender\n"
        "system. For EACH item, output short lowercase tags for the franchise, series, or\n"
        "shared universe it belongs to AND its closely related installments (sequels,\n"
        "prequels, spin-offs, same-creator signature series). If standalone, give its 2-3\n"
        "closest franchise-adjacent reference points. e.g. 'star-wars-saga', 'mcu',\n"
        "'before-trilogy', 'tarantino-canon'. Output ONLY a JSON object mapping each item's\n"
        "ID to its tag array.\n\nItems:\n{items_block}\n\nJSON:"
    ),
    "tone": (
        "You are labeling the TONE/MOOD of catalog items for a recommender system.\n"
        "For EACH item, give 6-10 short lowercase tone and mood tags -- the emotional\n"
        "texture, NOT the plot. e.g. 'bleak', 'whimsical', 'tense', 'uplifting',\n"
        "'melancholic', 'campy', 'cozy'. Output ONLY a JSON object mapping each item's ID\n"
        "to its tag array.\n\nItems:\n{items_block}\n\nJSON:"
    ),
    "use_context": (
        "You are labeling the VIEWING CONTEXT of catalog items for a recommender system.\n"
        "For EACH item, give 8-12 short lowercase tags for the mood, occasion, and context\n"
        "in which it is chosen -- when, with whom, and why someone picks it, NOT the plot.\n"
        "e.g. 'lazy-sunday', 'group-comedy-night', 'comfort-rewatch', 'background-noise',\n"
        "'tearjerker-mood'. Output ONLY a JSON object mapping each item's ID to its tag array.\n\nItems:\n{items_block}\n\nJSON:"
    ),
}


def _F_from_keywords(kw_by_id, item_to_idx, n_items):
    df = pd.DataFrame({"item_id": list(kw_by_id.keys()),
                       "content": [" ".join(v) for v in kw_by_id.values()]})
    feat = ItemFeatureExtractor().fit_transform(df, item_to_idx, n_items)
    return sp.csr_matrix((feat.data, feat.indices, feat.indptr), shape=(n_items, feat.n_features))


def _F_from_cols(items, cols, item_to_idx, n_items):
    df = items[["item_id", *cols]].copy()
    feat = ItemFeatureExtractor().fit_transform(df, item_to_idx, n_items)
    return sp.csr_matrix((feat.data, feat.indices, feat.indptr), shape=(n_items, feat.n_features))


def main():
    framing = os.environ.get("FRAMING", "audience")
    sample_n = int(os.environ.get("SAMPLE_N", "1200"))
    batch = int(os.environ.get("BATCH", "12"))

    split = movielens.load_1m(test_fraction=0.1)
    items = split.items
    item_ids = pd.Index(split.train["item_id"].unique())
    item_to_idx = {int(it): i for i, it in enumerate(item_ids)}
    n_items = len(item_ids)
    iidx = split.train["item_id"].map(item_to_idx).to_numpy()
    uidx = pd.factorize(split.train["entity_id"])[0]
    n_users = int(uidx.max()) + 1
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    warm = np.where(d >= WARM_MIN)[0]
    warm = warm[np.argsort(-d[warm])]
    sample = warm[:sample_n]  # top-N warm items get enriched

    S = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)), shape=(n_users, n_items))
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    Sw = S[:, warm]
    C = (Sw.T @ Sw).tocoo()
    keep = C.row != C.col
    C = sp.coo_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=(len(warm), len(warm)))
    cooc_emb = svd_emb(ppmi(C, d[warm], n_users), DIM)[:sample_n]  # sample = warm[:N]

    # generate the requested framing (resumable)
    sample_movie_ids = set(int(item_ids[g]) for g in sample)
    sdf = items[items["item_id"].isin(sample_movie_ids)][["item_id", "title", "genres"]].copy()
    print(f"[gen] framing={framing} sample={len(sdf)} batch={batch}", flush=True)
    enr = LLMEnricher(CACHE / f"ml1m_enrich_{framing}.jsonl",
                      prompt_template=FRAMINGS[framing], batch_size=batch, max_tokens=700)
    enr.enrich(sdf, describe_cols=["title", "genres"])

    # evaluate ALL available reps on the same sample
    rng = np.random.default_rng(0)
    perm = rng.permutation(sample_n)
    tr, te = perm[: int(0.7 * sample_n)], perm[int(0.7 * sample_n):]

    # Each metadata FACET as a separate embedding (genre = native genres; the rest
    # = LLM facet-prompts). Then fuse by marginal contribution (forward selection),
    # = "use each as a layer based on how much it improves" measured on held-out R^2.
    facets = {"genre": svd_emb(_F_from_cols(items, ["genres"], item_to_idx, n_items)[sample], DIM)}
    for fr in FRAMINGS:
        p = CACHE / (f"ml1m_enrich_{fr}.jsonl" if fr != "description" else "ml1m_keywords.jsonl")
        if not p.exists():
            continue
        kw = {int(json.loads(l)["item_id"]): json.loads(l)["keywords"]
              for l in p.read_text().splitlines() if l.strip()}
        facets[fr] = svd_emb(_F_from_keywords(kw, item_to_idx, n_items)[sample], DIM)

    print(f"\nfacet fusion (ml1m, sample={sample_n}) — per-facet mapping-R^2:")
    for name, E in facets.items():
        r2, rec = ridge_transfer(E, cooc_emb, tr, te)
        print(f"  {name:14s} R2={r2:+.4f}  nbr={rec:.3f}")

    print("forward-selection fusion (greedy add by marginal held-out R^2):")
    selected, cur, remaining = [], -1e9, set(facets)
    while remaining:
        best, best_r2 = None, cur
        for f in remaining:
            rr = ridge_transfer(np.hstack([facets[s] for s in [*selected, f]]), cooc_emb, tr, te)[0]
            if rr > best_r2 + 1e-4:
                best, best_r2 = f, rr
        if best is None:
            break
        selected.append(best)
        remaining.discard(best)
        cur = best_r2
        print(f"  + {best:14s} -> fused R2 {cur:+.4f}")
    print(f"FUSED [{' + '.join(selected)}] = {cur:.4f}  (genre-only = "
          f"{ridge_transfer(facets['genre'], cooc_emb, tr, te)[0]:.4f})")


if __name__ == "__main__":
    main()
