"""Stage-1 enrichment probe: is LLM keyword enrichment worth the cost?

Answers, from a small warm-item sample and WITHOUT running the engine:
do the LLM's keywords carry interaction-relevant taste signal?

Method
------
Take ~N warm items (enough interactions that the co-occurrence graph is
trustworthy ground truth) that have keywords. Within the sample:

  1. **Interaction similarity** — Otsuka-Ochiai cosine over user sets:
     ``|U_i ∩ U_j| / sqrt(|U_i|·|U_j|)``.
  2. **Keyword similarity** — cosine over IDF-weighted multi-hot
     keyword vectors (via ItemFeatureExtractor, same featurization the
     content channel uses).

Metrics
-------
  - **separation_d** (headline): Cohen's d between keyword-similarity
    of *interaction-neighbor* pairs vs *random* pairs. Keywords that
    can't tell taste-neighbors from random items can't help any channel.
  - **substitution_p_at_k**: precision@k of keyword-kNN at recovering
    interaction-kNN within the sample — "could keywords stand in for
    missing interaction data on cold items?" Reported against the
    random-chance baseline k/(N-1).
  - **degeneracy checks**: mean random-pair keyword similarity (high =
    the model says the same thing about everything), vocab diversity,
    modal-keyword share.

Verdict gates (defaults; structural, not temporal — these don't suffer
the internal-holdout window-displacement failure that sank per-fit
calibration):

    pass iff separation_d >= 0.5
         and random_pair_mean_sim <= 0.30
         and substitution_p_at_k >= 2 x random_chance

Stage-0 priors (cost, cold population) are the caller's job — this
module measures keyword signal quality only.

VALIDATED 2026-06-11 (ml1m end-to-end): passing gates means keywords
CAN stand in for interaction signal — it does NOT mean they lift warm
protocols. ml1m passed (d=0.60) yet the keyword channel was flat-to-
negative on warm NDCG (0.2931 -> 0.2914): with dense interactions the
keywords are aligned-but-redundant, same as beauty's curated metadata.
Curated-metadata thinness is NOT a valid substitute for the cold-
population requirement. Hence the verdict vocabulary below:

    SIGNAL_OK   gates pass -> enrich IF a cold/sparse population exists
                (cold items, item churn, cold-start serving)
    SKIP        gates fail -> keywords carry no taste signal; don't pay
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kindling.item_features import ItemFeatureExtractor


def load_keywords_jsonl(path: Path) -> dict[object, list[str]]:
    """Read a {item_id, keywords} JSONL cache (as written by llm_enrich)."""
    out: dict[object, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            kws = d.get("keywords")
            if isinstance(kws, list) and kws:
                out[d.get("item_id")] = [str(k).strip().lower() for k in kws if str(k).strip()]
    return out


def probe(
    keywords: dict[object, list[str]],
    train: pd.DataFrame,
    sample_size: int = 200,
    knn_k: int = 10,
    min_interactions: int = 20,
    seed: int = 0,
) -> dict:
    """Run the stage-1 probe. Returns a metrics + verdict dict."""
    rng = np.random.RandomState(seed)

    # ── Warm eligible items: enough interactions AND keyworded.
    counts = train.groupby("item_id").size()
    eligible = [
        i for i, c in counts.items() if c >= min_interactions and i in keywords
    ]
    eligible.sort(key=lambda i: -counts[i])
    if len(eligible) < max(30, knn_k * 3):
        return {"ok": False, "reason": f"only {len(eligible)} warm keyworded items"}
    # Strided sample over the popularity-sorted eligible list: warm but
    # spread across the popularity range, deterministic.
    if len(eligible) > sample_size:
        step = len(eligible) / sample_size
        sample = [eligible[int(j * step)] for j in range(sample_size)]
    else:
        sample = eligible
    n = len(sample)

    # ── Interaction similarity (Otsuka-Ochiai over user sets).
    users_of = {
        i: set(train.loc[train["item_id"] == i, "entity_id"].tolist())
        for i in sample
    }
    inter_sim = np.zeros((n, n))
    for a in range(n):
        ua = users_of[sample[a]]
        for b in range(a + 1, n):
            ub = users_of[sample[b]]
            inter = len(ua & ub)
            if inter:
                s = inter / np.sqrt(len(ua) * len(ub))
                inter_sim[a, b] = inter_sim[b, a] = s

    # ── Keyword similarity (same featurization as the content channel).
    frame = pd.DataFrame(
        {"item_id": sample, "keywords": [keywords[i] for i in sample]}
    )
    idx_map = {i: r for r, i in enumerate(sample)}
    feats = ItemFeatureExtractor(min_df=1).fit_transform(frame, idx_map, n)
    dense = np.zeros((n, feats.n_features), dtype=np.float64)
    for r in range(n):
        s_, e_ = int(feats.indptr[r]), int(feats.indptr[r + 1])
        dense[r, feats.indices[s_:e_]] = feats.data[s_:e_]
    kw_sim = dense @ dense.T
    np.fill_diagonal(kw_sim, 0.0)

    # ── Neighbor vs random pairs.
    neighbor_pairs: set[tuple[int, int]] = set()
    for a in range(n):
        order = np.argsort(-inter_sim[a])
        picked = [b for b in order[:knn_k] if inter_sim[a, b] > 0]
        for b in picked:
            neighbor_pairs.add((min(a, int(b)), max(a, int(b))))
    all_pairs = {(a, b) for a in range(n) for b in range(a + 1, n)}
    random_pool = list(all_pairs - neighbor_pairs)
    rng.shuffle(random_pool)
    random_pairs = random_pool[: max(len(neighbor_pairs) * 4, 1000)]
    if not neighbor_pairs or not random_pairs:
        return {"ok": False, "reason": "no neighbor/random pairs formed"}

    nb = np.array([kw_sim[a, b] for a, b in neighbor_pairs])
    rd = np.array([kw_sim[a, b] for a, b in random_pairs])
    pooled = np.sqrt((nb.var() + rd.var()) / 2)
    separation_d = float((nb.mean() - rd.mean()) / pooled) if pooled > 0 else 0.0

    # ── Substitution: keyword-kNN recovering interaction-kNN.
    precisions = []
    for a in range(n):
        true_nbrs = set(
            int(b) for b in np.argsort(-inter_sim[a])[:knn_k] if inter_sim[a, b] > 0
        )
        if not true_nbrs:
            continue
        kw_nbrs = set(int(b) for b in np.argsort(-kw_sim[a])[:knn_k])
        precisions.append(len(true_nbrs & kw_nbrs) / knn_k)
    substitution = float(np.mean(precisions)) if precisions else 0.0
    chance = knn_k / (n - 1)

    # ── Degeneracy.
    all_kws = [k for i in sample for k in keywords[i]]
    uniq = len(set(all_kws))
    modal_share = max(
        (all_kws.count(k) for k in set(all_kws)), default=0
    ) / max(n, 1)

    gates = {
        "separation": separation_d >= 0.5,
        "non_degenerate": float(rd.mean()) <= 0.30,
        "substitution": substitution >= 2 * chance,
    }
    return {
        "ok": True,
        "n_sample": n,
        "n_keyworded_total": len(keywords),
        "knn_k": knn_k,
        "separation_d": separation_d,
        "neighbor_pair_mean_sim": float(nb.mean()),
        "random_pair_mean_sim": float(rd.mean()),
        "substitution_p_at_k": substitution,
        "substitution_chance": chance,
        "vocab_unique": uniq,
        "keywords_per_item_mean": len(all_kws) / max(n, 1),
        "modal_keyword_share": float(modal_share),
        "gates": gates,
        "verdict": "SIGNAL_OK" if all(gates.values()) else "SKIP",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", default="movielens-1m")
    parser.add_argument("--keywords", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset(args.loader, test_fraction=0.1)
    kws = load_keywords_jsonl(args.keywords)
    report = probe(
        kws, split.train, sample_size=args.sample_size,
        knn_k=args.knn_k, seed=args.seed,
    )
    report["loader"] = args.loader
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        print(f"wrote {args.output}")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
