"""Screen MovieLens-25M + tag-genome with the validated metadata->cooc mapping
metric. Decisive question: does the RICHEST content-coherent metadata (1128
relevance-scored tags/movie) break past the ~0.08 mapping-R^2 ceiling we hit on
ml1m/steam/book, or cap there?

  break ~0.15+  => the ceiling is metadata-quality-limited (enrichment has headroom)
  cap ~0.08     => the ceiling is fundamental (consumption has irreducible latent
                   structure content can't reach)

Reuses ppmi / svd_emb / ridge_transfer from run_meta_cooc_map (same protocol:
cooc_emb = PPMI-SVD64 of warm cooc; map = ridge meta->cooc; transfer R^2 +
neighbor-recovery on held-out warm items). Genome metadata as SVD-64 and raw-1128.

Run: SUB_USERS=30000 .venv/bin/python bench/run_ml25m_screen.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from run_meta_cooc_map import DIM, N_WARM, WARM_MIN, ppmi, ridge_transfer, svd_emb

D = Path("~/.cache/kindling/ml-25m").expanduser()


def main():
    sub = int(os.environ.get("SUB_USERS", "30000"))
    rng = np.random.default_rng(0)

    print("[load] ratings.csv ...", flush=True)
    r = pd.read_csv(D / "ratings.csv", usecols=["userId", "movieId"], dtype=np.int32)
    users = r["userId"].unique()
    if len(users) > sub:
        keep = set(rng.choice(users, sub, replace=False).tolist())
        r = r[r["userId"].isin(keep)]
    print(f"[load] genome-scores.csv ...", flush=True)
    g = pd.read_csv(D / "genome-scores.csv")  # movieId, tagId, relevance
    genome_movies = set(g["movieId"].unique().tolist())
    n_tags = int(g["tagId"].max())

    # restrict to movies that HAVE genome metadata
    r = r[r["movieId"].isin(genome_movies)]
    movies = pd.Index(r["movieId"].unique())
    m2i = {int(m): i for i, m in enumerate(movies)}
    n_items = len(movies)
    iidx = r["movieId"].map(m2i).to_numpy()
    uidx = pd.factorize(r["userId"])[0]
    n_users = int(uidx.max()) + 1
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    warm = np.where(d >= WARM_MIN)[0]
    warm = warm[np.argsort(-d[warm])][:N_WARM]
    warm_movies = movies.to_numpy()[warm]
    print(f"[graph] users={n_users} genome_movies={n_items} warm={len(warm)} tags={n_tags}", flush=True)

    S = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)), shape=(n_users, n_items))
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    Sw = S[:, warm]
    C = (Sw.T @ Sw).tocoo()
    keep = C.row != C.col
    C = sp.coo_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=(len(warm), len(warm)))
    cooc_emb = svd_emb(ppmi(C, d[warm], n_users), DIM)

    # genome metadata for warm movies: warm x tags relevance matrix
    gw = g[g["movieId"].isin(set(int(m) for m in warm_movies))]
    wm2pos = {int(m): i for i, m in enumerate(warm_movies)}
    rows = gw["movieId"].map(wm2pos).to_numpy()
    cols = gw["tagId"].to_numpy() - 1
    G = sp.csr_matrix((gw["relevance"].to_numpy(), (rows, cols)), shape=(len(warm), n_tags))

    perm = rng.permutation(len(warm))
    cut = int(0.7 * len(warm))
    tr, te = perm[:cut], perm[cut:]

    print("\nmovielens-25m tag-genome mapping quality (vs ~0.08 ceiling):")
    for label, M in [("genome_svd64", svd_emb(G, DIM)), ("genome_raw1128", G.toarray())]:
        r2, rec = ridge_transfer(M, cooc_emb, tr, te)
        print(f"  {label:16s} transfer_R2={r2:+.4f}  cooc-nbr-recovery@10={rec:.3f}  (dim={M.shape[1]})")


if __name__ == "__main__":
    main()
