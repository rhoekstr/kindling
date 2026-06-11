"""Generic item feature extraction.

Turns an arbitrary item-metadata DataFrame into a sparse item × feature
matrix usable as a content-similarity channel. No dataset-specific
schema: column roles are inferred —

  - **numeric** dtype            → quantile-binned one-hot
  - **list / delimited string**  → multi-hot over the parts
    (delimiters tried: ``| ; > /`` — chosen when most non-null values
    split into >1 part)
  - **low-cardinality string**   → one-hot over distinct values
  - **high-cardinality string**  → bag-of-tokens text features
    (lowercased alphanumeric tokens, document-frequency capped)
  - constant / unique-per-item / mostly-null columns → ignored

Features are IDF-weighted (rare attributes are more informative than
ubiquitous ones) and rows are L2-normalized, so a dot product between
two item rows is their cosine content similarity.

Output is a plain CSR triple over *engine catalog indices*: rows are
``0..n_items-1`` in the engine's item index space; catalog items with
no metadata row get an empty feature row (content channel scores them
0 — the interaction channels still cover them).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DELIMITERS = ("|", ";", ">", "/")


@dataclass
class ColumnSpec:
    """Inferred role of one metadata column (diagnostics + repr)."""

    column: str
    kind: str  # "numeric" | "multi_categorical" | "categorical" | "text" | "ignored"
    detail: str = ""
    n_features: int = 0


@dataclass
class ItemFeatures:
    """Sparse item × feature matrix in CSR form (engine item index space)."""

    data: np.ndarray          # f32
    indices: np.ndarray       # i32
    indptr: np.ndarray        # i32, length n_items + 1
    n_features: int
    feature_names: list[str]
    specs: list[ColumnSpec] = field(default_factory=list)
    coverage: float = 0.0     # fraction of catalog items with ≥1 feature

    @property
    def n_items(self) -> int:
        return len(self.indptr) - 1


class ItemFeatureExtractor:
    """Schema-inferring extractor. One instance per fit.

    Parameters
    ----------
    max_text_features:
        Vocabulary cap per text column (kept by document frequency).
    numeric_bins:
        Quantile bins per numeric column.
    min_df:
        Drop features appearing in fewer than this many items.
    max_df_fraction:
        Drop features appearing in more than this fraction of items
        (a genre shared by 95% of the catalog carries no signal).
    categorical_max_cardinality_fraction:
        A string column is one-hot ("categorical") when its distinct-
        value count is below this fraction of the item count; above it,
        the column is treated as free text.
    """

    def __init__(
        self,
        max_text_features: int = 4096,
        numeric_bins: int = 8,
        min_df: int = 2,
        max_df_fraction: float = 0.5,
        categorical_max_cardinality_fraction: float = 0.5,
    ):
        self.max_text_features = max_text_features
        self.numeric_bins = numeric_bins
        self.min_df = min_df
        self.max_df_fraction = max_df_fraction
        self.categorical_max_cardinality_fraction = (
            categorical_max_cardinality_fraction
        )

    # ── column-role inference ──────────────────────────────────────

    def _infer_kind(self, col: pd.Series, n_rows: int) -> tuple[str, str]:
        """Return (kind, detail) for a metadata column."""
        non_null = col.dropna()
        if len(non_null) < max(2, n_rows // 100):
            return "ignored", "mostly null"
        if pd.api.types.is_numeric_dtype(non_null):
            if non_null.nunique() <= 1:
                return "ignored", "constant"
            return "numeric", ""
        # List-valued cells (real Python lists / tuples / ndarrays).
        sample = non_null.iloc[: min(len(non_null), 200)]
        if sample.map(lambda v: isinstance(v, (list, tuple, np.ndarray))).mean() > 0.5:
            return "multi_categorical", "list"
        as_str = non_null.astype(str)
        nunique = as_str.nunique()
        if nunique <= 1:
            return "ignored", "constant"
        # Delimited multi-value? Checked BEFORE the unique-per-item rule:
        # a column of unique delimited combinations (genre combos,
        # category paths) is still multi-categorical — the parts repeat
        # even when the combinations don't.
        str_sample = as_str.iloc[: min(len(as_str), 500)]
        best_delim, best_rate = None, 0.0
        for d in _DELIMITERS:
            rate = (str_sample.str.count(re.escape(d)) > 0).mean()
            if rate > best_rate:
                best_delim, best_rate = d, rate
        if best_delim is not None and best_rate >= 0.3:
            return "multi_categorical", best_delim
        if nunique == len(as_str):
            # Unique per item — an id or free text. Treat as text (ids
            # tokenize into junk but get df-filtered away).
            return "text", "unique-per-item"
        if nunique / len(as_str) <= self.categorical_max_cardinality_fraction:
            return "categorical", ""
        return "text", ""

    # ── extraction ─────────────────────────────────────────────────

    def fit_transform(
        self,
        items: pd.DataFrame,
        item_to_idx: dict[Any, int],
        n_items: int,
        item_id_column: str = "item_id",
    ) -> ItemFeatures:
        """Extract features for catalog items present in `items`.

        `item_to_idx` maps raw item ids → engine catalog index. Metadata
        rows for items outside the catalog are dropped; catalog items
        without metadata get empty feature rows.
        """
        specs: list[ColumnSpec] = []
        feature_names: list[str] = []
        # Per-item token lists: row → list of feature ids (then weighted).
        row_features: list[list[int]] = [[] for _ in range(n_items)]

        if item_id_column not in items.columns:
            raise ValueError(
                f"metadata frame has no {item_id_column!r} column; "
                f"got {list(items.columns)}"
            )
        # Keep only catalog rows; first metadata row wins on duplicates.
        meta = items.drop_duplicates(subset=item_id_column, keep="first")
        row_idx = meta[item_id_column].map(item_to_idx)
        keep = row_idx.notna()
        meta = meta.loc[keep]
        row_idx = row_idx.loc[keep].astype(np.int64).to_numpy()
        n_meta = len(meta)

        def _add_feature(name: str) -> int:
            feature_names.append(name)
            return len(feature_names) - 1

        for col_name in meta.columns:
            if col_name == item_id_column:
                continue
            col = meta[col_name]
            kind, detail = self._infer_kind(col, n_meta)
            spec = ColumnSpec(column=col_name, kind=kind, detail=detail)
            n_before = len(feature_names)

            if kind == "numeric":
                vals = pd.to_numeric(col, errors="coerce")
                ok = vals.notna().to_numpy()
                if ok.sum() >= 2:
                    arr = vals.to_numpy(dtype=np.float64)
                    qs = np.nanquantile(
                        arr, np.linspace(0, 1, self.numeric_bins + 1)[1:-1]
                    )
                    bins = np.searchsorted(np.unique(qs), arr[ok])
                    bin_fids: dict[int, int] = {}
                    for r, b in zip(row_idx[ok], bins):
                        fid = bin_fids.get(int(b))
                        if fid is None:
                            fid = _add_feature(f"{col_name}=bin{int(b)}")
                            bin_fids[int(b)] = fid
                        row_features[r].append(fid)

            elif kind in ("categorical", "multi_categorical"):
                if kind == "categorical":
                    parts_per_row = [
                        [str(v).strip().lower()] if pd.notna(v) else []
                        for v in col
                    ]
                elif detail == "list":
                    parts_per_row = [
                        [str(p).strip().lower() for p in v]
                        if isinstance(v, (list, tuple, np.ndarray))
                        else []
                        for v in col
                    ]
                else:
                    parts_per_row = [
                        [p.strip().lower() for p in str(v).split(detail) if p.strip()]
                        if pd.notna(v)
                        else []
                        for v in col
                    ]
                vocab: dict[str, int] = {}
                for r, parts in zip(row_idx, parts_per_row):
                    for p in set(parts):
                        fid = vocab.get(p)
                        if fid is None:
                            fid = _add_feature(f"{col_name}={p}")
                            vocab[p] = fid
                        row_features[r].append(fid)

            elif kind == "text":
                token_rows = [
                    set(_TOKEN_RE.findall(str(v).lower())) if pd.notna(v) else set()
                    for v in col
                ]
                df_counts: dict[str, int] = {}
                for toks in token_rows:
                    for t in toks:
                        df_counts[t] = df_counts.get(t, 0) + 1
                max_df = int(self.max_df_fraction * n_meta)
                eligible = {
                    t: c
                    for t, c in df_counts.items()
                    if self.min_df <= c <= max_df
                }
                kept = sorted(
                    eligible, key=lambda t: (-eligible[t], t)
                )[: self.max_text_features]
                vocab = {t: _add_feature(f"{col_name}:{t}") for t in kept}
                for r, toks in zip(row_idx, token_rows):
                    for t in toks:
                        fid = vocab.get(t)
                        if fid is not None:
                            row_features[r].append(fid)

            spec.n_features = len(feature_names) - n_before
            specs.append(spec)

        n_features = len(feature_names)
        if n_features == 0:
            return ItemFeatures(
                data=np.array([], dtype=np.float32),
                indices=np.array([], dtype=np.int32),
                indptr=np.zeros(n_items + 1, dtype=np.int32),
                n_features=0,
                feature_names=[],
                specs=specs,
                coverage=0.0,
            )

        # ── global df-filter + IDF weighting + L2 rows → CSR.
        df = np.zeros(n_features, dtype=np.int64)
        for feats in row_features:
            for f in set(feats):
                df[f] += 1
        max_df_global = max(int(self.max_df_fraction * max(n_meta, 1)), 1)
        alive = (df >= self.min_df) & (df <= max_df_global)
        idf = np.zeros(n_features, dtype=np.float64)
        idf[alive] = np.log((1.0 + n_meta) / (1.0 + df[alive])) + 1.0

        data_out: list[np.ndarray] = []
        indices_out: list[np.ndarray] = []
        indptr = np.zeros(n_items + 1, dtype=np.int32)
        covered = 0
        for r in range(n_items):
            feats = sorted(set(f for f in row_features[r] if alive[f]))
            if feats:
                w = idf[feats]
                norm = np.sqrt((w * w).sum())
                if norm > 0:
                    w = w / norm
                data_out.append(w.astype(np.float32))
                indices_out.append(np.asarray(feats, dtype=np.int32))
                covered += 1
            indptr[r + 1] = indptr[r] + len(feats)
        return ItemFeatures(
            data=(
                np.concatenate(data_out)
                if data_out
                else np.array([], dtype=np.float32)
            ),
            indices=(
                np.concatenate(indices_out)
                if indices_out
                else np.array([], dtype=np.int32)
            ),
            indptr=indptr,
            n_features=n_features,
            feature_names=feature_names,
            specs=specs,
            coverage=covered / max(n_items, 1),
        )


def content_scores(
    feats: ItemFeatures,
    owned: np.ndarray,
    owned_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Full-catalog content-similarity scores for one user.

    profile = Σ w_i · F[i, :]  over owned items, then score = F · profile.
    Both steps are vectorized; cost is O(nnz of owned rows + nnz of F).
    Rows are L2-normalized so scores are cosine-weighted sums.
    """
    n_items = feats.n_items
    if feats.n_features == 0 or feats.data.size == 0 or owned.size == 0:
        return np.zeros(n_items, dtype=np.float64)
    profile = np.zeros(feats.n_features, dtype=np.float64)
    for k, item in enumerate(owned.tolist()):
        s_, e_ = int(feats.indptr[item]), int(feats.indptr[item + 1])
        if e_ > s_:
            w = 1.0 if owned_weights is None else float(owned_weights[k])
            np.add.at(profile, feats.indices[s_:e_], w * feats.data[s_:e_])
    if not profile.any():
        return np.zeros(n_items, dtype=np.float64)
    # score[j] = Σ_k data[k]·profile[indices[k]] segment-summed per row.
    contrib = feats.data * profile[feats.indices]
    starts = feats.indptr[:-1].astype(np.int64)
    empty = starts == feats.indptr[1:].astype(np.int64)
    # reduceat quirks: empty rows copy the next element, and a start
    # equal to nnz (trailing empty rows) is out of bounds — clip, then
    # zero all empty rows.
    scores = np.add.reduceat(
        contrib, np.minimum(starts, contrib.size - 1), dtype=np.float64
    )
    scores[empty] = 0.0
    return scores
