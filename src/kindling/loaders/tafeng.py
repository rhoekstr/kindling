"""Ta Feng grocery loader (plan Phase 7).

Ta Feng is a Taiwanese supermarket transaction log spanning Nov 2000 -
Feb 2001. It's popular for basket-level recommender benchmarks because
each row is a (customer, transaction date, product) triple and items
naturally cluster into baskets per customer-day.

Source: Kaggle (search "Ta Feng Grocery Dataset"). The expected file
is ``ta_feng_all_months_merged.csv``. Columns:

    TRANSACTION_DT, CUSTOMER_ID, AGE_GROUP, PIN_CODE,
    PRODUCT_SUBCLASS, PRODUCT_ID, AMOUNT, ASSET, SALES_PRICE

We map ``CUSTOMER_ID -> entity_id``, ``PRODUCT_ID -> item_id``, the
date as timestamp, and synthesize ``session_id`` as
``(customer, transaction_date)`` so the basket index sees real
co-occurrence baskets.

CLI:
    python -m kindling.benchmarks.harness --dataset tafeng \\
        --data-dir ~/.cache/kindling/tafeng
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit


class TafengDataNotAvailableError(RuntimeError):
    pass


CANDIDATE_FILES = (
    "ta_feng_all_months_merged.csv",
    "tafeng.csv",
)


def _find_csv(base: Path) -> Path | None:
    for name in CANDIDATE_FILES:
        path = base / name
        if path.exists():
            return path
    return None


def load(data_dir: str | Path, test_fraction: float = 0.1) -> DatasetSplit:
    base = Path(data_dir)
    path = _find_csv(base)
    if path is None:
        raise TafengDataNotAvailableError(
            f"Ta Feng data not found under {base}. Provide one of {CANDIDATE_FILES}. "
            "Source: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset"
        )

    df = pd.read_csv(path)
    # Normalize column names — Kaggle file ships with mixed cases.
    df.columns = [c.strip().upper() for c in df.columns]
    required = {"TRANSACTION_DT", "CUSTOMER_ID", "PRODUCT_ID"}
    if not required.issubset(df.columns):
        raise TafengDataNotAvailableError(
            f"Ta Feng CSV at {path} missing columns {required - set(df.columns)}. "
            f"Found: {sorted(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["TRANSACTION_DT"], errors="coerce")
    df = df.dropna(subset=["timestamp", "CUSTOMER_ID", "PRODUCT_ID"]).reset_index(drop=True)

    # session = (customer, transaction_day) — the natural basket grouping.
    df["session_id"] = (
        df["CUSTOMER_ID"].astype(str)
        + "_"
        + df["timestamp"].dt.strftime("%Y%m%d")
    )

    canonical = pd.DataFrame(
        {
            "entity_id": df["CUSTOMER_ID"].astype("int64"),
            "item_id": df["PRODUCT_ID"].astype("int64"),
            "timestamp": df["timestamp"].to_numpy(),
            "session_id": df["session_id"].to_numpy(),
            "action_type": "purchase",
        }
    ).sort_values(["entity_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Chronological per-user split.
    cutoff = (
        canonical.groupby("entity_id")["timestamp"]
        .quantile(1.0 - test_fraction)
        .to_dict()
    )
    train_mask = canonical["timestamp"] <= canonical["entity_id"].map(cutoff)
    train = canonical[train_mask].reset_index(drop=True)
    test = canonical[~train_mask].reset_index(drop=True)

    items = None
    if "PRODUCT_SUBCLASS" in df.columns:
        items = (
            df[["PRODUCT_ID", "PRODUCT_SUBCLASS"]]
            .drop_duplicates(subset=["PRODUCT_ID"])
            .rename(columns={"PRODUCT_ID": "item_id", "PRODUCT_SUBCLASS": "category"})
            .reset_index(drop=True)
        )
        items["item_id"] = items["item_id"].astype("int64")
        items["category"] = items["category"].astype(str)

    return DatasetSplit(
        name="tafeng",
        train=train,
        test=test,
        items=items,
        description=(
            "Ta Feng 4-month Taiwanese supermarket transactions; "
            "real per-day baskets feed the basket index; product "
            "subclass available as calibration category."
        ),
    )
