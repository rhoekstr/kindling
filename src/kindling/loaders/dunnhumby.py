"""Dunnhumby Complete Journey loader (plan Phase 7).

The "Complete Journey" dataset is 2 years of household-level grocery
transactions from a US retailer. It's the most session-realistic of
the public grocery datasets because BASKET_ID is a real point-of-sale
basket id (not a synthesized session).

Source: Kaggle ("dunnhumby's The Complete Journey"). Expected file:
``transaction_data.csv`` with columns:

    household_key, BASKET_ID, DAY, QUANTITY, PRODUCT_ID,
    SALES_VALUE, STORE_ID, RETAIL_DISC, TRANS_TIME, WEEK_NO, COUPON_DISC, COUPON_MATCH_DISC

Optional companion file: ``product.csv`` with COMMODITY_DESC for
calibration categories.

We synthesize a timestamp from DAY + TRANS_TIME so paths get real
ordering. Session is BASKET_ID directly.

CLI:
    python -m kindling.benchmarks.harness --dataset dunnhumby \\
        --data-dir ~/.cache/kindling/dunnhumby
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit


class DunnhumbyDataNotAvailableError(RuntimeError):
    pass


TRANSACTIONS_FILE = "transaction_data.csv"
PRODUCT_FILE = "product.csv"


def load(data_dir: str | Path, test_fraction: float = 0.1) -> DatasetSplit:
    base = Path(data_dir)
    tx_path = base / TRANSACTIONS_FILE
    if not tx_path.exists():
        raise DunnhumbyDataNotAvailableError(
            f"Dunnhumby transaction file not found at {tx_path}. "
            "Source: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey"
        )

    df = pd.read_csv(tx_path)
    df.columns = [c.strip().upper() for c in df.columns]
    required = {"HOUSEHOLD_KEY", "BASKET_ID", "DAY", "PRODUCT_ID"}
    if not required.issubset(df.columns):
        raise DunnhumbyDataNotAvailableError(
            f"Dunnhumby CSV at {tx_path} missing columns {required - set(df.columns)}. "
            f"Found: {sorted(df.columns)}"
        )

    # Build a synthetic timestamp: base date + DAY (1..711) + TRANS_TIME (HHMM int).
    base_date = pd.Timestamp("2017-01-01")
    days = pd.to_numeric(df["DAY"], errors="coerce").fillna(0).astype(int)
    if "TRANS_TIME" in df.columns:
        t = pd.to_numeric(df["TRANS_TIME"], errors="coerce").fillna(0).astype(int)
        minutes = (t // 100) * 60 + (t % 100)
    else:
        minutes = pd.Series([0] * len(df))
    timestamps = base_date + pd.to_timedelta(days, unit="D") + pd.to_timedelta(minutes, unit="min")

    canonical = pd.DataFrame(
        {
            "entity_id": df["HOUSEHOLD_KEY"].astype("int64"),
            "item_id": df["PRODUCT_ID"].astype("int64"),
            "timestamp": timestamps.to_numpy(),
            "session_id": df["BASKET_ID"].astype("int64").to_numpy(),
            "action_type": "purchase",
        }
    ).sort_values(["entity_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    cutoff = (
        canonical.groupby("entity_id")["timestamp"]
        .quantile(1.0 - test_fraction)
        .to_dict()
    )
    train_mask = canonical["timestamp"] <= canonical["entity_id"].map(cutoff)
    train = canonical[train_mask].reset_index(drop=True)
    test = canonical[~train_mask].reset_index(drop=True)

    items = None
    product_path = base / PRODUCT_FILE
    if product_path.exists():
        prod = pd.read_csv(product_path)
        prod.columns = [c.strip().upper() for c in prod.columns]
        if {"PRODUCT_ID", "COMMODITY_DESC"}.issubset(prod.columns):
            items = (
                prod[["PRODUCT_ID", "COMMODITY_DESC"]]
                .drop_duplicates(subset=["PRODUCT_ID"])
                .rename(columns={"PRODUCT_ID": "item_id", "COMMODITY_DESC": "category"})
                .reset_index(drop=True)
            )
            items["item_id"] = items["item_id"].astype("int64")
            items["category"] = items["category"].astype(str)

    return DatasetSplit(
        name="dunnhumby",
        train=train,
        test=test,
        items=items,
        description=(
            "Dunnhumby Complete Journey 2-year household-level grocery "
            "transactions; real BASKET_ID drives the basket index; "
            "COMMODITY_DESC available as calibration category."
        ),
    )
