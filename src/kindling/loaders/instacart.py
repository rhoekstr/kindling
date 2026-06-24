"""Instacart grocery orders loader (plan Phase 7).

Instacart is session-heavy (each order is a session/basket) which makes
it the right dataset to exercise the basket and path signals that
ML-1M under-exercises.

Source: Kaggle competition ``instacart-market-basket-analysis``. The
expected file layout after unpacking:

    ${cache}/instacart/
      orders.csv
      order_products__prior.csv
      order_products__train.csv
      products.csv
      aisles.csv
      departments.csv

Because the data sits behind Kaggle authentication and our loader must
run on read-only environments, we do NOT auto-download. Users point the
loader at a local directory where they unpacked the competition zip.
CLI:

    python -m kindling.benchmarks.harness --dataset instacart \\
        --data-dir /path/to/instacart
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit


class InstacartDataNotAvailableError(RuntimeError):
    """Raised when the Instacart competition files aren't on disk."""


REQUIRED_FILES = (
    "orders.csv",
    "order_products__prior.csv",
    "products.csv",
)


def load(data_dir: str | Path, test_fraction: float = 0.1) -> DatasetSplit:
    """Build a DatasetSplit from a local Instacart unpack.

    Uses the ``prior`` history as train and holds out the last
    ``test_fraction`` of each user's orders (chronological-by-
    order_number) for test. Item metadata includes the aisle as the
    primary category for calibration.
    """
    base = Path(data_dir)
    missing = [f for f in REQUIRED_FILES if not (base / f).exists()]
    if missing:
        raise InstacartDataNotAvailableError(
            f"Missing Instacart files under {base}: {missing}. "
            "Download from Kaggle (instacart-market-basket-analysis) and "
            "unpack into the data_dir."
        )

    orders = pd.read_csv(
        base / "orders.csv",
        usecols=[
            "order_id",
            "user_id",
            "order_number",
            "days_since_prior_order",
            "eval_set",
        ],
    )
    order_products = pd.read_csv(
        base / "order_products__prior.csv",
        usecols=["order_id", "product_id", "add_to_cart_order"],
    )
    products = pd.read_csv(
        base / "products.csv", usecols=["product_id", "product_name", "aisle_id"]
    )

    # Join order -> (user, order_number) onto the product events.
    prior_orders = orders[orders["eval_set"] == "prior"]
    events = order_products.merge(
        prior_orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="inner",
    )
    if events.empty:
        raise InstacartDataNotAvailableError("No 'prior' orders found in the dataset")

    # Synthesize a timestamp from the order_number sequence per user so
    # sessions infer naturally. Instacart doesn't ship absolute
    # timestamps.
    base_time = pd.Timestamp("2024-01-01")
    events = events.sort_values(["user_id", "order_number", "add_to_cart_order"])
    events["timestamp"] = base_time + pd.to_timedelta(
        events["order_number"].astype(int) * 3, unit="D"
    )

    canonical = pd.DataFrame(
        {
            "entity_id": events["user_id"].astype("int64").to_numpy(),
            "item_id": events["product_id"].astype("int64").to_numpy(),
            "timestamp": events["timestamp"].to_numpy(),
            "session_id": events["order_id"].astype("int64").to_numpy(),
            "action_type": "add",
        }
    )

    # Chronological split per user.
    cutoff_order = (
        prior_orders.groupby("user_id")["order_number"]
        .apply(lambda s: s.quantile(1.0 - test_fraction))
        .to_dict()
    )
    train_mask = events.apply(
        lambda r: r["order_number"] <= cutoff_order.get(r["user_id"], 0), axis=1
    )
    train = canonical[train_mask.to_numpy()].reset_index(drop=True)
    test = canonical[(~train_mask).to_numpy()].reset_index(drop=True)

    items = pd.DataFrame(
        {
            "item_id": products["product_id"].astype("int64"),
            "title": products["product_name"],
            "category": products["aisle_id"].astype(str),  # aisle as calibration category
        }
    )

    return DatasetSplit(
        name="instacart",
        train=train,
        test=test,
        items=items,
        description=(
            "Instacart 3.4M grocery orders; session-structured baskets; "
            "tests path_basket + basket_index heavily. Aisle id used as the "
            "calibration category."
        ),
    )
