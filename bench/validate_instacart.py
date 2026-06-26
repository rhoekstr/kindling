"""Instacart Market Basket loader (kagglehub: psparks/instacart-market-basket-analysis).

Builds user→product interactions: the `prior` baskets are the training history,
and each user's single final `train` basket is the held-out test
(leave-last-basket-out — the standard Instacart eval). Items = the product
catalog. Matches the validate_hm contract: returns
(train, test, products) with entity_id / item_id [/ timestamp] columns.

Real grocery-reorder log: ~3.2M orders, ~32M prior basket lines, ~206k users,
~50k products — a genuine high-repeat retail dataset (vs the academic cuts).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import kagglehub
import pandas as pd

SLUG = "psparks/instacart-market-basket-analysis"


@lru_cache(maxsize=1)
def _dir() -> Path:
    # Download the whole dataset once (cached), then read files locally — avoids
    # re-fetching per file and the PANDAS adapter's flaky per-call downloads.
    return Path(kagglehub.dataset_download(SLUG))


def _csv(name: str, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(_dir() / name, usecols=usecols)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = _csv("orders.csv", ["order_id", "user_id", "eval_set"])
    o = orders[["order_id", "user_id"]]
    prior = _csv("order_products__prior.csv", ["order_id", "product_id"])
    final = _csv("order_products__train.csv", ["order_id", "product_id"])

    def _join(op: pd.DataFrame) -> pd.DataFrame:
        df = op.merge(o, on="order_id")[["user_id", "product_id"]].rename(
            columns={"user_id": "entity_id", "product_id": "item_id"}
        )
        # No absolute timestamp in the log; constant so the harness treats it as
        # timestamp-less (channels gate off, like book) and WARM=random splits.
        df["timestamp"] = 0.0
        return df

    train = _join(prior)
    test = _join(final)
    products = _csv(
        "products.csv", ["product_id", "product_name", "aisle_id", "department_id"]
    ).rename(columns={"product_id": "item_id"})
    return train, test, products


if __name__ == "__main__":
    tr, te, it = _load()
    print(
        f"train {len(tr):,} rows  users {tr.entity_id.nunique():,}  items {tr.item_id.nunique():,}  "
        f"test {len(te):,} rows  products {len(it):,}"
    )
