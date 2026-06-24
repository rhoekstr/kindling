"""RetailRocket event-stream loader (plan Phase 7).

RetailRocket publishes a Kaggle dataset of e-commerce events: view,
addtocart, transaction. ``addtocart`` without a following transaction is
the canonical soft-negative signal the PRD cost graph is designed for,
so RetailRocket is the definitive cost-graph exercise.

Source: Kaggle dataset ``retailrocket/ecommerce-dataset``. Expected
files:

    ${cache}/retailrocket/
      events.csv
      item_properties_part1.csv  (optional, provides category)
      category_tree.csv          (optional)

Action mapping:
  - ``view``        -> action_type = "view"
  - ``addtocart``   -> action_type = "add"
  - ``transaction`` -> action_type = "add" (stronger positive)

Cost-graph trigger: items with ``addtocart`` but no subsequent
``transaction`` within the same visit are re-labeled ``"remove"``. This
captures the "rejected after consideration" semantic the PRD wants in
its cost graph.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit


class RetailRocketDataNotAvailableError(RuntimeError):
    pass


def load(data_dir: str | Path, test_fraction: float = 0.1) -> DatasetSplit:
    """Parse RetailRocket events.csv into canonical format."""
    base = Path(data_dir)
    events_path = base / "events.csv"
    if not events_path.exists():
        raise RetailRocketDataNotAvailableError(
            f"RetailRocket events.csv not found at {events_path}. Download "
            "the Kaggle dataset ``retailrocket/ecommerce-dataset`` and "
            "unpack into the data_dir."
        )

    events = pd.read_csv(
        events_path,
        usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
    events = events.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Convert the three raw event types to kindling action_type values.
    # addtocart that never becomes a transaction is the cost-graph
    # negative signal.
    events = _relabel_cost_events(events)

    canonical = pd.DataFrame(
        {
            "entity_id": events["visitorid"].astype("int64").to_numpy(),
            "item_id": events["itemid"].astype("int64").to_numpy(),
            "timestamp": events["timestamp"].to_numpy(),
            "action_type": events["action_type"].to_numpy(),
        }
    )

    cutoff = int(len(canonical) * (1.0 - test_fraction))
    train = canonical.iloc[:cutoff].copy().reset_index(drop=True)
    test = canonical.iloc[cutoff:].copy().reset_index(drop=True)

    items = _load_item_metadata(base)
    return DatasetSplit(
        name="retailrocket",
        train=train,
        test=test,
        items=items,
        description=(
            "RetailRocket 2.7M events; view / addtocart / transaction "
            "flow; abandoned carts map to action_type='remove' to "
            "exercise the cost graph."
        ),
    )


def _relabel_cost_events(events: pd.DataFrame) -> pd.DataFrame:
    """An addtocart with no matching transaction within the same visitor's
    same-day window becomes 'remove' (cost-graph negative). Matching
    transactions stay as 'add'."""
    events = events.copy()
    events["action_type"] = "view"
    events.loc[events["event"] == "transaction", "action_type"] = "add"

    # For each (visitor, item) with addtocart but no transaction within
    # ~24 hours, label the addtocart event as "remove".
    add_events = events[events["event"] == "addtocart"][
        ["timestamp", "visitorid", "itemid"]
    ].reset_index()
    if not add_events.empty:
        transactions = events[events["event"] == "transaction"][
            ["timestamp", "visitorid", "itemid"]
        ]
        merged = add_events.merge(
            transactions,
            on=["visitorid", "itemid"],
            how="left",
            suffixes=("_add", "_txn"),
        )
        within_24h = (merged["timestamp_txn"] >= merged["timestamp_add"]) & (
            merged["timestamp_txn"] - merged["timestamp_add"] <= pd.Timedelta(hours=24)
        )
        # Rows with NO transaction within the window -> remove.
        no_txn = merged.groupby("index")["timestamp_txn"].apply(
            lambda s: not within_24h.loc[s.index].any()
        )
        abandon_indices = no_txn[no_txn].index.tolist()
        events.loc[abandon_indices, "action_type"] = "remove"
        # Addtocart followed by transaction -> "add".
        events.loc[
            (events["event"] == "addtocart") & (~events.index.isin(abandon_indices)),
            "action_type",
        ] = "add"
    return events


def _load_item_metadata(base: Path) -> pd.DataFrame | None:
    """RetailRocket's item_properties* files encode a category reference
    via the ``categoryid`` property. Optional - returns ``None`` when
    the file is missing."""
    prop_files = sorted(base.glob("item_properties*.csv"))
    if not prop_files:
        return None
    frames = [
        pd.read_csv(f, usecols=["timestamp", "itemid", "property", "value"]) for f in prop_files
    ]
    props = pd.concat(frames, ignore_index=True)
    cat_rows = props[props["property"] == "categoryid"]
    if cat_rows.empty:
        return None
    # Keep the latest known categoryid per item.
    cat_rows = cat_rows.sort_values("timestamp").drop_duplicates(subset=["itemid"], keep="last")
    return pd.DataFrame(
        {
            "item_id": cat_rows["itemid"].astype("int64").to_numpy(),
            "category": cat_rows["value"].astype(str).to_numpy(),
        }
    )
