"""Shared loader protocol (plan Phase 7).

Every dataset loader must produce a ``DatasetSplit`` - train / test
interactions and optional item metadata in kindling's canonical input
format. This lets the benchmark harness run dataset-agnostically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test split for a reference dataset.

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"movielens-1m"``.
    train:
        Interaction records in canonical input format (entity_id,
        item_id, optional timestamp/action_type/etc.).
    test:
        Held-out interactions. Same format as ``train``.
    items:
        Optional item-metadata frame (at least an ``item_id`` column;
        category / title / etc. as available).
    description:
        One-line plain-language summary of the dataset.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame | None
    description: str


class DatasetLoader(Protocol):
    """Any loader callable returning a ``DatasetSplit``."""

    name: str

    def __call__(self, test_fraction: float = 0.1) -> DatasetSplit: ...
