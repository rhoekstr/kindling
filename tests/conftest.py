"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _skip_loader_checksum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't fail tests on unpinned dataset checksums."""
    monkeypatch.setenv("KINDLING_SKIP_CHECKSUM", "1")


@pytest.fixture
def tiny_interactions() -> pd.DataFrame:
    """A minimal interaction set with 4 entities x 5 items.

    entity_a and entity_b overlap on items 1 and 2. entity_c is a singleton.
    """
    return pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b", "b", "c", "d", "d"],
            "item_id": [1, 2, 3, 1, 2, 4, 5, 3, 4],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                ]
            ),
        }
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=int(os.environ.get("KINDLING_TEST_SEED", "42")))
