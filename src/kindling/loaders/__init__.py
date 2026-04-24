"""Reference dataset loaders.

Phase 1 shipped MovieLens-1M. Phase 7 adds Instacart, Amazon Reviews,
RetailRocket, and two synthetic generators. Each loader returns a
``DatasetSplit``.
"""

from kindling.loaders import (
    amazon,
    dunnhumby,
    gowalla,
    instacart,
    movielens,
    retailrocket,
    synthetic,
    tafeng,
    yelp,
)
from kindling.loaders._base import DatasetLoader, DatasetSplit

__all__ = [
    "DatasetLoader",
    "DatasetSplit",
    "amazon",
    "dunnhumby",
    "gowalla",
    "instacart",
    "movielens",
    "retailrocket",
    "synthetic",
    "tafeng",
    "yelp",
]
