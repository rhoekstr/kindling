"""MovieLens-1M loader.

Downloads and caches the dataset, parses into kindling's canonical input
format, and provides a chronological train/test split for the benchmark
harness.

The ML-1M dataset is distributed by GroupLens under a research-use license.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# Pinned after first local download on 2026-04-20.
# Set KINDLING_SKIP_CHECKSUM=1 to skip verification during offline dev.
ML_1M_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"

DEFAULT_CACHE_DIR = Path(os.environ.get("KINDLING_CACHE_DIR", Path.home() / ".cache" / "kindling"))


@dataclass(frozen=True)
class MovieLensSplit:
    """Chronological train/test split of MovieLens interactions."""

    train: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame  # item metadata (movie_id, title, genres)


def _cache_dir(base: Path | None = None) -> Path:
    root = base if base is not None else DEFAULT_CACHE_DIR
    d = root / "movielens-1m"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_or_warn(path: Path, expected: str) -> None:
    if os.environ.get("KINDLING_SKIP_CHECKSUM") == "1":
        return
    actual = _sha256(path)
    if not expected:
        warnings.warn(
            f"No pinned checksum for {path.name} (got {actual}). "
            "Pin ML_1M_SHA256 in kindling.loaders.movielens to enable verification.",
            stacklevel=2,
        )
        return
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path.name}: expected {expected}, got {actual}")


def _download(cache_dir: Path | None = None) -> Path:
    """Download the ML-1M zip to the cache dir and return the extracted path."""
    cache = _cache_dir(cache_dir)
    zip_path = cache / "ml-1m.zip"
    extracted = cache / "ml-1m"

    if not extracted.exists():
        if not zip_path.exists():
            urllib.request.urlretrieve(ML_1M_URL, zip_path)
        _verify_or_warn(zip_path, ML_1M_SHA256)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache)
    return extracted


def load_raw(cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw ML-1M ratings and movies DataFrames.

    Returns ratings (UserID, MovieID, Rating, Timestamp) and movies
    (MovieID, Title, Genres).
    """
    base = _download(cache_dir=cache_dir)

    ratings = pd.read_csv(
        base / "ratings.dat",
        sep="::",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    movies = pd.read_csv(
        base / "movies.dat",
        sep="::",
        header=None,
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    return ratings, movies


def to_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """Convert raw ML-1M ratings into kindling's canonical interaction format.

    Ratings >= 4 are treated as positive interactions; lower ratings are
    dropped. This is a standard implicit-feedback conversion for ML-1M.
    Phase 1 doesn't use ratings directly; later phases may re-introduce them
    via action_type="rate".
    """
    positive = ratings[ratings["rating"] >= 4].copy()
    return pd.DataFrame(
        {
            "entity_id": positive["user_id"].astype("int64").to_numpy(),
            "item_id": positive["movie_id"].astype("int64").to_numpy(),
            "timestamp": positive["timestamp"].to_numpy(),
        }
    )


def chronological_split(
    interactions: pd.DataFrame, test_fraction: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions chronologically — last `test_fraction` of events by
    timestamp go to the test set. Standard for sequential recommendation.
    """
    sorted_df = interactions.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cutoff = int(len(sorted_df) * (1.0 - test_fraction))
    return sorted_df.iloc[:cutoff].copy(), sorted_df.iloc[cutoff:].copy()


def load_1m(cache_dir: Path | None = None, test_fraction: float = 0.1) -> DatasetSplit:
    """Load ML-1M as a chronological train/test split in canonical format.

    Returns a ``DatasetSplit`` (Phase 7 refactor). The Phase 1 legacy
    attributes (train / test / items) remain on the returned object.
    """
    ratings, movies = load_raw(cache_dir=cache_dir)
    interactions = to_interactions(ratings)
    train, test = chronological_split(interactions, test_fraction=test_fraction)
    items = movies.rename(columns={"movie_id": "item_id"})
    # Genre is the natural category for calibration on ML-1M.
    items = items.assign(category=items["genres"].str.split("|").str[0])
    return DatasetSplit(
        name="movielens-1m",
        train=train,
        test=test,
        items=items,
        description=(
            "MovieLens 1M ratings (ratings>=4 treated as positive "
            "implicit feedback). Primary genre as calibration category."
        ),
    )
