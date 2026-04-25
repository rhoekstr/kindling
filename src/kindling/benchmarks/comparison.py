"""Apples-to-apples comparison: kindling vs. industry-standard baselines.

Runs kindling's Engine and the baselines defined in ``benchmarks.baselines``
against the same chronological train/test split on a reference dataset.
Emits accuracy metrics (NDCG, Recall, MRR, Hit), catalog coverage, fit
wall-time, and per-recommend p50/p95 latency.

CLI:
    python -m kindling.benchmarks.comparison --dataset movielens-1m \
        --output bench/reports/baselines_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd

from kindling import __version__
from kindling.benchmarks.baselines import (
    ImplicitALSBaseline,
    ItemItemKNN,
    PopularityBaseline,
)
from kindling.benchmarks.metrics import MetricReport, aggregate
from kindling.engine import Engine
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
from kindling.loaders._base import DatasetSplit


class Recommender(Protocol):
    name: str

    def fit(self, interactions: pd.DataFrame) -> object: ...
    def recommend(self, entity_id: object, n: int = ...) -> list[object]: ...


class _EngineAdapter:
    """Wrap Engine behind the Recommender protocol."""

    name = "kindling"

    def __init__(
        self,
        use_ranker: bool = False,
        use_personas: bool = False,
        persona_cluster_k: int = 30,
    ) -> None:
        persona_config = None
        if use_personas:
            from kindling.personas import KMeansClustering, PersonaConfig

            persona_config = PersonaConfig(
                enabled=True,
                clustering=KMeansClustering(n_clusters=persona_cluster_k, random_state=0),
                min_activation_users=100,
            )
        self._engine = Engine(use_ranker=use_ranker, persona_config=persona_config)
        suffix_parts: list[str] = []
        if use_personas:
            suffix_parts.append("persona")
        if use_ranker:
            suffix_parts.append("ranker")
        if suffix_parts:
            self.name = "kindling+" + "+".join(suffix_parts)

    def fit(self, interactions: pd.DataFrame) -> "_EngineAdapter":
        self._engine.fit(interactions)
        return self

    def recommend(self, entity_id: object, n: int = 10) -> list[object]:
        recs = self._engine.recommend(entity_id=entity_id, n=n)
        return [r.item_id for r in recs]

    @property
    def n_items(self) -> int:
        return self._engine.item_graph.n_items


@dataclass(frozen=True)
class ModelResult:
    name: str
    fit_seconds: float
    recommend_p50_ms: float
    recommend_p95_ms: float
    metrics: MetricReport

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fit_seconds": self.fit_seconds,
            "recommend_p50_ms": self.recommend_p50_ms,
            "recommend_p95_ms": self.recommend_p95_ms,
            "metrics": self.metrics.as_dict(),
        }


def _evaluate(
    model: Recommender,
    train: pd.DataFrame,
    eval_entities: list[object],
    test_items_by_entity: pd.Series,
    train_items_by_entity: pd.Series,
    catalog_size: int,
    k: int,
) -> ModelResult:
    fit_start = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - fit_start

    per_entity: list[tuple[list[object], set[object]]] = []
    latencies_ms: list[float] = []
    for entity in eval_entities:
        train_owned = train_items_by_entity.get(entity, set())
        test_owned = test_items_by_entity.get(entity, set())
        relevant = test_owned - train_owned
        t0 = time.perf_counter()
        rec_items = model.recommend(entity, n=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        per_entity.append((rec_items, relevant))

    metrics = aggregate(per_entity, catalog_size=catalog_size, k=k)
    return ModelResult(
        name=model.name,
        fit_seconds=fit_seconds,
        recommend_p50_ms=float(np.percentile(latencies_ms, 50)),
        recommend_p95_ms=float(np.percentile(latencies_ms, 95)),
        metrics=metrics,
    )


def _cache_dir() -> Path:
    import os

    return Path(os.environ.get("KINDLING_CACHE_DIR", Path.home() / ".cache" / "kindling"))


def _load_dataset(name: str, test_fraction: float) -> DatasetSplit:
    if name == "movielens-1m":
        return movielens.load_1m(test_fraction=test_fraction)
    if name == "synthetic-grocery":
        return synthetic.make_grocery(
            n_entities=1500,
            n_items_per_category=20,
            n_categories=8,
            n_sessions_per_entity=10,
            items_per_session=6,
            test_fraction=test_fraction,
        )
    if name == "synthetic-grocery-deep":
        # Longer sessions (10 items) give the path signals enough sequential
        # depth to separate from item-item cosine. Matches the "real session"
        # shape of grocery / e-commerce baskets.
        return synthetic.make_grocery(
            n_entities=1500,
            n_items_per_category=25,
            n_categories=8,
            n_sessions_per_entity=12,
            items_per_session=10,
            test_fraction=test_fraction,
        )
    cache = _cache_dir()
    if name == "retailrocket":
        return retailrocket.load(cache / "retailrocket", test_fraction=test_fraction)
    if name == "instacart":
        return instacart.load(cache / "instacart", test_fraction=test_fraction)
    if name == "gowalla":
        return gowalla.load(cache / "gowalla", test_fraction=test_fraction)
    if name == "yelp2018":
        return yelp.load(cache / "yelp2018", test_fraction=test_fraction)
    if name == "tafeng":
        return tafeng.load(cache / "tafeng", test_fraction=test_fraction)
    if name == "dunnhumby":
        return dunnhumby.load(cache / "dunnhumby", test_fraction=test_fraction)
    if name == "amazon-beauty":
        return _load_amazon_5core(cache / "amazon-beauty", test_fraction=test_fraction, label="amazon-beauty")
    if name == "amazon-book":
        return _load_amazon_5core(cache / "amazon-book", test_fraction=test_fraction, label="amazon-book")
    raise ValueError(f"Unknown dataset: {name}")


def _load_amazon_5core(data_dir: Path, test_fraction: float, label: str) -> DatasetSplit:
    """Resolve an Amazon dataset under ``data_dir``.

    Two formats supported, in priority order:
    1. McAuley 5-core JSONL.gz (preferred - has timestamps + ratings).
    2. LightGCN academic train.txt/test.txt split (no timestamps;
       used as a fallback when the JSONL isn't available locally).
    """
    if not data_dir.exists():
        raise amazon.AmazonReviewsDataNotAvailableError(
            f"Amazon data dir {data_dir} does not exist."
        )
    candidates = sorted(data_dir.glob("*5*.json*"))
    if candidates:
        split = amazon.load(candidates[0], test_fraction=test_fraction)
        return DatasetSplit(
            name=label,
            train=split.train,
            test=split.test,
            items=split.items,
            description=f"{label}: {split.description}",
        )
    # LightGCN academic split fallback.
    train_path = data_dir / "train.txt"
    test_path = data_dir / "test.txt"
    if train_path.exists() and test_path.exists():
        return _load_academic_split(train_path, test_path, name=label, action_type="rate")
    raise amazon.AmazonReviewsDataNotAvailableError(
        f"No 5-core JSON file or LightGCN academic split (train.txt/test.txt) "
        f"under {data_dir} for {label}. "
        "Download a 5-core category JSONL.gz from "
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ "
        "or the academic split from "
        "https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data"
    )


def _load_academic_split(
    train_path: Path, test_path: Path, name: str, action_type: str
) -> DatasetSplit:
    """Parse a LightGCN-style train.txt/test.txt pair.

    Each line: ``user_id item_id1 item_id2 ...``. No timestamps; path
    signals will degrade to manual_fallback sessions.
    """
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    for path, sink in [(train_path, train_rows), (test_path, test_rows)]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                user = parts[0]
                for it in parts[1:]:
                    sink.append((user, it))
    train = pd.DataFrame(train_rows, columns=["entity_id", "item_id"])
    train["action_type"] = action_type
    test = pd.DataFrame(test_rows, columns=["entity_id", "item_id"])
    test["action_type"] = action_type
    return DatasetSplit(
        name=name,
        train=train,
        test=test,
        items=None,
        description=(
            f"{name}: LightGCN academic split (NGCF/LightGCN benchmark); "
            "no timestamps, path signals degrade to manual_fallback sessions."
        ),
    )


def run_comparison(
    k: int = 10,
    max_eval_entities: int = 2000,
    test_fraction: float = 0.1,
    include_als: bool = True,
    dataset: str = "movielens-1m",
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction)

    train_items_by_entity = cast(
        pd.Series,
        split.train.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    test_items_by_entity = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )

    eval_entities_all: list[object] = sorted(
        set(train_items_by_entity.index).intersection(test_items_by_entity.index)
    )
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities: list[object] = eval_entities_all[::step][:max_eval_entities]

    catalog_size = int(split.train["item_id"].nunique())

    # Build models. Kindling first; baselines after for predictable report order.
    models: list[Recommender] = [
        _EngineAdapter(),
        PopularityBaseline(),
        ItemItemKNN(k_neighbors=200),
    ]
    if include_als:
        models.append(ImplicitALSBaseline(factors=64, iterations=15))

    results: list[ModelResult] = []
    for m in models:
        print(f"  evaluating {m.name} ...", flush=True)
        res = _evaluate(
            m,
            train=split.train,
            eval_entities=eval_entities,
            test_items_by_entity=test_items_by_entity,
            train_items_by_entity=train_items_by_entity,
            catalog_size=catalog_size,
            k=k,
        )
        results.append(res)

    return {
        "dataset": dataset,
        "k": k,
        "n_eval_entities": len(eval_entities),
        "catalog_size": catalog_size,
        "kindling_version": __version__,
        "results": [r.as_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare kindling against popularity, item-item kNN, and implicit ALS."
    )
    parser.add_argument(
        "--dataset",
        default="movielens-1m",
        choices=[
            "movielens-1m",
            "synthetic-grocery",
            "synthetic-grocery-deep",
            "retailrocket",
            "instacart",
            "gowalla",
            "yelp2018",
            "tafeng",
            "dunnhumby",
            "amazon-beauty",
            "amazon-book",
        ],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-eval-entities", type=int, default=2000)
    parser.add_argument("--no-als", action="store_true", help="Skip the ALS baseline")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_comparison(
        k=args.k,
        max_eval_entities=args.max_eval_entities,
        include_als=not args.no_als,
        dataset=args.dataset,
    )
    pretty = json.dumps(report, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"Wrote {args.output}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
