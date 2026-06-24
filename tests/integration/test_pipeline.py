"""End-to-end integration tests.

The MovieLens-1M test is marked slow + integration because it downloads
the dataset. CI runs it behind a cache; local runs pull it once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kindling.engine import Engine
from kindling.benchmarks.harness import run_movielens_1m


def test_end_to_end_synthetic() -> None:
    """Synthetic two-cluster dataset: users in cluster A all interact with
    items 1-3, users in cluster B with items 4-6. Recommendations for a
    user in A should favor items from A that they don't own yet."""
    rng = np.random.default_rng(seed=0)
    rows: list[tuple[str, int]] = []
    for user in range(20):
        cluster = "A" if user < 10 else "B"
        items = [1, 2, 3] if cluster == "A" else [4, 5, 6]
        # Each user interacts with 2 of their cluster's items.
        chosen = rng.choice(items, size=2, replace=False)
        for item in chosen:
            rows.append((f"{cluster}_{user}", int(item)))

    df = pd.DataFrame({"entity_id": [r[0] for r in rows], "item_id": [r[1] for r in rows]})
    engine = Engine().fit(df)
    recs = engine.recommend(entity_id="A_0", n=3)
    assert len(recs) > 0
    # The top recommendation must be an A-cluster item the entity doesn't
    # own - that's the only item ranked purely on signal strength. Trailing
    # recs are padded from whatever the retriever stack surfaces next
    # when the tiny 40-interaction fixture doesn't give every retriever
    # a non-trivial signal (which is OK; on real data these contribute).
    owned = {2, 3}  # A_0 owns 2 of the 3 A-items (deterministic from seed)
    a_cluster_unowned = {1, 2, 3} - owned
    assert recs[0].item_id in a_cluster_unowned


@pytest.mark.integration
@pytest.mark.slow
def test_movielens_1m_pipeline_runs(tmp_path: Path) -> None:
    """Acceptance gate for Phase 1: the benchmark harness runs end-to-end on
    MovieLens-1M and produces non-degenerate metrics. The actual metric
    values are only sanity checks at this stage; the point is the pipeline
    completes without error."""
    result = run_movielens_1m(k=10, max_eval_entities=200)
    assert result.metrics.n_entities_evaluated > 0
    assert result.metrics.coverage > 0.0
    assert result.fit_seconds > 0.0
    assert result.recommend_seconds > 0.0
    # At minimum the trivial co-occurrence recommender should beat random.
    # Random NDCG@10 on ML-1M is ~0.002; heuristic co-occurrence should
    # clear at least 10x that.
    assert result.metrics.ndcg_at_k > 0.02, (
        f"NDCG@10 suspiciously low ({result.metrics.ndcg_at_k}); "
        "either the dataset loaded wrong or the pipeline regressed."
    )
