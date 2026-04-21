"""Arrow IPC export tests (plan Phase 10, PRD §10.4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kindling import Engine

pyarrow = pytest.importorskip("pyarrow")


def _engine() -> Engine:
    df = pd.DataFrame(
        {
            "entity_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "item_id": [1, 2, 3, 4, 5, 6, 1, 2, 4, 7, 8, 9, 2, 3, 5, 8, 10, 11],
            "timestamp": pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 7)] * 3),
        }
    )
    return Engine(vi_max_iter=20).fit(df)


def test_arrow_export_writes_files(tmp_path: Path) -> None:
    engine = _engine()
    path = tmp_path / "engine.arrow"
    engine.export_arrow(path)
    assert path.exists()
    graph_path = path.with_suffix(path.suffix + ".graph")
    assert graph_path.exists()


def test_arrow_export_items_table(tmp_path: Path) -> None:
    engine = _engine()
    path = tmp_path / "engine.arrow"
    engine.export_arrow(path)
    import pyarrow.ipc as ipc

    with pyarrow.OSFile(str(path), "rb") as src:
        with ipc.open_file(src) as reader:
            items = reader.read_all()
    assert "item_id" in items.column_names
    assert "internal_index" in items.column_names
    assert items.num_rows == engine.item_graph.n_items


def test_arrow_export_graph_table_has_edges(tmp_path: Path) -> None:
    engine = _engine()
    path = tmp_path / "engine.arrow"
    engine.export_arrow(path)
    import pyarrow.ipc as ipc

    graph_path = path.with_suffix(path.suffix + ".graph")
    with pyarrow.OSFile(str(graph_path), "rb") as src:
        with ipc.open_file(src) as reader:
            graph = reader.read_all()
    assert set(graph.column_names) == {"src", "dst", "weight"}
    assert graph.num_rows > 0


def test_arrow_export_posterior_table(tmp_path: Path) -> None:
    engine = _engine()
    path = tmp_path / "engine.arrow"
    engine.export_arrow(path)
    posterior_path = path.with_suffix(path.suffix + ".posterior")
    if not posterior_path.exists():
        pytest.skip("no Bayesian blend in this engine")
    import pyarrow.ipc as ipc

    with pyarrow.OSFile(str(posterior_path), "rb") as src:
        with ipc.open_file(src) as reader:
            posterior = reader.read_all()
    assert "signal" in posterior.column_names
    assert "posterior_mean" in posterior.column_names
    assert "credible_lower" in posterior.column_names
