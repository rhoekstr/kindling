"""Arrow IPC export (plan Phase 10, PRD §10.4).

Cross-language interop for the engine's frozen state. Unlike
``persist.format`` (pickle), the Arrow path exports a platform-neutral
representation that C++/Java/Rust consumers can read via Apache Arrow
IPC. Trades full round-trip fidelity for interoperability - pluggable
components and outcome-log contents don't export.

Requires ``pyarrow`` at runtime. Optional - the Arrow export lives
behind a helpful ImportError when the dep isn't installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from kindling.engine import Engine


def export_arrow(engine: "Engine", path: str | Path) -> None:
    """Write the engine's item graph + posterior params to an Arrow IPC
    file. Useful for sharing a fitted engine with a non-Python
    consumer.

    Format (single-file IPC with multiple tables):
    - ``items``: item_id column keyed by internal index.
    - ``item_graph``: (src, dst, weight) edge triples.
    - ``posterior``: signal_name + posterior_mean + credible_lower/upper.
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Arrow export requires the optional 'pyarrow' package. "
            "Install with ``pip install pyarrow``."
        ) from exc

    engine._require_fitted()
    assert engine._item_graph is not None

    items_table = pa.table(
        {
            "item_id": pa.array(
                [str(i) for i in engine._item_graph.item_ids], type=pa.string()
            ),
            "internal_index": pa.array(
                list(range(len(engine._item_graph.item_ids))), type=pa.int64()
            ),
        }
    )

    adj = engine._item_graph.adjacency.tocoo()
    graph_table = pa.table(
        {
            "src": pa.array(adj.row.astype("int64"), type=pa.int64()),
            "dst": pa.array(adj.col.astype("int64"), type=pa.int64()),
            "weight": pa.array(adj.data.astype("float64"), type=pa.float64()),
        }
    )

    posterior_table = _posterior_table(engine, pa)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, items_table.schema) as writer:
            writer.write_table(items_table)
    # Append graph + posterior as separate files next to the primary.
    with pa.OSFile(str(path.with_suffix(path.suffix + ".graph")), "wb") as sink:
        with ipc.new_file(sink, graph_table.schema) as writer:
            writer.write_table(graph_table)
    if posterior_table is not None:
        with pa.OSFile(str(path.with_suffix(path.suffix + ".posterior")), "wb") as sink:
            with ipc.new_file(sink, posterior_table.schema) as writer:
                writer.write_table(posterior_table)


def _posterior_table(engine: "Engine", pa):  # type: ignore[no-untyped-def]
    """Serialize the Bayesian blend posterior if present."""
    blend = engine._bayesian_blend
    if blend is None:
        return None
    ci = blend.credible_interval(coverage=engine.credible_coverage)
    return pa.table(
        {
            "signal": pa.array(list(blend.signal_names), type=pa.string()),
            "posterior_mean": pa.array(
                blend.posterior_mean.tolist(), type=pa.float64()
            ),
            "credible_lower": pa.array(ci[:, 0].tolist(), type=pa.float64()),
            "credible_upper": pa.array(ci[:, 1].tolist(), type=pa.float64()),
            "prior_alpha": pa.array(blend.prior_alpha.tolist(), type=pa.float64()),
        }
    )
