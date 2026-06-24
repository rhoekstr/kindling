"""Outcome log + replay tests (PRD §6.6, plan Phase 5).

Invariants:
- Dedup via (entity_id, recommendation_id, item_id) primary key.
- Corrections supersede prior rows with the same key.
- Simple-mode records are flagged so posterior_summary can warn.
- Replay from log is deterministic and matches the insertion-order batch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from kindling.engine import Engine
from kindling.outcomes.log import OutcomeLog, ReportingMode
from kindling.outcomes.replay import replay_to_batch


def test_empty_log_has_zero_rows() -> None:
    log = OutcomeLog()
    assert len(log) == 0
    log.close()


def test_precise_report_inserts_one_row_per_item() -> None:
    with OutcomeLog() as log:
        inserted = log.report_precise(
            entity_id="e1",
            recommendation_id="rec1",
            shown_items=["a", "b", "c"],
            selected_items=["a"],
        )
        assert inserted == 3
        assert len(log) == 3


def test_precise_report_dedups_on_primary_key() -> None:
    with OutcomeLog() as log:
        log.report_precise(
            entity_id="e1",
            recommendation_id="rec1",
            shown_items=["a", "b"],
            selected_items=["a"],
        )
        # Re-report the same (entity, rec, items).
        again = log.report_precise(
            entity_id="e1",
            recommendation_id="rec1",
            shown_items=["a", "b"],
            selected_items=["b"],  # different outcome
        )
        # Dedup means zero new rows.
        assert again == 0
        assert len(log) == 2


def test_correction_supersedes_prior_row() -> None:
    with OutcomeLog() as log:
        log.report_precise(
            entity_id="e1",
            recommendation_id="rec1",
            shown_items=["a"],
            selected_items=[],
        )
        # Initially selected=False; correct to True.
        log.report_correction(
            entity_id="e1",
            recommendation_id="rec1",
            item_id="a",
            shown=True,
            selected=True,
        )
        records = list(log.iter_records())
        assert len(records) == 1
        assert records[0].selected is True


def test_simple_report_flags_mode() -> None:
    with OutcomeLog() as log:
        log.report_simple(
            entity_id="e1",
            item_id="a",
            action="selected",
        )
        assert log.has_simple_mode_records()
        counts = log.count_by_mode()
        assert counts[ReportingMode.SIMPLE] == 1
        assert ReportingMode.PRECISE not in counts


def test_iter_records_in_insertion_order() -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    with OutcomeLog() as log:
        log.report_precise(
            entity_id="e1",
            recommendation_id="rec1",
            shown_items=["a", "b"],
            selected_items=["a"],
            timestamp=now,
        )
        log.report_simple(
            entity_id="e2",
            item_id="c",
            action="rejected",
            timestamp=now + timedelta(seconds=1),
        )
        records = list(log.iter_records())
        assert [r.item_id for r in records] == ["a", "b", "c"]


def test_simple_rejects_unknown_action() -> None:
    with OutcomeLog() as log, pytest.raises(ValueError, match="action"):
        log.report_simple(entity_id="e", item_id="a", action="purchased")


def test_positions_list_length_checked() -> None:
    with OutcomeLog() as log, pytest.raises(ValueError, match="positions length"):
        log.report_precise(
            entity_id="e",
            recommendation_id="r",
            shown_items=["a", "b"],
            positions=[1],  # wrong length
        )


# ---- Engine integration -------------------------------------------------


@pytest.fixture
def phase5_engine() -> Engine:
    df = pd.DataFrame(
        {
            "entity_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "item_id": [1, 2, 3, 4, 5, 6, 1, 2, 4, 7, 8, 9, 2, 3, 5, 8, 10, 11],
            "timestamp": pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 7)] * 3),
        }
    )
    return Engine(vi_max_iter=30).fit(df)


def test_engine_report_outcome_populates_log(phase5_engine: Engine) -> None:
    inserted = phase5_engine.report_outcome(
        entity_id="a",
        recommendation_id="r1",
        shown_items=[100, 200, 300],
        selected_items=[100],
    )
    assert inserted == 3
    assert len(phase5_engine.outcome_log) == 3


def test_engine_report_interaction_simple_mode(phase5_engine: Engine) -> None:
    phase5_engine.report_interaction(
        entity_id="a",
        item_id=100,
        action="selected",
    )
    assert phase5_engine.outcome_log.has_simple_mode_records()
    summary = phase5_engine.posterior_summary()
    # Warning should surface in the diagnostics warnings list.
    diag = summary.get("diagnostics", {})
    warns = diag.get("warnings", []) if isinstance(diag, dict) else []
    assert any("Simple-mode" in w for w in warns)


def test_engine_refit_posterior_runs(phase5_engine: Engine) -> None:
    """Reporting outcomes and refitting should leave the posterior in a
    valid state (no exceptions, still has per-signal weights)."""
    # Report outcomes for known items.
    item_ids = list(phase5_engine._item_graph.item_index.keys())[:4]
    phase5_engine.report_outcome(
        entity_id="a",
        recommendation_id="r1",
        shown_items=item_ids,
        selected_items=[item_ids[0]],
    )
    initial_mean = phase5_engine._bayesian_blend.posterior_mean.copy()  # type: ignore[union-attr]
    report = phase5_engine.refit_posterior(max_iter=30)
    # Diagnostics may or may not pass on such a tiny batch, but must exist.
    assert report is not None
    final_mean = phase5_engine._bayesian_blend.posterior_mean  # type: ignore[union-attr]
    # Weights should sum to 1 regardless.
    assert abs(float(final_mean.sum()) - 1.0) < 1e-9
    # Either changed or stayed near initial - both are valid posteriors.
    assert final_mean.shape == initial_mean.shape


def test_replay_deterministic(phase5_engine: Engine) -> None:
    item_ids = list(phase5_engine._item_graph.item_index.keys())[:3]
    phase5_engine.report_outcome(
        entity_id="a",
        recommendation_id="r1",
        shown_items=item_ids,
        selected_items=[item_ids[0]],
    )

    def builder(entity: object, item: object) -> np.ndarray | None:
        return phase5_engine._build_signal_row_for_outcome(entity, item)

    batch1 = replay_to_batch(phase5_engine.outcome_log, builder)
    batch2 = replay_to_batch(phase5_engine.outcome_log, builder)
    np.testing.assert_array_equal(batch1.signal_matrix, batch2.signal_matrix)
    np.testing.assert_array_equal(batch1.selected, batch2.selected)
    np.testing.assert_array_equal(batch1.list_ids, batch2.list_ids)
