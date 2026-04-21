"""Constraint filter tests."""

from __future__ import annotations

from kindling.rerank.constraints import apply_constraints
from kindling.retrieve.protocol import Candidate


def _candidates() -> list[Candidate]:
    return [
        Candidate(item_id="a", score=0.9, source="test"),
        Candidate(item_id="b", score=0.8, source="test"),
        Candidate(item_id="c", score=0.7, source="test"),
    ]


def test_no_predicates_passthrough() -> None:
    cands = _candidates()
    assert apply_constraints(cands, []) == cands


def test_single_predicate_filters() -> None:
    cands = _candidates()
    result = apply_constraints(cands, [lambda item: item != "b"])
    assert [c.item_id for c in result] == ["a", "c"]


def test_multiple_predicates_composed() -> None:
    cands = _candidates()
    result = apply_constraints(
        cands,
        [lambda item: item != "b", lambda item: item != "c"],
    )
    assert [c.item_id for c in result] == ["a"]


def test_predicate_short_circuit() -> None:
    """If the first predicate returns False, subsequent predicates must not run.
    Verified by a predicate that raises if called."""

    def first(item: object) -> bool:
        return False

    def second(item: object) -> bool:
        raise AssertionError("should not be called")

    cands = _candidates()
    result = apply_constraints(cands, [first, second])
    assert result == []
