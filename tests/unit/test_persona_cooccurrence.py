"""Tests for per-persona item cooccurrence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from kindling.graph.persona_cooccurrence import (
    PersonaCooccurrenceGraph,
    build_persona_cooccurrence_graph,
)


@dataclass
class _MockPersonaIndex:
    """Minimal stand-in for PersonaIndex with the fields we read."""
    n_personas: int
    user_to_persona: np.ndarray
    entity_id_to_user_idx: dict


def test_build_returns_none_when_no_personas() -> None:
    df = pd.DataFrame({"entity_id": [1, 2], "item_id": [1, 2]})
    g = build_persona_cooccurrence_graph(
        df,
        item_index={1: 0, 2: 1},
        persona_index=_MockPersonaIndex(
            n_personas=0,
            user_to_persona=np.array([], dtype=np.int64),
            entity_id_to_user_idx={},
        ),
    )
    assert g is None


def test_build_per_persona_adjacencies() -> None:
    """Build a clean two-persona toy. Items 0,1 only co-occur in persona 0;
    items 2,3 only co-occur in persona 1."""
    rows = []
    # Persona 0: users 1,2,3 each touch items 0,1
    for u in [1, 2, 3]:
        rows.append({"entity_id": u, "item_id": 0})
        rows.append({"entity_id": u, "item_id": 1})
    # Persona 1: users 4,5,6 each touch items 2,3
    for u in [4, 5, 6]:
        rows.append({"entity_id": u, "item_id": 2})
        rows.append({"entity_id": u, "item_id": 3})
    df = pd.DataFrame(rows)
    user_to_persona = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    entity_id_to_user_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    persona_index = _MockPersonaIndex(
        n_personas=2,
        user_to_persona=user_to_persona,
        entity_id_to_user_idx=entity_id_to_user_idx,
    )
    g = build_persona_cooccurrence_graph(
        df, item_index={0: 0, 1: 1, 2: 2, 3: 3},
        persona_index=persona_index, min_persona_users=2,
    )
    assert g is not None
    assert g.n_personas == 2
    # Persona 0 cooc: items 0,1 co-occur 3 times (3 users).
    A0 = g.per_persona_adjacency[0].toarray()
    assert A0[0, 1] == 3.0
    assert A0[2, 3] == 0.0  # no persona-0 user touched 2,3
    # Persona 1 cooc: items 2,3 co-occur 3 times.
    A1 = g.per_persona_adjacency[1].toarray()
    assert A1[2, 3] == 3.0
    assert A1[0, 1] == 0.0


def test_score_against_owned_soft_combines_personas() -> None:
    """Soft-weighted scoring uses match weights to blend persona contributions."""
    # Build the 2-persona toy from the previous test.
    rows = []
    for u in [1, 2, 3]:
        rows.append({"entity_id": u, "item_id": 0})
        rows.append({"entity_id": u, "item_id": 1})
    for u in [4, 5, 6]:
        rows.append({"entity_id": u, "item_id": 2})
        rows.append({"entity_id": u, "item_id": 3})
    df = pd.DataFrame(rows)
    persona_index = _MockPersonaIndex(
        n_personas=2,
        user_to_persona=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        entity_id_to_user_idx={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    )
    g = build_persona_cooccurrence_graph(
        df, item_index={0: 0, 1: 1, 2: 2, 3: 3},
        persona_index=persona_index, min_persona_users=2,
    )
    # Owned = [item 0]. With match=[1, 0], should score item 1 high (cooc with 0
    # in persona 0) and items 2, 3 zero.
    scores_p0 = g.score_against_owned_soft(
        owned_indices=np.array([0]),
        match_weights=np.array([1.0, 0.0]),
        exclude_indices={0},
    )
    assert scores_p0[1] > 0
    assert scores_p0[2] == 0
    assert scores_p0[3] == 0

    # With match=[0, 1], item 0's owned doesn't co-occur in persona 1, so
    # everything is zero.
    scores_p1 = g.score_against_owned_soft(
        owned_indices=np.array([0]),
        match_weights=np.array([0.0, 1.0]),
        exclude_indices={0},
    )
    assert scores_p1.max() == 0

    # With match=[0.5, 0.5], item 1 still scores via persona 0 contribution.
    scores_blend = g.score_against_owned_soft(
        owned_indices=np.array([0]),
        match_weights=np.array([0.5, 0.5]),
        exclude_indices={0},
    )
    assert scores_blend[1] > 0


def test_min_persona_users_skips_tiny_clusters() -> None:
    """Personas below min_persona_users get empty matrices."""
    df = pd.DataFrame([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 1},
    ])
    persona_index = _MockPersonaIndex(
        n_personas=1,
        user_to_persona=np.array([0]),
        entity_id_to_user_idx={1: 0},
    )
    g = build_persona_cooccurrence_graph(
        df, item_index={0: 0, 1: 1},
        persona_index=persona_index, min_persona_users=5,
    )
    assert g is not None
    assert g.per_persona_adjacency[0].nnz == 0


def test_score_against_owned_soft_empty_inputs() -> None:
    """Empty owned items or empty match weights returns zeros."""
    df = pd.DataFrame([
        {"entity_id": u, "item_id": i} for u in [1, 2, 3] for i in [0, 1]
    ])
    persona_index = _MockPersonaIndex(
        n_personas=1,
        user_to_persona=np.array([0, 0, 0]),
        entity_id_to_user_idx={1: 0, 2: 1, 3: 2},
    )
    g = build_persona_cooccurrence_graph(
        df, item_index={0: 0, 1: 1}, persona_index=persona_index,
        min_persona_users=2,
    )
    # Empty owned.
    scores = g.score_against_owned_soft(
        owned_indices=np.array([], dtype=np.int64),
        match_weights=np.array([1.0]),
    )
    assert scores.max() == 0
    # All-zero match.
    scores = g.score_against_owned_soft(
        owned_indices=np.array([0]),
        match_weights=np.array([0.0]),
    )
    assert scores.max() == 0


def test_score_normalization_to_unit() -> None:
    df = pd.DataFrame([
        {"entity_id": u, "item_id": i} for u in [1, 2, 3] for i in [0, 1, 2]
    ])
    persona_index = _MockPersonaIndex(
        n_personas=1,
        user_to_persona=np.array([0, 0, 0]),
        entity_id_to_user_idx={1: 0, 2: 1, 3: 2},
    )
    g = build_persona_cooccurrence_graph(
        df, item_index={0: 0, 1: 1, 2: 2}, persona_index=persona_index,
        min_persona_users=2,
    )
    scores = g.score_against_owned_soft(
        owned_indices=np.array([0]),
        match_weights=np.array([1.0]),
        exclude_indices={0},
    )
    assert 0 <= scores.max() <= 1.0
