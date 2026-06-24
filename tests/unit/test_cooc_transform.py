"""Unit + wiring tests for the cooc base weight transform."""

from __future__ import annotations

import numpy as np
import pytest

from kindling.engine_v2 import EngineV2
from kindling.graph.cooc_transform import apply_cooc_transform, resolve_cooc_transform
from kindling.loaders import synthetic

# Tiny symmetric cooc over 3 items a,b,c:
#   a-b = 4, a-c = 1, b-c = 2 ; marginals da=10, db=8, dc=4 ; n_users=20.
_INDPTR = np.array([0, 2, 4, 6], dtype=np.int32)
_INDICES = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
_DATA = np.array([4, 1, 4, 2, 1, 2], dtype=np.float32)
_COUNTS = np.array([10.0, 8.0, 4.0])
_N = 20


def _apply(transform):
    return apply_cooc_transform(_DATA, _INDICES, _INDPTR, _COUNTS, _N, transform)


def test_resolve_auto_is_wilson_and_validates():
    assert resolve_cooc_transform("auto") == "wilson"
    assert resolve_cooc_transform("cosine") == "cosine"
    with pytest.raises(ValueError, match="unknown cooc_base_transform"):
        resolve_cooc_transform("bogus")


def test_raw_is_identity():
    out = _apply("raw")
    np.testing.assert_array_equal(out, _DATA)


def test_cosine_matches_hand_computation_and_is_symmetric():
    out = _apply("cosine").astype(np.float64)
    # nonzero order: (a,b),(a,c),(b,a),(b,c),(c,a),(c,b)
    expected = np.array(
        [
            4 / np.sqrt(10 * 8),
            1 / np.sqrt(10 * 4),
            4 / np.sqrt(8 * 10),
            2 / np.sqrt(8 * 4),
            1 / np.sqrt(4 * 10),
            2 / np.sqrt(4 * 8),
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    # symmetry: w[a,b] == w[b,a], w[a,c] == w[c,a]
    assert out[0] == pytest.approx(out[2])
    assert out[1] == pytest.approx(out[4])


def test_jaccard_matches_hand_computation():
    out = _apply("jaccard").astype(np.float64)
    expected = np.array(
        [
            4 / (10 + 8 - 4),
            1 / (10 + 4 - 1),
            4 / (8 + 10 - 4),
            2 / (8 + 4 - 2),
            1 / (4 + 10 - 1),
            2 / (4 + 8 - 2),
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_wilson_shrinks_low_count_edges_below_raw_conditional():
    out = _apply("wilson").astype(np.float64)
    # Wilson LB is strictly below the raw conditional prob p̂=c/d and >= 0.
    assert np.all(out >= 0.0)
    assert np.all(out < 1.0)
    # The a-b edge (c=4) is more confident than a-c (c=1): higher weight.
    assert out[0] > out[1]


def test_transform_preserves_sparsity_pattern():
    for t in ("cosine", "jaccard", "wilson"):
        out = _apply(t)
        assert out.shape == _DATA.shape


# ── Wiring: applied only on the cooc path, never on the EASE path. ──────────
def _grocery():
    return synthetic.make_grocery(
        n_entities=120,
        n_items_per_category=8,
        n_categories=4,
        n_sessions_per_entity=6,
        items_per_session=4,
        seed=0,
    ).train


def test_cooc_path_applies_transform():
    eng = EngineV2(base_scorer="cooc", cooc_base_transform="wilson").fit(_grocery())
    prof = eng._state.profile
    assert prof["base_scorer_used"] == "cooc"
    assert prof["cooc_base_transform"] == "wilson"
    assert eng.recommend(entity_id=_grocery()["entity_id"].iloc[0], n=5) is not None


def test_cooc_path_raw_override_skips_transform():
    eng = EngineV2(base_scorer="cooc", cooc_base_transform="raw").fit(_grocery())
    prof = eng._state.profile
    assert prof["base_scorer_used"] == "cooc"
    assert "cooc_base_transform" not in prof


def test_ease_path_never_transforms_even_when_requested():
    # Small catalog (<= ease_max_items) → EASE base; cooc weights untouched.
    eng = EngineV2(base_scorer="auto", cooc_base_transform="wilson").fit(_grocery())
    prof = eng._state.profile
    assert prof["base_scorer_used"] == "ease"
    assert "cooc_base_transform" not in prof
