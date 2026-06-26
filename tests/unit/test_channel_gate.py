"""Held-out channel-activation gate (Engine.channel_gate)."""

from __future__ import annotations

from kindling import Engine
from kindling.loaders import synthetic


def _data():
    return synthetic.make_ratings(
        n_entities=300, n_items=80, ratings_per_entity=20, seed=7
    ).train


def test_gate_drops_overweighted_channel():
    """An absurdly over-weighted last_item channel hurts held-out NDCG, so the
    gate should eliminate it."""
    train = _data()
    eng = Engine(random_state=0, channel_gate=True, last_item_alpha=50.0).fit(train)
    assert "last_item" in eng._state.profile.get("channels_gated", [])
    assert eng._state.last_item_alpha == 0.0
    # Engine still serves after gating.
    ent = next(iter(eng._state.owned_by_entity))
    assert isinstance(eng.recommend(ent, 5), list)


def test_gate_off_keeps_channels():
    train = _data()
    eng = Engine(random_state=0, channel_gate=False, last_item_alpha=50.0).fit(train)
    assert eng._state.last_item_alpha == 50.0
    assert "channels_gated" not in eng._state.profile


def test_gate_validation():
    import pytest

    with pytest.raises(ValueError, match="channel_gate"):
        Engine(channel_gate="sometimes")
