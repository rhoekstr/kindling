"""Held-out repeat gate regression test.

Locks the two directions of the gate: KEEP when reorders predict the recent
(global-time) future, DECLINE when they don't — the discriminator a repeat-rate
threshold can't make (cf. docs/REPEAT-GATE.md, the steam case). Plus the native
repeat toggles used by the gate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kindling import Engine


def _true_repeat() -> pd.DataFrame:
    """Each user rebuys a small staple set every cycle — the recent (held-out)
    items ARE repeats, so the reorder module predicts them."""
    rows = []
    for u in range(300):
        staples = [u % 8, (u + 1) % 8, (u + 2) % 8]
        t = 0
        for _cycle in range(7):
            for it in staples:
                rows.append((u, it, float(t)))
                t += 1
    return pd.DataFrame(rows, columns=["entity_id", "item_id", "timestamp"])


def _useless_repeat() -> pd.DataFrame:
    """Early staples create a high repeat-rate, but each user's recent items are
    fresh and idiosyncratic — past reorders DON'T predict the future, so the gate
    should decline despite the duplicates."""
    rows = []
    for u in range(300):
        staples = [u % 8, (u + 1) % 8]
        t = 0
        for _cycle in range(6):
            for it in staples:
                rows.append((u, it, float(t)))
                t += 1
        for j in range(3):  # recent: unique, unpredictable, never a repeat
            rows.append((u, 1000 + u * 3 + j, float(t)))
            t += 1
    return pd.DataFrame(rows, columns=["entity_id", "item_id", "timestamp"])


def test_repeat_gate_keeps_true_repeat():
    eng = Engine(random_state=0, channel_gate=False).fit(_true_repeat())
    g = eng._state.profile.get("repeat_gated")
    assert g is not None, "gate did not run (pre-filter/held-out too small)"
    assert g["kept"] is True and g["ndcg_on"] > g["ndcg_off"]
    assert eng._state.repeat_active is True


def test_repeat_gate_declines_when_useless():
    eng = Engine(random_state=0, channel_gate=False).fit(_useless_repeat())
    g = eng._state.profile.get("repeat_gated")
    assert g is not None
    assert g["kept"] is False  # reorders don't predict the recent future
    assert eng._state.repeat_active is False


def test_native_repeat_toggle_changes_recs():
    eng = Engine(random_state=0, channel_gate=False).fit(_true_repeat())
    if not eng._state.repeat_active:
        pytest.skip("repeat inactive on this fixture")
    from kindling._native_engine import build_native_engine

    native = build_native_engine(eng)
    ent = 0
    owned = [int(x) for x in eng._state.owned_by_entity[ent]]
    ur = int(eng._state.entity_to_user_idx.get(ent, -1))
    native.set_repeat_active(True)
    on_items, _, _ = native.recommend(owned, ur, 10, 0.0)
    native.set_repeat_active(False)
    off_items, _, _ = native.recommend(owned, ur, 10, 0.0)
    assert on_items != off_items, "repeat toggle had no effect on recommendations"
