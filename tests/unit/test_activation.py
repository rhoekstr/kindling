"""The activation plan reports the regime-based gating faithfully."""

from __future__ import annotations

from kindling import ActivationPlan, Engine
from kindling.loaders import synthetic


def _plan(**fit_kw):
    s = synthetic.make_ratings(n_entities=150, n_items=90, ratings_per_entity=25, seed=0)
    e = Engine(random_state=0, **fit_kw)
    e.fit(s.train)
    return e.activation_plan


def test_activation_plan_shape():
    plan = _plan()
    assert isinstance(plan, ActivationPlan)
    assert plan.base_scorer == "ease"  # small catalog → EASE base
    assert plan.n_users == 150
    assert isinstance(plan.summary(), str) and plan.summary()


def test_timestamped_data_activates_trend():
    plan = _plan()
    assert "trend" in plan.active_channels
    trend = next(c for c in plan.channels if c.name == "trend")
    assert trend.active and trend.weight == 0.5


def test_dense_history_gates_off_user_cf():
    # median history 25 > the 20-item gate → user_cf must be off, with a reason.
    plan = _plan()
    ucf = next(c for c in plan.channels if c.name == "user_cf")
    assert not ucf.active
    assert "gate" in ucf.reason


def test_each_channel_has_a_reason():
    plan = _plan()
    assert {"trend", "last_item", "transitions", "user_cf", "content"} == {
        c.name for c in plan.channels
    }
    assert all(c.reason for c in plan.channels)
