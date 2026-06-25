"""Open-catalog extension memory cap: fail-safe on constrained machines.

The cap bounds the metadata-only extension so the fit stays under a RAM
budget. It reserves the estimated interaction-fit peak plus an OS/runtime
floor; on a borderline machine it must shrink the extension (or drop to
catalog-only) rather than risk an OOM. See engine_v2._open_catalog_extension_cap.
"""

from __future__ import annotations

import os

import pytest

from kindling import Engine

GB = 1024**3
_PAGE = 4096


def _engine(**kw):
    return Engine(persona_min_users=10**9, random_state=0, **kw)


def _patch_total(monkeypatch, total_bytes: int) -> None:
    def fake_sysconf(name: str) -> int:
        if name == "SC_PAGE_SIZE":
            return _PAGE
        if name == "SC_PHYS_PAGES":
            return total_bytes // _PAGE
        raise ValueError(name)

    monkeypatch.setattr(os, "sysconf", fake_sysconf)


def test_explicit_override_wins():
    assert (
        _engine(open_catalog_max_extension=12_345)._open_catalog_extension_cap(10**6, 50_000)
        == 12_345
    )
    assert _engine(open_catalog_max_extension=0)._open_catalog_extension_cap(10**6, 50_000) == 0


def test_os_reserve_tightens_on_24gb(monkeypatch):
    # Book-scale fit on 24 GB: interaction peak ~17.4 GB. The OS-reserve
    # ceiling (total − 6 GB = 18 GB) is tighter than 0.80×24 (19.2 GB), so
    # the extension is strictly smaller than the percentage-only cap.
    _patch_total(monkeypatch, 24 * GB)
    e = _engine()
    peak = e._PEAK_BYTES_PER_OBS * 8_000_000 + e._PEAK_BYTES_PER_TRAIN_ITEM * 357_000
    pct_only_cap = int((0.80 * 24 * GB - peak) / e._EXTENSION_BYTES_PER_ITEM)
    cap = e._open_catalog_extension_cap(8_000_000, 357_000)
    assert 0 < cap < pct_only_cap  # the OS reserve binds and shrinks it


def test_returns_zero_when_interaction_fit_exceeds_budget(monkeypatch):
    # A very large train catalog: the interaction fit alone blows the
    # budget → catalog-only (no extension), never an OOM.
    _patch_total(monkeypatch, 24 * GB)
    assert _engine()._open_catalog_extension_cap(8_000_000, 600_000) == 0


def test_large_machine_allows_large_extension(monkeypatch):
    # 128 GB box, modest catalog: the cap should not bind meaningfully.
    _patch_total(monkeypatch, 128 * GB)
    assert _engine()._open_catalog_extension_cap(1_000_000, 50_000) > 1_000_000


@pytest.mark.parametrize("total_gb", [8, 16, 24, 64])
def test_cap_keeps_extension_within_budget(monkeypatch, total_gb):
    # Whenever the cap admits an extension (cap > 0), interaction_peak +
    # extension_bytes must stay under the ceiling. When the interaction fit
    # alone exceeds budget the cap returns 0 (catalog-only) — the cap can't
    # shrink the interaction fit, only the extension.
    _patch_total(monkeypatch, total_gb * GB)
    n_obs, n_train = 2_000_000, 80_000
    e = _engine()
    cap = e._open_catalog_extension_cap(n_obs, n_train)
    peak = e._PEAK_BYTES_PER_OBS * n_obs + e._PEAK_BYTES_PER_TRAIN_ITEM * n_train
    ceiling = min(0.80 * total_gb * GB, total_gb * GB - e._OS_RESERVE_BYTES)
    if cap > 0:
        assert peak + cap * e._EXTENSION_BYTES_PER_ITEM <= ceiling + 1
    else:
        assert peak >= ceiling  # 0 only because the fit itself blows budget
