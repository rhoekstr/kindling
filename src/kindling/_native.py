"""Rust core import.

``kindling_core`` is the (PyO3) Rust extension that owns the numeric
kernels — EASE, cooccurrence, directional cooc, layered scoring, retrieval.
The v2 engine requires it; there is no pure-Python fallback. Call sites
check ``CORE_AVAILABLE`` before dereferencing ``kindling_core`` so that
``import kindling`` still succeeds on a partial install (with a clear
error at ``fit`` time rather than at import).
"""

from __future__ import annotations

try:
    import kindling_core  # type: ignore[import-untyped]

    CORE_AVAILABLE = True
except ImportError:  # pragma: no cover - platform-specific
    kindling_core = None
    CORE_AVAILABLE = False


__all__ = [
    "CORE_AVAILABLE",
    "kindling_core",
]
