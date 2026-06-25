"""Rust core import.

The (PyO3) Rust extension owns the numeric kernels — EASE, cooccurrence,
directional cooc, layered scoring, retrieval. The v2 engine requires it;
there is no pure-Python fallback. Call sites check ``CORE_AVAILABLE``
before dereferencing ``kindling_core`` so that ``import kindling`` still
succeeds on a partial install (with a clear error at ``fit`` time rather
than at import).

The extension is packaged inside the wheel as ``kindling._core``. The
legacy top-level ``kindling_core`` import is kept as a fallback so a
dev environment with the standalone extension installed keeps working.
"""

from __future__ import annotations

try:
    from kindling import _core as kindling_core  # packaged in the wheel

    CORE_AVAILABLE = True
except ImportError:  # pragma: no cover - dev / partial install
    try:
        import kindling_core  # type: ignore[import-untyped]  # standalone dev extension

        CORE_AVAILABLE = True
    except ImportError:
        kindling_core = None
        CORE_AVAILABLE = False


__all__ = [
    "CORE_AVAILABLE",
    "kindling_core",
]
