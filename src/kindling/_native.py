"""Optional Rust extension imports.

Two extensions live alongside each other during the v1→v2 migration:

- ``kindling_native`` — legacy hot-path crate (cooc/path_family/dpp_kernel/
  dedup/personas). Used by the v1 Engine path. Will be deleted after the
  v2 cutover.
- ``kindling_core`` — v2 Rust core. Owns signals/clustering/scoring/
  retrieval/repeat/loaders. Used by the v2 Engine path
  (``Engine(use_v2_core=True)``).

Either or both may be missing on a given install; call sites must check
``NATIVE_AVAILABLE`` (v1) or ``CORE_AVAILABLE`` (v2) before dereferencing.
The pure-Python fallback only exists for the v1 path; v2 requires the
``kindling_core`` wheel.
"""

from __future__ import annotations

try:
    import kindling_native  # type: ignore[import-untyped]

    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - platform-specific
    kindling_native = None
    NATIVE_AVAILABLE = False

try:
    import kindling_core  # type: ignore[import-untyped]

    CORE_AVAILABLE = True
except ImportError:  # pragma: no cover - platform-specific
    kindling_core = None
    CORE_AVAILABLE = False


__all__ = [
    "CORE_AVAILABLE",
    "NATIVE_AVAILABLE",
    "kindling_core",
    "kindling_native",
]
