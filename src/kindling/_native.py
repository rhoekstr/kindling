"""Optional Rust extension import.

The kindling Python package never hard-requires ``kindling_native`` - if
the extension isn't available (no wheel for the user's platform, source
install without a Rust toolchain, intentional opt-out), the call sites
fall back to the pure-Python implementations. This module owns the
import guard so every consumer has a single check.
"""

from __future__ import annotations

try:
    import kindling_native  # type: ignore[import-untyped]

    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - platform-specific
    kindling_native = None
    NATIVE_AVAILABLE = False


__all__ = ["NATIVE_AVAILABLE", "kindling_native"]
