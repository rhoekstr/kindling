"""Benchmark harness and metrics.

Submodules (``metrics``, ``harness``) are imported lazily via explicit
submodule imports by callers. Importing them here would cause a
RuntimeWarning when ``harness`` is run as a script via ``python -m``.
"""
