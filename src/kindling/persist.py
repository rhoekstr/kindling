"""Save / load a fitted engine.

A fitted :class:`~kindling.engine.Engine` is serialized with a small
JSON header (a magic marker, a format version, and the kindling version)
followed by a pickle of the engine. The header lets ``load`` reject an
incompatible format *before* unpickling, and warn on a kindling-version
mismatch.

Format: pickle, the way scikit-learn / joblib persist fitted estimators.
It is fast and lossless (the whole fitted state round-trips identically),
but it is **not** portable across incompatible kindling versions and, like
all pickles, **executes code on load — only load files you trust.** A
portable columnar format (npz/Arrow) is a future enhancement.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kindling import __version__

if TYPE_CHECKING:
    from kindling.engine import Engine

_MAGIC = "kindling-engine"
FORMAT_VERSION = 1


def save_engine(engine: Engine, path: str | Path) -> None:
    """Serialize a fitted engine to ``path`` (header line + pickle)."""
    if getattr(engine, "_state", None) is None:
        raise RuntimeError("Engine is not fitted; nothing to save.")
    header = json.dumps(
        {
            "magic": _MAGIC,
            "format_version": FORMAT_VERSION,
            "kindling_version": __version__,
        }
    ).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(header + b"\n")
        pickle.dump(engine, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_engine(path: str | Path) -> Engine:
    """Load an engine saved by :func:`save_engine`.

    Raises ``ValueError`` on an unrecognized or incompatible format;
    warns (does not fail) on a kindling-version mismatch.
    """
    with open(path, "rb") as fh:
        first = fh.readline()
        try:
            header: dict[str, Any] = json.loads(first)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path} is not a kindling engine file.") from exc
        if header.get("magic") != _MAGIC:
            raise ValueError(f"{path} is not a kindling engine file.")
        fmt = header.get("format_version")
        if fmt != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported engine format version {fmt} (this build reads "
                f"{FORMAT_VERSION}). Re-fit and re-save with the current version."
            )
        saved_ver = header.get("kindling_version")
        if saved_ver != __version__:
            warnings.warn(
                f"Engine was saved with kindling {saved_ver}, loading with "
                f"{__version__}; pickle compatibility is best-effort.",
                stacklevel=2,
            )
        return cast("Engine", pickle.load(fh))
