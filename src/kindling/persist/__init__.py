"""Engine persistence (plan Phase 10, PRD §10.4).

Plan rework of the PRD's bincode plan: Python closures (lambda
constraints, user-supplied retrievers / rankers / kernels) can't be
serialized generically. The Phase 10 design splits persistence in two:

- **Core state** (graphs, path indexes, cost graph, posterior params,
  orthogonalization basis, outcome log reference): versioned pickle.
- **Pluggable components**: saved as a manifest of qualified names +
  configuration dicts. ``Engine.load(path, registry={...})`` takes a
  caller-supplied registry mapping names to factory functions. User
  closures save with a warning; loaded engine rebuilds without them.

Optional Arrow IPC export is available for cross-language interop on
the frozen state (graphs + posterior params).
"""

from kindling.persist.arrow import export_arrow
from kindling.persist.format import (
    SCHEMA_VERSION,
    EngineState,
    PluginManifest,
    load_engine,
    save_engine,
)

__all__ = [
    "SCHEMA_VERSION",
    "EngineState",
    "PluginManifest",
    "export_arrow",
    "load_engine",
    "save_engine",
]
