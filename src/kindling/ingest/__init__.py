"""Input ingestion: schema validation, canonicalization, session inference."""

from kindling.ingest.contract import (
    InteractionSchema,
    canonicalize,
    validate_interactions,
)

__all__ = ["InteractionSchema", "canonicalize", "validate_interactions"]
