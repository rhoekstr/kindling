"""Stage 1 retrieval — candidate generation."""

from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.protocol import Candidate, RetrieverProtocol

__all__ = ["Candidate", "CoOccurrenceRetriever", "RetrieverProtocol"]
