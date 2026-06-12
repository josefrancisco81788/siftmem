"""Siftmem — append-only JSONL memory store with type-aware indexing and BM25 search."""

from siftmem.lib import IMPORTANCE_FLOORS
from siftmem.store import MemoryStore

__version__ = "0.2.0"
__all__ = ["MemoryStore", "IMPORTANCE_FLOORS", "__version__"]
