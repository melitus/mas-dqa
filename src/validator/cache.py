"""Async-safe caching utilities for Semantic Validator.

Provides LRU caching that works with async functions.
Falls back gracefully if optional dependencies are missing.

Reference: MAS-DQA Knowledge Base §4 (Performance Constraints)
"""
import hashlib
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AsyncCache:
    """Async-safe LRU cache with hash-based keys."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: list[str] = []
        
        # Try to use cachetools if available (better performance)
        try:
            from cachetools import LRUCache
            self._backend = LRUCache(maxsize=max_size)
            self._use_cachetools = True
            logger.info("Using cachetools for async caching")
        except ImportError:
            self._backend = None
            self._use_cachetools = False
            logger.warning("cachetools not installed; using fallback dict cache")

    @staticmethod
    def make_key(record: Dict, domain_context_dict: Dict, profiler_score: Optional[float]) -> str:
        """Create a deterministic hash key for caching."""
        raw = json.dumps({
            "record": record,
            "domain_context": domain_context_dict,
            "profiler_score": profiler_score,
        }, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if self._use_cachetools:
            return self._backend.get(key)
        
        if key in self._cache:
            # Update access order for LRU
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set item in cache with LRU eviction."""
        if self._use_cachetools:
            self._backend[key] = value
            return
        
        # Fallback dict-based LRU
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict oldest
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = value
        self._access_order.append(key)

    def __contains__(self, key: str) -> bool:
        return key in (self._backend if self._use_cachetools else self._cache)