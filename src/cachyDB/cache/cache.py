"""Module for the Cache class in the cachyDB database.

This module provides a thread-safe LRU cache implementation for storing key-value pairs.
"""

import threading
from collections import OrderedDict
from typing import Any


class Cache:
    """Thread-safe Least Recently Used (LRU) Cache implementation.

    Stores the most frequently and recently accessed key-value pairs in memory.
    Provides O(1) time complexity for GET and PUT operations.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self.lock: threading.Lock = threading.Lock()

    def get(self, key: tuple[str, str]) -> Any | None:
        """Retrieves a value and marks it as recently used."""
        with self.lock:
            if key not in self.cache:
                return None
            # Move the accessed item to the end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: tuple[str, str], value: Any) -> None:
        """Inserts or updates a value, maintaining the LRU capacity limit."""
        with self.lock:
            if key in self.cache:
                # If updating, move to the end to mark as recently used
                self.cache.move_to_end(key)
            self.cache[key] = value
            # Enforce capacity limit by removing the least recently used item (from the beginning)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def invalidate(self, key: tuple[str, str]) -> None:
        """Removes a specific key from the cache (e.g., on deletion)."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self) -> None:
        """Clears all cached items."""
        with self.lock:
            self.cache.clear()
