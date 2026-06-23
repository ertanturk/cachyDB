"""Module for the Cache class in the cachyDB database.

This module provides a thread-safe LFU cache implementation for storing key-value pairs.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict
from typing import Any


class Cache:
    """Thread-safe Least Frequently Used (LFU) Cache implementation.

    Stores the most frequently accessed key-value pairs in memory.
    Provides O(1) time complexity for GET and PUT operations.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize the LFU Cache.

        Args:
            capacity: Maximum items the cache can hold.
        """
        self.capacity: int = capacity
        self.cache: dict[tuple[str, str], Any] = {}
        self.frequencies: dict[tuple[str, str], int] = {}
        self.freq_groups: dict[int, OrderedDict[tuple[str, str], None]] = defaultdict(OrderedDict)
        self.min_freq: int = 0
        self.lock: threading.Lock = threading.Lock()

    def get(self, key: tuple[str, str]) -> Any | None:
        """Retrieves a value and increments its access frequency."""
        with self.lock:
            if key not in self.cache:
                return None

            freq: int = self.frequencies[key]
            self.frequencies[key] = freq + 1

            del self.freq_groups[freq][key]
            if not self.freq_groups[freq] and self.min_freq == freq:
                self.min_freq += 1

            self.freq_groups[freq + 1][key] = None
            return self.cache[key]

    def put(self, key: tuple[str, str], value: Any) -> None:
        """Inserts or updates a value, maintaining the LFU capacity limit."""
        if self.capacity <= 0:
            return

        with self.lock:
            if key in self.cache:
                self.cache[key] = value

                freq: int = self.frequencies[key]
                self.frequencies[key] = freq + 1

                del self.freq_groups[freq][key]
                if not self.freq_groups[freq] and self.min_freq == freq:
                    self.min_freq += 1

                self.freq_groups[freq + 1][key] = None
                return

            if len(self.cache) >= self.capacity:
                evict_key, _ = self.freq_groups[self.min_freq].popitem(last=False)
                del self.cache[evict_key]
                del self.frequencies[evict_key]

            self.cache[key] = value
            self.frequencies[key] = 1
            self.min_freq = 1
            self.freq_groups[1][key] = None

    def invalidate(self, key: tuple[str, str]) -> None:
        """Removes a specific key from the cache (e.g., on deletion)."""
        with self.lock:
            if key in self.cache:
                freq: int = self.frequencies[key]
                del self.freq_groups[freq][key]
                del self.cache[key]
                del self.frequencies[key]

    def clear(self) -> None:
        """Clears all cached items."""
        with self.lock:
            self.cache.clear()
            self.frequencies.clear()
            self.freq_groups.clear()
            self.min_freq = 0
