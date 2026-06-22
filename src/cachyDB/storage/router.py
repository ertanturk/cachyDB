"""Module for the Router (Database Engine Core) in cachyDB.

This module orchestrates the data pipeline, routing requests between the
in-memory B-Tree (MemTable), the Write-Ahead Log (WAL) for durability,
and the on-disk storage (TableHandler/Pager).
"""

from __future__ import annotations

import json
import logging
import struct
import threading
from typing import TYPE_CHECKING

from cachyDB.cache.cache import Cache
from cachyDB.index.btree import BTreeMemTable
from cachyDB.storage.table_handler import TableHandler

if TYPE_CHECKING:
    from typing import Any

    from cachyDB.recovery.wal import WAL

# Maximum UTF-8 byte length for a key, constrained by the 2-byte length field.
_MAX_KEY_BYTES: int = 0xFFFF  # 65 535


class RouterError(Exception):
    """Base exception for Router operations."""


class Router:
    """Orchestrates database operations (GET, SET, DELETE) across modules.

    The Router manages a collection of in-memory B-Trees (one per table),
    ensures durability by communicating with the WAL, routes hot data through
    an LRU Cache, and triggers flushes to disk when MemTables reach their capacity.

    Attributes:
        wal (WAL): The Write-Ahead Log manager for durability.
        memtables (dict[str, BTreeMemTable]): Active in-memory trees per table.
        locks (dict[str, threading.RLock]): Table-specific concurrency locks.
        table_handlers (dict[str, TableHandler]): On-disk handlers per table.
        cache (Cache): Global LRU cache for fast read access.
        max_memtable_size (int): Maximum records in a MemTable before flushing.
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, wal_instance: WAL, max_memtable_size: int = 10000, cache_capacity: int = 5000) -> None:
        """Initializes the Router with a WAL instance and configuration.

        Args:
            wal_instance: The WAL manager used for durability.
            max_memtable_size: Number of records that triggers a MemTable
                flush to disk.  Must be >= 1.

        Raises:
            ValueError: If ``max_memtable_size`` is less than 1.
        """
        if max_memtable_size < 1:
            raise ValueError(f"max_memtable_size must be >= 1, got {max_memtable_size}.")
        if cache_capacity < 1:
            raise ValueError(f"cache_capacity must be >= 1, got {cache_capacity}.")

        self.wal: WAL = wal_instance
        self.memtables: dict[str, BTreeMemTable] = {}
        self.locks: dict[str, threading.RLock] = {}
        self.table_handlers: dict[str, TableHandler] = {}
        self.max_memtable_size: int = max_memtable_size
        self.cache: Cache = Cache(capacity=cache_capacity)
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Non-reentrant lock that guards only the dict initialisation step.
        # Never hold this lock while calling methods that also acquire it
        # (e.g., flush_memtable_to_disk → _get_table_state).
        self.global_lock: threading.Lock = threading.Lock()
        self.logger.info(
            "cachyDB Router initialized (MemTable Max: %d, Cache Cap: %d).",
            max_memtable_size,
            cache_capacity,
        )

    def _pack_record(self, key: str, value: Any) -> bytes:
        """Packs a key and value into a binary payload for WAL and Pager storage.

        Binary layout::

            [Key Length: 2 bytes, big-endian unsigned short]
            [Key: UTF-8 bytes, length = Key Length]
            [Value: JSON UTF-8 bytes (deterministic, sort_keys=True)]

        Args:
            key: The key to pack.
            value: The value to serialise as JSON.

        Returns:
            The packed binary payload.

        Raises:
            ValueError: If the UTF-8 encoding of ``key`` exceeds
                ``_MAX_KEY_BYTES`` (65 535 bytes).
            RouterError: If ``value`` is not JSON-serialisable.
        """
        k_bytes: bytes = key.encode("utf-8")
        k_len: int = len(k_bytes)
        if k_len > _MAX_KEY_BYTES:
            raise ValueError(
                f"Key UTF-8 encoding is {k_len} bytes, which exceeds the maximum of {_MAX_KEY_BYTES} bytes."
            )

        try:
            v_bytes: bytes = json.dumps(value, sort_keys=True).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RouterError(f"Value for key '{key}' is not JSON-serialisable: {exc}") from exc

        return struct.pack(">H", k_len) + k_bytes + v_bytes

    def _unpack_record(self, payload: bytes) -> tuple[str, Any]:
        """Unpacks a binary payload back into a ``(key, value)`` tuple.

        Args:
            payload: The raw bytes previously produced by :meth:`_pack_record`.

        Returns:
            A ``(key, value)`` tuple.

        Raises:
            ValueError: If the payload is too short, the key region is
                truncated, or the value region is not valid JSON.
        """
        if len(payload) < 2:
            raise ValueError(f"Payload too short to contain a key-length header: {len(payload)} byte(s).")
        k_len: int = struct.unpack(">H", payload[:2])[0]
        min_length: int = 2 + k_len + 1  # header + key + at least 1 byte of JSON
        if len(payload) < min_length:
            raise ValueError(
                f"Payload truncated: key claims {k_len} bytes but only "
                f"{len(payload) - 2} byte(s) remain after the header "
                f"(need {min_length} total)."
            )
        try:
            key: str = payload[2 : 2 + k_len].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Key bytes are not valid UTF-8: {exc}") from exc
        try:
            value: Any = json.loads(payload[2 + k_len :].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Value bytes are not valid JSON: {exc}") from exc
        return key, value

    def _get_table_state(self, table_name: str) -> tuple[BTreeMemTable, threading.RLock, TableHandler]:
        """Returns (or lazily initialises) the MemTable, lock, and handler for a table.

        Args:
            table_name: Name of the table whose state to retrieve.

        Returns:
            A ``(memtable, lock, handler)`` triple for ``table_name``.
        """
        with self.global_lock:
            if table_name not in self.memtables:
                self.memtables[table_name] = BTreeMemTable()
                self.locks[table_name] = threading.RLock()
                self.table_handlers[table_name] = TableHandler(table_name)
                self.logger.debug(
                    "Initialized new MemTable and TableHandler for table '%s'.",
                    table_name,
                )
            return (
                self.memtables[table_name],
                self.locks[table_name],
                self.table_handlers[table_name],
            )

    def set(self, table_name: str, key: str, value: Any) -> bool:
        """Writes a key-value pair to the database with WAL-backed durability.

        Pipeline:

        1. Validate inputs and pack the record into binary format.
        2. Append to the WAL — this is the durability point.
        3. Insert into the in-memory MemTable (B-Tree).
        4. Update the LRU Cache.
        5. Mark the WAL entry as committed.
        6. Trigger a MemTable flush to disk if capacity is reached.

        Args:
            table_name: Target table name.  Must be a non-empty string.
            key: Record key.  Must be a non-empty string.
            value: Value to store.  Must be JSON-serialisable.

        Returns:
            ``True`` on success.

        Raises:
            TypeError: If ``table_name`` or ``key`` are not strings.
            ValueError: If ``table_name`` or ``key`` are empty.
            RouterError: If the WAL write or MemTable insert fails.
        """
        if not isinstance(table_name, str) or not isinstance(key, str):
            raise TypeError("table_name and key must both be str.")
        if not table_name or not key:
            raise ValueError("table_name and key must not be empty.")

        memtable, table_lock, _ = self._get_table_state(table_name)

        with table_lock:
            # 1. Serialise the record to binary
            packed_data: bytes = self._pack_record(key, value)

            # 2. Write to WAL — guarantees durability even if the process crashes
            #    before the MemTable insert completes.
            try:
                tx_id: int = self.wal.append_to_buffer(packed_data)
            except Exception as exc:
                self.logger.error(
                    "WAL write failed (table='%s', key='%s'): %s",
                    table_name,
                    key,
                    exc,
                )
                raise RouterError(f"WAL operation failed: {exc}") from exc

            # 3. Insert into the in-memory B-Tree.  If this raises, the WAL
            #    entry is left uncommitted — recovery will replay it.
            try:
                memtable.insert(key, value)
            except Exception as exc:
                self.logger.error(
                    "MemTable insert failed for key '%s' in table '%s'. "
                    "WAL tx_id %d left uncommitted for recovery replay: %s",
                    key,
                    table_name,
                    tx_id,
                    exc,
                )
                raise RouterError(f"MemTable insert failed: {exc}") from exc

            self.logger.debug("Inserted key '%s' into MemTable '%s'.", key, table_name)

            # 4. Update the LRU Cache
            self.cache.put((table_name, key), value)

            # 5. Mark the WAL transaction as committed
            self.wal.mark_job_complete(tx_id)

            # 6. Flush to disk if the MemTable has reached capacity
            if memtable.size >= self.max_memtable_size:
                self.flush_memtable_to_disk(table_name)

            return True

    def get(self, table_name: str, key: str) -> Any | None:
        """Retrieves a value by key, checking Cache, MemTable, then disk.

        Args:
            table_name: Table to query.  Must be a non-empty string.
            key: Key to look up.  Must be a non-empty string.

        Returns:
            The associated value, or ``None`` if the key does not exist.

        Raises:
            TypeError: If ``table_name`` or ``key`` are not strings.
            ValueError: If ``table_name`` or ``key`` are empty.
        """
        if not isinstance(table_name, str) or not isinstance(key, str):
            raise TypeError("table_name and key must both be str.")
        if not table_name or not key:
            raise ValueError("table_name and key must not be empty.")

        cached_value = self.cache.get((table_name, key))
        if cached_value is not None:
            self.logger.debug("LRU Cache HIT for key '%s' in table '%s'.", key, table_name)
            return cached_value

        memtable, table_lock, _ = self._get_table_state(table_name)

        with table_lock:
            # 1. Check the in-memory MemTable first (fast path).
            #    Use contains() to correctly handle keys stored with value=None.
            if memtable.contains(key):
                self.logger.debug("Cache hit for key '%s' in table '%s'.", key, table_name)
                val = memtable.search(key)
                self.cache.put((table_name, key), val)
                return val

            # 2. Key is not in RAM — fall back to a linear disk scan.
            self.logger.debug(
                "Cache/MemTable MISS for key '%s' in table '%s'. Scanning disk...",
                key,
                table_name,
            )
            disk_result = self._read_from_disk(table_name, key)
            if disk_result is not None:
                self.cache.put((table_name, key), disk_result)

            return disk_result

    def _read_from_disk(self, table_name: str, key: str) -> Any | None:
        """Scans on-disk pages to find the most recent record for ``key``.

        Pages are scanned newest-first (highest page ID to 0) and slots are
        scanned newest-first within each page, so the first match found is
        the most recently written version.

        Note:
            This is a linear O(n) scan.  A dedicated on-disk B-Tree index
            is not yet implemented.

        Args:
            table_name: Table to scan.
            key: Key to search for.

        Returns:
            The most recently stored value, or ``None`` if not found.
        """
        _, _, handler = self._get_table_state(table_name)
        total_pages: int = handler.total_pages

        if total_pages == 0:
            return None

        for page_id in range(total_pages - 1, -1, -1):
            try:
                page = handler.read_page(page_id)
            except Exception as exc:
                self.logger.error(
                    "Error reading page %d from table '%s': %s",
                    page_id,
                    table_name,
                    exc,
                )
                continue

            for slot_id in range(len(page.slots) - 1, -1, -1):
                payload: bytes | None = page.get_record(slot_id)
                if payload is not None:
                    try:
                        p_key: str
                        p_val: Any
                        p_key, p_val = self._unpack_record(payload)
                        if p_key == key:
                            return p_val
                    except Exception as exc:
                        self.logger.warning(
                            "Error unpacking record on page %d, slot %d in table '%s': %s",
                            page_id,
                            slot_id,
                            table_name,
                            exc,
                        )
                        continue

        return None

    def flush_memtable_to_disk(self, table_name: str) -> None:
        """Flushes the MemTable for ``table_name`` to the on-disk TableHandler.

        Extracts all records from the B-Tree in ascending key order, packs
        them, fills existing pages before allocating new ones, and clears
        the MemTable only after all writes succeed.

        Args:
            table_name: Name of the table whose MemTable to flush.
        """
        memtable, table_lock, handler = self._get_table_state(table_name)

        with table_lock:
            if memtable.size == 0:
                return

            self.logger.info(
                "Flushing MemTable '%s' (%d record(s)) to disk...",
                table_name,
                memtable.size,
            )

            # 1. Collect all records in sorted key order before modifying anything
            sorted_records: list[tuple[str, Any]] = list(memtable.in_order_traversal())

            # 2. Position on the last existing page, or allocate the first one
            if handler.total_pages == 0:
                current_page_id: int = handler.allocate_page()
            else:
                current_page_id = handler.total_pages - 1

            page = handler.read_page(current_page_id)
            is_page_dirty: bool = False

            # 3. Write each record, spilling to a new page when the current one is full
            for rec_key, rec_value in sorted_records:
                packed_record: bytes = self._pack_record(rec_key, rec_value)

                if not page.has_space_for(len(packed_record)):
                    # Persist the current page before moving on
                    if is_page_dirty:
                        handler.write_page(page)

                    current_page_id = handler.allocate_page()
                    page = handler.read_page(current_page_id)
                    is_page_dirty = False

                page.insert_record(packed_record)
                is_page_dirty = True

            # 4. Persist the final (possibly partially filled) page
            if is_page_dirty:
                handler.write_page(page)

            # 5. Clear the MemTable only after all writes have succeeded
            memtable.clear()
            self.logger.info("MemTable '%s' flushed and cleared successfully.", table_name)

    def shutdown(self) -> None:
        """Safely shuts down the router, flushing all MemTables and closing files.

        Table names are collected under ``global_lock`` and then each table is
        flushed *outside* the lock to avoid a deadlock: ``flush_memtable_to_disk``
        calls ``_get_table_state`` which must acquire ``global_lock`` itself.
        """
        # Snapshot table names atomically, then release global_lock before
        # flushing — holding global_lock while flushing would deadlock because
        # flush_memtable_to_disk → _get_table_state also acquires global_lock.
        with self.global_lock:
            table_names: list[str] = list(self.memtables.keys())

        for table_name in table_names:
            try:
                self.flush_memtable_to_disk(table_name)
            except Exception as exc:
                self.logger.error(
                    "Failed to flush table '%s' during shutdown: %s",
                    table_name,
                    exc,
                )
            handler: TableHandler | None = self.table_handlers.get(table_name)
            if handler is not None:
                handler.close()

        self.cache.clear()
        self.logger.info("Router shut down successfully.")
