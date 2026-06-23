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
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from cachyDB.cache.cache import Cache
from cachyDB.index.btree import BTreeMemTable
from cachyDB.logger.structured_logger import OperationType, get_structured_logger
from cachyDB.storage.table_handler import TableHandler

if TYPE_CHECKING:
    from typing import Any

    from cachyDB.pager.pager import Pager
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

    def __init__(self, wal_instance: WAL, max_memtable_size: int = 20000, cache_capacity: int = 5000) -> None:
        """Initializes the Router with a WAL instance and configuration.

        Args:
            wal_instance: The WAL manager used for durability.
            max_memtable_size: Number of records that triggers a MemTable
                flush to disk.  Must be >= 1.
            cache_capacity: Maximum items in the LRU cache.

        Raises:
            ValueError: If ``max_memtable_size`` or ``cache_capacity`` is less than 1.
        """
        if max_memtable_size < 1:
            raise ValueError(f"max_memtable_size must be >= 1, got {max_memtable_size}.")
        if cache_capacity < 1:
            raise ValueError(f"cache_capacity must be >= 1, got {cache_capacity}.")

        self.wal: WAL = wal_instance
        self.active_memtables: dict[str, BTreeMemTable] = {}
        self.immutable_memtables: dict[str, BTreeMemTable | None] = {}
        # Add these below self.immutable_memtables
        self.active_tx_ids: dict[str, list[int]] = {}
        self.immutable_tx_ids: dict[str, list[int] | None] = {}
        self.locks: dict[str, threading.RLock] = {}
        self.table_handlers: dict[str, TableHandler] = {}
        self.disk_index: dict[str, dict[str, tuple[int, int]]] = {}
        self.max_memtable_size: int = max_memtable_size
        self.cache: Cache = Cache(capacity=cache_capacity)
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.structured_logger = get_structured_logger("router")
        # Non-reentrant lock that guards only the table initialization step.
        self.global_lock: threading.RLock = threading.RLock()
        self.flush_condition: threading.Condition = threading.Condition(self.global_lock)
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
        self.flush_futures: dict[str, Future[None]] = {}
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
            v_bytes: bytes = json.dumps(value).encode("utf-8")
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

    def _get_page_slot_record(self, page: Pager, slot_id: int) -> tuple[str, Any] | None:
        """Helper to get and cache unpacked (key, value) for a page slot."""
        if slot_id in page.cached_metadata:
            return page.cached_metadata[slot_id]

        payload = page.get_record(slot_id)
        if payload is None:
            page.cached_metadata[slot_id] = None
            return None

        try:
            unpacked = self._unpack_record(payload)
            page.cached_metadata[slot_id] = unpacked
            return unpacked
        except Exception:
            page.cached_metadata[slot_id] = None
            return None

    def set_many(self, table_name: str, records: dict[str, Any]) -> bool:
        """Writes multiple key-value pairs to the database efficiently.

        Args:
            table_name: Target table name. Must be a non-empty string.
            records: Dictionary of key-value pairs to store.

        Returns:
            True on success.

        Raises:
            TypeError: If table_name is not a string or records is not a dict.
            ValueError: If table_name or records are empty.
        """
        if type(table_name) is not str:
            raise TypeError("table_name must be a str.")
        if type(records) is not dict:
            raise TypeError("records must be a dict.")
        if not table_name or not records:
            raise ValueError("table_name and records must not be empty.")

        return self._set_many_internal(table_name, records)

    def _set_many_internal(self, table_name: str, records: dict[str, Any]) -> bool:
        """Executes a bulk write efficiently under a single lock.

        Args:
            table_name: Target table name.
            records: Dictionary of key-value pairs to store.

        Returns:
            True on success.

        Raises:
            RouterError: If WAL or MemTable operations fail.
        """
        trace_id: str = ""
        start_time: float = time.perf_counter()

        # 1. Pre-pack all records outside the lock
        packed_records: list[tuple[str, bytes]] = []
        payload_list: list[bytes] = []
        total_data_size: int = 0

        for key, value in records.items():
            packed_data: bytes = self._pack_record(key, value)
            packed_records.append((key, packed_data))
            payload_list.append(packed_data)
            total_data_size += len(packed_data)

        records_count: int = len(packed_records)

        # 2. Batch write to the WAL buffer natively (1 tx_id per record)
        try:
            tx_ids: list[int] = self.wal.append_many_to_buffer(payload_list)
        except Exception as exc:
            self.logger.error("Bulk WAL write failed: %s", exc)
            raise RouterError(f"Bulk WAL operation failed: {exc}") from exc

        table_lock, _ = self._get_table_state(table_name)

        with table_lock:
            active_memtable: BTreeMemTable = self.active_memtables[table_name]

            # 3. Batch insert into the B-Tree
            try:
                for key, packed_data in packed_records:
                    active_memtable.insert(key, packed_data)
                    self.cache.put((table_name, key), records[key])
                self.active_tx_ids[table_name].extend(tx_ids)
            except Exception as exc:
                self.logger.error("Bulk MemTable insert failed: %s", exc)
                raise RouterError(f"Bulk MemTable insert failed: {exc}") from exc

            needs_flush: bool = active_memtable.size >= self.max_memtable_size

        # 5. Check capacity and trigger flush
        if needs_flush:
            future: Future[None] | None = self.flush_futures.get(table_name)
            if future is not None and not future.done():
                future.result()

            with table_lock:
                if self.active_memtables[table_name].size >= self.max_memtable_size:
                    self.immutable_memtables[table_name] = self.active_memtables[table_name]
                    self.immutable_tx_ids[table_name] = self.active_tx_ids[table_name]
                    self.active_memtables[table_name] = BTreeMemTable()
                    self.active_tx_ids[table_name] = []

                    with self.global_lock:
                        self.flush_futures[table_name] = self.executor.submit(self.flush_memtable_to_disk, table_name)

        if self.logger.isEnabledFor(logging.DEBUG):
            latency_ms: float = (time.perf_counter() - start_time) * 1000
            self.structured_logger.log_write_operation(
                trace_id=trace_id,
                operation_type=OperationType.SET,
                table_name=table_name,
                records_count=records_count,
                data_size_bytes=total_data_size,
                latency_ms=latency_ms,
                error_code=0,
            )

        return True

    def _init_disk_index(self, table_name: str, handler: TableHandler) -> dict[str, tuple[int, int]]:
        index: dict[str, tuple[int, int]] = {}
        total_pages = handler.total_pages
        for page_id in range(total_pages):
            try:
                page = handler.read_page(page_id)
            except Exception:
                continue
            for slot_id in range(len(page.slots)):
                payload = page.get_record(slot_id)
                if payload is not None:
                    try:
                        if len(payload) >= 2:
                            k_len = struct.unpack(">H", payload[:2])[0]
                            k = payload[2 : 2 + k_len].decode("utf-8")
                            if payload[2 + k_len :] == b"null":
                                index.pop(k, None)
                            else:
                                index[k] = (page_id, slot_id)
                    except Exception:
                        pass
        return index

    def _get_table_state(self, table_name: str) -> tuple[threading.RLock, TableHandler]:
        """Returns (or lazily initialises) the lock and handler for a table.

        Optimized to use a double-checked locking pattern to minimize global lock contention.

        Args:
            table_name: Name of the table whose state to retrieve.

        Returns:
            A ``(lock, handler)`` tuple for ``table_name``.
        """
        if table_name in self.active_memtables:
            return self.locks[table_name], self.table_handlers[table_name]

        with self.global_lock:
            if table_name not in self.active_memtables:
                self.active_memtables[table_name] = BTreeMemTable()
                self.immutable_memtables[table_name] = None
                self.active_tx_ids[table_name] = []
                self.immutable_tx_ids[table_name] = None
                self.locks[table_name] = threading.RLock()
                self.table_handlers[table_name] = handler = TableHandler(table_name)
                self.disk_index[table_name] = self._init_disk_index(table_name, handler)
                self.logger.info(
                    "Initialized active and immutable MemTables for table '%s'.",
                    table_name,
                )
            return self.locks[table_name], self.table_handlers[table_name]

    def set(self, table_name: str, key: str, value: Any) -> bool:
        """Writes a key-value pair to the database.

        If the value is large, it is automatically compressed and chunked into
        smaller records to ensure that it fits within the page-size limits on disk.

        Args:
            table_name: Target table name. Must be a non-empty string.
            key: Record key. Must be a non-empty string.
            value: Value to store. Must be JSON-serialisable.

        Returns:
            True on success.

        Raises:
            TypeError: If table_name or key are not strings.
            ValueError: If table_name or key are empty.
        """
        if not isinstance(table_name, str) or not isinstance(key, str):
            raise TypeError("table_name and key must both be str.")
        if not table_name or not key:
            raise ValueError("table_name and key must not be empty.")

        # Chunking mechanism for huge records
        is_large = False
        if not key.startswith("__chunk__:") and value is not None:
            import base64
            import zlib

            try:
                value_json_str: str = json.dumps(value)
                value_bytes: bytes = value_json_str.encode("utf-8")
                if len(value_bytes) > 2000:
                    is_large = True
                    # Clean up old chunks first if they existed
                    try:
                        old_val: Any = self._get_raw_manifest(table_name, key)
                        if isinstance(old_val, dict) and old_val.get("__cachy_overflow__") is True:
                            old_num_chunks: int = old_val["num_chunks"]
                            for i in range(old_num_chunks):
                                self._set_internal(table_name, f"__chunk__:{key}:{i}", None)
                    except Exception as old_exc:
                        self.logger.debug("No old chunks to clean up or failed to read them: %s", old_exc)

                    # Compress the serialized bytes
                    compressed_bytes: bytes = zlib.compress(value_bytes)
                    chunk_size: int = 2000
                    chunks: list[bytes] = [
                        compressed_bytes[i : i + chunk_size] for i in range(0, len(compressed_bytes), chunk_size)
                    ]

                    for i, chunk in enumerate(chunks):
                        chunk_b64: str = base64.b64encode(chunk).decode("utf-8")
                        self._set_internal(table_name, f"__chunk__:{key}:{i}", chunk_b64)

                    manifest: dict[str, Any] = {
                        "__cachy_overflow__": True,
                        "num_chunks": len(chunks),
                        "original_size": len(value_bytes),
                        "compressed": True,
                    }

                    # Write the manifest
                    res: bool = self._set_internal(table_name, key, manifest)
                    # Force original value into cache instead of the manifest dictionary
                    self.cache.put((table_name, key), value)
                    return res
            except Exception as e:
                self.logger.warning("Failed to chunk large record for key '%s': %s", key, e)

        # Deletion or small-value update: clean up chunks if the record was previously large
        if not key.startswith("__chunk__:") and not is_large:
            try:
                old_val_del: Any = self._get_raw_manifest(table_name, key)
                if isinstance(old_val_del, dict) and old_val_del.get("__cachy_overflow__") is True:
                    old_num_chunks_del: int = old_val_del["num_chunks"]
                    for i in range(old_num_chunks_del):
                        self._set_internal(table_name, f"__chunk__:{key}:{i}", None)
            except Exception as del_exc:
                self.logger.debug("Failed to clean up old chunks during cleanup: %s", del_exc)

        return self._set_internal(table_name, key, value)

    def _set_internal(self, table_name: str, key: str, value: Any) -> bool:
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

        trace_id: str = ""
        start_time: float = time.perf_counter()
        packed_data: bytes = self._pack_record(key, value)
        data_size: int = len(packed_data)
        table_lock, _ = self._get_table_state(table_name)

        with table_lock:
            # 1. Serialise the record to binary
            active_memtable: BTreeMemTable = self.active_memtables[table_name]
            packed_data: bytes = self._pack_record(key, value)
            data_size: int = len(packed_data)

            # 2. Write to WAL
            try:
                tx_id: int = self.wal.append_to_buffer(packed_data)
            except Exception as exc:
                self.logger.error(
                    "WAL write failed (table='%s', key='%s'): %s",
                    table_name,
                    key,
                    exc,
                )
                latency_ms: float = (time.perf_counter() - start_time) * 1000
                self.structured_logger.log_write_operation(
                    trace_id=trace_id,
                    operation_type=OperationType.SET,
                    table_name=table_name,
                    records_count=1,
                    data_size_bytes=data_size,
                    latency_ms=latency_ms,
                    error_code=1,
                    error_message=f"WAL operation failed: {exc}",
                )
                raise RouterError(f"WAL operation failed: {exc}") from exc

            # 3. Insert into the in-memory B-Tree.
            try:
                active_memtable.insert(key, packed_data)
                self.active_tx_ids[table_name].append(tx_id)
            except Exception as exc:
                self.logger.error(
                    "MemTable insert failed for key '%s' in table '%s'. WAL tx_id %d left uncommitted: %s",
                    key,
                    table_name,
                    tx_id,
                    exc,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.structured_logger.log_write_operation(
                    trace_id=trace_id,
                    operation_type=OperationType.SET,
                    table_name=table_name,
                    records_count=1,
                    data_size_bytes=data_size,
                    latency_ms=latency_ms,
                    error_code=2,
                    error_message=f"MemTable insert failed: {exc}",
                )
                raise RouterError(f"MemTable insert failed: {exc}") from exc

            # Log progress every 1,000 inserts to avoid spamming the log
            if active_memtable.size % 1000 == 0:
                self.logger.debug("Progress: Processed %d records in table '%s'.", active_memtable.size, table_name)

            # 4. Update the LRU Cache
            self.cache.put((table_name, key), value)

            needs_flush: bool = active_memtable.size >= self.max_memtable_size

        # 6. Apply backpressure and trigger flush OUTSIDE the table lock to prevent deadlocks
        if needs_flush:
            future = self.flush_futures.get(table_name)
            if future is not None and not future.done():
                # Block incoming writes until the active disk flush completes
                future.result()

            with table_lock:
                if self.active_memtables[table_name].size >= self.max_memtable_size:
                    self.immutable_memtables[table_name] = self.active_memtables[table_name]
                    self.immutable_tx_ids[table_name] = self.active_tx_ids[table_name]
                    self.active_memtables[table_name] = BTreeMemTable()
                    self.active_tx_ids[table_name] = []

                    with self.global_lock:
                        self.flush_futures[table_name] = self.executor.submit(self.flush_memtable_to_disk, table_name)

        # Log successful SET operation with structured logging
        if self.logger.isEnabledFor(logging.DEBUG):
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.structured_logger.log_write_operation(
                trace_id=trace_id,
                operation_type=OperationType.SET,
                table_name=table_name,
                records_count=1,
                data_size_bytes=data_size,
                latency_ms=latency_ms,
                error_code=0,
            )

        return True

    def get(self, table_name: str, key: str) -> Any | None:
        """Retrieves a value by key, checking Cache, MemTable, then disk.

        If the retrieved value is an overflow record, its chunks are automatically
        reconstructed, decompressed, and deserialised.

        Args:
            table_name: Table to query. Must be a non-empty string.
            key: Key to look up. Must be a non-empty string.

        Returns:
            The associated value, or None if the key does not exist.

        Raises:
            TypeError: If table_name or key are not strings.
            ValueError: If table_name or key are empty.
        """
        if not isinstance(table_name, str) or not isinstance(key, str):
            raise TypeError("table_name and key must both be str.")
        if not table_name or not key:
            raise ValueError("table_name and key must not be empty.")

        val: Any | None = self._get_internal(table_name, key)

        if isinstance(val, dict) and val.get("__cachy_overflow__") is True:
            import base64
            import zlib

            try:
                num_chunks: int = val["num_chunks"]
                chunks_data: list[bytes] = []
                for i in range(num_chunks):
                    chunk_key: str = f"__chunk__:{key}:{i}"
                    chunk_val: Any | None = self._get_internal(table_name, chunk_key)
                    if chunk_val is None:
                        self.logger.warning("Missing chunk '%s' for large record.", chunk_key)
                        return None
                    chunks_data.append(base64.b64decode(chunk_val.encode("utf-8")))

                compressed_bytes: bytes = b"".join(chunks_data)
                if val.get("compressed") is True:
                    original_bytes: bytes = zlib.decompress(compressed_bytes)
                else:
                    original_bytes = compressed_bytes

                original_val: Any = json.loads(original_bytes.decode("utf-8"))
                # Cache the fully reconstructed value for future fast gets
                self.cache.put((table_name, key), original_val)
                return original_val
            except Exception as e:
                self.logger.error("Failed to reconstruct chunked record for key '%s': %s", key, e)
                return None

        return val

    def _get_raw_manifest(self, table_name: str, key: str) -> Any | None:
        """Retrieves raw manifest from MemTable or disk, bypassing the LRU cache.

        Args:
            table_name: Target table name.
            key: Target key.

        Returns:
            The raw stored value (e.g. manifest dictionary) or None.
        """
        table_lock, _ = self._get_table_state(table_name)
        with table_lock:
            val = None
            active_memtable: BTreeMemTable = self.active_memtables[table_name]
            immutable_memtable: BTreeMemTable | None = self.immutable_memtables[table_name]

            if active_memtable.contains(key):
                val = active_memtable.search(key)
            elif immutable_memtable is not None and immutable_memtable.contains(key):
                val = immutable_memtable.search(key)
            else:
                return self._read_from_disk(table_name, key)

            if isinstance(val, bytes):
                try:
                    _, unpacked_val = self._unpack_record(val)
                    return unpacked_val
                except Exception as exc:
                    self.logger.error("Failed to unpack raw manifest for key '%s': %s", key, exc)
                    return None
            return val

    def _get_internal(self, table_name: str, key: str) -> Any | None:
        """Retrieves a value by key, checking Cache, MemTable, then disk.

        Args:
            table_name: Table to query.  Must be a non-empty string.
            key: Key to look up.  Must be a non-empty string.

        Returns:
            The associated value, or ``None`` if the key does not exist.
        """
        if not isinstance(table_name, str) or not isinstance(key, str):
            raise TypeError("table_name and key must both be str.")
        if not table_name or not key:
            raise ValueError("table_name and key must not be empty.")

        trace_id: str = ""
        start_time: float = time.perf_counter()

        cached_value = self.cache.get((table_name, key))
        if cached_value is not None:
            # Log the cache hit with structured logging
            latency_ms: float = (time.perf_counter() - start_time) * 1000
            self.structured_logger.log_cache_operation(
                trace_id=trace_id,
                hit=True,
                key=key,
                table_name=table_name,
                latency_ms=latency_ms,
            )
            self.logger.debug("Cache HIT for key '%s' in table '%s'.", key, table_name)
            return cached_value

        table_lock, _ = self._get_table_state(table_name)

        with table_lock:
            active_memtable: BTreeMemTable = self.active_memtables[table_name]
            immutable_memtable: BTreeMemTable | None = self.immutable_memtables[table_name]
            # 1. Check the in-memory MemTable first (fast path).
            if active_memtable.contains(key):
                packed_val = active_memtable.search(key)
                if isinstance(packed_val, bytes):
                    _, val = self._unpack_record(packed_val)
                else:
                    val = packed_val

                self.cache.put((table_name, key), val)
                return val

            if immutable_memtable is not None and immutable_memtable.contains(key):
                packed_val = immutable_memtable.search(key)
                if isinstance(packed_val, bytes):
                    _, val = self._unpack_record(packed_val)
                else:
                    val = packed_val

                if val is not None:
                    self.cache.put((table_name, key), val)
                return val

            # 2. Key is not in RAM — fall back to disk.
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
        """Reads a record from disk using the in-memory disk index for O(1) reads."""
        _, handler = self._get_table_state(table_name)

        index = self.disk_index.get(table_name)
        if index is not None:
            if key not in index:
                return None

            page_id, slot_id = index[key]
            try:
                page = handler.read_page(page_id)
                record = self._get_page_slot_record(page, slot_id)
                if record is not None:
                    return record[1]
            except Exception as exc:
                self.logger.error("Error reading page %d from table '%s': %s", page_id, table_name, exc)

        # Fallback if index fails or is unavailable
        total_pages = handler.total_pages
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
                record = self._get_page_slot_record(page, slot_id)
                if record is not None:
                    p_key, p_val = record
                    if p_key == key:
                        return p_val

        return None

    def flush_memtable_to_disk(self, table_name: str) -> None:
        """Flushes the MemTable for ``table_name`` to disk."""

        trace_id: str = ""
        start_time: float = time.perf_counter()

        table_lock, handler = self._get_table_state(table_name)

        records_count: int = 0
        total_data_size: int = 0

        try:
            with table_lock:
                immutable_memtable: BTreeMemTable | None = self.immutable_memtables.get(table_name)
                tx_ids_to_mark = self.immutable_tx_ids.get(table_name, [])

                if immutable_memtable is None or immutable_memtable.size == 0:
                    return

                records_count = immutable_memtable.size
                self.logger.info(
                    "Flushing MemTable '%s' (%d record(s)) to disk...",
                    table_name,
                    records_count,
                )
                records: tuple[tuple[str, Any], ...] = tuple(immutable_memtable.in_order_traversal())

            flush_lock = getattr(self.wal, "flush_lock", None)

            if flush_lock is not None:
                with flush_lock:
                    if hasattr(self.wal, "active_flushes"):
                        self.wal.active_flushes += 1

            current_page_id = handler.allocate_page() if handler.total_pages == 0 else handler.total_pages - 1
            page = handler.read_page(current_page_id)

            pending_updates: dict[str, tuple[int, int]] = {}
            pending_deletions: list[str] = []

            page_dirty = False

            for rec_key, packed_record in records:
                record_size = len(packed_record)
                total_data_size += record_size

                if not page.has_space_for(record_size):
                    if page_dirty:
                        handler.write_page(page)

                    current_page_id = handler.allocate_page()
                    page = handler.read_page(current_page_id)

                    page_dirty = False

                slot_id = page.insert_record(packed_record)

                key_len = len(rec_key.encode("utf-8"))

                is_deletion = packed_record[2 + key_len :] == b"null"

                if is_deletion:
                    pending_deletions.append(rec_key)
                else:
                    pending_updates[rec_key] = (
                        current_page_id,
                        slot_id,
                    )

                page_dirty = True

            if page_dirty:
                handler.write_page(page)

            disk_index = self.disk_index[table_name]

            for key in pending_deletions:
                disk_index.pop(key, None)

            disk_index.update(pending_updates)
            handler.sync()

            if tx_ids_to_mark:
                self.wal.mark_many_jobs_complete(tx_ids_to_mark)

            with table_lock:
                self.immutable_memtables[table_name] = None
                self.immutable_tx_ids[table_name] = None

            latency_ms = (time.perf_counter() - start_time) * 1000
            throughput_kops = (records_count / latency_ms) if latency_ms > 0 else 0.0
            self.structured_logger.log_write_operation(
                trace_id=trace_id,
                operation_type=OperationType.FLUSH,
                table_name=table_name,
                records_count=records_count,
                data_size_bytes=total_data_size,
                latency_ms=latency_ms,
                throughput_kops=throughput_kops,
                error_code=0,
            )

            self.logger.info(
                "MemTable '%s' flushed successfully.",
                table_name,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self.structured_logger.log_write_operation(
                trace_id=trace_id,
                operation_type=OperationType.FLUSH,
                table_name=table_name,
                records_count=records_count,
                data_size_bytes=0,
                latency_ms=latency_ms,
                error_code=1,
                error_message=f"Flush operation failed: {exc}",
            )

            self.logger.error(
                "Failed to flush MemTable '%s': %s",
                table_name,
                exc,
            )
            raise
        finally:
            flush_lock = getattr(
                self.wal,
                "flush_lock",
                None,
            )
            if flush_lock is not None:
                with flush_lock:
                    if hasattr(self.wal, "active_flushes"):
                        self.wal.active_flushes -= 1
            with self.flush_condition:
                self.flush_condition.notify_all()

    def shutdown(self) -> None:
        """Safely shuts down the router."""
        # 1. Wait for any active background flushes to complete before closing files
        for future in list(self.flush_futures.values()):
            try:
                future.result()
            except Exception:
                pass

        with self.global_lock:
            table_names: list[str] = list(self.immutable_memtables.keys())

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

        self.executor.shutdown(wait=True)
        if self.wal:
            self.wal.shutdown()
        self.cache.clear()
        self.logger.info("Router shut down successfully.")
