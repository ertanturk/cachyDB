"""Main entry point and Facade API for the cachyDB database engine.

This module unifies the underlying WAL, Router, Pager, and Compactor layers
behind a single high-level interface, enabling users to perform read and
write operations directly without manual component configuration.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
import uuid
from typing import Any

from cachyDB.garbage.compactor import Compactor
from cachyDB.logger.log_handler import LogHandler
from cachyDB.logger.structured_logger import get_structured_logger
from cachyDB.recovery.checkpoint import WALCheckpoint
from cachyDB.recovery.wal import WAL
from cachyDB.storage.router import Router


class CachyDB:
    """Main management and coordination facade for cachyDB.

    Fuses underlying WAL, Router, Pager, and Compactor layers into a unified,
    high-level API. This class automatically orchestrates background execution
    threads, transaction logging, and locking mechanisms.

    Attributes:
        max_memtable_size (int): The maximum record capacity threshold for an in-memory MemTable.
        cache_capacity (int): The maximum entry capacity limit for the LRU cache.
    """

    max_memtable_size: int
    cache_capacity: int
    _wal: WAL | None
    _router: Router | None
    _checkpoint: WALCheckpoint | None
    _compactor: Compactor | None
    _lock: threading.Lock
    _logger: logging.Logger | None
    _structured_logger: logging.Logger | None

    def __init__(self, max_memtable_size: int = 20000, cache_capacity: int = 5000) -> None:
        """Initializes the CachyDB instance facade.

        Args:
            max_memtable_size: Capacity limit of the MemTable before flushing to disk.
            cache_capacity: Element count threshold for the LRU Cache layer.
        """
        self.max_memtable_size = max_memtable_size
        self.cache_capacity = cache_capacity

        self._wal = None
        self._router = None
        self._checkpoint = None
        self._compactor = None

        self._lock = threading.Lock()
        self._logger = None
        self._structured_logger = get_structured_logger("database")  # type: ignore

        # Automatically register graceful shutdown tasks on runtime exit
        atexit.register(self.close)

    def _ensure_initialized(self) -> None:
        """Safely and lazily initializes database engine sub-layers using double-checked locking.

        Raises:
            RuntimeError: If any sub-layer component fails to initialize or start up.
        """
        if self._router is not None:
            return

        str(uuid.uuid4())[:8]
        time.perf_counter()

        with self._lock:
            # Double-checked locking validation step
            if self._router is not None:
                return

            LogHandler()
            logger: logging.Logger = logging.getLogger("cachyDB")
            logger.info("Initializing cachyDB sub-layers lazily...")

            try:
                # Build everything out in local variables to protect atomicity
                wal_inst: WAL = WAL()
                router_inst: Router = Router(
                    wal_instance=wal_inst,
                    max_memtable_size=self.max_memtable_size,
                    cache_capacity=self.cache_capacity,
                )
                checkpoint_inst: WALCheckpoint = WALCheckpoint(wal_inst)
                checkpoint_inst.start_thread()

                compactor_inst: Compactor = Compactor(router_instance=router_inst)
                compactor_inst.start_background()

                # Atomically expose objects only after entire configuration pipeline succeeds
                self._wal = wal_inst
                self._checkpoint = checkpoint_inst
                self._compactor = compactor_inst
                self._logger = logger
                self._router = router_inst

                logger.info("cachyDB sub-layers initialized successfully.")

                # Trigger the automatic transaction recovery phase
                self._recover_transactions()

            except Exception as exc:
                logger.error("Failed to initialize cachyDB sub-layers: %s", exc, exc_info=True)
                self._router = None
                self._wal = None
                self._checkpoint = None
                self._compactor = None
                raise RuntimeError("Database initialization sequence failed.") from exc

    def _recover_transactions(self) -> None:
        """Recovers and replays uncommitted or partial transactions found in the WAL logs."""
        if self._wal is None or self._router is None or self._logger is None:
            return

        trace_id: str = str(uuid.uuid4())[:8]
        start_time: float = time.perf_counter()
        self._logger.info("Starting automatic transaction recovery via WAL logs...")
        recovered_count: int = 0

        try:
            # Read incomplete transaction items from the WAL file
            tx_id: int
            packed_record: bytes
            for tx_id, packed_record in self._wal.read_wal():
                try:
                    # Unpack the raw transaction binary payload
                    key, value = self._router._unpack_record(packed_record)

                    # Parse namespace prefix to support multi-table partitioning (e.g., "users:user_1")
                    table_name: str = "default"
                    if ":" in key:
                        parts: list[str] = key.split(":", 1)
                        if parts[0]:
                            table_name = parts[0]

                    # Retrieve targeted table metadata and recover state into memory
                    table_lock, _ = self._router._get_table_state(table_name)
                    with table_lock:
                        self._router.active_memtables[table_name].insert(key, packed_record)

                    # Mark the transaction job as completed in the log
                    self._wal.mark_job_complete(tx_id, self._wal.get_file_path(tx_id))
                    recovered_count += 1
                    self._logger.info(
                        "Transaction successfully recovered (Replayed tx_id=%d, Table=%s): %s",
                        tx_id,
                        table_name,
                        key,
                    )
                except Exception as record_exc:
                    self._logger.error("Failed to recover transaction tx_id=%d: %s", tx_id, record_exc)
        except Exception as exc:
            self._logger.error("Critical error encountered during transaction recovery phase: %s", exc)

        # Log recovery operation with structured logging
        latency_ms: float = (time.perf_counter() - start_time) * 1000
        self._structured_logger.log_recovery_operation(  # type: ignore
            trace_id=trace_id,
            records_recovered=recovered_count,
            latency_ms=latency_ms,
            error_code=0,
        )

        if recovered_count > 0:
            self._logger.info(
                "Transaction recovery completed successfully. Replayed %d transaction(s).", recovered_count
            )
        else:
            self._logger.info("No incomplete or pending transactions found to recover.")

    def set(self, table: str, key: str, value: Any) -> bool:
        """Writes a key-value record to the specified database table.

        Args:
            table: Target database table name. Must not be empty.
            key: Unique identifier key for the record. Must not be empty.
            value: Arbitrary value payload compatible with JSON serialization.

        Returns:
            True if the write operation completes successfully.

        Raises:
            TypeError: If table or key is not a string.
            ValueError: If table or key is empty.
        """
        if not isinstance(table, str) or not isinstance(key, str):
            raise TypeError("Table name and key parameters must be strings.")
        if not table or not key:
            raise ValueError("Table name and key parameters must not be empty.")

        self._ensure_initialized()
        assert self._router is not None

        # Automatically prepend the table name as a namespace for recovery tracking stability
        namespaced_key: str = f"{table}:{key}"
        return self._router.set(table, namespaced_key, value)

    def set_many(self, table: str, records: dict[str, Any]) -> bool:
        """Writes multiple key-value records to the specified database table.

        Args:
            table: Target database table name. Must not be empty.
            records: Dictionary of key-value pairs to store. Must not be empty.

        Returns:
            True if the batch write operation completes successfully.

        Raises:
            TypeError: If table is not a string or records is not a dictionary.
            ValueError: If table or records is empty.
        """
        if type(table) is not str:
            raise TypeError("Table name must be a string.")
        if type(records) is not dict:
            raise TypeError("Records parameter must be a dictionary.")
        if not table or not records:
            raise ValueError("Table name and records must not be empty.")

        self._ensure_initialized()
        assert self._router is not None

        # Automatically prepend the table name as a namespace for recovery tracking stability
        namespaced_records: dict[str, Any] = {f"{table}:{k}": v for k, v in records.items()}

        return self._router.set_many(table, namespaced_records)

    def get(self, table: str, key: str) -> Any | None:
        """Retrieves a record value by key from the database hierarchy.

        Looks up data across the LRU Cache, the active B-Tree MemTable, and
        falls back to an on-disk pages scan if omitted in RAM.

        Args:
            table: Target database table name. Must not be empty.
            key: Record identifier key to search for. Must not be empty.

        Returns:
            The stored value payload if found, or None if it does not exist.

        Raises:
            TypeError: If table or key is not a string.
            ValueError: If table or key is empty.
        """
        if not isinstance(table, str) or not isinstance(key, str):
            raise TypeError("Table name and key parameters must be strings.")
        if not table or not key:
            raise ValueError("Table name and key parameters must not be empty.")

        self._ensure_initialized()
        assert self._router is not None

        namespaced_key: str = f"{table}:{key}"
        return self._router.get(table, namespaced_key)

    def delete(self, table: str, key: str) -> bool:
        """Logically deletes a key from the specified database table.

        Appends a tombstone designator record to the engine data pipeline. Space
        reclamation occurs asynchronously via the background Compactor subsystem.

        Args:
            table: Target database table name. Must not be empty.
            key: Record identifier key to delete. Must not be empty.

        Returns:
            True if the deletion command completes successfully.

        Raises:
            TypeError: If table or key is not a string.
            ValueError: If table or key is empty.
        """
        if not isinstance(table, str) or not isinstance(key, str):
            raise TypeError("Table name and key parameters must be strings.")
        if not table or not key:
            raise ValueError("Table name and key parameters must not be empty.")

        return self.set(table, key, None)

    def close(self) -> None:
        """Performs a graceful shutdown of all background worker threads and active file handles."""
        with self._lock:
            if self._router is None:
                return

            assert self._logger is not None
            self._logger.info("Initiating graceful shutdown sequence for cachyDB...")

            if self._compactor is not None:
                self._compactor.stop_background()

            if self._checkpoint is not None:
                self._checkpoint.stop_thread()

            if self._router is not None:
                self._router.shutdown()

            if self._wal is not None:
                self._wal.force_flush()

            self._logger.info("cachyDB engine shut down cleanly and securely.")

            self._router = None
            self._wal = None
            self._checkpoint = None
            self._compactor = None


_default_db: CachyDB = CachyDB()

set = _default_db.set
set_many = _default_db.set_many
get = _default_db.get
delete = _default_db.delete
close = _default_db.close
