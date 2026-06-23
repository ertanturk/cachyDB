"""Module for the Compactor (Garbage Collector) in the cachyDB database.

This module provides background and manual compaction for table storage files.
It reclaims fragmented space by discarding tombstoned (deleted) records and
obsolete historical versions of updated keys, writing the active records
into a clean temporary table file and atomically swapping it.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from cachyDB.pager.pager import Pager
from cachyDB.storage.table_handler import TableHandler

if TYPE_CHECKING:
    from pathlib import Path

    from cachyDB.storage.router import Router


class CompactorError(Exception):
    """Base exception for compactor and garbage collection operations."""


class Compactor:
    """cachyDB Table Compactor and Garbage Collector.

    Periodically or manually scans and compacts table files to remove
    tombstoned (deleted) records and obsolete historical updates.

    Attributes:
        DEFAULT_THRESHOLD (float): Default fragmentation ratio threshold.
        DEFAULT_INTERVAL (int): Default loop wait interval in seconds.
        TOMBSTONE_OFFSET (int): File offset value signifying a deleted record.
        TOMBSTONE_LENGTH (int): Data length value signifying a deleted record.
        router (Router): Active Router instance orchestrating table states and locks.
        threshold (float): Fragmentation ratio threshold below which compaction is triggered.
        logger (logging.Logger): Logger instance.
        _stop_event (threading.Event): Signal to stop the background thread.
        _thread (threading.Thread | None): The background compaction thread.
    """

    DEFAULT_THRESHOLD: float = 0.25
    DEFAULT_INTERVAL: int = 600
    TOMBSTONE_OFFSET: int = 0
    TOMBSTONE_LENGTH: int = 0

    router: Router
    threshold: float
    logger: logging.Logger
    _stop_event: threading.Event
    _thread: threading.Thread | None

    def __init__(self, router_instance: Router, compaction_threshold: float = DEFAULT_THRESHOLD) -> None:
        """Initializes the Compactor.

        Args:
            router_instance: Active Router instance managing table state and locking.
            compaction_threshold: Fragmentation ratio threshold to trigger automatic compaction.
        """
        self.router = router_instance
        self.threshold = compaction_threshold
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread = None

    def start_background(self, interval_seconds: int = DEFAULT_INTERVAL) -> None:
        """Starts the compaction process as an asynchronous background thread.

        Args:
            interval_seconds: Sleep interval for the background loop in seconds.
        """
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(interval_seconds,),
                daemon=True,
                name="cachyDB-Compactor",
            )
            self._thread.start()
            self.logger.info("Background Compactor (Garbage Collector) thread started.")

    def stop_background(self) -> None:
        """Safely stops the background compaction thread."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            self.logger.info("Background Compactor thread stopped.")

    def _run_loop(self, interval: int) -> None:
        """Main loop executed by the background thread to periodically inspect tables.

        Args:
            interval: Sleep interval between checks in seconds.
        """
        while not self._stop_event.is_set():
            if self._stop_event.wait(interval):
                break

            try:
                tables_to_compact: list[str]
                with self.router.global_lock:
                    tables_to_compact = list(self.router.memtables.keys())

                table_name: str
                for table_name in tables_to_compact:
                    if self._stop_event.is_set():
                        break

                    if self._should_compact(table_name):
                        self.logger.info("Starting automatic compaction for table '%s'...", table_name)
                        self.compact_table(table_name)
            except Exception as exc:
                self.logger.error("Critical error in background compaction loop: %s", exc, exc_info=True)

        self.logger.info("Background compactor loop terminated.")

    def _should_compact(self, table_name: str) -> bool:
        """Analyzes whether the given table needs compaction.

        Returns True if the ratio of tombstoned slots to total slots exceeds
        the configured threshold.

        Args:
            table_name: The name of the table to analyze.

        Returns:
            True if compaction is required, False otherwise.
        """
        memtable: Any
        table_lock: threading.RLock
        handler: TableHandler

        try:
            memtable, table_lock, handler = self.router._get_table_state(table_name)
        except KeyError:
            self.logger.warning("Table '%s' was dropped or is missing during analysis.", table_name)
            return False

        with table_lock:
            total_pages: int = handler.total_pages
            if total_pages == 0:
                return False

            tombstone_count: int = 0
            total_slots: int = 0

            page_id: int
            for page_id in range(total_pages):
                try:
                    page: Pager = handler.read_page(page_id)
                    total_slots += len(page.slots)

                    offset: int
                    length: int
                    for offset, length in page.slots:
                        if offset == self.TOMBSTONE_OFFSET and length == self.TOMBSTONE_LENGTH:
                            tombstone_count += 1
                except Exception as exc:
                    self.logger.warning(
                        "Failed to read page %d of table '%s' during compaction analysis: %s",
                        page_id,
                        table_name,
                        exc,
                    )
                    continue

            if total_slots == 0:
                return False

            ratio: float = tombstone_count / total_slots
            self.logger.debug(
                "Table '%s' fragmentation ratio: %.2f (Threshold: %.2f)", table_name, ratio, self.threshold
            )
            return ratio >= self.threshold

    def compact_table(self, table_name: str) -> None:
        """Compact the chosen table using Copy-on-Write (Out-of-Place) strategy.

        Reads active records, filters them to retain only the most recent version,
        writes them to a temporary compacted file, and atomically swaps it with the
        original file under table lock.

        Args:
            table_name: The name of the table to compact.

        Raises:
            CompactorError: If compaction fails or is aborted.
        """
        memtable: Any
        table_lock: threading.RLock
        handler: TableHandler

        with self.router.global_lock:
            try:
                memtable, table_lock, handler = self.router._get_table_state(table_name)
            except KeyError as exc:
                raise CompactorError(f"Table '{table_name}' state could not be fetched for compaction.") from exc
            original_file_path: Path = handler.file_path

        self.logger.info("Starting background compaction for '%s'...", table_name)
        temp_handler_name: str = f"{table_name}.compact"
        temp_handler: TableHandler = TableHandler(temp_handler_name, tables_dir=original_file_path.parent)

        total_pages: int = handler.total_pages()  # ty: ignore[call-non-callable]
        if total_pages == 0:
            return

        try:
            active_records: dict[str, Any] = {}

            page_id: int
            for page_id in range(total_pages):
                try:
                    page: Pager = handler.read_page(page_id)
                except Exception as exc:
                    raise CompactorError(
                        f"Error reading page {page_id} of table '{table_name}' during compaction."
                    ) from exc

                slot_id: int
                for slot_id in range(len(page.slots)):
                    payload: bytes | None = page.get_record(slot_id)
                    if payload is not None:
                        try:
                            p_key: str
                            p_val: Any
                            p_key, p_val = self.router._unpack_record(payload)
                            active_records[p_key] = p_val
                        except Exception as exc:
                            self.logger.warning(
                                "Skipping corrupted record (Page: %d, Slot: %d) in table '%s': %s",
                                page_id,
                                slot_id,
                                table_name,
                                exc,
                            )

            current_page_id: int = temp_handler.allocate_page()
            page_temp: Pager = temp_handler.read_page(current_page_id)
            is_page_dirty: bool = False

            for key, value in active_records.items():
                packed_record: bytes = self.router._pack_record(key, value)
                if not page_temp.has_space_for(len(packed_record)):
                    if is_page_dirty:
                        temp_handler.write_page(page_temp)
                    current_page_id = temp_handler.allocate_page()
                    page_temp = temp_handler.read_page(current_page_id)
                    is_page_dirty = False
                page_temp.insert_record(packed_record)
                is_page_dirty = True

            if is_page_dirty:
                temp_handler.write_page(page_temp)

            temp_handler.close()

            with self.router.global_lock:
                backup_file_path: Path = original_file_path.with_suffix(".old")
                compacted_file_path: Path = temp_handler.file_path

                # Close the original handler file before renaming
                handler.close()

                if original_file_path.exists():
                    original_file_path.rename(backup_file_path)

                compacted_file_path.rename(original_file_path)

                # Re-open the new file for the handler
                handler.file = original_file_path.open("r+b")

                if backup_file_path.exists():
                    backup_file_path.unlink()

                self.router.cache.clear()
                self.logger.info("Table '%s' compacted successfully.", table_name)
        except Exception as exc:
            try:
                temp_handler.close()
            except Exception:
                pass
            if temp_handler.file_path.exists():
                try:
                    temp_handler.file_path.unlink()
                except Exception:
                    pass
            if not isinstance(exc, CompactorError):
                raise CompactorError(f"Compaction aborted for table '{table_name}' due to error: {exc}") from exc
            raise
