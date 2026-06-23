"""Module for the WAL class in the cachyDB database.

This module provides Write-Ahead Log (WAL) functionality for transaction recovery
in the cachyDB database. It manages transaction logs with checksums for data
integrity verification.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import IO, Iterator

from cachyDB.logger.structured_logger import OperationType, get_structured_logger


class WALError(Exception):
    """Base exception for WAL-related errors."""


class WALCorruptionError(WALError):
    """Raised when WAL data corruption is detected."""


class WALCapacityError(WALError):
    """Raised when WAL capacity is exceeded."""


class WAL:
    """Manages the Write-Ahead Log (WAL) for transaction recovery.

    The WAL provides durability and crash recovery for database transactions.
    Each transaction is written to the log before being committed to the main
    storage, allowing recovery of uncommitted transactions after a crash.

    Attributes:
        WAL_FILE_PATH (str): Directory path for WAL files.
        WAL_CAPACITY (int): Maximum number of entries per WAL file.
        WAL_FILE_HEADER_SIZE (int): Size of the file header in bytes.
        WAL_ENTRY_HEADER_SIZE (int): Size of each entry header in bytes.
        PAYLOAD_AREA_SIZE (int): Total size of the payload storage area.
        FILE_HEADER_FMT (str): Struct format for file header.
        ENTRY_HEADER_FMT (str): Struct format for entry header.
        global_tx_id (int): Current global transaction ID counter.
        wal_buffer (list[tuple[int, bytes]]): In-memory buffer for pending writes.
        ram_completed_tx (set[int]): Set of completed transaction IDs in buffer.
        last_flush_time (float): Timestamp of last buffer flush.
        flush_interval (float): Time interval between automatic flushes.
        batch_size (int): Number of records to batch before flushing.
        logger (logging.Logger): Logger instance for this module.
        lock (threading.Lock): Thread lock for concurrent access protection.

    Note:
        WAL files are pre-allocated and use a fixed-size entry header area
        with a variable-size payload area. Payloads grow upward from the
        beginning of the payload area.
    """

    WAL_FILE_PATH: str = "data/wal/"
    WAL_CAPACITY: int = 50000
    WAL_FILE_HEADER_SIZE: int = 32
    WAL_ENTRY_HEADER_SIZE: int = 25
    PAYLOAD_AREA_SIZE: int = 100 * 1024 * 1024  # 100MB
    WAL_FILE_PERMISSIONS: int = 0o600  # Owner read/write only

    # File Header: Magic(4s) + Capacity(I) + Current_Records(I) + Used_Space(Q) + Checksum(I) + Padding(8x) = 32 Byte
    FILE_HEADER_FMT: str = "> 4s I I Q I 8x"
    # Entry Header: tx_id(Q) + timestamp(d) + is_completed(?) + payload_offset(Q) = 25 Byte
    ENTRY_HEADER_FMT: str = "> Q d ? Q"

    def __init__(self) -> None:
        """Initializes the WAL manager.

        Creates the WAL directory if it doesn't exist, restores the global
        transaction ID from existing WAL files, and initializes internal state.
        """
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.structured_logger = get_structured_logger("wal")
        self.global_tx_id: int = 0
        self.wal_buffer: list[tuple[int, bytes]] = []
        self.ram_completed_tx: set[int] = set()
        self.last_flush_time: float = time.time()
        self.flush_interval: float = 1.0  # 1 second
        self.batch_size: int = 50000  # number of records to batch before flushing
        self.lock: threading.Lock = threading.Lock()
        self.io_lock: threading.Lock = threading.Lock()
        self.flush_lock: threading.Lock = threading.Lock()
        self.active_flushes: int = 0
        self._current_file_path: Path | None = None
        self._current_file: IO[bytes] | None = None
        self.create_wal_directory()
        self.restore_global_tx_id()
        self._flush_event: threading.Event = threading.Event()
        self._stop_event: threading.Event = threading.Event()
        self._writer_thread = threading.Thread(target=self._wal_writer_loop, daemon=True, name="WAL-Writer")
        self._writer_thread.start()

    def restore_global_tx_id(self) -> None:
        """Scans WAL files to restore the current tx_id.

        Examines the last WAL file to determine the highest transaction ID
        and resumes from that point. If no WAL files exist, starts from 0.

        Raises:
            WALCorruptionError: If the WAL file header is corrupted.
        """
        wal_dir: Path = Path(self.WAL_FILE_PATH)
        wal_files: list[Path] = sorted(wal_dir.glob("global_*.wal"), key=lambda p: int(p.stem.split("_")[1]))

        if not wal_files:
            self.global_tx_id = 0
            self.logger.info("No existing WAL files found. Starting with tx_id=0")
            return

        last_file: Path = wal_files[-1]
        file_id: int = int(last_file.stem.split("_")[1])

        try:
            with last_file.open("rb") as f:
                header_data: bytes = f.read(self.WAL_FILE_HEADER_SIZE)
                if len(header_data) == self.WAL_FILE_HEADER_SIZE:
                    magic, capacity, current_rec, used_space, checksum = struct.unpack(
                        self.FILE_HEADER_FMT, header_data
                    )

                    # Verify header checksum
                    header_without_checksum: bytes = struct.pack("> 4s I I Q", magic, capacity, current_rec, used_space)
                    computed_checksum: int = zlib.crc32(header_without_checksum) & 0xFFFFFFFF

                    if computed_checksum != checksum:
                        raise WALCorruptionError(
                            f"WAL file {last_file} header checksum mismatch: "
                            f"expected 0x{checksum:08X}, got 0x{computed_checksum:08X}"
                        )

                    if magic != b"WAL1":
                        raise WALCorruptionError(f"WAL file {last_file} has invalid magic number: {magic!r}")

                    self.global_tx_id = (file_id * self.WAL_CAPACITY) + current_rec
                else:
                    self.logger.warning(
                        "WAL file %s header truncated (expected %d bytes, got %d). Starting from file boundary.",
                        last_file,
                        self.WAL_FILE_HEADER_SIZE,
                        len(header_data),
                    )
                    self.global_tx_id = file_id * self.WAL_CAPACITY
        except (OSError, struct.error) as exc:
            raise WALCorruptionError(f"Failed to restore transaction ID from {last_file}: {exc}") from exc

        self.logger.info("Transaction system restored. Current tx_id=%d", self.global_tx_id)

    def _get_open_file(self, file_path: Path) -> IO[bytes]:
        """Returns a cached open file handle for the WAL file."""
        if (
            getattr(self, "_current_file_path", None) == file_path
            and getattr(self, "_current_file", None) is not None
            and not self._current_file.closed  # type: ignore
        ):
            return self._current_file  # type: ignore

        # Close existing cached file handle if any
        existing_file = getattr(self, "_current_file", None)
        if existing_file is not None and not existing_file.closed:
            try:
                existing_file.close()
            except Exception:
                pass

        # Open and cache the new file handle
        self._current_file_path = file_path
        self._current_file = file_path.open("r+b")
        return self._current_file

    def validate_tx_id(self, tx_id: int) -> None:
        """Validates that tx_id is non-negative and within reasonable bounds.

        Args:
            tx_id: The transaction ID to validate.

        Raises:
            ValueError: If tx_id is negative or invalid.
        """
        if tx_id < 0:
            raise ValueError(f"Transaction ID must be non-negative, got {tx_id}")
        if tx_id > (2**63 - 1):  # Maximum value for signed 64-bit integer
            raise ValueError(f"Transaction ID {tx_id} exceeds maximum value {2**63 - 1}")

    def get_file_path(self, tx_id: int) -> Path:
        """Returns the path of the WAL file for the given tx_id.

        Args:
            tx_id: The transaction ID.

        Returns:
            Path object for the WAL file containing this transaction.

        Raises:
            ValueError: If tx_id is invalid.
        """
        self.validate_tx_id(tx_id)
        file_id: int = tx_id // self.WAL_CAPACITY
        return Path(self.WAL_FILE_PATH) / f"global_{file_id}.wal"

    def get_slot_index(self, tx_id: int) -> int:
        """Returns the slot index within the WAL file for the given tx_id.

        Args:
            tx_id: The transaction ID.

        Returns:
            Slot index (0-based) within the WAL file.

        Raises:
            ValueError: If tx_id is invalid.
        """
        self.validate_tx_id(tx_id)
        return tx_id % self.WAL_CAPACITY

    def create_wal_directory(self) -> None:
        """Creates the WAL directory if it does not exist.

        Raises:
            OSError: If directory creation fails.
        """
        try:
            wal_path: Path = Path(self.WAL_FILE_PATH)
            wal_path.mkdir(parents=True, exist_ok=True)
            # Set directory permissions (owner read/write/execute only)
            wal_path.chmod(0o700)
        except OSError as exc:
            raise OSError(f"Failed to create WAL directory {self.WAL_FILE_PATH}: {exc}") from exc

    def pre_allocate_wal(self, file_path: Path, size_bytes: int) -> None:
        """Pre-allocates the WAL file to the maximum capacity.

        Creates a new WAL file with pre-allocated space for the header block
        and payload area. The file is initialized with zeros and a valid header.

        Args:
            file_path: Path where the WAL file should be created.

        Raises:
            OSError: If file creation or allocation fails.
        """

        try:
            with file_path.open("wb") as f:
                os.posix_fallocate(f.fileno(), 0, size_bytes)
                f.flush()
                os.fsync(f.fileno())

            # Set file permissions (owner read/write only)
            file_path.chmod(self.WAL_FILE_PERMISSIONS)

            # Initialize with header pointing to start of payload area
            self.create_file_header(file_path, current_records=0, used_space=0)

            self.logger.info("Pre-allocated WAL file %s (size: %d bytes)", file_path, size_bytes)
        except OSError as exc:
            raise OSError(f"Failed to pre-allocate WAL file {file_path}: {exc}") from exc

    def create_file_header(self, file_path: Path, current_records: int, used_space: int) -> None:
        """Creates the file header for the WAL file.

        Args:
            file_path: Path to the WAL file.
            current_records: Number of records currently in the file.
            used_space: Number of bytes used in the payload area.

        Raises:
            ValueError: If current_records or used_space are invalid.
            OSError: If file write fails.
        """
        if current_records < 0 or current_records > self.WAL_CAPACITY:
            raise ValueError(f"current_records must be between 0 and {self.WAL_CAPACITY}, got {current_records}")
        if used_space < 0 or used_space > self.PAYLOAD_AREA_SIZE:
            raise ValueError(f"used_space must be between 0 and {self.PAYLOAD_AREA_SIZE}, got {used_space}")

        magic: bytes = b"WAL1"
        # Create header without checksum first
        header_without_checksum: bytes = struct.pack(
            "> 4s I I Q", magic, self.WAL_CAPACITY, current_records, used_space
        )
        # Calculate checksum
        checksum: int = zlib.crc32(header_without_checksum) & 0xFFFFFFFF
        # Create full header with checksum
        header_bytes: bytes = struct.pack(
            self.FILE_HEADER_FMT, magic, self.WAL_CAPACITY, current_records, used_space, checksum
        )

        try:
            with file_path.open("r+b") as f:
                f.seek(0)
                f.write(header_bytes)
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise OSError(f"Failed to write header to {file_path}: {exc}") from exc

    def create_entry_header(self, tx_id: int, payload_offset: int, is_completed: bool) -> bytes:
        """Creates the entry header for a WAL record.

        Args:
            tx_id: The transaction ID.
            payload_offset: Offset in the file where the payload is stored.
            is_completed: Whether the transaction is completed.

        Returns:
            Packed entry header as bytes.

        Raises:
            ValueError: If tx_id or payload_offset are invalid.
        """
        self.validate_tx_id(tx_id)
        if payload_offset < 0:
            raise ValueError(f"payload_offset must be non-negative, got {payload_offset}")

        timestamp: float = time.time()
        return struct.pack(self.ENTRY_HEADER_FMT, tx_id, timestamp, is_completed, payload_offset)

    def validate_payload_size(self, payload: bytes, current_used_space: int) -> None:
        """Validates that the payload fits in the available space.

        Args:
            payload: The payload bytes to validate.
            current_used_space: Currently used space in the payload area.

        Raises:
            WALCapacityError: If the payload doesn't fit.
        """
        payload_len: int = len(payload)
        if payload_len > self.PAYLOAD_AREA_SIZE:
            raise WALCapacityError(f"Payload size {payload_len} bytes exceeds maximum {self.PAYLOAD_AREA_SIZE} bytes")
        if current_used_space + payload_len > self.PAYLOAD_AREA_SIZE:
            raise WALCapacityError(
                f"Payload size {payload_len} bytes would exceed available space. "
                f"Used: {current_used_space}, Available: {self.PAYLOAD_AREA_SIZE}"
            )

    def append_to_buffer(self, packed_record: bytes) -> int:
        """Appends a packed record to the WAL buffer.

        The record is buffered in memory and flushed to disk when either the
        batch size is reached or the flush interval expires.

        Args:
            packed_record: The serialized record to append.

        Returns:
            The transaction ID assigned to this record.

        Raises:
            WALCapacityError: If the record is too large.
        """
        while len(self.wal_buffer) >= self.batch_size * 2:
            time.sleep(0.001)

        with self.lock:
            tx_id: int = self.global_tx_id
            self.global_tx_id += 1
            self.wal_buffer.append((tx_id, packed_record))

            if len(self.wal_buffer) >= self.batch_size:
                self._flush_event.set()

            return tx_id

    def append_many_to_buffer(self, packed_records: list[bytes]) -> list[int]:
        """Appends multiple records to the WAL buffer in a single atomic lock operation.

        Args:
            packed_records: A list of serialized records to append.

        Returns:
            A list of transaction IDs assigned to these records.
        """
        while len(self.wal_buffer) >= self.batch_size * 2:
            time.sleep(0.001)

        tx_ids: list[int] = []
        with self.lock:
            for packed_record in packed_records:
                tx_id: int = self.global_tx_id
                self.global_tx_id += 1
                self.wal_buffer.append((tx_id, packed_record))
                tx_ids.append(tx_id)

            self._flush_event.set()

        return tx_ids

    def mark_many_jobs_complete(self, tx_ids: list[int]) -> None:
        """Marks multiple jobs as complete efficiently to avoid I/O blocking."""
        if not tx_ids:
            return

        with self.lock:
            self.ram_completed_tx.update(tx_ids)

            if self.wal_buffer:
                buffer_min: int = self.wal_buffer[0][0]
                buffer_max: int = self.wal_buffer[-1][0]
                disk_tx_ids: list[int] = [t for t in tx_ids if not (buffer_min <= t <= buffer_max)]
            else:
                disk_tx_ids = tx_ids

        if not disk_tx_ids:
            return

        files_to_update = defaultdict(list)

        # Only touch the disk for transactions that already flushed to background files
        for tx_id in disk_tx_ids:
            try:
                files_to_update[self.get_file_path(tx_id)].append(tx_id)
            except ValueError:
                continue

        for file_path, file_tx_ids in files_to_update.items():
            if not file_path.exists():
                continue

            try:
                with getattr(self, "io_lock", self.lock):
                    f = self._get_open_file(file_path)
                    for tx_id in file_tx_ids:
                        slot_index: int = self.get_slot_index(tx_id)
                        entry_offset: int = self.WAL_FILE_HEADER_SIZE + (slot_index * self.WAL_ENTRY_HEADER_SIZE)
                        f.seek(entry_offset + 16)
                        f.write(struct.pack("?", True))
                    f.flush()
            except Exception as exc:
                self.logger.warning("Failed to mark transaction %d complete: %s", tx_id, exc)

    def _wal_writer_loop(self) -> None:
        """Background thread loop for asynchronous WAL fsyncing."""
        while not self._stop_event.is_set():
            # Wait for batch threshold or time interval
            self._flush_event.wait(timeout=self.flush_interval)
            self._flush_event.clear()

            buffer_copy: list[tuple[int, bytes]] = []

            # O(1) buffer swap under lock
            with self.lock:
                if not self.wal_buffer:
                    continue
                buffer_copy = self.wal_buffer
                self.wal_buffer = []
                self.last_flush_time = time.time()

            # Perform heavy disk I/O and os.fsync WITHOUT holding the main lock
            try:
                self._write_buffer_to_disk(buffer_copy)
            except Exception as exc:
                self.logger.error("Critical failure in background WAL writer: %s", exc)

    def _pre_allocate_async(self, file_path: Path) -> None:
        """Safely pre-allocates a WAL file in the background."""
        try:
            if not file_path.exists():
                self.pre_allocate_wal(
                    file_path,
                    self.WAL_FILE_HEADER_SIZE
                    + (self.WAL_CAPACITY * self.WAL_ENTRY_HEADER_SIZE)
                    + self.PAYLOAD_AREA_SIZE,
                )
        except Exception as exc:
            self.logger.warning("Background pre-allocation failed for %s: %s", file_path, exc)

    def _write_buffer_to_disk(self, buffer_copy: list[tuple[int, bytes]]) -> None:
        """Appends the WAL buffer to the WAL file.

        Writes all buffered records to disk atomically. Updates the file header
        with the new record count and used space. This method should be called
        while holding the instance lock.

        Raises:
            WALCapacityError: If records don't fit in available space.
            WALCorruptionError: If the WAL file header is corrupted.
            OSError: If file operations fail.
        """
        if not buffer_copy:
            return

        file_batches = defaultdict(list)

        with self.lock:
            completed_copy = self.ram_completed_tx.copy()
            self.ram_completed_tx.clear()

        # 2. Split the buffer into batches by file boundary
        for tx_id, payload in buffer_copy:
            file_id = tx_id // self.WAL_CAPACITY
            file_batches[file_id].append((tx_id, payload))

        # 3. Write each batch to its respective file
        for file_id, batch in file_batches.items():
            self._write_batch_to_disk(batch, completed_copy)

    def _write_batch_to_disk(self, batch: list[tuple[int, bytes]], completed_copy: set[int]) -> None:
        """Appends a localized batch of records to a single WAL file."""
        trace_id: str = ""
        start_time: float = time.perf_counter()

        buffer_size: int = len(batch)
        total_payload_size: int = sum(len(payload) + 4 for _, payload in batch)

        first_tx_id: int = batch[0][0]
        file_path: Path = self.get_file_path(first_tx_id)

        if not file_path.exists():
            self.pre_allocate_wal(file_path, self.WAL_FILE_HEADER_SIZE + total_payload_size)
            next_file_path: Path = self.get_file_path(first_tx_id + self.WAL_CAPACITY)
            threading.Thread(
                target=self._pre_allocate_async, args=(next_file_path,), daemon=True, name="WAL-Prealloc"
            ).start()

        try:
            with getattr(self, "io_lock", self.lock):
                f = self._get_open_file(file_path)
                # Read and validate full header
                f.seek(0)
                header_data: bytes = f.read(self.WAL_FILE_HEADER_SIZE)
                if len(header_data) != self.WAL_FILE_HEADER_SIZE:
                    raise WALCorruptionError(
                        f"WAL file {file_path} header truncated: "
                        f"expected {self.WAL_FILE_HEADER_SIZE} bytes, got {len(header_data)}"
                    )

                magic, cap, current_rec, current_used_space, checksum = struct.unpack(self.FILE_HEADER_FMT, header_data)

                # Verify header checksum
                header_without_checksum: bytes = struct.pack("> 4s I I Q", magic, cap, current_rec, current_used_space)
                computed_checksum: int = zlib.crc32(header_without_checksum) & 0xFFFFFFFF

                if computed_checksum != checksum:
                    raise WALCorruptionError(
                        f"WAL file {file_path} header checksum mismatch: "
                        f"expected 0x{checksum:08X}, got 0x{computed_checksum:08X}"
                    )

                if magic != b"WAL1":
                    raise WALCorruptionError(f"WAL file {file_path} has invalid magic number: {magic!r}")

                # Calculate payload area start
                header_block_size: int = self.WAL_FILE_HEADER_SIZE + (self.WAL_CAPACITY * self.WAL_ENTRY_HEADER_SIZE)

                # Phase 1: Write all payloads sequentially
                payload_offsets = []
                payload_block = bytearray()
                start_space = current_used_space

                for tx_id, payload in batch:
                    payload_with_len = struct.pack(">I", len(payload)) + payload
                    self.validate_payload_size(payload_with_len, current_used_space)
                    payload_len: int = len(payload_with_len)

                    payload_offsets.append(header_block_size + current_used_space)
                    payload_block.extend(payload_with_len)
                    current_used_space += payload_len

                f.seek(header_block_size + start_space)
                f.write(payload_block)

                first_slot_index: int = self.get_slot_index(batch[0][0])
                entry_offset_start: int = self.WAL_FILE_HEADER_SIZE + (first_slot_index * self.WAL_ENTRY_HEADER_SIZE)

                header_block = bytearray()

                # Phase 2: Write all entry headers sequentially
                for i, (tx_id, payload) in enumerate(batch):
                    is_completed: bool = tx_id in completed_copy
                    entry_header: bytes = self.create_entry_header(tx_id, payload_offsets[i], is_completed)
                    header_block.extend(entry_header)

                f.seek(entry_offset_start)
                f.write(header_block)
                current_rec += len(batch)

                # Update file header
                f.seek(0)
                header_without_checksum = struct.pack("> 4s I I Q", magic, cap, current_rec, current_used_space)
                new_checksum: int = zlib.crc32(header_without_checksum) & 0xFFFFFFFF
                header_bytes: bytes = struct.pack(
                    self.FILE_HEADER_FMT, magic, cap, current_rec, current_used_space, new_checksum
                )
                f.write(header_bytes)

                f.flush()

            os.fsync(f.fileno())

            self.logger.debug(
                "Flushed %d records to %s (used space: %d bytes)",
                buffer_size,
                file_path,
                current_used_space,
            )
            # Log successful WAL append
            latency_ms: float = (time.perf_counter() - start_time) * 1000
            throughput_kops: float = (buffer_size / latency_ms * 1000) / 1000 if latency_ms > 0 else 0
            self.structured_logger.log_write_operation(
                trace_id=trace_id,
                operation_type=OperationType.WAL_APPEND,
                table_name="_wal_internal",
                records_count=buffer_size,
                data_size_bytes=total_payload_size,
                latency_ms=latency_ms,
                throughput_kops=throughput_kops,
                error_code=0,
            )
        except (OSError, struct.error) as exc:
            latency_ms: float = (time.perf_counter() - start_time) * 1000
            self.structured_logger.log_write_operation(
                trace_id=trace_id,
                operation_type=OperationType.WAL_APPEND,
                table_name="_wal_internal",
                records_count=buffer_size,
                data_size_bytes=total_payload_size,
                latency_ms=latency_ms,
                error_code=1,
                error_message=str(exc),
            )
            raise OSError(f"Failed to append to WAL file {file_path}: {exc}") from exc

    def mark_job_complete(self, tx_id: int, file_path: Path) -> None:
        """Marks a job as complete in the WAL buffer and file.

        If the transaction is in the current buffer, marks it in RAM. Otherwise,
        updates the on-disk entry header to mark it as completed.

        Args:
            tx_id: The transaction ID to mark as complete.

        Raises:
            ValueError: If tx_id is invalid.
        """
        self.validate_tx_id(tx_id)

        with self.lock:
            self.ram_completed_tx.add(tx_id)

            # Check if tx_id is in the buffer
            if self.wal_buffer and self.wal_buffer[0][0] <= tx_id <= self.wal_buffer[-1][0]:
                return

        if not file_path.exists():
            return

        # Transaction already flushed, update on disk
        slot_index: int = self.get_slot_index(tx_id)
        entry_offset: int = self.WAL_FILE_HEADER_SIZE + (slot_index * self.WAL_ENTRY_HEADER_SIZE)

        try:
            with getattr(self, "io_lock", self.lock):
                f = self._get_open_file(file_path)
                f.seek(entry_offset + 16)
                f.write(struct.pack("?", True))
                f.flush()
        except Exception as exc:
            self.logger.warning("Failed to mark transaction %d complete in %s: %s", tx_id, file_path, exc)

    def read_wal(self) -> Iterator[tuple[int, bytes]]:
        """Reads the WAL file and yields the incomplete records.

        Iterates through all WAL files in order, yielding transaction ID and
        packed record for each incomplete transaction. This is used during
        recovery to replay uncommitted transactions.

        Yields:
            Tuple of (tx_id, packed_record) for each incomplete transaction.

        Raises:
            WALCorruptionError: If WAL data is corrupted.
        """
        wal_dir: Path = Path(self.WAL_FILE_PATH)

        if not wal_dir.exists():
            self.logger.info("WAL directory does not exist, no records to read")
            return

        wal_files: list[Path] = sorted(wal_dir.glob("global_*.wal"), key=lambda p: int(p.stem.split("_")[1]))

        if not wal_files:
            self.logger.info("No WAL files found")
            return

        for file_path in wal_files:
            try:
                with file_path.open("rb") as f:
                    header_data: bytes = f.read(self.WAL_FILE_HEADER_SIZE)
                    if len(header_data) < self.WAL_FILE_HEADER_SIZE:
                        self.logger.warning(
                            "WAL file %s header truncated (expected %d bytes, got %d), skipping",
                            file_path,
                            self.WAL_FILE_HEADER_SIZE,
                            len(header_data),
                        )
                        continue

                    magic, cap, current_rec, used_space, checksum = struct.unpack(self.FILE_HEADER_FMT, header_data)

                    # Verify header checksum
                    header_without_checksum: bytes = struct.pack("> 4s I I Q", magic, cap, current_rec, used_space)
                    computed_checksum: int = zlib.crc32(header_without_checksum) & 0xFFFFFFFF

                    if computed_checksum != checksum:
                        self.logger.error(
                            "WAL file %s header checksum mismatch (expected 0x%08X, got 0x%08X), skipping",
                            file_path,
                            checksum,
                            computed_checksum,
                        )
                        continue

                    if magic != b"WAL1":
                        self.logger.warning("WAL file %s has invalid magic number %r, skipping", file_path, magic)
                        continue

                    # Read entries (only up to current_rec, not all slots)
                    for i in range(current_rec):
                        entry_offset: int = self.WAL_FILE_HEADER_SIZE + (i * self.WAL_ENTRY_HEADER_SIZE)
                        f.seek(entry_offset)
                        entry_data: bytes = f.read(self.WAL_ENTRY_HEADER_SIZE)
                        if len(entry_data) < self.WAL_ENTRY_HEADER_SIZE:
                            self.logger.warning(
                                "WAL file %s entry %d truncated, stopping at this entry",
                                file_path,
                                i,
                            )
                            break

                        tx_id, timestamp, is_completed, payload_offset = struct.unpack(
                            self.ENTRY_HEADER_FMT, entry_data
                        )

                        # Only yield incomplete transactions
                        if not is_completed:
                            f.seek(payload_offset)
                            length_bytes: bytes = f.read(4)
                            if len(length_bytes) < 4:
                                self.logger.warning(
                                    "WAL file %s tx_id %d payload length truncated, skipping",
                                    file_path,
                                    tx_id,
                                )
                                continue

                            record_length: int = int.from_bytes(length_bytes, byteorder="big")

                            # Validate record length
                            if record_length <= 0 or record_length > self.PAYLOAD_AREA_SIZE:
                                self.logger.warning(
                                    "WAL file %s tx_id %d has invalid record length %d, skipping",
                                    file_path,
                                    tx_id,
                                    record_length,
                                )
                                continue

                            packed_record: bytes = f.read(record_length)

                            if len(packed_record) < record_length:
                                self.logger.warning(
                                    "WAL file %s tx_id %d payload truncated (expected %d bytes, got %d), skipping",
                                    file_path,
                                    tx_id,
                                    record_length,
                                    len(packed_record),
                                )
                                continue

                            yield tx_id, packed_record

                self.logger.info("WAL file %s read successfully", file_path)
            except (OSError, struct.error) as exc:
                self.logger.error("Failed to read WAL file %s: %s", file_path, exc)
                continue

    def force_flush(self) -> None:
        """Forces an immediate flush of the WAL buffer to disk.

        This method should be called before shutdown or when durability
        guarantees are required.
        """
        with self.lock:
            if self.wal_buffer:
                self._write_buffer_to_disk(self.wal_buffer)
                self.wal_buffer.clear()
                self.logger.info("Forced flush completed")
            # Close the cached file handle during force flush/shutdown
            existing_file = getattr(self, "_current_file", None)
            if existing_file is not None and not existing_file.closed:
                try:
                    existing_file.close()
                except Exception:
                    pass
                self._current_file = None
                self._current_file_path = None

    def shutdown(self) -> None:
        """Safely stops the background writer thread and flushes remaining data."""
        self.logger.info("Shutting down WAL background writer...")
        self._stop_event.set()
        self._flush_event.set()  # Wake the thread up so it can exit

        if getattr(self, "_writer_thread", None) and self._writer_thread.is_alive():
            self._writer_thread.join()

        self.force_flush()
