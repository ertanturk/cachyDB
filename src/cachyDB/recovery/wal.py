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
from pathlib import Path
from typing import Iterator


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
        self.global_tx_id: int = 0
        self.wal_buffer: list[tuple[int, bytes]] = []
        self.ram_completed_tx: set[int] = set()
        self.last_flush_time: float = time.time()
        self.flush_interval: float = 0.01  # 10 milliseconds
        self.batch_size: int = 100  # number of records to batch before flushing
        self.lock: threading.Lock = threading.Lock()
        self.create_wal_directory()
        self.restore_global_tx_id()

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

    def pre_allocate_wal(self, file_path: Path) -> None:
        """Pre-allocates the WAL file to the maximum capacity.

        Creates a new WAL file with pre-allocated space for the header block
        and payload area. The file is initialized with zeros and a valid header.

        Args:
            file_path: Path where the WAL file should be created.

        Raises:
            OSError: If file creation or allocation fails.
        """
        header_block_size: int = self.WAL_FILE_HEADER_SIZE + (self.WAL_CAPACITY * self.WAL_ENTRY_HEADER_SIZE)
        total_file_size: int = header_block_size + self.PAYLOAD_AREA_SIZE

        try:
            with file_path.open("wb") as f:
                f.truncate(total_file_size)

            # Set file permissions (owner read/write only)
            file_path.chmod(self.WAL_FILE_PERMISSIONS)

            # Initialize with header pointing to start of payload area
            self.create_file_header(file_path, current_records=0, used_space=0)

            self.logger.info("Pre-allocated WAL file %s (size: %d bytes)", file_path, total_file_size)
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
        with self.lock:
            tx_id: int = self.global_tx_id
            self.global_tx_id += 1
            self.wal_buffer.append((tx_id, packed_record))

            should_flush: bool = (
                len(self.wal_buffer) >= self.batch_size or (time.time() - self.last_flush_time) >= self.flush_interval
            )

            if should_flush:
                self.append_to_wal()

            return tx_id

    def append_to_wal(self) -> None:
        """Appends the WAL buffer to the WAL file.

        Writes all buffered records to disk atomically. Updates the file header
        with the new record count and used space. This method should be called
        while holding the instance lock.

        Raises:
            WALCapacityError: If records don't fit in available space.
            WALCorruptionError: If the WAL file header is corrupted.
            OSError: If file operations fail.
        """
        if not self.wal_buffer:
            return

        first_tx_id: int = self.wal_buffer[0][0]
        file_path: Path = self.get_file_path(first_tx_id)

        if not file_path.exists():
            self.pre_allocate_wal(file_path)

        try:
            with file_path.open("r+b") as f:
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

                for tx_id, payload in self.wal_buffer:
                    # Validate payload fits
                    self.validate_payload_size(payload, current_used_space)

                    slot_index: int = self.get_slot_index(tx_id)
                    payload_len: int = len(payload)

                    # Write payload at the current used space position
                    payload_offset: int = header_block_size + current_used_space
                    f.seek(payload_offset)
                    f.write(payload)

                    # Create and write entry header
                    is_completed: bool = tx_id in self.ram_completed_tx
                    entry_header: bytes = self.create_entry_header(tx_id, payload_offset, is_completed)
                    entry_offset: int = self.WAL_FILE_HEADER_SIZE + (slot_index * self.WAL_ENTRY_HEADER_SIZE)
                    f.seek(entry_offset)
                    f.write(entry_header)

                    current_used_space += payload_len
                    current_rec += 1

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
                len(self.wal_buffer),
                file_path,
                current_used_space,
            )
        except (OSError, struct.error) as exc:
            raise OSError(f"Failed to append to WAL file {file_path}: {exc}") from exc
        finally:
            self.wal_buffer.clear()
            self.ram_completed_tx.clear()
            self.last_flush_time = time.time()

    def mark_job_complete(self, tx_id: int) -> None:
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
            # Check if tx_id is in the buffer
            if self.wal_buffer and self.wal_buffer[0][0] <= tx_id <= self.wal_buffer[-1][0]:
                self.ram_completed_tx.add(tx_id)
                return

        # Transaction already flushed, update on disk
        file_path: Path = self.get_file_path(tx_id)
        slot_index: int = self.get_slot_index(tx_id)
        entry_offset: int = self.WAL_FILE_HEADER_SIZE + (slot_index * self.WAL_ENTRY_HEADER_SIZE)

        try:
            with file_path.open("r+b") as f:
                # Seek to is_completed flag (offset 16 in entry header: 8 bytes tx_id + 8 bytes timestamp)
                f.seek(entry_offset + 16)
                f.write(struct.pack("?", True))
                f.flush()
                os.fsync(f.fileno())
            self.logger.debug("Marked transaction %d as complete", tx_id)
        except FileNotFoundError:
            self.logger.warning("WAL file %s not found when marking tx_id %d complete", file_path, tx_id)
            raise WALError(f"Cannot mark transaction {tx_id} complete: WAL file {file_path} not found") from None
        except OSError as exc:
            raise OSError(f"Failed to mark transaction {tx_id} complete in {file_path}: {exc}") from exc

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

                            f.seek(payload_offset)
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
                self.append_to_wal()
                self.logger.info("Forced flush completed")
