"""Module for WAL checkpoint management in the cachyDB database.

This module provides checkpoint functionality for the Write-Ahead Log (WAL).
It periodically scans sealed WAL files (all except the currently-active one),
verifies their integrity via header checksum, compresses them into gzip archives,
and removes the originals to reclaim disk space.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import struct
import threading
import zlib
from pathlib import Path

from cachyDB.recovery.wal import WAL


class WALCheckpoint:
    """Manages WAL file checkpointing and archiving.

    Runs a background thread that wakes up at a fixed interval and archives
    every sealed WAL file whose entries are all marked as completed. Each WAL
    file is validated (magic byte + CRC-32 header checksum) before being
    compressed and removed.

    Attributes:
        ARCHIVE_DIR (str): Directory path for compressed WAL archives.
        CHECK_INTERVAL (int): Seconds between successive checkpoint runs.
        wal (WAL): Reference to the active WAL manager.
        logger (logging.Logger): Logger instance for this module.
        stop_event (threading.Event): Signals the checkpoint thread to stop.
        thread (threading.Thread | None): The background checkpoint thread.
    """

    ARCHIVE_DIR: str = "data/wal_archives"
    CHECK_INTERVAL: int = 300  # 5 minutes

    def __init__(self, wal_instance: WAL) -> None:
        """Initializes the WALCheckpoint manager.

        Creates the archive directory with restricted permissions and stores a
        reference to the active WAL instance.

        Args:
            wal_instance: An active WAL instance whose sealed files will be archived.
        """
        self.wal: WAL = wal_instance
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.stop_event: threading.Event = threading.Event()
        self.thread: threading.Thread | None = None
        self.create_wal_archive_directory()

    def create_wal_archive_directory(self) -> None:
        """Creates the WAL archive directory if it does not exist.

        The directory is created with mode 0o700 (owner read/write/execute only)
        to restrict access to archived WAL data.

        Raises:
            OSError: If directory creation or permission change fails.
        """
        try:
            wal_path: Path = Path(self.ARCHIVE_DIR)
            wal_path.mkdir(parents=True, exist_ok=True)
            wal_path.chmod(0o700)
        except OSError as exc:
            raise OSError(f"Failed to create WAL archive directory {self.ARCHIVE_DIR}: {exc}") from exc

    def start_thread(self) -> None:
        """Starts the background checkpoint thread if it is not already running.

        Clears the stop event before starting so that a previously stopped thread
        can be restarted cleanly.
        """
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.run_loop, daemon=True, name="WAL-Checkpoint-Thread")
            self.thread.start()
            self.logger.info("Checkpoint thread started.")

    def stop_thread(self) -> None:
        """Signals the checkpoint thread to stop and blocks until it terminates."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
            self.logger.info("Checkpoint thread stopped.")

    def run_loop(self) -> None:
        """Main loop executed by the background checkpoint thread.

        Runs checkpoint operations at CHECK_INTERVAL-second intervals until the
        stop event is set. Errors are logged but do not terminate the loop.
        """
        while not self.stop_event.is_set():
            try:
                self.run_checkpoint()
            except Exception as e:
                self.logger.error("Checkpoint failed: %s", e)
            self.stop_event.wait(self.CHECK_INTERVAL)

    def run_checkpoint(self) -> None:
        """Scans for sealed WAL files and archives each fully-completed one.

        Only files that are not the current active WAL (i.e. all except the
        newest) are considered. Files with unparseable names are skipped with a
        warning rather than aborting the entire checkpoint cycle.
        """

        if getattr(self.wal, "active_flushes", 0) > 0:
            self.logger.debug("Skipping checkpoint: Router is currently flushing MemTables.")
            return

        wal_dir: Path = Path(self.wal.WAL_FILE_PATH)
        if not wal_dir.exists():
            return

        valid_files: list[Path] = []
        for p in wal_dir.glob("global_*.wal"):
            parts: list[str] = p.stem.split("_")
            if len(parts) != 2 or not parts[1].isdigit():
                self.logger.warning("Skipping WAL file with malformed name: %s", p.name)
                continue
            valid_files.append(p)

        wal_files: list[Path] = sorted(valid_files, key=lambda p: int(p.stem.split("_")[1]))

        if len(wal_files) <= 1:
            return

        sealed_files: list[Path] = wal_files[:-1]

        for file_path in sealed_files:
            if self.is_file_fully_completed(file_path):
                self.archive_file(file_path)

    def is_file_fully_completed(self, file_path: Path) -> bool:
        """Checks whether all entries in a WAL file are marked as completed.

        Validates the file header magic bytes and CRC-32 checksum before
        iterating over entry headers. Returns False on any I/O error or
        data-integrity failure to prevent archiving of suspect files.

        Args:
            file_path: Path to the WAL file to inspect.

        Returns:
            True if every entry in the file is marked completed, False otherwise.
        """
        try:
            with file_path.open("rb") as f:
                header_data: bytes = f.read(self.wal.WAL_FILE_HEADER_SIZE)
                if len(header_data) < self.wal.WAL_FILE_HEADER_SIZE:
                    self.logger.warning(
                        "WAL file %s header is truncated (%d bytes). Skipping.",
                        file_path.name,
                        len(header_data),
                    )
                    return False

                magic, capacity, current_rec, used_space, checksum = struct.unpack(
                    self.wal.FILE_HEADER_FMT, header_data
                )

                if magic != b"WAL1":
                    self.logger.warning(
                        "WAL file %s has invalid magic number %r. Skipping.",
                        file_path.name,
                        magic,
                    )
                    return False

                # Verify header integrity before trusting any header field.
                header_without_checksum: bytes = struct.pack("> 4s I I Q", magic, capacity, current_rec, used_space)
                computed_checksum: int = zlib.crc32(header_without_checksum) & 0xFFFFFFFF
                if computed_checksum != checksum:
                    self.logger.warning(
                        "WAL file %s header checksum mismatch (expected 0x%08X, got 0x%08X). Skipping.",
                        file_path.name,
                        checksum,
                        computed_checksum,
                    )
                    return False

                if current_rec == 0:
                    return True

                for i in range(current_rec):
                    entry_offset: int = self.wal.WAL_FILE_HEADER_SIZE + (i * self.wal.WAL_ENTRY_HEADER_SIZE)
                    f.seek(entry_offset)

                    entry_data: bytes = f.read(self.wal.WAL_ENTRY_HEADER_SIZE)
                    if len(entry_data) < self.wal.WAL_ENTRY_HEADER_SIZE:
                        self.logger.warning(
                            "WAL file %s entry %d is truncated. Skipping file.",
                            file_path.name,
                            i,
                        )
                        return False

                    _tx_id, _timestamp, is_completed, _payload_offset = struct.unpack(
                        self.wal.ENTRY_HEADER_FMT, entry_data
                    )

                    if not is_completed:
                        self.logger.debug("WAL file %s has uncompleted records. Skipping.", file_path.name)
                        return False

                return True
        except (OSError, struct.error) as e:
            self.logger.error("I/O error while checking file status %s: %s", file_path.name, e)
            return False

    def archive_file(self, file_path: Path) -> None:
        """Compresses a sealed WAL file into a gzip archive and removes the original.

        The archive is written atomically with exclusive creation (``"xb"`` mode)
        so that an existing archive from a previous partial run is never silently
        overwritten. The original WAL file is only deleted after the archive has
        been fully written and closed. If deletion of the original fails, the
        valid archive is preserved.

        Args:
            file_path: Path to the sealed WAL file to archive.
        """
        file_name: str = file_path.name
        archive_path: Path = Path(self.ARCHIVE_DIR) / f"archive_{file_name}.gz"

        if archive_path.exists():
            self.logger.warning(
                "Archive %s already exists; skipping %s to avoid overwriting existing data.",
                archive_path.name,
                file_name,
            )
            return

        self.logger.info("Archiving started: %s -> %s", file_name, archive_path.name)
        try:
            with file_path.open("rb") as f_in:
                with gzip.open(archive_path, "xb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except FileExistsError:
            self.logger.warning(
                "Archive %s was created concurrently; skipping %s.",
                archive_path.name,
                file_name,
            )
            return
        except OSError as e:
            self.logger.error("Compression failed for %s: %s", file_name, e)
            if archive_path.exists():
                archive_path.unlink()
            return
        try:
            file_path.unlink()
            self.logger.info("Archiving completed: %s -> %s", file_name, archive_path.name)
        except OSError as e:
            self.logger.error(
                "Failed to remove original WAL file %s after successful archiving: %s. The archive at %s is intact.",
                file_name,
                e,
                archive_path.name,
            )
