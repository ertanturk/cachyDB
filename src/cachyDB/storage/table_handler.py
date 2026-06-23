"""Module for the TableHandler class in the cachyDB database.

This module manages the on-disk representation of a single table as a file
of fixed-size pages.  Each page is a serialised
:class:`~cachyDB.pager.pager.Pager` object of exactly ``PAGE_SIZE`` bytes.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import IO

from cachyDB.pager.pager import PageError, Pager


class TableHandlerError(Exception):
    """Base exception for table-handler errors."""


class TableHandler:
    """Manages a single table stored as a file of fixed-size Pager pages.

    The table is persisted as a file named ``<table_name>.cachy`` inside
    ``tables_dir`` (or ``TABLES_DIR`` by default).  Every page is a fully
    serialised :class:`~cachyDB.pager.pager.Pager`; pages are allocated,
    read, and written through the methods below.

    Attributes:
        table_name (str): Logical name of the table; also used as the file
            stem.
        file_path (Path): Path to the ``.cachy`` file on disk.
        file (IO[bytes] | None): Open binary file handle, or ``None`` if the
            handler was never successfully opened or has already been closed.

    Example:
        >>> h = TableHandler("users", tables_dir=Path("/tmp/cachy"))
        >>> pid = h.allocate_page()
        >>> page = h.read_page(pid)
        >>> h.write_page(page)
        >>> h.close()
    """

    TABLES_DIR: str = "data/tables"
    PAGE_SIZE: int = Pager.PAGE_SIZE

    def __init__(
        self,
        table_name: str,
        tables_dir: str | Path | None = None,
    ) -> None:
        """Opens or creates a table file, creating parent directories as needed.

        Args:
            table_name: Logical name of the table.  Used as the ``.cachy``
                file stem.  Must not be empty.
            tables_dir: Directory in which to store ``.cachy`` files.
                Defaults to ``TABLES_DIR`` relative to the current working
                directory when ``None``.

        Raises:
            ValueError: If ``table_name`` is empty.
            TableHandlerError: If the directory or file cannot be created, or
                if the file cannot be opened.
        """
        if not table_name:
            raise ValueError("table_name must not be empty.")

        self.table_name: str = table_name
        _dir: Path = Path(tables_dir) if tables_dir is not None else Path(self.TABLES_DIR)
        self.file_path: Path = _dir / f"{table_name}.cachy"
        self.logger: logging.Logger = logging.getLogger(__name__)
        self._lock: threading.RLock = threading.RLock()
        self._page_cache: dict[int, Pager] = {}

        # Initialize to None before create_table() so close() is always safe
        # even when __init__ raises partway through.
        self.file: IO[bytes] | None = None
        self.create_table()
        self.logger.debug(f"Opened table file '{self.file_path}' for table '{self.table_name}'.")
        try:
            self.file = self.file_path.open("r+b")
            self.logger.debug(f"Opened table file '{self.file_path}' for table '{self.table_name}'.")
        except OSError as exc:
            raise TableHandlerError(f"Failed to open table file '{self.file_path}': {exc}") from exc

    def create_table(self) -> None:
        """Creates the table directory and file on disk if they do not exist.

        The parent directory receives ``0o700`` permissions (owner
        read/write/execute only).  The ``.cachy`` file receives ``0o600``
        permissions (owner read/write only) when first created.

        Raises:
            TableHandlerError: If the directory or file cannot be created.
        """
        try:
            # Create the *parent directory*, not the file itself.
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.parent.chmod(0o700)
        except OSError as exc:
            raise TableHandlerError(f"Failed to create table directory '{self.file_path.parent}': {exc}") from exc

        try:
            if not self.file_path.exists():
                self.file_path.touch()
                self.file_path.chmod(0o600)
        except OSError as exc:
            raise TableHandlerError(f"Failed to create table file '{self.file_path}': {exc}") from exc

    @property
    def total_pages(self) -> int:
        """Returns the number of complete pages stored in the table file.

        Returns:
            Total page count (zero for an empty or newly created file).

        Raises:
            TableHandlerError: If the file is not open.
        """
        with self._lock:
            self.require_open()
            self.file.seek(0, os.SEEK_END)  # ty: ignore[unresolved-attribute]
            file_size: int = self.file.tell()  # ty: ignore[unresolved-attribute]
            return file_size // self.PAGE_SIZE

    def allocate_page(self) -> int:
        """Appends a new empty page to the table file and returns its page ID.

        The page is written as a fully serialised, empty
        :class:`~cachyDB.pager.pager.Pager` so that :meth:`read_page` can
        load it via :meth:`~cachyDB.pager.pager.Pager.from_bytes` without
        any special-casing.

        Returns:
            The zero-based page ID of the newly allocated page.

        Raises:
            TableHandlerError: If the file is not open.
        """
        with self._lock:
            self.require_open()
            new_page_id: int = self.total_pages
            page: Pager = Pager(page_id=new_page_id)
            self._page_cache[new_page_id] = page
            self.file.seek(new_page_id * self.PAGE_SIZE)  # ty: ignore[unresolved-attribute]
            self.file.write(page.to_bytes())  # ty: ignore[unresolved-attribute]
            self.logger.debug(f"Allocated page {new_page_id}.")
            return new_page_id

    def read_page(self, page_id: int) -> Pager:
        """Reads and deserialises a page from the table file.

        Args:
            page_id: Zero-based index of the page to read.  Must be a
                non-negative integer less than :attr:`total_pages`.

        Returns:
            A fully populated :class:`~cachyDB.pager.pager.Pager` instance.

        Raises:
            ValueError: If ``page_id`` is negative.
            TableHandlerError: If the file is not open, ``page_id`` is out of
                range, the page data is truncated, or the page is corrupt.
        """
        with self._lock:
            self.require_open()
            if page_id < 0:
                raise ValueError(f"page_id must be >= 0, got {page_id}.")

            if page_id in self._page_cache:
                return self._page_cache[page_id]

            total: int = self.total_pages
            if page_id >= total:
                raise TableHandlerError(
                    f"Page ID {page_id} is out of range: table '{self.table_name}' has {total} page(s)."
                )

            self.file.seek(page_id * self.PAGE_SIZE)  # ty: ignore[unresolved-attribute]
            raw_data: bytes = self.file.read(self.PAGE_SIZE)  # ty: ignore[unresolved-attribute]
            if len(raw_data) != self.PAGE_SIZE:
                raise TableHandlerError(
                    f"Page {page_id} in table '{self.table_name}' is truncated: "
                    f"expected {self.PAGE_SIZE} bytes, read {len(raw_data)}."
                )
            try:
                page = Pager.from_bytes(raw_data)
                self.logger.debug(f"Deserialised page {page_id}.")
                self._page_cache[page_id] = page
                return page
            except PageError as exc:
                raise TableHandlerError(
                    f"Failed to deserialise page {page_id} in table '{self.table_name}': {exc}"
                ) from exc

    def write_page(self, page: Pager) -> None:
        """Serialises and writes a page to its position in the table file.

        The page must already have been allocated (``page.page_id`` must be
        in the range ``[0, total_pages)``).

        Args:
            page: The :class:`~cachyDB.pager.pager.Pager` instance to write.

        Raises:
            TableHandlerError: If the file is not open or ``page.page_id``
                has not yet been allocated.
        """
        with self._lock:
            self.require_open()
            total: int = self.total_pages
            if page.page_id >= total:
                raise TableHandlerError(
                    f"Page ID {page.page_id} is out of range: table '{self.table_name}' has {total} page(s)."
                )
            self._page_cache[page.page_id] = page
            self.file.seek(page.page_id * self.PAGE_SIZE)  # ty: ignore[unresolved-attribute]
            self.file.write(page.to_bytes())  # ty: ignore[unresolved-attribute]
            self.logger.debug(f"Written page {page.page_id}.")

    def close(self) -> None:
        """Flushes, syncs, and closes the underlying table file.

        Safe to call even if the file was never successfully opened or has
        already been closed; subsequent calls are a no-op.
        """
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                getattr(self, "_page_cache", {}).clear()
                if self.file is not None and not self.file.closed:
                    self.file.flush()
                    os.fsync(self.file.fileno())
                    self.file.close()
                    self.logger.debug(f"Closed table file '{self.file_path}'.")
        else:
            getattr(self, "_page_cache", {}).clear()
            if self.file is not None and not self.file.closed:
                self.file.flush()
                os.fsync(self.file.fileno())
                self.file.close()
                self.logger.debug(f"Closed table file '{self.file_path}'.")

    def sync(self) -> None:
        """Forces the operating system to write file buffers to physical disk.

        Raises:
            TableHandlerError: If the file is not open.
        """
        with self._lock:
            self.require_open()
            self.file.flush()  # ty: ignore[unresolved-attribute]
            os.fsync(self.file.fileno())  # ty: ignore[unresolved-attribute]
            self.logger.debug(f"Synced table file '{self.file_path}' to physical disk.")

    def require_open(self) -> None:
        """Raises TableHandlerError if the file handle is not currently open.

        Raises:
            TableHandlerError: If ``self.file`` is ``None`` or closed.
        """
        if self.file is None or self.file.closed:
            raise TableHandlerError(f"Table '{self.table_name}' file is not open. Has the handler already been closed?")
