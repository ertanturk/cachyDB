"""Tests for the TableHandler class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cachyDB.pager.pager import Pager
from cachyDB.storage.table_handler import TableHandler, TableHandlerError


class TestInit:
    """Tests for TableHandler.__init__ and create_table."""

    def test_creates_tables_directory(self, tmp_path: Path) -> None:
        """Init must create the tables directory."""
        tables_dir = tmp_path / "tables"
        h = TableHandler("t", tables_dir=tables_dir)
        assert tables_dir.is_dir()
        h.close()

    def test_creates_cachy_file(self, tmp_path: Path) -> None:
        """Init must create a .cachy file inside the tables directory."""
        h = TableHandler("users", tables_dir=tmp_path)
        assert (tmp_path / "users.cachy").is_file()
        h.close()

    def test_file_is_not_a_directory(self, tmp_path: Path) -> None:
        """The .cachy path must be a regular file, not a directory."""
        h = TableHandler("t", tables_dir=tmp_path)
        assert (tmp_path / "t.cachy").is_file()
        assert not (tmp_path / "t.cachy").is_dir()
        h.close()

    def test_file_handle_is_open(self, tmp_path: Path) -> None:
        """After init, self.file must be a non-None open file handle."""
        h = TableHandler("t", tables_dir=tmp_path)
        assert h.file is not None
        assert not h.file.closed
        h.close()

    def test_empty_table_name_raises(self, tmp_path: Path) -> None:
        """An empty table_name must raise ValueError."""
        with pytest.raises(ValueError, match="table_name must not be empty"):
            TableHandler("", tables_dir=tmp_path)

    def test_idempotent_creation(self, tmp_path: Path) -> None:
        """Opening the same table twice must not raise (existing file/dir ok)."""
        h1 = TableHandler("idm", tables_dir=tmp_path)
        h1.close()
        h2 = TableHandler("idm", tables_dir=tmp_path)
        h2.close()

    def test_file_initialised_to_none_before_open(self, tmp_path: Path) -> None:
        """self.file must be set to None initially so close() never crashes."""
        # Simulate a failed open by pointing to a non-creatable path
        h = TableHandler.__new__(TableHandler)
        h.table_name = "ghost"
        h.file_path = tmp_path / "ghost.cachy"
        h.file = None
        # close() must be safe even without a real file handle
        h.close()

    def test_tables_dir_defaults_to_class_constant(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no tables_dir is supplied, TABLES_DIR is used (relative to CWD)."""
        monkeypatch.chdir(tmp_path)
        h = TableHandler("default_dir_test")
        assert (tmp_path / TableHandler.TABLES_DIR / "default_dir_test.cachy").is_file()
        h.close()


class TestTotalPages:
    """Tests for TableHandler.total_pages."""

    def test_empty_file_has_zero_pages(self, tmp_path: Path) -> None:
        """A freshly created table must have 0 pages."""
        h = TableHandler("t", tables_dir=tmp_path)
        assert h.total_pages == 0
        h.close()

    def test_pages_increase_after_allocation(self, tmp_path: Path) -> None:
        """total_pages must increment by 1 after each allocate_page call."""
        h = TableHandler("t", tables_dir=tmp_path)
        for expected in range(1, 5):
            h.allocate_page()
            assert h.total_pages == expected
        h.close()

    def test_total_pages_after_close_raises(self, tmp_path: Path) -> None:
        """Accessing total_pages after close must raise TableHandlerError."""
        h = TableHandler("t", tables_dir=tmp_path)
        h.close()
        with pytest.raises(TableHandlerError, match="not open"):
            _ = h.total_pages


class TestAllocatePage:
    """Tests for TableHandler.allocate_page."""

    def test_first_allocation_returns_zero(self, tmp_path: Path) -> None:
        """The first allocation must return page ID 0."""
        h = TableHandler("t", tables_dir=tmp_path)
        assert h.allocate_page() == 0
        h.close()

    def test_sequential_allocation_ids(self, tmp_path: Path) -> None:
        """Successive allocations must return consecutive page IDs."""
        h = TableHandler("t", tables_dir=tmp_path)
        ids = [h.allocate_page() for _ in range(5)]
        assert ids == list(range(5))
        h.close()

    def test_allocated_page_is_valid_pager(self, tmp_path: Path) -> None:
        """A freshly allocated page must deserialise to a valid empty Pager."""
        h = TableHandler("t", tables_dir=tmp_path)
        pid = h.allocate_page()
        page = h.read_page(pid)
        assert page.page_id == pid
        assert page.slots == []
        assert page.free_space == Pager.PAGE_SIZE - Pager.HEADER_SIZE
        h.close()

    def test_allocated_page_has_correct_page_id(self, tmp_path: Path) -> None:
        """The page_id inside the serialised page must match the allocation ID."""
        h = TableHandler("t", tables_dir=tmp_path)
        for _ in range(3):
            pid = h.allocate_page()
            page = h.read_page(pid)
            assert page.page_id == pid
        h.close()

    def test_allocate_after_close_raises(self, tmp_path: Path) -> None:
        """Allocating after close must raise TableHandlerError."""
        h = TableHandler("t", tables_dir=tmp_path)
        h.close()
        with pytest.raises(TableHandlerError, match="not open"):
            h.allocate_page()

    def test_allocated_page_passes_from_bytes(self, tmp_path: Path) -> None:
        """Raw bytes of an allocated page must pass Pager.from_bytes directly."""
        h = TableHandler("t", tables_dir=tmp_path)
        pid = h.allocate_page()
        h.file.seek(pid * Pager.PAGE_SIZE)  # ty: ignore[unresolved-attribute]
        raw = h.file.read(Pager.PAGE_SIZE)  # ty: ignore[unresolved-attribute]
        # Must not raise PageCorruptionError
        page = Pager.from_bytes(raw)
        assert page.page_id == pid
        h.close()


class TestReadPage:
    """Tests for TableHandler.read_page."""

    def test_read_returns_pager(self, tmp_path: Path) -> None:
        """read_page must return a Pager instance."""
        h = TableHandler("t", tables_dir=tmp_path)
        pid = h.allocate_page()
        page = h.read_page(pid)
        assert isinstance(page, Pager)
        h.close()

    def test_read_reflects_written_records(self, tmp_path: Path) -> None:
        """Records written via write_page must be readable back with read_page."""
        h = TableHandler("t", tables_dir=tmp_path)
        pid = h.allocate_page()
        page = h.read_page(pid)
        sid = page.insert_record(b"hello world")
        h.write_page(page)

        page2 = h.read_page(pid)
        assert page2.get_record(sid) == b"hello world"
        h.close()

    def test_negative_page_id_raises_value_error(self, tmp_path: Path) -> None:
        """Negative page_id must raise ValueError."""
        h = TableHandler("t", tables_dir=tmp_path)
        with pytest.raises(ValueError, match="page_id must be >= 0"):
            h.read_page(-1)
        h.close()

    def test_out_of_range_page_id_raises(self, tmp_path: Path) -> None:
        """page_id >= total_pages must raise TableHandlerError."""
        h = TableHandler("t", tables_dir=tmp_path)
        with pytest.raises(TableHandlerError, match="out of range"):
            h.read_page(0)
        h.close()

    def test_read_after_close_raises(self, tmp_path: Path) -> None:
        """Reading after close must raise TableHandlerError."""
        h = TableHandler("t", tables_dir=tmp_path)
        h.allocate_page()
        h.close()
        with pytest.raises(TableHandlerError, match="not open"):
            h.read_page(0)

    def test_read_multiple_pages_independent(self, tmp_path: Path) -> None:
        """Each page must hold its own data independently."""
        h = TableHandler("t", tables_dir=tmp_path)
        pids = [h.allocate_page() for _ in range(3)]
        for i, pid in enumerate(pids):
            page = h.read_page(pid)
            page.insert_record(f"page-{i}".encode())
            h.write_page(page)

        for i, pid in enumerate(pids):
            page = h.read_page(pid)
            assert page.get_record(0) == f"page-{i}".encode()
        h.close()


class TestWritePage:
    """Tests for TableHandler.write_page."""

    def test_write_and_read_back(self, tmp_path: Path) -> None:
        """A page written via write_page must be readable back via read_page."""
        h = TableHandler("t", tables_dir=tmp_path)
        pid = h.allocate_page()
        page = h.read_page(pid)
        page.insert_record(b"data")
        h.write_page(page)

        restored = h.read_page(pid)
        assert restored.get_record(0) == b"data"
        h.close()

    def test_write_unallocated_page_raises(self, tmp_path: Path) -> None:
        """Writing a Pager whose page_id >= total_pages must raise TableHandlerError."""
        h = TableHandler("t", tables_dir=tmp_path)
        phantom = Pager(page_id=99)
        with pytest.raises(TableHandlerError, match="out of range"):
            h.write_page(phantom)
        h.close()

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        """Writing after close must raise TableHandlerError."""
        h = TableHandler("t", tables_dir=tmp_path)
        pid = h.allocate_page()
        page = h.read_page(pid)
        h.close()
        with pytest.raises(TableHandlerError, match="not open"):
            h.write_page(page)

    def test_overwrite_preserves_page_id(self, tmp_path: Path) -> None:
        """Re-writing a page must not corrupt its page_id."""
        h = TableHandler("t", tables_dir=tmp_path)
        pid = h.allocate_page()
        page = h.read_page(pid)
        page.insert_record(b"overwrite-me")
        h.write_page(page)
        page.delete_record(0)
        h.write_page(page)

        restored = h.read_page(pid)
        assert restored.page_id == pid
        assert restored.get_record(0) is None
        h.close()


class TestClose:
    """Tests for TableHandler.close."""

    def test_close_marks_file_closed(self, tmp_path: Path) -> None:
        """After close(), the file handle must be closed."""
        h = TableHandler("t", tables_dir=tmp_path)
        h.close()
        assert h.file is None or h.file.closed

    def test_double_close_is_noop(self, tmp_path: Path) -> None:
        """Calling close() twice must not raise."""
        h = TableHandler("t", tables_dir=tmp_path)
        h.close()
        h.close()

    def test_close_with_none_file_is_safe(self, tmp_path: Path) -> None:
        """close() must be safe when self.file is still None."""
        h = TableHandler.__new__(TableHandler)
        h.table_name = "ghost"
        h.file = None
        h.close()

    def test_data_persists_across_reopen(self, tmp_path: Path) -> None:
        """Data written before close must be readable after reopening the table."""
        h = TableHandler("persist", tables_dir=tmp_path)
        pid = h.allocate_page()
        page = h.read_page(pid)
        page.insert_record(b"persistent")
        h.write_page(page)
        h.close()

        h2 = TableHandler("persist", tables_dir=tmp_path)
        page2 = h2.read_page(pid)
        assert page2.get_record(0) == b"persistent"
        h2.close()

    def test_all_operations_raise_after_close(self, tmp_path: Path) -> None:
        """total_pages, allocate_page, read_page, and write_page must all raise after close."""
        h = TableHandler("t", tables_dir=tmp_path)
        h.allocate_page()
        page = h.read_page(0)
        h.close()

        with pytest.raises(TableHandlerError, match="not open"):
            _ = h.total_pages
        with pytest.raises(TableHandlerError, match="not open"):
            h.allocate_page()
        with pytest.raises(TableHandlerError, match="not open"):
            h.read_page(0)
        with pytest.raises(TableHandlerError, match="not open"):
            h.write_page(page)
