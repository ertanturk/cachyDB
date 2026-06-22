"""Unit tests for the Router engine module in the cachyDB database."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import pytest

from cachyDB.recovery.wal import WAL
from cachyDB.storage.router import _MAX_KEY_BYTES, Router, RouterError
from cachyDB.storage.table_handler import TableHandler

if TYPE_CHECKING:
    from pathlib import Path


class TestRouterInitialization:
    """Tests for Router initialization and configuration."""

    @pytest.fixture(autouse=True)
    def isolate_storage(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with isolated WAL and TableHandler directories."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))
        monkeypatch.setattr(TableHandler, "TABLES_DIR", str(tmp_path / "tables"))

    def test_init_sets_correct_attributes(self) -> None:
        """Router initializes with valid configuration."""
        wal: WAL = WAL()
        router: Router = Router(wal_instance=wal, max_memtable_size=100, cache_capacity=50)

        assert router.wal is wal
        assert router.max_memtable_size == 100
        assert router.cache.capacity == 50
        assert isinstance(router.memtables, dict)
        assert len(router.memtables) == 0

    def test_init_rejects_invalid_memtable_size(self) -> None:
        """Router rejects max_memtable_size less than 1."""
        wal: WAL = WAL()
        with pytest.raises(ValueError, match="must be >= 1"):
            Router(wal_instance=wal, max_memtable_size=0)

    def test_init_rejects_invalid_cache_capacity(self) -> None:
        """Router rejects cache_capacity less than 1."""
        wal: WAL = WAL()
        with pytest.raises(ValueError, match="must be >= 1"):
            Router(wal_instance=wal, cache_capacity=0)


class TestRecordPacking:
    """Tests for packing and unpacking records into binary format."""

    @pytest.fixture
    def router(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Router:
        """Provides a configured Router instance for packing tests."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))
        monkeypatch.setattr(TableHandler, "TABLES_DIR", str(tmp_path / "tables"))
        return Router(wal_instance=WAL())

    def test_pack_unpack_roundtrip(self, router: Router) -> None:
        """Data packed and unpacked should remain identical."""
        key: str = "test_key_123"
        value: dict[str, Any] = {"id": 1, "name": "Alice", "active": True}

        packed: bytes = router._pack_record(key, value)
        unpacked_key, unpacked_value = router._unpack_record(packed)

        assert unpacked_key == key
        assert unpacked_value == value

    def test_pack_rejects_oversized_key(self, router: Router) -> None:
        """Packing rejects keys that exceed the maximum UTF-8 byte limit."""
        long_key: str = "a" * (_MAX_KEY_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds the maximum"):
            router._pack_record(long_key, {"data": 1})

    def test_pack_rejects_unserializable_value(self, router: Router) -> None:
        """Packing rejects values that cannot be JSON serialized."""

        class UnserializableObject:
            pass

        with pytest.raises(RouterError, match="not JSON-serialisable"):
            router._pack_record("key", UnserializableObject())

    def test_unpack_rejects_truncated_header(self, router: Router) -> None:
        """Unpacking rejects payloads that are too short to contain a header."""
        with pytest.raises(ValueError, match="too short to contain a key-length header"):
            router._unpack_record(b"\x00")

    def test_unpack_rejects_truncated_payload(self, router: Router) -> None:
        """Unpacking rejects payloads that claim a length longer than provided bytes."""
        key: str = "test"
        k_len: int = len(key)
        # Create a payload claiming to have a key, but lacking the JSON value byte
        invalid_payload: bytes = struct.pack(">H", k_len) + key.encode("utf-8")

        with pytest.raises(ValueError, match="Payload truncated"):
            router._unpack_record(invalid_payload)


class TestRouterOperations:
    """Tests for GET and SET operations, including cache interactions."""

    @pytest.fixture
    def router(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Router:
        """Provides a Router instance with small limits to test flushes."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))
        monkeypatch.setattr(TableHandler, "TABLES_DIR", str(tmp_path / "tables"))
        # Set max_memtable_size to 3 for easy flushing
        return Router(wal_instance=WAL(), max_memtable_size=3, cache_capacity=10)

    def test_set_validates_inputs(self, router: Router) -> None:
        """SET operation validates input types and emptiness."""
        with pytest.raises(TypeError, match="must both be str"):
            router.set(123, "key", "val")  # ty: ignore[invalid-argument-type]

        with pytest.raises(ValueError, match="must not be empty"):
            router.set("", "key", "val")

    def test_set_populates_memtable_and_cache(self, router: Router) -> None:
        """SET operation successfully writes to MemTable and warms LRU Cache."""
        table: str = "users"
        key: str = "user_1"
        value: str = "Alice"

        result: bool = router.set(table, key, value)
        assert result is True

        # Verify it exists in LRU cache
        assert router.cache.get((table, key)) == value

        # Verify it exists in MemTable
        memtable, _, _ = router._get_table_state(table)
        assert memtable.search(key) == value

    def test_get_cache_hit(self, router: Router) -> None:
        """GET operation retrieves data from LRU Cache successfully."""
        table: str = "users"
        key: str = "user_1"
        value: str = "Alice"

        # Manually inject into cache
        router.cache.put((table, key), value)

        # GET should fetch from cache (returns value directly)
        assert router.get(table, key) == value

    def test_get_memtable_hit(self, router: Router) -> None:
        """GET operation retrieves data from MemTable when not in cache."""
        table: str = "users"
        key: str = "user_2"
        value: str = "Bob"

        router.set(table, key, value)

        # Clear cache to force MemTable lookup
        router.cache.clear()
        assert router.cache.get((table, key)) is None

        # GET should fetch from MemTable and warm up the cache
        assert router.get(table, key) == value
        assert router.cache.get((table, key)) == value

    def test_get_disk_hit(self, router: Router) -> None:
        """GET operation scans disk and retrieves data if missing from RAM."""
        table: str = "users"
        key: str = "user_3"
        value: str = "Charlie"

        router.set(table, key, value)

        # Force flush MemTable to disk
        router.flush_memtable_to_disk(table)

        # Clear cache to force Disk lookup
        router.cache.clear()

        # Verify MemTable is empty and Cache is empty
        memtable, _, _ = router._get_table_state(table)
        assert memtable.size == 0
        assert router.cache.get((table, key)) is None

        # GET should fallback to disk, find it, and warm up the cache
        assert router.get(table, key) == value
        assert router.cache.get((table, key)) == value

    def test_get_miss(self, router: Router) -> None:
        """GET operation returns None for non-existent keys."""
        assert router.get("users", "non_existent") is None


class TestRouterFlushAndShutdown:
    """Tests for automated flushing and shutdown procedures."""

    @pytest.fixture
    def router(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Router:
        """Provides a Router instance."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))
        monkeypatch.setattr(TableHandler, "TABLES_DIR", str(tmp_path / "tables"))
        return Router(wal_instance=WAL(), max_memtable_size=2)

    def test_automatic_flush_on_capacity(self, router: Router) -> None:
        """MemTable flushes to disk automatically when capacity is reached."""
        table: str = "users"

        # Set 1st item -> MemTable size 1
        router.set(table, "user1", "val1")
        memtable, _, handler = router._get_table_state(table)
        assert memtable.size == 1
        assert handler.total_pages == 0

        # Set 2nd item -> Capacity reached (2), automatic flush triggered!
        router.set(table, "user2", "val2")

        # MemTable should be cleared, and pages should be allocated on disk
        assert memtable.size == 0
        assert handler.total_pages > 0

    def test_shutdown_flushes_all_tables_and_clears_cache(self, router: Router) -> None:
        """Shutdown safely flushes remaining MemTables and clears cache."""
        router.set("users", "user1", "val1")
        router.set("orders", "order1", "val1")

        # Manually add something to cache
        router.cache.put(("users", "user1"), "val1")

        # Run shutdown
        router.shutdown()

        # Cache should be empty
        # Notice we test internal dictionary size since there is no cache size property
        assert len(router.cache.cache) == 0

        # MemTables should be empty
        user_memtable, _, user_handler = router._get_table_state("users")
        order_memtable, _, order_handler = router._get_table_state("orders")

        assert user_memtable.size == 0
        assert order_memtable.size == 0

        # Since the files are safely closed by shutdown(), calling total_pages would raise an error.
        # Instead, we directly verify the underlying physical file sizes using pathlib.
        assert user_handler.file_path.exists()
        assert user_handler.file_path.stat().st_size > 0
        assert order_handler.file_path.exists()
        assert order_handler.file_path.stat().st_size > 0

        # Verify that handlers correctly report closed states
        assert user_handler.file is not None
        assert user_handler.file.closed is True
        assert order_handler.file is not None
