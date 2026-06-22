"""Tests for the Pager class and related exceptions."""

from __future__ import annotations

import struct
import zlib

import pytest

from cachyDB.pager.pager import PageCorruptionError, PageOverflowError, Pager


def _make_page_bytes(page_id: int = 0, flags: int = 0) -> bytes:
    """Return a valid serialised empty page."""
    return Pager(page_id, flags).to_bytes()


def _corrupt_checksum(raw: bytes) -> bytes:
    """Flip one bit in the checksum field of a serialised page."""
    data = bytearray(raw)
    data[Pager.CHECKSUM_OFFSET] ^= 0xFF
    return bytes(data)


class TestConstants:
    """Verify that derived class constants have the expected values."""

    def test_header_size(self) -> None:
        """HEADER_SIZE must be 20 bytes for the defined format."""
        assert Pager.HEADER_SIZE == 20

    def test_slot_size(self) -> None:
        """SLOT_SIZE must be 4 bytes (two unsigned shorts)."""
        assert Pager.SLOT_SIZE == 4

    def test_checksum_offset(self) -> None:
        """CHECKSUM_OFFSET must be HEADER_SIZE minus CHECKSUM_SIZE."""
        assert Pager.CHECKSUM_OFFSET == Pager.HEADER_SIZE - Pager.CHECKSUM_SIZE

    def test_header_format_consistency(self) -> None:
        """HEADER_FORMAT must be HEADER_FORMAT_NO_CHECKSUM with a trailing I field."""
        assert Pager.HEADER_FORMAT == Pager.HEADER_FORMAT_NO_CHECKSUM + " I"


class TestInit:
    """Tests for Pager.__init__ parameter validation."""

    def test_valid_page(self) -> None:
        """A page with page_id=0 and default flags should be created without error."""
        p = Pager(page_id=0)
        assert p.page_id == 0
        assert p.flags == 0
        assert p.lower_offset == Pager.HEADER_SIZE
        assert p.upper_offset == Pager.PAGE_SIZE
        assert p.slots == []
        assert len(p.data) == Pager.PAGE_SIZE

    def test_negative_page_id_raises(self) -> None:
        """Negative page_id must raise ValueError."""
        with pytest.raises(ValueError, match="page_id must be >= 0"):
            Pager(page_id=-1)

    def test_flags_too_large_raises(self) -> None:
        """flags exceeding MAX_FLAGS must raise ValueError."""
        with pytest.raises(ValueError, match="flags must be in"):
            Pager(page_id=0, flags=Pager.MAX_FLAGS + 1)

    def test_max_flags_accepted(self) -> None:
        """flags == MAX_FLAGS is valid."""
        p = Pager(page_id=0, flags=Pager.MAX_FLAGS)
        assert p.flags == Pager.MAX_FLAGS


class TestSpaceAccounting:
    """Tests for free_space and has_space_for."""

    def test_initial_free_space(self) -> None:
        """A fresh page must have PAGE_SIZE - HEADER_SIZE bytes of free space."""
        p = Pager(page_id=0)
        assert p.free_space == Pager.PAGE_SIZE - Pager.HEADER_SIZE

    def test_free_space_decreases_after_insert(self) -> None:
        """Inserting a record must reduce free_space by SLOT_SIZE + payload size."""
        p = Pager(page_id=0)
        before = p.free_space
        p.insert_record(b"hello")
        assert p.free_space == before - Pager.SLOT_SIZE - len(b"hello")

    def test_has_space_for_negative_raises(self) -> None:
        """Negative payload_length must raise ValueError."""
        p = Pager(page_id=0)
        with pytest.raises(ValueError, match="payload_length must be >= 0"):
            p.has_space_for(-1)

    def test_has_space_for_zero(self) -> None:
        """Zero-length query is valid and returns True on an empty page."""
        p = Pager(page_id=0)
        assert p.has_space_for(0) is True

    def test_has_space_for_exact_fit(self) -> None:
        """A payload that exactly fills the remaining space must return True."""
        p = Pager(page_id=0)
        exact = p.free_space - Pager.SLOT_SIZE
        assert p.has_space_for(exact) is True

    def test_has_space_for_one_byte_over(self) -> None:
        """A payload one byte larger than available space must return False."""
        p = Pager(page_id=0)
        over = p.free_space - Pager.SLOT_SIZE + 1
        assert p.has_space_for(over) is False


class TestInsertRecord:
    """Tests for Pager.insert_record."""

    def test_insert_returns_slot_id_zero(self) -> None:
        """First insert must return slot ID 0."""
        p = Pager(page_id=0)
        assert p.insert_record(b"a") == 0

    def test_insert_sequential_slot_ids(self) -> None:
        """Subsequent inserts must return incrementing slot IDs."""
        p = Pager(page_id=0)
        for expected_id in range(5):
            assert p.insert_record(f"record-{expected_id}".encode()) == expected_id

    def test_insert_non_bytes_raises_type_error(self) -> None:
        """Passing a non-bytes payload must raise TypeError."""
        p = Pager(page_id=0)
        with pytest.raises(TypeError, match="payload must be bytes"):
            p.insert_record("not bytes")  # ty: ignore[invalid-argument-type]

    def test_insert_empty_bytes_raises_value_error(self) -> None:
        """An empty payload must raise ValueError."""
        p = Pager(page_id=0)
        with pytest.raises(ValueError, match="payload must not be empty"):
            p.insert_record(b"")

    def test_insert_overflow_raises(self) -> None:
        """Inserting when there is no space left must raise PageOverflowError."""
        p = Pager(page_id=0)
        big = b"x" * (p.free_space - Pager.SLOT_SIZE)
        p.insert_record(big)
        with pytest.raises(PageOverflowError):
            p.insert_record(b"y")

    def test_insert_updates_offsets(self) -> None:
        """After insert, lower_offset must grow and upper_offset must shrink."""
        p = Pager(page_id=0)
        payload = b"test"
        p.insert_record(payload)
        assert p.lower_offset == Pager.HEADER_SIZE + Pager.SLOT_SIZE
        assert p.upper_offset == Pager.PAGE_SIZE - len(payload)


class TestGetRecord:
    """Tests for Pager.get_record."""

    def test_get_returns_inserted_payload(self) -> None:
        """get_record must return the exact bytes that were inserted."""
        p = Pager(page_id=0)
        sid = p.insert_record(b"payload")
        assert p.get_record(sid) == b"payload"

    def test_get_deleted_returns_none(self) -> None:
        """get_record must return None for a deleted slot."""
        p = Pager(page_id=0)
        sid = p.insert_record(b"data")
        p.delete_record(sid)
        assert p.get_record(sid) is None

    def test_get_negative_slot_id_raises(self) -> None:
        """Negative slot_id must raise ValueError."""
        p = Pager(page_id=0)
        with pytest.raises(ValueError, match="Invalid slot ID"):
            p.get_record(-1)

    def test_get_out_of_range_slot_id_raises(self) -> None:
        """slot_id >= number of slots must raise ValueError."""
        p = Pager(page_id=0)
        p.insert_record(b"x")
        with pytest.raises(ValueError, match="Invalid slot ID"):
            p.get_record(1)

    def test_get_on_empty_page_raises(self) -> None:
        """Any slot_id on an empty page must raise ValueError."""
        p = Pager(page_id=0)
        with pytest.raises(ValueError, match="Invalid slot ID"):
            p.get_record(0)

    def test_get_multiple_records_independent(self) -> None:
        """Multiple records must be retrievable independently."""
        p = Pager(page_id=0)
        ids = [p.insert_record(f"rec{i}".encode()) for i in range(10)]
        for i, sid in enumerate(ids):
            assert p.get_record(sid) == f"rec{i}".encode()


class TestDeleteRecord:
    """Tests for Pager.delete_record."""

    def test_delete_marks_slot_tombstone(self) -> None:
        """After deletion, the slot entry must be (0, 0)."""
        p = Pager(page_id=0)
        sid = p.insert_record(b"secret")
        p.delete_record(sid)
        assert p.slots[sid] == (0, 0)

    def test_delete_zeros_payload_bytes(self) -> None:
        """The payload bytes in data must be zeroed after deletion."""
        p = Pager(page_id=0)
        payload = b"sensitive"
        sid = p.insert_record(payload)
        offset = p.upper_offset  # captured after insert moved it back
        p.delete_record(sid)
        assert p.data[offset : offset + len(payload)] == bytes(len(payload))

    def test_delete_negative_slot_id_raises(self) -> None:
        """Negative slot_id must raise ValueError."""
        p = Pager(page_id=0)
        with pytest.raises(ValueError, match="Invalid slot ID"):
            p.delete_record(-1)

    def test_delete_out_of_range_raises(self) -> None:
        """slot_id >= number of slots must raise ValueError."""
        p = Pager(page_id=0)
        with pytest.raises(ValueError, match="Invalid slot ID"):
            p.delete_record(0)

    def test_double_delete_raises(self) -> None:
        """Deleting an already-deleted slot must raise ValueError."""
        p = Pager(page_id=0)
        sid = p.insert_record(b"once")
        p.delete_record(sid)
        with pytest.raises(ValueError, match="already deleted"):
            p.delete_record(sid)

    def test_delete_does_not_affect_other_records(self) -> None:
        """Deleting one slot must not corrupt neighbouring records."""
        p = Pager(page_id=0)
        s0 = p.insert_record(b"keep-me")
        s1 = p.insert_record(b"delete-me")
        s2 = p.insert_record(b"also-keep")
        p.delete_record(s1)
        assert p.get_record(s0) == b"keep-me"
        assert p.get_record(s2) == b"also-keep"


class TestCompact:
    """Tests for Pager.compact."""

    def test_compact_preserves_active_records(self) -> None:
        """Active records must be retrievable after compaction."""
        p = Pager(page_id=0)
        s0 = p.insert_record(b"alpha")
        s1 = p.insert_record(b"beta")
        s2 = p.insert_record(b"gamma")
        p.delete_record(s1)
        p.compact()
        assert p.get_record(s0) == b"alpha"
        assert p.get_record(s1) is None
        assert p.get_record(s2) == b"gamma"

    def test_compact_increases_free_space(self) -> None:
        """Compacting a page with deletions must produce more free space."""
        p = Pager(page_id=0)
        s0 = p.insert_record(b"x" * 100)
        p.insert_record(b"y" * 100)
        p.delete_record(s0)
        free_before = p.free_space
        p.compact()
        assert p.free_space > free_before

    def test_compact_zeros_free_area(self) -> None:
        """The free area between lower_offset and upper_offset must be all zeros."""
        p = Pager(page_id=0)
        sid = p.insert_record(b"stale")
        p.delete_record(sid)
        p.compact()
        free_region = p.data[p.lower_offset : p.upper_offset]
        assert free_region == bytes(len(free_region))

    def test_compact_on_empty_page_is_noop(self) -> None:
        """Compacting an empty page must not raise and must not change offsets."""
        p = Pager(page_id=0)
        lo, up = p.lower_offset, p.upper_offset
        p.compact()
        assert p.lower_offset == lo
        assert p.upper_offset == up

    def test_compact_preserves_slot_count(self) -> None:
        """Slot count must not change after compaction."""
        p = Pager(page_id=0)
        for _ in range(5):
            p.insert_record(b"item")
        p.delete_record(2)
        p.compact()
        assert len(p.slots) == 5


class TestSerialisationRoundTrip:
    """Tests for Pager.to_bytes and Pager.from_bytes."""

    def test_round_trip_empty_page(self) -> None:
        """An empty page must survive a serialise/deserialise cycle."""
        original = Pager(page_id=42, flags=3)
        restored = Pager.from_bytes(original.to_bytes())
        assert restored.page_id == 42
        assert restored.flags == 3
        assert restored.slots == []
        assert restored.lower_offset == Pager.HEADER_SIZE
        assert restored.upper_offset == Pager.PAGE_SIZE

    def test_round_trip_with_records(self) -> None:
        """Records must survive a serialise/deserialise cycle intact."""
        p = Pager(page_id=7)
        payloads = [b"first", b"second", b"third"]
        ids = [p.insert_record(pl) for pl in payloads]
        restored = Pager.from_bytes(p.to_bytes())
        for sid, pl in zip(ids, payloads):
            assert restored.get_record(sid) == pl

    def test_round_trip_with_deletions(self) -> None:
        """Deleted slots must remain deleted after a round-trip."""
        p = Pager(page_id=1)
        s0 = p.insert_record(b"keep")
        s1 = p.insert_record(b"gone")
        p.delete_record(s1)
        restored = Pager.from_bytes(p.to_bytes())
        assert restored.get_record(s0) == b"keep"
        assert restored.get_record(s1) is None

    def test_to_bytes_length(self) -> None:
        """to_bytes must return exactly PAGE_SIZE bytes."""
        raw = Pager(page_id=0).to_bytes()
        assert len(raw) == Pager.PAGE_SIZE

    def test_to_bytes_is_idempotent(self) -> None:
        """Calling to_bytes twice must return the same bytes."""
        p = Pager(page_id=0)
        p.insert_record(b"data")
        assert p.to_bytes() == p.to_bytes()


class TestFromBytesCorruption:
    """Tests for PageCorruptionError detection in Pager.from_bytes."""

    def test_wrong_size_raises(self) -> None:
        """Data shorter than PAGE_SIZE must raise PageCorruptionError."""
        with pytest.raises(PageCorruptionError, match="Invalid page size"):
            Pager.from_bytes(b"\x00" * (Pager.PAGE_SIZE - 1))

    def test_wrong_magic_raises(self) -> None:
        """Wrong magic bytes must raise PageCorruptionError."""
        data = bytearray(_make_page_bytes())
        data[0:4] = b"XXXX"
        # Fix checksum so magic check is reached
        data[Pager.CHECKSUM_OFFSET : Pager.HEADER_SIZE] = b"\x00\x00\x00\x00"
        checksum = zlib.crc32(data) & 0xFFFFFFFF
        data[Pager.CHECKSUM_OFFSET : Pager.HEADER_SIZE] = struct.pack("> I", checksum)
        with pytest.raises(PageCorruptionError, match="magic"):
            Pager.from_bytes(bytes(data))

    def test_bad_checksum_raises(self) -> None:
        """A flipped checksum byte must raise PageCorruptionError."""
        with pytest.raises(PageCorruptionError, match="Checksum mismatch"):
            Pager.from_bytes(_corrupt_checksum(_make_page_bytes()))

    def test_inconsistent_lower_offset_raises(self) -> None:
        """A lower_offset that does not match num_slots must raise PageCorruptionError."""
        p = Pager(page_id=0)
        raw = bytearray(p.to_bytes())
        # Manually set lower_offset to an inconsistent value (e.g. HEADER_SIZE + 1)
        # while num_slots stays 0; then recompute the checksum.
        struct.pack_into("> H", raw, 8, Pager.HEADER_SIZE + 1)  # lower field offset = 8
        raw[Pager.CHECKSUM_OFFSET : Pager.HEADER_SIZE] = b"\x00\x00\x00\x00"
        csum = zlib.crc32(raw) & 0xFFFFFFFF
        struct.pack_into("> I", raw, Pager.CHECKSUM_OFFSET, csum)
        with pytest.raises(PageCorruptionError, match="inconsistent"):
            Pager.from_bytes(bytes(raw))

    def test_upper_below_lower_raises(self) -> None:
        """upper_offset < lower_offset must raise PageCorruptionError."""
        p = Pager(page_id=0)
        raw = bytearray(p.to_bytes())
        # Set upper_offset to HEADER_SIZE - 1 (below lower_offset = HEADER_SIZE)
        struct.pack_into("> H", raw, 10, Pager.HEADER_SIZE - 1)  # upper field offset = 10
        raw[Pager.CHECKSUM_OFFSET : Pager.HEADER_SIZE] = b"\x00\x00\x00\x00"
        csum = zlib.crc32(raw) & 0xFFFFFFFF
        struct.pack_into("> I", raw, Pager.CHECKSUM_OFFSET, csum)
        with pytest.raises(PageCorruptionError, match="upper_offset"):
            Pager.from_bytes(bytes(raw))

    def test_slot_with_out_of_bounds_offset_raises(self) -> None:
        """A slot whose record region extends past PAGE_SIZE must raise PageCorruptionError."""
        p = Pager(page_id=0)
        p.insert_record(b"legit")
        raw = bytearray(p.to_bytes())
        # Overwrite the slot entry: set offset to PAGE_SIZE - 1, length to 10
        # (so offset + length > PAGE_SIZE)
        slot_start = Pager.HEADER_SIZE
        struct.pack_into("> H H", raw, slot_start, Pager.PAGE_SIZE - 1, 10)
        raw[Pager.CHECKSUM_OFFSET : Pager.HEADER_SIZE] = b"\x00\x00\x00\x00"
        csum = zlib.crc32(raw) & 0xFFFFFFFF
        struct.pack_into("> I", raw, Pager.CHECKSUM_OFFSET, csum)
        with pytest.raises(PageCorruptionError, match="outside the record area"):
            Pager.from_bytes(bytes(raw))

    def test_tombstone_with_nonzero_length_raises(self) -> None:
        """A slot with offset=0 but non-zero length must raise PageCorruptionError."""
        p = Pager(page_id=0)
        p.insert_record(b"data")
        raw = bytearray(p.to_bytes())
        # Overwrite slot: offset=0, length=5 — invalid tombstone
        struct.pack_into("> H H", raw, Pager.HEADER_SIZE, 0, 5)
        raw[Pager.CHECKSUM_OFFSET : Pager.HEADER_SIZE] = b"\x00\x00\x00\x00"
        csum = zlib.crc32(raw) & 0xFFFFFFFF
        struct.pack_into("> I", raw, Pager.CHECKSUM_OFFSET, csum)
        with pytest.raises(PageCorruptionError, match="non-zero length"):
            Pager.from_bytes(bytes(raw))
