"""Unit tests for the WAL recovery module."""

from __future__ import annotations

import struct
import threading
import time
import zlib
from pathlib import Path

import pytest

from cachyDB.recovery.wal import WAL, WALCapacityError, WALCorruptionError, WALError


def create_test_payload(data: bytes) -> bytes:
    """Create a test payload with length prefix (simulating Record.pack_record structure).

    Args:
        data: Raw data bytes to wrap.

    Returns:
        Payload with 4-byte big-endian length prefix.
    """
    length = len(data) + 4  # Include length prefix itself
    return length.to_bytes(4, byteorder="big") + data


class TestWALInitialization:
    """Tests for WAL initialization and directory management."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_init_creates_wal_directory(self, tmp_path: Path) -> None:
        """WAL initialization creates the directory if it doesn't exist."""
        wal = WAL()
        assert Path(wal.WAL_FILE_PATH).exists()
        assert Path(wal.WAL_FILE_PATH).is_dir()

    def test_init_sets_directory_permissions(self, tmp_path: Path) -> None:
        """WAL directory has restricted permissions (owner only)."""
        wal = WAL()
        wal_path = Path(wal.WAL_FILE_PATH)
        permissions = oct(wal_path.stat().st_mode)[-3:]
        assert permissions == "700"

    def test_init_starts_with_tx_id_zero_when_no_files(self) -> None:
        """WAL starts with tx_id=0 when no existing WAL files."""
        wal = WAL()
        assert wal.global_tx_id == 0

    def test_init_initializes_empty_buffer(self) -> None:
        """WAL initializes with empty buffer and state."""
        wal = WAL()
        assert wal.wal_buffer == []
        assert wal.ram_completed_tx == set()
        assert isinstance(wal.lock, threading.Lock)


class TestTransactionIDValidation:
    """Tests for transaction ID validation."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_validate_tx_id_accepts_zero(self) -> None:
        """Transaction ID 0 is valid."""
        wal = WAL()
        wal.validate_tx_id(0)  # Should not raise

    def test_validate_tx_id_accepts_positive_values(self) -> None:
        """Positive transaction IDs are valid."""
        wal = WAL()
        wal.validate_tx_id(1)
        wal.validate_tx_id(1000)
        wal.validate_tx_id(1000000)

    def test_validate_tx_id_rejects_negative_values(self) -> None:
        """Negative transaction IDs are rejected."""
        wal = WAL()
        with pytest.raises(ValueError, match="non-negative"):
            wal.validate_tx_id(-1)
        with pytest.raises(ValueError, match="non-negative"):
            wal.validate_tx_id(-100)

    def test_validate_tx_id_rejects_excessive_values(self) -> None:
        """Transaction IDs exceeding 2^63-1 are rejected."""
        wal = WAL()
        with pytest.raises(ValueError, match="exceeds maximum"):
            wal.validate_tx_id(2**63)
        with pytest.raises(ValueError, match="exceeds maximum"):
            wal.validate_tx_id(2**64)

    def test_validate_tx_id_accepts_maximum_valid_value(self) -> None:
        """Transaction ID 2^63-1 (max signed 64-bit) is valid."""
        wal = WAL()
        wal.validate_tx_id(2**63 - 1)  # Should not raise


class TestFilePathAndSlotIndexCalculation:
    """Tests for file path and slot index calculation."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_get_file_path_for_first_file(self) -> None:
        """Transaction IDs in first file map to global_0.wal."""
        wal = WAL()
        assert wal.get_file_path(0).name == "global_0.wal"
        assert wal.get_file_path(1).name == "global_0.wal"
        assert wal.get_file_path(WAL.WAL_CAPACITY - 1).name == "global_0.wal"

    def test_get_file_path_for_second_file(self) -> None:
        """Transaction IDs in second file map to global_1.wal."""
        wal = WAL()
        assert wal.get_file_path(WAL.WAL_CAPACITY).name == "global_1.wal"
        assert wal.get_file_path(WAL.WAL_CAPACITY + 1).name == "global_1.wal"
        assert wal.get_file_path(2 * WAL.WAL_CAPACITY - 1).name == "global_1.wal"

    def test_get_file_path_rejects_negative_tx_id(self) -> None:
        """get_file_path rejects negative transaction IDs."""
        wal = WAL()
        with pytest.raises(ValueError, match="non-negative"):
            wal.get_file_path(-1)

    def test_get_slot_index_calculates_correctly(self) -> None:
        """Slot index is tx_id modulo capacity."""
        wal = WAL()
        assert wal.get_slot_index(0) == 0
        assert wal.get_slot_index(1) == 1
        assert wal.get_slot_index(WAL.WAL_CAPACITY - 1) == WAL.WAL_CAPACITY - 1
        assert wal.get_slot_index(WAL.WAL_CAPACITY) == 0
        assert wal.get_slot_index(WAL.WAL_CAPACITY + 5) == 5

    def test_get_slot_index_rejects_negative_tx_id(self) -> None:
        """get_slot_index rejects negative transaction IDs."""
        wal = WAL()
        with pytest.raises(ValueError, match="non-negative"):
            wal.get_slot_index(-1)


class TestWALFilePreallocation:
    """Tests for WAL file pre-allocation."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_pre_allocate_wal_creates_file(self) -> None:
        """pre_allocate_wal creates a new file."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)
        assert file_path.exists()

    def test_pre_allocate_wal_sets_correct_size(self) -> None:
        """pre_allocate_wal creates file with correct size."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)
        expected_size = (
            WAL.WAL_FILE_HEADER_SIZE + (WAL.WAL_CAPACITY * WAL.WAL_ENTRY_HEADER_SIZE) + WAL.PAYLOAD_AREA_SIZE
        )
        assert file_path.stat().st_size == expected_size

    def test_pre_allocate_wal_sets_file_permissions(self) -> None:
        """pre_allocate_wal sets file permissions to owner read/write only."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)
        permissions = oct(file_path.stat().st_mode)[-3:]
        assert permissions == "600"

    def test_pre_allocate_wal_initializes_header(self) -> None:
        """pre_allocate_wal initializes file with valid header."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)

        with file_path.open("rb") as f:
            header_data = f.read(WAL.WAL_FILE_HEADER_SIZE)
            magic, cap, current_rec, used_space, checksum = struct.unpack(WAL.FILE_HEADER_FMT, header_data)

            assert magic == b"WAL1"
            assert cap == WAL.WAL_CAPACITY
            assert current_rec == 0
            assert used_space == 0

            # Verify checksum
            header_without_checksum = struct.pack("> 4s I I Q", magic, cap, current_rec, used_space)
            expected_checksum = zlib.crc32(header_without_checksum) & 0xFFFFFFFF
            assert checksum == expected_checksum


class TestFileHeaderCreation:
    """Tests for file header creation and validation."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_create_file_header_writes_valid_header(self) -> None:
        """create_file_header writes a valid header with correct checksum."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)
        wal.create_file_header(file_path, current_records=10, used_space=1024)

        with file_path.open("rb") as f:
            header_data = f.read(WAL.WAL_FILE_HEADER_SIZE)
            magic, cap, current_rec, used_space, checksum = struct.unpack(WAL.FILE_HEADER_FMT, header_data)

            assert magic == b"WAL1"
            assert cap == WAL.WAL_CAPACITY
            assert current_rec == 10
            assert used_space == 1024

            # Verify checksum
            header_without_checksum = struct.pack("> 4s I I Q", magic, cap, current_rec, used_space)
            expected_checksum = zlib.crc32(header_without_checksum) & 0xFFFFFFFF
            assert checksum == expected_checksum

    def test_create_file_header_rejects_negative_current_records(self) -> None:
        """create_file_header rejects negative current_records."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)

        with pytest.raises(ValueError, match="must be between 0"):
            wal.create_file_header(file_path, current_records=-1, used_space=0)

    def test_create_file_header_rejects_excessive_current_records(self) -> None:
        """create_file_header rejects current_records exceeding capacity."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)

        with pytest.raises(ValueError, match="must be between 0"):
            wal.create_file_header(file_path, current_records=WAL.WAL_CAPACITY + 1, used_space=0)

    def test_create_file_header_rejects_negative_used_space(self) -> None:
        """create_file_header rejects negative used_space."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)

        with pytest.raises(ValueError, match="must be between 0"):
            wal.create_file_header(file_path, current_records=0, used_space=-1)

    def test_create_file_header_rejects_excessive_used_space(self) -> None:
        """create_file_header rejects used_space exceeding payload area size."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)

        with pytest.raises(ValueError, match="must be between 0"):
            wal.create_file_header(file_path, current_records=0, used_space=WAL.PAYLOAD_AREA_SIZE + 1)

    def test_create_file_header_accepts_boundary_values(self) -> None:
        """create_file_header accepts boundary values."""
        wal = WAL()
        file_path = Path(wal.WAL_FILE_PATH) / "test.wal"
        wal.pre_allocate_wal(file_path)

        wal.create_file_header(file_path, current_records=WAL.WAL_CAPACITY, used_space=WAL.PAYLOAD_AREA_SIZE)
        # Should not raise


class TestEntryHeaderCreation:
    """Tests for entry header creation."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_create_entry_header_returns_correct_size(self) -> None:
        """create_entry_header returns header of correct size."""
        wal = WAL()
        entry_header = wal.create_entry_header(tx_id=10, payload_offset=1000, is_completed=False)
        assert len(entry_header) == WAL.WAL_ENTRY_HEADER_SIZE

    def test_create_entry_header_packs_values_correctly(self) -> None:
        """create_entry_header packs values in correct format."""
        wal = WAL()
        entry_header = wal.create_entry_header(tx_id=42, payload_offset=2048, is_completed=True)
        tx_id, timestamp, is_completed, payload_offset = struct.unpack(WAL.ENTRY_HEADER_FMT, entry_header)

        assert tx_id == 42
        assert isinstance(timestamp, float)
        assert timestamp > 0
        assert is_completed is True
        assert payload_offset == 2048

    def test_create_entry_header_rejects_negative_tx_id(self) -> None:
        """create_entry_header rejects negative transaction IDs."""
        wal = WAL()
        with pytest.raises(ValueError, match="non-negative"):
            wal.create_entry_header(tx_id=-1, payload_offset=0, is_completed=False)

    def test_create_entry_header_rejects_negative_payload_offset(self) -> None:
        """create_entry_header rejects negative payload offsets."""
        wal = WAL()
        with pytest.raises(ValueError, match="non-negative"):
            wal.create_entry_header(tx_id=0, payload_offset=-1, is_completed=False)


class TestPayloadValidation:
    """Tests for payload size validation."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_validate_payload_size_accepts_small_payload(self) -> None:
        """validate_payload_size accepts payloads smaller than capacity."""
        wal = WAL()
        payload = b"small payload"
        wal.validate_payload_size(payload, current_used_space=0)  # Should not raise

    def test_validate_payload_size_accepts_maximum_payload(self) -> None:
        """validate_payload_size accepts payload equal to capacity."""
        wal = WAL()
        payload = b"x" * WAL.PAYLOAD_AREA_SIZE
        wal.validate_payload_size(payload, current_used_space=0)  # Should not raise

    def test_validate_payload_size_rejects_oversized_payload(self) -> None:
        """validate_payload_size rejects payloads exceeding capacity."""
        wal = WAL()
        payload = b"x" * (WAL.PAYLOAD_AREA_SIZE + 1)
        with pytest.raises(WALCapacityError, match="exceeds maximum"):
            wal.validate_payload_size(payload, current_used_space=0)

    def test_validate_payload_size_checks_available_space(self) -> None:
        """validate_payload_size rejects payloads that don't fit in available space."""
        wal = WAL()
        payload = b"x" * 1000
        with pytest.raises(WALCapacityError, match="would exceed available space"):
            wal.validate_payload_size(payload, current_used_space=WAL.PAYLOAD_AREA_SIZE - 500)

    def test_validate_payload_size_accepts_payload_fitting_in_available_space(self) -> None:
        """validate_payload_size accepts payloads that fit in available space."""
        wal = WAL()
        payload = b"x" * 500
        wal.validate_payload_size(payload, current_used_space=WAL.PAYLOAD_AREA_SIZE - 1000)  # Should not raise

    def test_validate_payload_size_accepts_empty_payload(self) -> None:
        """validate_payload_size accepts empty payloads."""
        wal = WAL()
        payload = b""
        wal.validate_payload_size(payload, current_used_space=0)  # Should not raise


class TestAppendToBuffer:
    """Tests for appending records to WAL buffer."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_append_to_buffer_assigns_transaction_id(self) -> None:
        """append_to_buffer assigns sequential transaction IDs."""
        wal = WAL()
        tx_id1 = wal.append_to_buffer(b"record1")
        tx_id2 = wal.append_to_buffer(b"record2")
        tx_id3 = wal.append_to_buffer(b"record3")

        assert tx_id1 == 0
        assert tx_id2 == 1
        assert tx_id3 == 2
        assert wal.global_tx_id == 3

    def test_append_to_buffer_adds_to_buffer(self) -> None:
        """append_to_buffer adds records to internal buffer."""
        wal = WAL()
        # Set batch size high to prevent auto-flush
        wal.batch_size = 1000

        wal.append_to_buffer(b"record1")
        wal.append_to_buffer(b"record2")

        assert len(wal.wal_buffer) == 2
        assert wal.wal_buffer[0] == (0, b"record1")
        assert wal.wal_buffer[1] == (1, b"record2")

    def test_append_to_buffer_auto_flushes_on_batch_size(self) -> None:
        """append_to_buffer auto-flushes when batch size is reached."""
        wal = WAL()
        wal.batch_size = 2

        wal.append_to_buffer(b"record1")
        wal.append_to_buffer(b"record2")

        # Buffer should be cleared after flush
        assert len(wal.wal_buffer) == 0

    def test_append_to_buffer_auto_flushes_on_time_interval(self) -> None:
        """append_to_buffer auto-flushes when flush interval expires."""
        wal = WAL()
        wal.batch_size = 1000  # High batch size
        wal.flush_interval = 0.001  # 1 millisecond

        wal.append_to_buffer(b"record1")
        time.sleep(0.002)  # Wait for interval to expire
        wal.append_to_buffer(b"record2")

        # Buffer should be cleared after flush
        assert len(wal.wal_buffer) == 0

    def test_append_to_buffer_is_thread_safe(self) -> None:
        """append_to_buffer is thread-safe for concurrent appends."""
        wal = WAL()
        wal.batch_size = 1000  # Prevent auto-flush during test

        def append_records() -> None:
            for _ in range(100):
                wal.append_to_buffer(b"record")

        threads = [threading.Thread(target=append_records) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert wal.global_tx_id == 1000


class TestAppendToWAL:
    """Tests for flushing buffer to WAL file."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_append_to_wal_creates_file_if_not_exists(self) -> None:
        """append_to_wal creates WAL file if it doesn't exist."""
        wal = WAL()
        wal.batch_size = 1000
        wal.append_to_buffer(b"record")
        wal.force_flush()

        file_path = wal.get_file_path(0)
        assert file_path.exists()

    def test_append_to_wal_writes_payload(self) -> None:
        """append_to_wal writes payload to file."""
        wal = WAL()
        payload = struct.pack(">I", 10) + b"test_data"
        wal.append_to_buffer(payload)
        wal.force_flush()

        file_path = wal.get_file_path(0)
        header_block_size = WAL.WAL_FILE_HEADER_SIZE + (WAL.WAL_CAPACITY * WAL.WAL_ENTRY_HEADER_SIZE)

        with file_path.open("rb") as f:
            f.seek(header_block_size)
            written_payload = f.read(len(payload))
            assert written_payload == payload

    def test_append_to_wal_updates_file_header(self) -> None:
        """append_to_wal updates file header with record count and used space."""
        wal = WAL()
        payload1 = b"x" * 100
        payload2 = b"y" * 200

        wal.append_to_buffer(payload1)
        wal.append_to_buffer(payload2)
        wal.force_flush()

        file_path = wal.get_file_path(0)
        with file_path.open("rb") as f:
            header_data = f.read(WAL.WAL_FILE_HEADER_SIZE)
            magic, cap, current_rec, used_space, checksum = struct.unpack(WAL.FILE_HEADER_FMT, header_data)

            assert current_rec == 2
            assert used_space == 300

    def test_append_to_wal_clears_buffer(self) -> None:
        """append_to_wal clears buffer after successful flush."""
        wal = WAL()
        wal.batch_size = 1000

        wal.append_to_buffer(b"record1")
        wal.append_to_buffer(b"record2")
        assert len(wal.wal_buffer) > 0

        wal.force_flush()
        assert len(wal.wal_buffer) == 0

    def test_append_to_wal_handles_empty_buffer(self) -> None:
        """append_to_wal handles empty buffer gracefully."""
        wal = WAL()
        wal.append_to_wal()  # Should not raise

    def test_append_to_wal_raises_on_corrupted_header(self) -> None:
        """append_to_wal raises on corrupted WAL file header."""
        wal = WAL()
        wal.batch_size = 1000
        wal.append_to_buffer(b"record")
        wal.force_flush()

        # Corrupt the header
        file_path = wal.get_file_path(0)
        with file_path.open("r+b") as f:
            f.seek(20)  # Checksum position
            f.write(b"\x00\x00\x00\x00")

        wal.append_to_buffer(b"record2")
        with pytest.raises(WALCorruptionError, match="checksum mismatch"):
            wal.force_flush()


class TestMarkJobComplete:
    """Tests for marking transactions as complete."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_mark_job_complete_updates_ram_for_buffered_tx(self) -> None:
        """mark_job_complete marks buffered transactions in RAM."""
        wal = WAL()
        wal.batch_size = 1000  # Prevent auto-flush

        tx_id = wal.append_to_buffer(b"record")
        wal.mark_job_complete(tx_id)

        assert tx_id in wal.ram_completed_tx

    def test_mark_job_complete_updates_disk_for_flushed_tx(self) -> None:
        """mark_job_complete updates on-disk entry for flushed transactions."""
        wal = WAL()
        tx_id = wal.append_to_buffer(b"record")
        wal.force_flush()

        wal.mark_job_complete(tx_id)

        # Verify entry header is updated
        file_path = wal.get_file_path(tx_id)
        slot_index = wal.get_slot_index(tx_id)
        entry_offset = WAL.WAL_FILE_HEADER_SIZE + (slot_index * WAL.WAL_ENTRY_HEADER_SIZE)

        with file_path.open("rb") as f:
            f.seek(entry_offset)
            entry_data = f.read(WAL.WAL_ENTRY_HEADER_SIZE)
            tx_id_read, timestamp, is_completed, payload_offset = struct.unpack(WAL.ENTRY_HEADER_FMT, entry_data)
            assert is_completed is True

    def test_mark_job_complete_rejects_negative_tx_id(self) -> None:
        """mark_job_complete rejects negative transaction IDs."""
        wal = WAL()
        with pytest.raises(ValueError, match="non-negative"):
            wal.mark_job_complete(-1)

    def test_mark_job_complete_raises_on_missing_file(self) -> None:
        """mark_job_complete raises error if WAL file is missing."""
        wal = WAL()
        with pytest.raises(WALError, match="WAL file.*not found"):
            wal.mark_job_complete(1000)

    def test_mark_job_complete_persists_after_flush(self) -> None:
        """Completed status persists across buffer flush."""
        wal = WAL()
        wal.batch_size = 1000

        tx_id = wal.append_to_buffer(b"record")
        wal.mark_job_complete(tx_id)
        wal.force_flush()

        # Verify status persisted
        file_path = wal.get_file_path(tx_id)
        slot_index = wal.get_slot_index(tx_id)
        entry_offset = WAL.WAL_FILE_HEADER_SIZE + (slot_index * WAL.WAL_ENTRY_HEADER_SIZE)

        with file_path.open("rb") as f:
            f.seek(entry_offset)
            entry_data = f.read(WAL.WAL_ENTRY_HEADER_SIZE)
            _, _, is_completed, _ = struct.unpack(WAL.ENTRY_HEADER_FMT, entry_data)
            assert is_completed is True


class TestRestoreGlobalTxID:
    """Tests for restore_global_tx_id method."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_restore_global_tx_id_starts_at_zero_when_no_files(self) -> None:
        """Restore sets tx_id to 0 when no WAL files exist."""
        wal = WAL()
        wal.global_tx_id = 9999  # Set non-zero value
        wal.restore_global_tx_id()
        assert wal.global_tx_id == 0

    def test_restore_global_tx_id_from_single_file(self) -> None:
        """Restore correctly reads tx_id from a single WAL file."""
        wal = WAL()
        # Append some records
        for _ in range(5):
            wal.append_to_buffer(b"test_record")
        wal.force_flush()

        # Create new WAL instance and restore
        wal2 = WAL()
        assert wal2.global_tx_id == 5

    def test_restore_global_tx_id_from_multiple_files(self) -> None:
        """Restore correctly reads tx_id from the last WAL file."""
        wal = WAL()
        # Add some records to first file
        first_file_records = 10
        for _ in range(first_file_records):
            wal.append_to_buffer(b"record")
        wal.force_flush()

        # Manually set tx_id to create a second file
        wal.global_tx_id = WAL.WAL_CAPACITY
        second_file_records = 5
        for _ in range(second_file_records):
            wal.append_to_buffer(b"record")
        wal.force_flush()

        # Create new WAL instance and restore - should be at the end of second file
        wal2 = WAL()
        assert wal2.global_tx_id == WAL.WAL_CAPACITY + second_file_records

    def test_restore_global_tx_id_raises_on_corrupted_header_checksum(self) -> None:
        """Restore raises WALCorruptionError when header checksum is invalid."""
        wal = WAL()
        wal.append_to_buffer(b"record")
        wal.force_flush()

        # Corrupt the header checksum
        file_path = wal.get_file_path(0)
        with file_path.open("r+b") as f:
            f.seek(20)  # Checksum position (after magic + capacity + current_rec + used_space)
            f.write(b"\x00\x00\x00\x00")  # Invalid checksum

        # Create new WAL instance - restore is called in __init__ and should raise
        with pytest.raises(WALCorruptionError, match="checksum mismatch"):
            WAL()

    def test_restore_global_tx_id_raises_on_invalid_magic_number(self) -> None:
        """Restore raises WALCorruptionError when magic number is invalid."""
        wal = WAL()
        wal.append_to_buffer(b"record")
        wal.force_flush()

        # Corrupt the magic number but keep checksum valid
        file_path = wal.get_file_path(0)
        with file_path.open("r+b") as f:
            # Read current header
            header_data = f.read(WAL.WAL_FILE_HEADER_SIZE)
            magic, capacity, current_rec, used_space, checksum = struct.unpack(WAL.FILE_HEADER_FMT, header_data)

            # Write invalid magic with valid checksum
            f.seek(0)
            invalid_magic = b"XXXX"
            header_without_checksum = struct.pack("> 4s I I Q", invalid_magic, capacity, current_rec, used_space)
            new_checksum = zlib.crc32(header_without_checksum) & 0xFFFFFFFF
            f.write(struct.pack(WAL.FILE_HEADER_FMT, invalid_magic, capacity, current_rec, used_space, new_checksum))

        # Create new WAL instance - restore is called in __init__ and should raise
        with pytest.raises(WALCorruptionError, match="invalid magic number"):
            WAL()

    def test_restore_global_tx_id_handles_truncated_header(self) -> None:
        """Restore handles truncated header by falling back to file boundary."""
        wal = WAL()
        wal.append_to_buffer(b"record")
        wal.force_flush()

        # Truncate the file header
        file_path = wal.get_file_path(0)
        with file_path.open("r+b") as f:
            f.truncate(10)  # Less than WAL_FILE_HEADER_SIZE

        # Create new WAL instance and restore
        wal2 = WAL()
        wal2.restore_global_tx_id()
        # Should fall back to file_id * WAL_CAPACITY (0 * 50000 = 0)
        assert wal2.global_tx_id == 0

    def test_restore_global_tx_id_handles_io_error(self) -> None:
        """Restore raises WALCorruptionError on I/O errors."""
        wal = WAL()
        wal.append_to_buffer(b"record")
        wal.force_flush()

        # Remove read permission from file
        file_path = wal.get_file_path(0)
        file_path.chmod(0o000)

        try:
            # Create new WAL instance - restore is called in __init__ and should raise
            with pytest.raises(WALCorruptionError, match="Failed to restore transaction ID"):
                WAL()
        finally:
            # Restore permissions for cleanup
            file_path.chmod(0o600)


class TestCreateWALDirectory:
    """Tests for create_wal_directory method."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_create_wal_directory_creates_directory(self) -> None:
        """create_wal_directory creates the directory."""
        wal = WAL()
        wal_path = Path(wal.WAL_FILE_PATH)
        # Remove the directory created by __init__
        wal_path.rmdir()
        assert not wal_path.exists()

        wal.create_wal_directory()
        assert wal_path.exists()
        assert wal_path.is_dir()

    def test_create_wal_directory_creates_parent_directories(self, tmp_path: Path) -> None:
        """create_wal_directory creates parent directories recursively."""
        nested_path = tmp_path / "a" / "b" / "c" / "wal"
        wal = WAL()
        wal.WAL_FILE_PATH = str(nested_path)

        wal.create_wal_directory()
        assert nested_path.exists()
        assert nested_path.is_dir()

    def test_create_wal_directory_sets_permissions(self) -> None:
        """create_wal_directory sets correct permissions (0o700)."""
        wal = WAL()
        wal_path = Path(wal.WAL_FILE_PATH)
        permissions = oct(wal_path.stat().st_mode)[-3:]
        assert permissions == "700"

    def test_create_wal_directory_is_idempotent(self) -> None:
        """create_wal_directory can be called multiple times safely."""
        wal = WAL()
        wal.create_wal_directory()  # First call
        wal.create_wal_directory()  # Second call should not raise
        assert Path(wal.WAL_FILE_PATH).exists()


class TestReadWAL:
    """Tests for read_wal method."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_read_wal_returns_empty_when_no_directory(self, tmp_path: Path) -> None:
        """read_wal returns empty iterator when WAL directory doesn't exist."""
        wal = WAL()
        # Remove WAL directory
        Path(wal.WAL_FILE_PATH).rmdir()
        records = list(wal.read_wal())
        assert records == []

    def test_read_wal_returns_empty_when_no_files(self) -> None:
        """read_wal returns empty iterator when no WAL files exist."""
        wal = WAL()
        records = list(wal.read_wal())
        assert records == []

    def test_read_wal_yields_incomplete_records(self) -> None:
        """read_wal yields only incomplete records."""
        wal = WAL()
        # Add some records with length prefix
        tx_id1 = wal.append_to_buffer(create_test_payload(b"record1"))
        tx_id2 = wal.append_to_buffer(create_test_payload(b"record2"))
        tx_id3 = wal.append_to_buffer(create_test_payload(b"record3"))
        wal.force_flush()

        # Mark only tx_id2 as complete
        wal.mark_job_complete(tx_id2)

        # Read WAL
        records = list(wal.read_wal())
        tx_ids = [tx_id for tx_id, _ in records]

        # Should only yield tx_id1 and tx_id3 (incomplete)
        assert len(records) == 2
        assert tx_id1 in tx_ids
        assert tx_id3 in tx_ids
        assert tx_id2 not in tx_ids

    def test_read_wal_reads_correct_payload(self) -> None:
        """read_wal returns correct payload data for incomplete records."""
        wal = WAL()
        test_data = b"test_payload_data"
        test_payload = create_test_payload(test_data)
        tx_id = wal.append_to_buffer(test_payload)
        wal.force_flush()

        records = list(wal.read_wal())
        assert len(records) == 1
        returned_tx_id, returned_payload = records[0]
        assert returned_tx_id == tx_id
        assert returned_payload == test_payload

    def test_read_wal_reads_multiple_files_in_order(self) -> None:
        """read_wal reads multiple WAL files in correct order."""
        wal = WAL()
        # Add records to first file
        first_batch_count = 100
        first_payloads = []
        for i in range(first_batch_count):
            payload = create_test_payload(f"rec{i}".encode())
            first_payloads.append(payload)
            wal.append_to_buffer(payload)
        wal.force_flush()

        # Manually set tx_id to force next batch into second file
        wal.global_tx_id = WAL.WAL_CAPACITY
        second_batch_count = 50
        second_payloads = []
        for i in range(second_batch_count):
            payload = create_test_payload(f"rec{i + first_batch_count}".encode())
            second_payloads.append(payload)
            wal.append_to_buffer(payload)
        wal.force_flush()

        # Read all records from both files
        records = list(wal.read_wal())
        assert len(records) == first_batch_count + second_batch_count

        # Verify records from first file are in order
        for i in range(first_batch_count):
            assert records[i][0] == i
            assert records[i][1] == first_payloads[i]

        # Verify records from second file are in order
        for i in range(second_batch_count):
            idx = first_batch_count + i
            assert records[idx][0] == WAL.WAL_CAPACITY + i
            assert records[idx][1] == second_payloads[i]

    def test_read_wal_skips_completed_records(self) -> None:
        """read_wal skips all completed records."""
        wal = WAL()
        # Add and complete all records
        tx_ids = []
        for i in range(5):
            tx_id = wal.append_to_buffer(create_test_payload(f"record_{i}".encode()))
            tx_ids.append(tx_id)
        wal.force_flush()

        # Mark all as complete
        for tx_id in tx_ids:
            wal.mark_job_complete(tx_id)

        # Read WAL
        records = list(wal.read_wal())
        assert records == []  # All completed, should be empty

    def test_read_wal_handles_corrupted_header_checksum(self) -> None:
        """read_wal skips files with corrupted header checksums."""
        wal = WAL()
        wal.append_to_buffer(create_test_payload(b"record1"))
        wal.force_flush()

        # Corrupt the header checksum
        file_path = wal.get_file_path(0)
        with file_path.open("r+b") as f:
            f.seek(20)  # Checksum position
            f.write(b"\x00\x00\x00\x00")

        # Read WAL - should skip corrupted file
        records = list(wal.read_wal())
        assert records == []

    def test_read_wal_handles_invalid_magic_number(self) -> None:
        """read_wal skips files with invalid magic numbers."""
        wal = WAL()
        wal.append_to_buffer(create_test_payload(b"record1"))
        wal.force_flush()

        # Write invalid magic but keep checksum valid
        file_path = wal.get_file_path(0)
        with file_path.open("r+b") as f:
            header_data = f.read(WAL.WAL_FILE_HEADER_SIZE)
            magic, capacity, current_rec, used_space, checksum = struct.unpack(WAL.FILE_HEADER_FMT, header_data)

            f.seek(0)
            invalid_magic = b"XXXX"
            header_without_checksum = struct.pack("> 4s I I Q", invalid_magic, capacity, current_rec, used_space)
            new_checksum = zlib.crc32(header_without_checksum) & 0xFFFFFFFF
            f.write(struct.pack(WAL.FILE_HEADER_FMT, invalid_magic, capacity, current_rec, used_space, new_checksum))

        # Read WAL - should skip file with invalid magic
        records = list(wal.read_wal())
        assert records == []

    def test_read_wal_handles_truncated_header(self) -> None:
        """read_wal skips files with truncated headers."""
        wal = WAL()
        wal.append_to_buffer(create_test_payload(b"record1"))
        wal.force_flush()

        # Truncate the file header
        file_path = wal.get_file_path(0)
        with file_path.open("r+b") as f:
            f.truncate(10)  # Less than WAL_FILE_HEADER_SIZE

        # Read WAL - should skip truncated file
        records = list(wal.read_wal())
        assert records == []

    def test_read_wal_handles_truncated_payload_length(self) -> None:
        """read_wal skips records with truncated payload length."""
        wal = WAL()
        wal.append_to_buffer(create_test_payload(b"record1"))
        wal.force_flush()

        # Get payload offset and corrupt it
        file_path = wal.get_file_path(0)
        slot_index = wal.get_slot_index(0)
        entry_offset = WAL.WAL_FILE_HEADER_SIZE + (slot_index * WAL.WAL_ENTRY_HEADER_SIZE)

        with file_path.open("r+b") as f:
            f.seek(entry_offset)
            entry_data = f.read(WAL.WAL_ENTRY_HEADER_SIZE)
            tx_id, timestamp, is_completed, payload_offset = struct.unpack(WAL.ENTRY_HEADER_FMT, entry_data)

            # Truncate at payload offset (no room for length bytes)
            f.truncate(payload_offset + 2)  # Less than 4 bytes for length

        # Read WAL - should skip record with truncated payload length
        records = list(wal.read_wal())
        assert records == []

    def test_read_wal_handles_invalid_payload_length(self) -> None:
        """read_wal skips records with invalid payload length."""
        wal = WAL()
        wal.append_to_buffer(create_test_payload(b"record1"))
        wal.force_flush()

        # Corrupt payload length
        file_path = wal.get_file_path(0)
        slot_index = wal.get_slot_index(0)
        entry_offset = WAL.WAL_FILE_HEADER_SIZE + (slot_index * WAL.WAL_ENTRY_HEADER_SIZE)

        with file_path.open("r+b") as f:
            f.seek(entry_offset)
            entry_data = f.read(WAL.WAL_ENTRY_HEADER_SIZE)
            tx_id, timestamp, is_completed, payload_offset = struct.unpack(WAL.ENTRY_HEADER_FMT, entry_data)

            # Write invalid length (exceeds PAYLOAD_AREA_SIZE)
            f.seek(payload_offset)
            invalid_length = WAL.PAYLOAD_AREA_SIZE + 1000
            f.write(invalid_length.to_bytes(4, byteorder="big"))

        # Read WAL - should skip record with invalid length
        records = list(wal.read_wal())
        assert records == []

    def test_read_wal_handles_truncated_payload_data(self) -> None:
        """read_wal skips records with truncated payload data."""
        wal = WAL()
        wal.append_to_buffer(create_test_payload(b"record1_with_long_data"))
        wal.force_flush()

        # Get payload offset
        file_path = wal.get_file_path(0)
        slot_index = wal.get_slot_index(0)
        entry_offset = WAL.WAL_FILE_HEADER_SIZE + (slot_index * WAL.WAL_ENTRY_HEADER_SIZE)

        with file_path.open("r+b") as f:
            f.seek(entry_offset)
            entry_data = f.read(WAL.WAL_ENTRY_HEADER_SIZE)
            tx_id, timestamp, is_completed, payload_offset = struct.unpack(WAL.ENTRY_HEADER_FMT, entry_data)

            # Read the length, then truncate the payload
            f.seek(payload_offset)
            length_bytes = f.read(4)
            record_length = int.from_bytes(length_bytes, byteorder="big")

            # Truncate payload data (keep only half)
            f.truncate(payload_offset + record_length // 2)

        # Read WAL - should skip record with truncated payload
        records = list(wal.read_wal())
        assert records == []


class TestForceFlush:
    """Tests for force_flush method."""

    @pytest.fixture(autouse=True)
    def isolate_wal_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run each test with an isolated WAL directory."""
        monkeypatch.setattr(WAL, "WAL_FILE_PATH", str(tmp_path / "wal"))

    def test_force_flush_writes_buffer_to_disk(self) -> None:
        """force_flush writes all buffered records to disk."""
        wal = WAL()
        wal.batch_size = 1000  # Prevent auto-flush

        tx_id = wal.append_to_buffer(b"record1")
        assert len(wal.wal_buffer) == 1

        wal.force_flush()
        assert len(wal.wal_buffer) == 0

        # Verify record was written
        file_path = wal.get_file_path(tx_id)
        assert file_path.exists()

    def test_force_flush_clears_buffer(self) -> None:
        """force_flush clears the buffer after writing."""
        wal = WAL()
        wal.batch_size = 1000

        for i in range(5):
            wal.append_to_buffer(f"record_{i}".encode())
        assert len(wal.wal_buffer) == 5

        wal.force_flush()
        assert wal.wal_buffer == []

    def test_force_flush_handles_empty_buffer(self) -> None:
        """force_flush handles empty buffer gracefully."""
        wal = WAL()
        assert len(wal.wal_buffer) == 0

        wal.force_flush()  # Should not raise
        assert len(wal.wal_buffer) == 0

    def test_force_flush_updates_last_flush_time(self) -> None:
        """force_flush updates the last_flush_time."""
        wal = WAL()
        wal.batch_size = 1000

        initial_flush_time = wal.last_flush_time
        time.sleep(0.01)  # Small delay

        wal.append_to_buffer(b"record")
        wal.force_flush()

        assert wal.last_flush_time > initial_flush_time

    def test_force_flush_is_thread_safe(self) -> None:
        """force_flush is safe to call from multiple threads."""
        wal = WAL()
        wal.batch_size = 1000

        # Add some records
        for i in range(10):
            wal.append_to_buffer(f"record_{i}".encode())

        # Create multiple threads that call force_flush
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=wal.force_flush)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Buffer should be empty
        assert wal.wal_buffer == []

    def test_force_flush_persists_completion_status(self) -> None:
        """force_flush persists transaction completion status."""
        wal = WAL()
        wal.batch_size = 1000

        tx_id = wal.append_to_buffer(b"record")
        wal.mark_job_complete(tx_id)
        wal.force_flush()

        # Read back and verify completion status
        records = list(wal.read_wal())
        assert records == []  # Completed records should not be returned

    def test_force_flush_with_multiple_records(self) -> None:
        """force_flush correctly writes multiple buffered records."""
        wal = WAL()
        wal.batch_size = 1000

        tx_ids = []
        payloads = []
        for i in range(10):
            payload = create_test_payload(f"record_{i}".encode())
            payloads.append(payload)
            tx_id = wal.append_to_buffer(payload)
            tx_ids.append(tx_id)

        wal.force_flush()

        # Verify all records were written
        records = list(wal.read_wal())
        assert len(records) == 10
        for i, (tx_id, payload) in enumerate(records):
            assert tx_id == tx_ids[i]
            assert payload == payloads[i]
