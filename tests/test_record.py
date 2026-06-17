"""Unit tests for the Record class."""

import os
from typing import Any

import pytest

from cachyDB.record.record import Record


@pytest.fixture
def encryption_key() -> bytes:
    """Generate a 32-byte encryption key using os.urandom.

    Returns:
        A 32-byte encryption key suitable for ChaCha20Poly1305.
    """
    return os.urandom(32)


class TestValidateKeyLength:
    """Tests for key length validation."""

    def test_valid_minimum_key_length(self) -> None:
        """Verify that minimum allowed key length passes validation."""
        Record.validate_key_length(Record.MIN_KEY_SIZE)

    def test_valid_maximum_key_length(self) -> None:
        """Verify that maximum allowed key length passes validation."""
        Record.validate_key_length(Record.MAX_KEY_SIZE)

    def test_valid_midrange_key_length(self) -> None:
        """Verify that midrange key lengths pass validation."""
        Record.validate_key_length(100)
        Record.validate_key_length(1024)
        Record.validate_key_length(2048)

    def test_key_length_below_minimum(self) -> None:
        """Verify that key length below minimum is rejected."""
        with pytest.raises(ValueError, match="must be at least"):
            Record.validate_key_length(0)

    def test_key_length_exceeds_maximum(self) -> None:
        """Verify that key length exceeding maximum is rejected."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            Record.validate_key_length(Record.MAX_KEY_SIZE + 1)


class TestValidateValueLength:
    """Tests for value length validation."""

    def test_valid_minimum_value_length(self) -> None:
        """Verify that minimum allowed value length passes validation."""
        Record.validate_value_length(Record.MIN_VALUE_SIZE)

    def test_valid_maximum_value_length(self) -> None:
        """Verify that maximum allowed value length passes validation."""
        Record.validate_value_length(Record.MAX_VALUE_SIZE)

    def test_valid_midrange_value_length(self) -> None:
        """Verify that midrange value lengths pass validation."""
        Record.validate_value_length(1024)
        Record.validate_value_length(1024 * 1024)
        Record.validate_value_length(8 * 1024 * 1024)

    def test_value_length_below_minimum(self) -> None:
        """Verify that value length below minimum is rejected."""
        with pytest.raises(ValueError, match="must be at least"):
            Record.validate_value_length(-1)

    def test_value_length_exceeds_maximum(self) -> None:
        """Verify that value length exceeding maximum is rejected."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            Record.validate_value_length(Record.MAX_VALUE_SIZE + 1)


class TestPackRecord:
    """Tests for record packing functionality."""

    def test_pack_simple_string_value(self, encryption_key: bytes) -> None:
        """Verify packing of simple string key-value pairs.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "test_key"
        value: str = "test_value"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)
        assert len(packed) >= Record.MIN_RECORD_SIZE

    def test_pack_complex_json_value(self, encryption_key: bytes) -> None:
        """Verify packing of complex nested JSON structures.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "complex_key"
        value: dict[str, Any] = {
            "nested": {"data": [1, 2, 3]},
            "string": "value",
            "number": 42,
        }
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)
        assert len(packed) >= Record.MIN_RECORD_SIZE

    def test_pack_empty_string_value(self, encryption_key: bytes) -> None:
        """Verify packing of records with empty string values.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "empty_value"
        value: str = ""
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)
        assert len(packed) >= Record.MIN_RECORD_SIZE

    def test_pack_null_value(self, encryption_key: bytes) -> None:
        """Verify packing of records with null values.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "null_value"
        value: None = None
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)

    def test_pack_numeric_value(self, encryption_key: bytes) -> None:
        """Verify packing of records with numeric values.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "numeric"
        value: int = 42
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)

    def test_pack_boolean_value(self, encryption_key: bytes) -> None:
        """Verify packing of records with boolean values.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "boolean"
        value: bool = True
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)

    def test_pack_list_value(self, encryption_key: bytes) -> None:
        """Verify packing of records with list values.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "list"
        value: list[int] = [1, 2, 3, 4, 5]
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)

    def test_pack_empty_key_rejected(self, encryption_key: bytes) -> None:
        """Verify that empty keys are rejected.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        with pytest.raises(ValueError, match="must be at least"):
            Record.pack_record("", "value", encryption_key)

    def test_pack_oversized_key_rejected(self, encryption_key: bytes) -> None:
        """Verify that keys exceeding size limit are rejected.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        oversized_key: str = "x" * (Record.MAX_KEY_SIZE + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            Record.pack_record(oversized_key, "value", encryption_key)

    def test_pack_oversized_value_rejected(self, encryption_key: bytes) -> None:
        """Verify that values exceeding size limit are rejected.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "test"
        oversized_value: list[int] = [0] * (16 * 1024 * 1024 + 1000)
        with pytest.raises(ValueError, match="exceeds maximum"):
            Record.pack_record(key, oversized_value, encryption_key)

    def test_pack_unicode_key(self, encryption_key: bytes) -> None:
        """Verify packing of records with Unicode characters in keys.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "key_🔑_emoji"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)

    def test_pack_unicode_value(self, encryption_key: bytes) -> None:
        """Verify packing of records with Unicode characters in values.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "test"
        value: str = "value_🎉_with_emoji_🌟"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)

    def test_pack_produces_different_ciphertext(self, encryption_key: bytes) -> None:
        """Verify that encryption produces different output each time due to random nonces.

        Args:
            encryption_key: Encryption key required for packing records.

        Note:
            Since encryption uses random nonces, each pack produces different output
            even for the same input key-value pair.
        """
        key: str = "deterministic"
        value: dict[str, int] = {"z": 1, "a": 2, "m": 3}
        packed1: bytes = Record.pack_record(key, value, encryption_key)
        packed2: bytes = Record.pack_record(key, value, encryption_key)
        assert packed1 != packed2

    def test_pack_record_structure(self, encryption_key: bytes) -> None:
        """Verify that packed record contains all required components.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)

        assert len(packed) >= Record.MIN_RECORD_SIZE
        record_length: int = int.from_bytes(packed[0:4], byteorder="big")
        assert record_length == len(packed)
        magic: int = int.from_bytes(packed[4:8], byteorder="big")
        assert magic == Record.MAGIC_VALUE


class TestUnpackRecord:
    """Tests for record unpacking functionality."""

    def test_unpack_simple_string_value(self, encryption_key: bytes) -> None:
        """Verify unpacking of simple string key-value pairs.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test_key"
        value: str = "test_value"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_unpack_complex_json_value(self, encryption_key: bytes) -> None:
        """Verify unpacking of complex nested JSON structures.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "complex"
        value: dict[str, Any] = {
            "nested": {"data": [1, 2, 3]},
            "string": "value",
        }
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_unpack_empty_string_value(self, encryption_key: bytes) -> None:
        """Verify unpacking of records with empty string values.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "empty"
        value: str = ""
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_unpack_null_value(self, encryption_key: bytes) -> None:
        """Verify unpacking of records with null values.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "null_value"
        value: None = None
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value is None

    def test_unpack_numeric_value(self, encryption_key: bytes) -> None:
        """Verify unpacking of records with numeric values.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "number"
        value: int = 42
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_unpack_unicode_key(self, encryption_key: bytes) -> None:
        """Verify unpacking of records with Unicode characters in keys.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "key_🔑"
        value: str = "test"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_unpack_unicode_value(self, encryption_key: bytes) -> None:
        """Verify unpacking of records with Unicode characters in values.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "value_🎉"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_unpack_incomplete_record(self, encryption_key: bytes) -> None:
        """Verify that incomplete records are rejected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        incomplete_data: bytes = b"short"
        with pytest.raises(ValueError, match="Incomplete record"):
            Record.unpack_record(incomplete_data, encryption_key)

    def test_unpack_truncated_key_data(self, encryption_key: bytes) -> None:
        """Verify that truncated key data is detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test_key"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        truncated: bytes = packed[:25]
        with pytest.raises(ValueError, match="Incomplete"):
            Record.unpack_record(truncated, encryption_key)

    def test_unpack_truncated_value_data(self, encryption_key: bytes) -> None:
        """Verify that truncated value data is detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test_key"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        truncated: bytes = packed[:-10]
        with pytest.raises(ValueError, match="Incomplete"):
            Record.unpack_record(truncated, encryption_key)

    def test_unpack_truncated_checksum(self, encryption_key: bytes) -> None:
        """Verify that truncated checksum is detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test_key"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        truncated: bytes = packed[:-1]
        with pytest.raises(ValueError, match="Incomplete checksum"):
            Record.unpack_record(truncated, encryption_key)

    def test_unpack_wrong_encryption_key(self, encryption_key: bytes) -> None:
        """Verify that using wrong encryption key fails to decrypt.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "test_key"
        value: str = "test_value"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        wrong_key: bytes = os.urandom(32)
        with pytest.raises(ValueError, match="Decryption failed"):
            Record.unpack_record(packed, wrong_key)


class TestCorruptionDetection:
    """Tests for corruption detection capabilities."""

    def test_detect_corrupted_record_length(self, encryption_key: bytes) -> None:
        """Verify that corruption in record length field is detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[0] ^= 0x01
        with pytest.raises(ValueError, match="Checksum mismatch"):
            Record.unpack_record(bytes(corrupted), encryption_key)

    def test_detect_corrupted_magic_value(self, encryption_key: bytes) -> None:
        """Verify that corruption in magic value field is detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[4:8] = b"\xff\xff\xff\xff"
        with pytest.raises(ValueError, match="Invalid magic value"):
            Record.unpack_record(bytes(corrupted), encryption_key)

    def test_detect_corrupted_key_data(self, encryption_key: bytes) -> None:
        """Verify that corruption in key data is detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[12] ^= 0x01
        with pytest.raises(ValueError):
            Record.unpack_record(bytes(corrupted), encryption_key)

    def test_detect_corrupted_value_data(self, encryption_key: bytes) -> None:
        """Verify that corruption in value data is detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[-8] ^= 0xFF
        with pytest.raises(ValueError, match="Checksum mismatch"):
            Record.unpack_record(bytes(corrupted), encryption_key)

    def test_detect_corrupted_checksum(self, encryption_key: bytes) -> None:
        """Verify that corruption in checksum is detected via mismatch.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[-4] ^= 0xFF
        with pytest.raises(ValueError, match="Checksum mismatch"):
            Record.unpack_record(bytes(corrupted), encryption_key)

    def test_detect_record_length_mismatch(self, encryption_key: bytes) -> None:
        """Verify that buffer length mismatch is detected after checksum validation.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        truncated: bytes = packed[:-5]
        with pytest.raises(ValueError):
            Record.unpack_record(truncated, encryption_key)

    def test_detect_invalid_utf8_in_key(self, encryption_key: bytes) -> None:
        """Verify that invalid UTF-8 sequences in keys are detected.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        key_offset: int = 12
        corrupted[key_offset] = 0xFF
        with pytest.raises(ValueError, match="invalid UTF-8"):
            Record.unpack_record(bytes(corrupted), encryption_key)

    def test_detect_invalid_json_payload(self, encryption_key: bytes) -> None:
        """Verify that invalid JSON payloads are detected with wrong decryption key.

        Args:
            encryption_key: Encryption key required for packing records.

        Note:
            This test uses a wrong key to trigger decryption failure, which results
            in corrupted data that cannot be decoded as valid JSON.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        wrong_key: bytes = os.urandom(32)
        with pytest.raises(ValueError, match="Decryption failed"):
            Record.unpack_record(packed, wrong_key)


class TestErrorHandling:
    """Tests for error handling and error messages."""

    def test_error_message_includes_hex_values(self, encryption_key: bytes) -> None:
        """Verify that error messages include hex format values for debugging.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[0] ^= 0x01
        with pytest.raises(ValueError) as exc_info:
            Record.unpack_record(bytes(corrupted), encryption_key)
        error_message: str = str(exc_info.value)
        assert "0x" in error_message

    def test_error_message_includes_descriptive_info(self, encryption_key: bytes) -> None:
        """Verify that error messages provide sufficient context.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[4:8] = b"\xff\xff\xff\xff"
        with pytest.raises(ValueError) as exc_info:
            Record.unpack_record(bytes(corrupted), encryption_key)
        error_message: str = str(exc_info.value)
        assert "expected" in error_message or "Invalid" in error_message

    def test_exception_chaining_for_utf8_errors(self, encryption_key: bytes) -> None:
        """Verify that UTF-8 decode errors are properly chained.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        corrupted: bytearray = bytearray(packed)
        corrupted[12] = 0xFF
        with pytest.raises(ValueError) as exc_info:
            Record.unpack_record(bytes(corrupted), encryption_key)
        assert exc_info.value.__cause__ is not None

    def test_exception_chaining_for_decryption_errors(self, encryption_key: bytes) -> None:
        """Verify that decryption errors are properly chained.

        Args:
            encryption_key: Encryption key required for unpacking records.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        wrong_key: bytes = os.urandom(32)
        with pytest.raises(ValueError) as exc_info:
            Record.unpack_record(packed, wrong_key)
        assert exc_info.value.__cause__ is not None


class TestBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_maximum_key_size(self, encryption_key: bytes) -> None:
        """Verify handling of records with maximum allowed key size.

        Args:
            encryption_key: Encryption key required for packing and unpacking records.
        """
        key: str = "x" * Record.MAX_KEY_SIZE
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_large_value_size(self, encryption_key: bytes) -> None:
        """Verify handling of records with large value sizes.

        Args:
            encryption_key: Encryption key required for packing and unpacking records.
        """
        key: str = "test"
        value: str = "x" * (1024 * 1024)
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_minimum_record_size_constant(self, encryption_key: bytes) -> None:
        """Verify that MIN_RECORD_SIZE constant is correctly defined.

        Args:
            encryption_key: Encryption key required for packing records.
        """
        key: str = "x"
        value: str = ""
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert len(packed) >= Record.MIN_RECORD_SIZE

    def test_multiple_pack_unpack_cycles(self, encryption_key: bytes) -> None:
        """Verify that multiple pack-unpack cycles preserve data integrity.

        Args:
            encryption_key: Encryption key required for packing and unpacking records.
        """
        key: str = "test"
        value: str = "data"
        for _ in range(10):
            packed: bytes = Record.pack_record(key, value, encryption_key)
            unpacked_key: str
            unpacked_value: Any
            unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
            assert unpacked_key == key
            assert unpacked_value == value

    def test_encrypted_value_minimum_size(self, encryption_key: bytes) -> None:
        """Verify that encrypted value meets minimum size requirements.

        Args:
            encryption_key: Encryption key required for packing records.

        Note:
            Encrypted values must be at least MIN_ENCRYPTED_SIZE bytes (nonce + MAC tag).
        """
        key: str = "test"
        value: str = ""
        packed: bytes = Record.pack_record(key, value, encryption_key)
        value_length_offset: int = 12 + len(key.encode("utf-8"))
        value_length: int = int.from_bytes(packed[value_length_offset : value_length_offset + 4], byteorder="big")
        assert value_length >= Record.MIN_ENCRYPTED_SIZE

    def test_nonce_size_in_encrypted_value(self, encryption_key: bytes) -> None:
        """Verify that encrypted value contains a nonce of correct size.

        Args:
            encryption_key: Encryption key required for packing and unpacking records.

        Note:
            The nonce should be exactly NONCE_SIZE bytes, located at the start of encrypted value.
        """
        key: str = "test"
        value: str = "data"
        packed: bytes = Record.pack_record(key, value, encryption_key)
        value_length_offset: int = 12 + len(key.encode("utf-8"))
        value_offset: int = value_length_offset + 4
        value_length: int = int.from_bytes(packed[value_length_offset : value_length_offset + 4], byteorder="big")
        encrypted_value: bytes = packed[value_offset : value_offset + value_length]
        assert len(encrypted_value) >= Record.NONCE_SIZE
        nonce: bytes = encrypted_value[: Record.NONCE_SIZE]
        assert len(nonce) == Record.NONCE_SIZE


class TestEncryption:
    """Tests for encryption functionality."""

    def test_encryption_key_requirement(self) -> None:
        """Verify that encryption key is required for packing records."""
        key: str = "test"
        value: str = "data"
        encryption_key: bytes = os.urandom(32)
        packed: bytes = Record.pack_record(key, value, encryption_key)
        assert isinstance(packed, bytes)

    def test_decryption_key_requirement(self) -> None:
        """Verify that encryption key is required for unpacking records."""
        key: str = "test"
        value: str = "data"
        encryption_key: bytes = os.urandom(32)
        packed: bytes = Record.pack_record(key, value, encryption_key)
        unpacked_key: str
        unpacked_value: Any
        unpacked_key, unpacked_value = Record.unpack_record(packed, encryption_key)
        assert unpacked_key == key
        assert unpacked_value == value

    def test_nonce_randomness(self) -> None:
        """Verify that nonce is random for each encryption operation."""
        key: str = "test"
        value: str = "data"
        encryption_key: bytes = os.urandom(32)
        packed1: bytes = Record.pack_record(key, value, encryption_key)
        packed2: bytes = Record.pack_record(key, value, encryption_key)
        value_length_offset: int = 12 + len(key.encode("utf-8"))
        value_offset1: int = value_length_offset + 4
        value_offset2: int = value_length_offset + 4
        encrypted_value1: bytes = packed1[value_offset1 : value_offset1 + Record.NONCE_SIZE]
        encrypted_value2: bytes = packed2[value_offset2 : value_offset2 + Record.NONCE_SIZE]
        assert encrypted_value1 != encrypted_value2

    def test_nonce_size_constant(self) -> None:
        """Verify that NONCE_SIZE constant is set correctly."""
        assert Record.NONCE_SIZE == 12

    def test_mac_tag_size_constant(self) -> None:
        """Verify that MAC_TAG_SIZE constant is set correctly."""
        assert Record.MAC_TAG_SIZE == 16

    def test_min_encrypted_size_constant(self) -> None:
        """Verify that MIN_ENCRYPTED_SIZE constant is correctly calculated."""
        expected_min_size: int = Record.NONCE_SIZE + Record.MAC_TAG_SIZE
        assert expected_min_size == Record.MIN_ENCRYPTED_SIZE


class TestCalculateChecksum:
    """Tests for checksum calculation functionality."""

    def test_checksum_is_unsigned_32bit(self) -> None:
        """Verify that checksum is properly masked to unsigned 32-bit value."""
        record_length_bytes: bytes = (100).to_bytes(4, byteorder="big")
        key_length_bytes: bytes = (4).to_bytes(4, byteorder="big")
        key_bytes: bytes = b"test"
        value_length_bytes: bytes = (4).to_bytes(4, byteorder="big")
        value_bytes: bytes = b"data"
        checksum: int = Record.calculate_checksum(
            record_length_bytes,
            key_length_bytes,
            key_bytes,
            value_length_bytes,
            value_bytes,
        )
        assert 0 <= checksum <= 0xFFFFFFFF

    def test_checksum_deterministic(self) -> None:
        """Verify that checksum calculation is deterministic."""
        record_length_bytes: bytes = (100).to_bytes(4, byteorder="big")
        key_length_bytes: bytes = (4).to_bytes(4, byteorder="big")
        key_bytes: bytes = b"test"
        value_length_bytes: bytes = (4).to_bytes(4, byteorder="big")
        value_bytes: bytes = b"data"
        checksum1: int = Record.calculate_checksum(
            record_length_bytes,
            key_length_bytes,
            key_bytes,
            value_length_bytes,
            value_bytes,
        )
        checksum2: int = Record.calculate_checksum(
            record_length_bytes,
            key_length_bytes,
            key_bytes,
            value_length_bytes,
            value_bytes,
        )
        assert checksum1 == checksum2

    def test_checksum_changes_with_different_input(self) -> None:
        """Verify that different inputs produce different checksums."""
        record_length_bytes: bytes = (100).to_bytes(4, byteorder="big")
        key_length_bytes: bytes = (4).to_bytes(4, byteorder="big")
        key_bytes1: bytes = b"test"
        key_bytes2: bytes = b"best"
        value_length_bytes: bytes = (4).to_bytes(4, byteorder="big")
        value_bytes: bytes = b"data"
        checksum1: int = Record.calculate_checksum(
            record_length_bytes,
            key_length_bytes,
            key_bytes1,
            value_length_bytes,
            value_bytes,
        )
        checksum2: int = Record.calculate_checksum(
            record_length_bytes,
            key_length_bytes,
            key_bytes2,
            value_length_bytes,
            value_bytes,
        )
        assert checksum1 != checksum2
