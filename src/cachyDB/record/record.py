"""Module for the Record class in the cachyDB database."""

import json
import os
import zlib
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class Record:
    """Represents a key-value record in the cachyDB database."""

    MAGIC_VALUE: int = 0x43444231
    MAX_KEY_SIZE: int = 4096  # 4KB
    MAX_VALUE_SIZE: int = 16 * 1024 * 1024  # 16MB
    MIN_KEY_SIZE: int = 1
    MIN_VALUE_SIZE: int = 0
    RECORD_LENGTH_SIZE: int = 4
    MAGIC_VALUE_SIZE: int = 4
    KEY_LENGTH_SIZE: int = 4
    VALUE_LENGTH_SIZE: int = 4
    CHECKSUM_SIZE: int = 4
    NONCE_SIZE: int = 12
    MAC_TAG_SIZE: int = 16
    MIN_ENCRYPTED_SIZE: int = NONCE_SIZE + MAC_TAG_SIZE
    MIN_RECORD_SIZE: int = RECORD_LENGTH_SIZE + MAGIC_VALUE_SIZE + KEY_LENGTH_SIZE + VALUE_LENGTH_SIZE + CHECKSUM_SIZE

    @staticmethod
    def validate_key_length(key_length: int) -> None:
        """Validates that key length is within allowed bounds.

        Args:
            key_length: Length of the key in bytes.

        Raises:
            ValueError: If key length is invalid.
        """
        if key_length < Record.MIN_KEY_SIZE:
            raise ValueError(f"Key length must be at least {Record.MIN_KEY_SIZE} byte(s).")
        if key_length > Record.MAX_KEY_SIZE:
            raise ValueError(f"Key length exceeds maximum allowed size of {Record.MAX_KEY_SIZE} bytes.")

    @staticmethod
    def validate_value_length(value_length: int) -> None:
        """Validates that value length is within allowed bounds.

        Args:
            value_length: Length of the value in bytes.

        Raises:
            ValueError: If value length is invalid.
        """
        if value_length < Record.MIN_VALUE_SIZE:
            raise ValueError(f"Value length must be at least {Record.MIN_VALUE_SIZE} byte(s).")
        if value_length > Record.MAX_VALUE_SIZE:
            raise ValueError(f"Value length exceeds maximum allowed size of {Record.MAX_VALUE_SIZE} bytes.")

    @staticmethod
    def pack_record(key: str, value: Any, encryption_key: bytes) -> bytes:
        """Packs the record into bytes for storage.

        Args:
            key: The key as a string.
            value: The value to serialize as JSON.

        Returns:
            Packed record bytes.

        Raises:
            ValueError: If key or value exceeds size limits or key is empty.
            UnicodeEncodeError: If key cannot be encoded as UTF-8.
        """
        key_bytes = key.encode("utf-8")
        value_json_str = json.dumps(value, sort_keys=True)
        raw_value_bytes = value_json_str.encode("utf-8")

        nonce = os.urandom(Record.NONCE_SIZE)
        chacha = ChaCha20Poly1305(encryption_key)

        ciphertext = chacha.encrypt(nonce, raw_value_bytes, associated_data=None)
        value_bytes = nonce + ciphertext

        key_length: int = len(key_bytes)
        Record.validate_key_length(key_length)
        value_length: int = len(value_bytes)
        Record.validate_value_length(value_length)

        key_length_bytes = key_length.to_bytes(Record.KEY_LENGTH_SIZE, byteorder="big")
        value_length_bytes = value_length.to_bytes(Record.VALUE_LENGTH_SIZE, byteorder="big")
        record_length: int = (
            Record.RECORD_LENGTH_SIZE
            + Record.MAGIC_VALUE_SIZE
            + key_length
            + Record.KEY_LENGTH_SIZE
            + value_length
            + Record.VALUE_LENGTH_SIZE
            + Record.CHECKSUM_SIZE
        )
        record_length_bytes = record_length.to_bytes(Record.RECORD_LENGTH_SIZE, byteorder="big")

        checksum: int = Record.calculate_checksum(
            record_length_bytes,
            key_length_bytes,
            key_bytes,
            value_length_bytes,
            value_bytes,
        )

        packed_record: bytes = (
            record_length_bytes
            + Record.MAGIC_VALUE.to_bytes(Record.MAGIC_VALUE_SIZE, byteorder="big")
            + key_length_bytes
            + key_bytes
            + value_length_bytes
            + value_bytes
            + checksum.to_bytes(Record.CHECKSUM_SIZE, byteorder="big")
        )
        return packed_record

    @staticmethod
    def calculate_checksum(
        record_length_bytes: bytes,
        key_length_bytes: bytes,
        key_bytes: bytes,
        value_length_bytes: bytes,
        value_bytes: bytes,
    ) -> int:
        """Calculates a CRC32 checksum for the record.

        Includes record_length in checksum to detect corruption of length field.
        Uses zlib.crc32() and masks result to ensure unsigned 32-bit value.

        Args:
            record_length_bytes: Serialized record length.
            key_length_bytes: Serialized key length.
            key_bytes: Raw key bytes.
            value_length_bytes: Serialized value length.
            value_bytes: Raw value bytes.

        Returns:
            Unsigned 32-bit CRC32 checksum.
        """
        record_body = record_length_bytes + key_length_bytes + key_bytes + value_length_bytes + value_bytes
        return zlib.crc32(record_body) & 0xFFFFFFFF

    @classmethod
    def unpack_record(cls, packed_record: bytes, encryption_key: bytes) -> tuple[str, Any]:
        """Unpacks the record from bytes.

        Args:
            packed_record: Packed record bytes.

        Returns:
            Tuple of (key, value).

        Raises:
            ValueError: If record is malformed, corrupted, or data is invalid.
        """
        if len(packed_record) < cls.MIN_RECORD_SIZE:
            raise ValueError(
                f"Incomplete record: expected at least {cls.MIN_RECORD_SIZE} bytes, got {len(packed_record)}"
            )

        record_length: int = int.from_bytes(packed_record[0 : cls.RECORD_LENGTH_SIZE], byteorder="big")

        magic_value: int = int.from_bytes(
            packed_record[cls.RECORD_LENGTH_SIZE : cls.RECORD_LENGTH_SIZE + cls.MAGIC_VALUE_SIZE],
            byteorder="big",
        )
        if magic_value != cls.MAGIC_VALUE:
            raise ValueError(
                f"Invalid magic value in record: expected 0x{cls.MAGIC_VALUE:08X}, got 0x{magic_value:08X}"
            )

        key_length_offset = cls.RECORD_LENGTH_SIZE + cls.MAGIC_VALUE_SIZE
        key_length: int = int.from_bytes(
            packed_record[key_length_offset : key_length_offset + cls.KEY_LENGTH_SIZE],
            byteorder="big",
        )
        try:
            cls.validate_key_length(key_length)
        except ValueError as exc:
            raise ValueError(f"Invalid key length in record: {exc}") from exc

        key_offset = key_length_offset + cls.KEY_LENGTH_SIZE
        key_end = key_offset + key_length
        if key_end > len(packed_record):
            raise ValueError(f"Incomplete key data: record too short to contain {key_length} bytes of key data")
        key_bytes: bytes = packed_record[key_offset:key_end]

        try:
            key = key_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Key contains invalid UTF-8 sequence: {exc.reason} at position {exc.start}") from exc

        value_length_offset = key_end
        if value_length_offset + cls.VALUE_LENGTH_SIZE > len(packed_record):
            raise ValueError("Incomplete value length field: record too short")
        value_length: int = int.from_bytes(
            packed_record[value_length_offset : value_length_offset + cls.VALUE_LENGTH_SIZE],
            byteorder="big",
        )
        try:
            cls.validate_value_length(value_length)
        except ValueError as exc:
            raise ValueError(f"Invalid value length in record: {exc}") from exc

        value_offset = value_length_offset + cls.VALUE_LENGTH_SIZE
        value_end = value_offset + value_length
        if value_end > len(packed_record):
            raise ValueError("Incomplete value data")
        encrypted_value_bytes: bytes = packed_record[value_offset:value_end]

        checksum_offset = value_end
        if checksum_offset + cls.CHECKSUM_SIZE > len(packed_record):
            raise ValueError("Incomplete checksum field: record too short")
        checksum: int = int.from_bytes(
            packed_record[checksum_offset : checksum_offset + cls.CHECKSUM_SIZE],
            byteorder="big",
        )

        calculated_checksum: int = cls.calculate_checksum(
            packed_record[0 : cls.RECORD_LENGTH_SIZE],
            packed_record[key_length_offset : key_length_offset + cls.KEY_LENGTH_SIZE],
            key_bytes,
            packed_record[value_length_offset : value_length_offset + cls.VALUE_LENGTH_SIZE],
            encrypted_value_bytes,
        )
        if checksum != calculated_checksum:
            raise ValueError(
                f"Checksum mismatch in record: expected 0x{calculated_checksum:08X}, "
                f"got 0x{checksum:08X}. Record may be corrupted."
            )

        if len(encrypted_value_bytes) < cls.MIN_ENCRYPTED_SIZE:
            raise ValueError("Invalid encrypted value. Too short to contain nonce and ciphertext.")

        nonce = encrypted_value_bytes[: cls.NONCE_SIZE]
        ciphertext = encrypted_value_bytes[cls.NONCE_SIZE :]
        chacha = ChaCha20Poly1305(encryption_key)

        try:
            decrypted_value_bytes = chacha.decrypt(nonce, ciphertext, associated_data=None)
        except Exception as exc:
            raise ValueError(f"Decryption failed: {exc}") from exc

        try:
            value_str = decrypted_value_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Value contains invalid UTF-8 sequence: {exc.reason} at position {exc.start}") from exc

        try:
            value = json.loads(value_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted JSON payload: {exc.msg} at line {exc.lineno}, column {exc.colno}") from exc

        if record_length != len(packed_record):
            raise ValueError(
                f"Record length mismatch: header claims {record_length} bytes, but received {len(packed_record)} bytes"
            )

        return (key, value)
