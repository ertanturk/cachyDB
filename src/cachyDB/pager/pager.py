"""Module for the Pager class in the cachyDB database.

This module provides page-level storage management using a slotted-page layout.
Each page is a fixed-size buffer divided into three regions:

    [Header (HEADER_SIZE bytes)] [Slot directory ↓] [Free space] [↑ Record data]

The slot directory grows downward from the header; record payloads are written
upward from the top of the page.  Integrity is verified via a CRC32 checksum
stored in the last four bytes of the header.
"""

from __future__ import annotations

import logging
import struct
import zlib


class PageError(Exception):
    """Base exception for page-level errors."""


class PageOverflowError(PageError):
    """Raised when a page does not have enough free space for a new record."""


class PageCorruptionError(PageError):
    """Raised when a page fails a size, magic-byte, checksum, or bounds check."""


class Pager:
    """Manages a single fixed-size database page with a slotted-page layout.

    Records are inserted into the page by writing payload bytes from the top of
    the free area downward (upper_offset decrements) and appending a slot entry
    to the directory (lower_offset increments).  Deleted records are marked with
    the tombstone ``(0, 0)`` in the slot directory; their bytes are zeroed
    immediately to prevent data leakage.

    Attributes:
        page_id (int): Non-negative unique identifier for this page.
        flags (int): Page flags bitmask in ``[0, MAX_FLAGS]``.
        lower_offset (int): Byte offset of the end of the slot directory
            (start of free space).
        upper_offset (int): Byte offset of the start of the record data area
            (end of free space).
        slots (list[tuple[int, int]]): Slot directory as ``(offset, length)``
            pairs; ``(0, 0)`` marks a deleted slot.
        data (bytearray): Raw page buffer of exactly ``PAGE_SIZE`` bytes.

    Example:
        >>> p = Pager(page_id=0)
        >>> sid = p.insert_record(b"hello")
        >>> p.get_record(sid)
        b'hello'
        >>> p.delete_record(sid)
        >>> p.get_record(sid) is None
        True
    """

    PAGE_SIZE: int = 4096
    MAGIC_DATA_PAGE: bytes = b"CDBP"

    # Header layout (big-endian):
    #   Magic(4s) + PageID(I) + Lower(H) + Upper(H) + NumSlots(H) + Flags(H) + Checksum(I)
    #   = 4 + 4 + 2 + 2 + 2 + 2 + 4 = 20 bytes
    HEADER_FORMAT_NO_CHECKSUM: str = "> 4s I H H H H"
    CHECKSUM_FORMAT: str = "> I"
    HEADER_FORMAT: str = HEADER_FORMAT_NO_CHECKSUM + " I"
    HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)  # 20 bytes
    CHECKSUM_SIZE: int = struct.calcsize(CHECKSUM_FORMAT)  # 4 bytes
    CHECKSUM_OFFSET: int = HEADER_SIZE - CHECKSUM_SIZE  # 16 bytes

    # Slot entry layout (big-endian): Offset(H) + Length(H) = 4 bytes
    SLOT_FORMAT: str = "> H H"
    SLOT_SIZE: int = struct.calcsize(SLOT_FORMAT)  # 4 bytes

    # Upper bound for the flags field (unsigned short)
    MAX_FLAGS: int = 0xFFFF

    def __init__(self, page_id: int, flags: int = 0) -> None:
        """Initializes a new, empty page buffer.

        Args:
            page_id: Non-negative unique identifier for this page.
            flags: Page flags bitmask.  Must be in ``[0, MAX_FLAGS]``.

        Raises:
            ValueError: If ``page_id`` is negative or ``flags`` exceeds
                ``MAX_FLAGS``.
        """
        if page_id < 0:
            raise ValueError(f"page_id must be >= 0, got {page_id}.")
        if not (0 <= flags <= self.MAX_FLAGS):
            raise ValueError(f"flags must be in [0, {self.MAX_FLAGS:#06x}], got {flags:#06x}.")

        self.page_id: int = page_id
        self.flags: int = flags
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.lower_offset: int = self.HEADER_SIZE
        self.upper_offset: int = self.PAGE_SIZE
        self.slots: list[tuple[int, int]] = []
        self.data: bytearray = bytearray(self.PAGE_SIZE)

    @property
    def free_space(self) -> int:
        """Returns the contiguous free space available on the page in bytes."""
        return self.upper_offset - self.lower_offset

    def has_space_for(self, payload_length: int) -> bool:
        """Returns ``True`` if the page can accommodate a record of this size.

        The required space includes the payload itself *and* one new slot
        directory entry.

        Args:
            payload_length: Size of the candidate payload in bytes.
                Must be >= 0.

        Returns:
            ``True`` if there is enough free space, ``False`` otherwise.

        Raises:
            ValueError: If ``payload_length`` is negative.
        """
        if payload_length < 0:
            raise ValueError(f"payload_length must be >= 0, got {payload_length}.")
        required_space: int = self.SLOT_SIZE + payload_length
        return self.free_space >= required_space

    def insert_record(self, payload: bytes) -> int:
        """Inserts a record into the page and returns the assigned slot ID.

        The payload is written at the current ``upper_offset`` (which then
        decrements), and a new slot entry is appended to the directory
        (``lower_offset`` increments).

        Args:
            payload: Raw bytes to store.  Must be a non-empty ``bytes``
                object.

        Returns:
            The non-negative slot ID assigned to the inserted record.

        Raises:
            TypeError: If ``payload`` is not a ``bytes`` object.
            ValueError: If ``payload`` is empty.
            PageOverflowError: If the page does not have enough free space
                for ``payload`` plus a slot directory entry.
        """
        if not isinstance(payload, bytes):
            raise TypeError(f"payload must be bytes, got {type(payload).__name__}.")
        payload_len: int = len(payload)
        if payload_len == 0:
            raise ValueError("payload must not be empty.")

        if not self.has_space_for(payload_len):
            raise PageOverflowError(
                f"Not enough space on page {self.page_id}: "
                f"need {self.SLOT_SIZE + payload_len} bytes, "
                f"only {self.free_space} available."
            )

        self.upper_offset -= payload_len
        self.data[self.upper_offset : self.upper_offset + payload_len] = payload
        slot_id: int = len(self.slots)
        self.slots.append((self.upper_offset, payload_len))
        self.lower_offset += self.SLOT_SIZE
        self.logger.debug(f"Inserted record of size {payload_len} into slot {slot_id} on page {self.page_id}.")
        return slot_id

    def get_record(self, slot_id: int) -> bytes | None:
        """Returns the payload bytes for the given slot, or ``None`` if deleted.

        Args:
            slot_id: The slot ID returned by a previous :meth:`insert_record`
                call.

        Returns:
            The raw record bytes, or ``None`` if the slot has been deleted.

        Raises:
            ValueError: If ``slot_id`` is out of the valid range.
        """
        if slot_id < 0 or slot_id >= len(self.slots):
            raise ValueError(f"Invalid slot ID {slot_id} on page {self.page_id}: page has {len(self.slots)} slot(s).")

        offset, length = self.slots[slot_id]

        if offset == 0 and length == 0:
            return None

        self.logger.debug(f"Retrieved record of size {length} from slot {slot_id} on page {self.page_id}.")
        return bytes(self.data[offset : offset + length])

    def delete_record(self, slot_id: int) -> None:
        """Marks a slot as deleted and zeros its payload bytes.

        The slot entry is preserved in the directory (as the tombstone
        ``(0, 0)``) to maintain stable slot IDs for all other records.
        The underlying payload bytes are zeroed immediately to prevent
        sensitive data from lingering in the page buffer.

        Args:
            slot_id: The slot ID of the record to delete.

        Raises:
            ValueError: If ``slot_id`` is out of the valid range or the
                slot is already deleted.
        """
        if slot_id < 0 or slot_id >= len(self.slots):
            raise ValueError(f"Invalid slot ID {slot_id} on page {self.page_id}: page has {len(self.slots)} slot(s).")

        offset, length = self.slots[slot_id]
        if offset == 0 and length == 0:
            raise ValueError(f"Slot {slot_id} on page {self.page_id} is already deleted.")

        # Zero payload bytes before marking deleted to prevent data leakage
        self.data[offset : offset + length] = bytes(length)
        self.slots[slot_id] = (0, 0)
        self.logger.debug(f"Deleted record from slot {slot_id} on page {self.page_id}.")

    def compact(self) -> None:
        """Reclaims fragmented space by repacking all active records.

        All non-deleted records are moved to be contiguous at the top of the
        page data area.  Deleted slot entries remain in place to preserve
        stable slot IDs.  After repacking, the entire free area between the
        slot directory and the new record area is zeroed to eliminate any
        stale bytes from previously deleted or relocated records.
        """
        active_records: list[bytes | None] = []
        for offset, length in self.slots:
            if offset == 0 and length == 0:
                active_records.append(None)
            else:
                active_records.append(bytes(self.data[offset : offset + length]))

        self.upper_offset = self.PAGE_SIZE
        for i, record_bytes in enumerate(active_records):
            if record_bytes is not None:
                record_len: int = len(record_bytes)
                self.upper_offset -= record_len
                self.data[self.upper_offset : self.upper_offset + record_len] = record_bytes
                self.slots[i] = (self.upper_offset, record_len)

        # Zero the free area to eliminate stale data from deleted/moved records
        free_size: int = self.upper_offset - self.lower_offset
        if free_size > 0:
            self.data[self.lower_offset : self.upper_offset] = bytes(free_size)
        self.logger.debug(f"Compacted page {self.page_id}, free space: {free_size} bytes.")

    def to_bytes(self) -> bytes:
        """Serialises the page to a fixed-size ``bytes`` object with a CRC32 checksum.

        The slot directory is written immediately after the header.  The
        checksum is computed over the entire ``PAGE_SIZE`` buffer with the
        checksum field itself zeroed, then stored back at ``CHECKSUM_OFFSET``.

        Returns:
            Exactly ``PAGE_SIZE`` bytes representing the serialised page.
        """
        slot_bytes: bytearray = bytearray()
        for offset, length in self.slots:
            slot_bytes.extend(struct.pack(self.SLOT_FORMAT, offset, length))

        self.data[self.HEADER_SIZE : self.HEADER_SIZE + len(slot_bytes)] = slot_bytes

        num_slots: int = len(self.slots)
        header_no_checksum: bytes = struct.pack(
            self.HEADER_FORMAT_NO_CHECKSUM,
            self.MAGIC_DATA_PAGE,
            self.page_id,
            self.lower_offset,
            self.upper_offset,
            num_slots,
            self.flags,
        )
        self.data[0 : self.CHECKSUM_OFFSET] = header_no_checksum
        self.data[self.CHECKSUM_OFFSET : self.HEADER_SIZE] = b"\x00\x00\x00\x00"

        checksum: int = zlib.crc32(self.data) & 0xFFFFFFFF
        self.data[self.CHECKSUM_OFFSET : self.HEADER_SIZE] = struct.pack(self.CHECKSUM_FORMAT, checksum)
        self.logger.debug(f"Serialised page {self.page_id} with checksum {checksum}.")
        return bytes(self.data)

    @classmethod
    def from_bytes(cls, raw_data: bytes) -> Pager:
        """Deserialises and validates a page from raw bytes.

        Validation order:

        1. Page size must equal ``PAGE_SIZE``.
        2. Magic bytes must equal ``MAGIC_DATA_PAGE``.
        3. CRC32 checksum must match the stored value.
        4. ``lower_offset`` must equal ``HEADER_SIZE + num_slots * SLOT_SIZE``.
        5. ``lower_offset`` must be in ``[HEADER_SIZE, PAGE_SIZE]``.
        6. ``upper_offset`` must be in ``[lower_offset, PAGE_SIZE]``.
        7. Every active slot entry must reference bytes within the record area.

        Args:
            raw_data: Exactly ``PAGE_SIZE`` bytes representing a serialised
                page.

        Returns:
            A fully populated :class:`Pager` instance.

        Raises:
            PageCorruptionError: If any integrity or bounds check fails.
        """
        if len(raw_data) != cls.PAGE_SIZE:
            raise PageCorruptionError(f"Invalid page size: expected {cls.PAGE_SIZE} bytes, got {len(raw_data)}.")

        magic, page_id, lower, upper, num_slots, flags, checksum = struct.unpack(
            cls.HEADER_FORMAT, raw_data[: cls.HEADER_SIZE]
        )

        if magic != cls.MAGIC_DATA_PAGE:
            raise PageCorruptionError(f"Invalid magic bytes: expected {cls.MAGIC_DATA_PAGE!r}, got {magic!r}.")

        temp_data: bytearray = bytearray(raw_data)
        temp_data[cls.CHECKSUM_OFFSET : cls.HEADER_SIZE] = b"\x00\x00\x00\x00"
        calculated_checksum: int = zlib.crc32(temp_data) & 0xFFFFFFFF

        if calculated_checksum != checksum:
            raise PageCorruptionError(
                f"Checksum mismatch on page {page_id}: stored {checksum:#010x}, calculated {calculated_checksum:#010x}."
            )

        # Validate that lower_offset is consistent with the declared slot count
        expected_lower: int = cls.HEADER_SIZE + num_slots * cls.SLOT_SIZE
        if lower != expected_lower:
            raise PageCorruptionError(
                f"lower_offset {lower} is inconsistent with num_slots {num_slots} "
                f"on page {page_id}: expected lower_offset {expected_lower}."
            )

        if not (cls.HEADER_SIZE <= lower <= cls.PAGE_SIZE):
            raise PageCorruptionError(
                f"lower_offset {lower} is out of valid range [{cls.HEADER_SIZE}, {cls.PAGE_SIZE}] on page {page_id}."
            )

        if not (lower <= upper <= cls.PAGE_SIZE):
            raise PageCorruptionError(
                f"upper_offset {upper} is invalid on page {page_id}: "
                f"must satisfy lower_offset ({lower}) <= upper_offset <= PAGE_SIZE ({cls.PAGE_SIZE})."
            )

        page: Pager = cls(page_id, flags)
        page.data = bytearray(raw_data)
        page.lower_offset = lower
        page.upper_offset = upper

        slot_area: bytes = raw_data[cls.HEADER_SIZE : lower]
        slot_offset: int
        slot_length: int
        for i in range(num_slots):
            start: int = i * cls.SLOT_SIZE
            slot_offset, slot_length = struct.unpack(cls.SLOT_FORMAT, slot_area[start : start + cls.SLOT_SIZE])

            if slot_offset == 0:
                # Tombstone: both fields must be zero
                if slot_length != 0:
                    raise PageCorruptionError(
                        f"Slot {i} on page {page_id} has offset 0 but "
                        f"non-zero length {slot_length}: expected a deleted slot (0, 0)."
                    )
            else:
                # Active slot: must lie entirely within the record data area
                if slot_offset < upper or slot_offset + slot_length > cls.PAGE_SIZE:
                    raise PageCorruptionError(
                        f"Slot {i} on page {page_id} references data outside the "
                        f"record area: offset={slot_offset}, length={slot_length}, "
                        f"upper_offset={upper}, PAGE_SIZE={cls.PAGE_SIZE}."
                    )

            page.slots.append((slot_offset, slot_length))

        page.logger.debug(f"Deserialised page {page_id} with {num_slots} slots.")
        return page
