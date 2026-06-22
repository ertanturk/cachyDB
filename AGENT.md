# cachyDB AI Code Generation Instructions

This document provides comprehensive guidelines for AI agents writing code for the cachyDB project. All generated code must adhere to these standards.

## Project Overview

**cachyDB** is an educational database engine implemented in Python. It provides:

- Key-value storage with record serialization/deserialization
- Low-level storage management with paging, indexing, and transaction support
- SQL query interface
- Garbage collection and recovery mechanisms

**Project Structure:**

```
cachyDB/
├── src/cachyDB/
│   ├── record/              # Record serialization/deserialization
│   ├── storage/             # Storage management
│   ├── pager/               # Paging layer
│   ├── index/               # Indexing structures
│   ├── transaction/         # Transaction management
│   ├── recovery/            # Recovery mechanisms
│   ├── garbage/             # Garbage collection
│   └── sql/                 # SQL interface
├── tests/                   # Pytest test suite
├── docs/                    # Architecture documentation
└── pyproject.toml           # Project configuration
```

## Code Style and Quality Standards

### 1. Python Version & Dependencies

- **Minimum Python version:** `3.14` (as specified in `pyproject.toml`)
- **No external dependencies for core modules** (except in dev environment)
- All imports must be standard library or from cachyDB itself
- Development tools: `pytest`, `ruff`, `bandit`, `hatchling`, `ty`

### 2. Type Annotations (Mandatory)

Every function, method, and variable must have explicit type annotations.

**Rules:**

- Use `from typing import` for complex types (e.g., `Any`, `Union`, `Optional`, `Callable`)
- Use logging for debugging and error messages instead of `print` and write detailed log messages
- Use english for log messages and comments
- Before writing a function, write a docstring describing what the function does and its parameters/return type
- Use modern type hint syntax: `list[T]` instead of `List[T]`, `tuple[T, ...]` instead of `Tuple[T, ...]`
- Annotate all function parameters and return types
- Annotate class variables with their types
- Annotate local variables in complex functions (especially loops and conditional blocks)

**Example:**

```python
from typing import Any

def pack_record(key: str, value: Any) -> bytes:
    """Pack a record into bytes."""
    key_bytes: bytes = key.encode("utf-8")
    value_length: int = len(key_bytes)
    packed_record: bytes = key_bytes + b'\x00'
    return packed_record
```

### 3. Docstrings (Mandatory for All Functions/Methods/Classes)

Every function, method, and class must have a docstring. Use Google-style docstrings.

**Format:**

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief one-line description.

    Longer description if needed. Explain the purpose, behavior, and any
    important implementation details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When specific error condition occurs.
        TypeError: When type mismatch occurs.

    Note:
        Any additional notes about usage or behavior.
    """
```

**Class Docstrings:**

```python
class ClassName:
    """Brief description of the class.

    Longer description of the class purpose, responsibilities, and usage.

    Attributes:
        attribute1 (str): Description of attribute1.
        attribute2 (int): Description of attribute2.

    Example:
        Usage example if applicable.
    """
```

### 4. Ruff Linter Configuration

The project uses `ruff` with the following configuration (from `pyproject.toml`):

**Line length:** 120 characters

**Enabled rules:**

- `F` - Pyflakes (undefined names, unused imports)
- `W` - Warnings (whitespace, blank lines)
- `E` - Errors (logical errors, syntax)
- `I` - isort (import sorting)
- `UP` - Upgrades (modern Python syntax)
- `C4` - Comprehensions (dict/list comprehensions)
- `FA` - Future annotations
- `ISC` - Implicit string concatenation
- `ICN` - Import conventions (e.g., `import numpy as np`)
- `RET` - Return statements
- `SIM` - Simplifications
- `TID` - Flake8-tidy (unused noqa)
- `TC` - Type checking (type checking imports)
- `PTH` - Pathlib (use pathlib instead of os.path)
- `TD` - Todos (properly formatted TODOs)
- `NPY` - NumPy (not applicable unless using NumPy)
- `FURB` - Refurb (modernizations)

**Ignored rules:**

- `E501` - Line length (handled manually due to readability)

**Per-file ignores:**

- `tests/**`: `S101` (assert used in tests - allowed)
- `__init__.py`: `F401` (unused imports - allowed for re-exports)

**Import Order:**
All imports must follow isort conventions:

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library imports
import json
import zlib
from typing import Any

# 3. Third-party imports
# (none currently in cachyDB core)

# 4. Local application imports
from cachydb.record import Record
```

### 5. Naming Conventions

Follow PEP 8 naming conventions:

- **Classes:** `PascalCase` (e.g., `Record`, `StorageEngine`)
- **Functions/methods:** `snake_case` (e.g., `pack_record`, `validate_key_length`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_KEY_SIZE`, `MAGIC_VALUE`)
- **Private methods/attributes:** Prefix with `_` (e.g., `_internal_method`)
- **Avoid single-letter variable names** except in loops or mathematical contexts

### 6. Error Handling

- **Provide meaningful error messages** with relevant context (values, limits, etc.)
- **Chain exceptions** when re-raising: `raise NewError(...) from original_error`
- **Use specific exception types**: `ValueError`, `TypeError`, `IOError`, etc.
- **Include hex values** for numeric data: `0x{value:08X}`
- **Reference constants** in messages\*\* instead of magic numbers

**Example:**

```python
if key_length < Record.MIN_KEY_SIZE:
    raise ValueError(
        f"Key length must be at least {Record.MIN_KEY_SIZE} byte(s), "
        f"got {key_length}."
    )

try:
    value = json.loads(value_str)
except json.JSONDecodeError as exc:
    raise ValueError(
        f"Corrupted JSON payload: {exc.msg} at line {exc.lineno}, "
        f"column {exc.colno}"
    ) from exc
```

### 7. Constants and Magic Numbers

- **Define all magic numbers as class or module-level constants**
- **Use descriptive names** for constants
- **Group related constants** in the class
- **Document the purpose** of significant constants

**Example:**

```python
class Record:
    """Represents a key-value record in the cachyDB database."""

    MAGIC_VALUE: int = 0x43444231
    MAX_KEY_SIZE: int = 4096  # 4KB
    MAX_VALUE_SIZE: int = 16 * 1024 * 1024  # 16MB
    MIN_KEY_SIZE: int = 1
    MIN_VALUE_SIZE: int = 0
    RECORD_LENGTH_SIZE: int = 4
    CHECKSUM_SIZE: int = 4
```

### 8. Static Methods and Class Methods

- Use `@staticmethod` for utility functions that don't need instance state
- Use `@classmethod` for methods that work with class-level data or serve as alternate constructors
- Document the distinction in docstrings

**Example:**

```python
@staticmethod
def validate_key_length(key_length: int) -> None:
    """Validates that key length is within allowed bounds.

    Args:
        key_length: Length of the key in bytes.

    Raises:
        ValueError: If key length is invalid.
    """

@classmethod
def unpack_record(cls, packed_record: bytes) -> tuple[str, Any]:
    """Unpacks the record from bytes.

    Args:
        packed_record: Packed record bytes.

    Returns:
        Tuple of (key, value).
    """
```

### 9. Testing Requirements

- **Test directory:** `tests/` at project root
- **Test framework:** `pytest`
- **File naming:** `test_*.py` (matches `record/record.py` → `test_record.py`)
- **Test organization:** Group tests into classes by functionality
- **Assertions:** Use `assert` statements (S101 is ignored in tests)
- **Fixtures:** Use pytest fixtures for common setup
- **Parametrize:** Use `@pytest.mark.parametrize` for testing multiple cases

**Example:**

```python
import pytest
from cachydb.record import Record

class TestValidateKeyLength:
    """Tests for Record.validate_key_length method."""

    def test_valid_key_length(self) -> None:
        """Test that valid key lengths are accepted."""
        Record.validate_key_length(100)  # Should not raise

    def test_key_too_short(self) -> None:
        """Test that too-short keys raise ValueError."""
        with pytest.raises(ValueError, match="must be at least"):
            Record.validate_key_length(0)
```

### 10. Code Organization

**Module structure:**

```python
"""Module docstring explaining the module's purpose."""

# Imports (sorted per isort rules)
from __future__ import annotations

import json
import zlib
from typing import Any

# Constants
SOME_CONSTANT: int = 42

# Classes
class MyClass:
    """Class docstring."""

    CONSTANT: int = 10

    def __init__(self) -> None:
        """Initialize the class."""
        self.value: int = 0

    @staticmethod
    def static_method() -> None:
        """Static method docstring."""

    @classmethod
    def class_method(cls) -> None:
        """Class method docstring."""

    def instance_method(self) -> None:
        """Instance method docstring."""

# Module-level functions
def module_function() -> None:
    """Module-level function docstring."""
```

### 11. Performance and Correctness

- **Validate inputs** before processing
- **Use efficient algorithms** (e.g., `zlib.crc32()` for checksums)
- **Avoid unnecessary copies** of large data structures
- **Document performance implications** for I/O-intensive operations
- **Use byte operations** for binary data handling (as in Record class)
- **Ensure deterministic behavior** (e.g., `json.dumps(..., sort_keys=True)`)

### 12. Type Checking with ty

The project uses `ty` for Python type checking:

**Configuration (from `pyproject.toml`):**

```toml
[tool.ty.environment]
python = "./.venv"

[tool.ty.rules]
unresolved-import = "warn"
```

- Type hints are validated at development time
- Unresolved imports generate warnings (not errors) to allow for optional dependencies

### 13. Module Docstrings

Every module must start with a docstring describing its purpose:

```python
"""Module for the Record class in the cachyDB database.

This module provides serialization and deserialization of key-value records
with checksums for data integrity verification.
"""
```

### 14. Common Patterns in cachyDB

#### Record Serialization

- Use `json.dumps(..., sort_keys=True)` for deterministic output
- Encode to UTF-8 bytes: `key.encode("utf-8")`
- Calculate checksums using `zlib.crc32()` with masking: `crc32(...) & 0xFFFFFFFF`
- Store sizes as 4-byte big-endian integers

#### Binary Data Handling

```python
# Converting integer to bytes
size_bytes: bytes = size.to_bytes(4, byteorder="big")

# Converting bytes to integer
size: int = int.from_bytes(data[0:4], byteorder="big")

# Bit masking for unsigned values
checksum: int = zlib.crc32(data) & 0xFFFFFFFF
```

#### Validation Pattern

```python
@staticmethod
def validate_parameter(param: int) -> None:
    """Validate parameter is within bounds."""
    if param < MIN_VALUE:
        raise ValueError(f"Parameter must be at least {MIN_VALUE}, got {param}.")
    if param > MAX_VALUE:
        raise ValueError(f"Parameter exceeds maximum {MAX_VALUE}, got {param}.")
```

#### Error Recovery Pattern

```python
try:
    result = perform_operation()
except SpecificError as exc:
    raise ContextualError(f"Operation failed: {exc.message}") from exc
```

## Checklist Before Committing Code

- [ ] All functions have type annotations (parameters and return types)
- [ ] All variables have type annotations in complex functions
- [ ] Every function/method/class has a Google-style docstring
- [ ] Error messages reference constants instead of magic numbers
- [ ] Error messages include relevant context values
- [ ] Exceptions are chained with `from original_exception`
- [ ] Code passes `ruff check` with no errors
- [ ] Code follows PEP 8 naming conventions
- [ ] No bare `except` statements
- [ ] No unused imports (ruff I)
- [ ] No security vulnerabilities
- [ ] No dry code and no copy-pasted logic
- [ ] Check for logical errors and edge cases and bugs
- [ ] Tests exist for new functionality
- [ ] Tests pass with `pytest`
- [ ] Code is deterministic (no random behavior unless documented)

## Resources

- **Python typing docs:** https://docs.python.org/3.14/library/typing.html
- **Google Docstring Style:** https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- **PEP 8 Style Guide:** https://pep8.org/
- **Ruff documentation:** https://docs.astral.sh/ruff/
- **pytest documentation:** https://docs.pytest.org/

---
