# cachyDB

cachyDB is an educational, high-performance, transactional key-value database engine implemented in Python — featuring Write-Ahead Logging, LFU caching, B-Tree indexing, and background compaction.

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Description

cachyDB is a layered database engine designed for learning database internals without sacrificing real functionality.

**Core design principles:**

| Principle         | Implementation                                  |
| ----------------- | ----------------------------------------------- |
| **Durability**    | Write-Ahead Logging (WAL) for crash recovery    |
| **Performance**   | LFU cache + in-memory B-Tree for O(log n) ops   |
| **Efficiency**    | Background compaction reclaims fragmented space |
| **Thread-Safety** | Internal locking for concurrent access          |
| **Security**      | Per-record encryption via ChaCha20Poly1305      |

---

## Features

- **Transactional Durability** — WAL ensures data survives crashes
- **LFU Cache** — Least Frequently Used eviction for hot-path reads
- **B-Tree Index** — In-memory indexing with O(log n) complexity
- **Background Compaction** — Async space reclamation for deleted/stale records
- **Thread-Safe Operations** — Concurrent read/write with internal locks
- **Slotted-Page Storage** — Page-level layout for efficient disk I/O
- **Data Encryption** — ChaCha20Poly1305 authenticated encryption per record
- **Structured Logging** — Rich-powered, enterprise-grade observability
- **Large Record Chunking** — Auto-chunk, compress, and serialize oversized values

---

## Tech Stack

| Category   | Library        |
| ---------- | -------------- |
| Encryption | `cryptography` |
| Logging UI | `rich`         |
| Testing    | `pytest`       |
| Linting    | `ruff`         |
| Build      | `hatchling`    |

---

## Project Structure

```
cachyDB/
├── __init__.py
├── cache/
│   ├── __init__.py
│   └── cache.py          # LFU cache implementation
├── database.py           # Public facade (CachyDB class)
├── garbage/
│   ├── __init__.py
│   └── compactor.py      # Background compaction worker
├── index/
│   ├── __init__.py
│   └── btree.py          # In-memory B-Tree index
├── logger/
│   ├── __init__.py
│   ├── log_handler.py
│   └── structured_logger.py
├── pager/
│   ├── __init__.py
│   └── pager.py          # Slotted-page storage manager
├── recovery/
│   ├── __init__.py
│   ├── checkpoint.py
│   └── wal.py            # Write-Ahead Log
└── storage/
    ├── __init__.py
    ├── router.py
    └── table_handler.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ertanturk/cachyDB.git
cd cachyDB
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install runtime dependencies

```bash
pip install cryptography rich
```

### 4. Install development dependencies _(optional)_

```bash
pip install bandit debugpy hatchling pytest ruff ty
```

### 5. If you have uv installed

```bash
uv sync
```

---

## Usage

### Quick start — top-level functions

The simplest way to use cachyDB is via module-level convenience functions backed by a shared default instance.

```python
from cachyDB import set, set_many, get, delete, close

# Write a single record
set('users', 'user:1', {'name': 'Alice', 'age': 30})

# Read a record
user = get('users', 'user:1')
# → {'name': 'Alice', 'age': 30}

# Write multiple records atomically
set_many('users', {
    'user:2': {'name': 'Bob',     'age': 25},
    'user:3': {'name': 'Charlie', 'age': 35},
})

# Delete a record (logical tombstone)
delete('users', 'user:3')

# Verify deletion
assert get('users', 'user:3') is None

# Flush WAL, stop compaction worker, close file handles
close()
```

---

### Custom instance

Use `CachyDB` directly when you need fine-grained control over memory or multiple isolated databases.

```python
from cachyDB.database import CachyDB

db = CachyDB(
    max_memtable_size=10_000,  # flush to disk after ~10k entries
    cache_capacity=2_000,      # LFU cache slots
)

db.set('products', 'sku:laptop-pro', {'name': 'Laptop Pro', 'price': 1_299.99})
db.set('products', 'sku:mouse-x',    {'name': 'Mouse X',    'price':    49.99})

product = db.get('products', 'sku:laptop-pro')
# → {'name': 'Laptop Pro', 'price': 1299.99}

db.delete('products', 'sku:mouse-x')

db.close()  # Always call close() for a clean shutdown
```

---

### Bulk operations

```python
from cachyDB.database import CachyDB

db = CachyDB()

inventory = {f'item:{i}': {'stock': i * 10} for i in range(1, 501)}
ok = db.set_many('inventory', inventory)

if not ok:
    print("Bulk write failed — check logs for details")

db.close()
```

---

### Error-aware pattern

`set`, `set_many`, and `delete` return `bool`. Always check the return value in production code.

```python
from cachyDB.database import CachyDB

db = CachyDB()

if not db.set('orders', 'ord:9001', {'item': 'Book', 'qty': 2}):
    raise RuntimeError("Write failed")

order = db.get('orders', 'ord:9001')
if order is None:
    raise KeyError("Record not found")

db.close()
```

---

## API Reference

### `CachyDB(max_memtable_size, cache_capacity)`

```python
from cachyDB.database import CachyDB
```

#### Constructor

```python
CachyDB(max_memtable_size: int = 20_000, cache_capacity: int = 5_000)
```

| Parameter           | Type  | Default | Description                                              |
| ------------------- | ----- | ------- | -------------------------------------------------------- |
| `max_memtable_size` | `int` | `20000` | Number of entries held in memory before flushing to disk |
| `cache_capacity`    | `int` | `5000`  | Maximum number of slots in the LFU read cache            |

---

#### `set(table, key, value) → bool`

Writes a single key-value record to the specified table.

```python
db.set(table: str, key: str, value: Any) -> bool
```

| Parameter | Type  | Description                |
| --------- | ----- | -------------------------- |
| `table`   | `str` | Logical table namespace    |
| `key`     | `str` | Unique record identifier   |
| `value`   | `Any` | Serializable Python object |

**Returns:** `True` on success, `False` on failure.

```python
ok = db.set('sessions', 'sess:abc123', {'user_id': 42, 'ttl': 3600})
```

---

#### `set_many(table, records) → bool`

Writes multiple records to the same table in one call.

```python
db.set_many(table: str, records: dict[str, Any]) -> bool
```

| Parameter | Type             | Description                    |
| --------- | ---------------- | ------------------------------ |
| `table`   | `str`            | Logical table namespace        |
| `records` | `dict[str, Any]` | Mapping of `key → value` pairs |

**Returns:** `True` if all records were written successfully, `False` otherwise.

```python
ok = db.set_many('metrics', {
    'cpu:host-1':  {'value': 72.4, 'ts': 1719000000},
    'mem:host-1':  {'value': 54.1, 'ts': 1719000000},
})
```

---

#### `get(table, key) → Any | None`

Retrieves a record by key, checking the LFU cache first, then the B-Tree index, then disk.

```python
db.get(table: str, key: str) -> Any | None
```

| Parameter | Type  | Description                  |
| --------- | ----- | ---------------------------- |
| `table`   | `str` | Logical table namespace      |
| `key`     | `str` | Record identifier to look up |

**Returns:** The stored value, or `None` if the key does not exist or was deleted.

```python
user = db.get('users', 'user:1')
if user:
    print(user['name'])
```

---

#### `delete(table, key) → bool`

Logically deletes a record by writing a tombstone. The record is physically removed during the next compaction cycle.

```python
db.delete(table: str, key: str) -> bool
```

| Parameter | Type  | Description                 |
| --------- | ----- | --------------------------- |
| `table`   | `str` | Logical table namespace     |
| `key`     | `str` | Record identifier to delete |

**Returns:** `True` on success, `False` on failure.

```python
ok = db.delete('sessions', 'sess:abc123')
```

---

#### `close() → None`

Flushes the WAL, waits for the compaction worker to finish, and releases all file handles. **Always call this before your process exits.**

```python
db.close()
```

---

### Top-level convenience functions

These functions delegate to a shared module-level `CachyDB` instance (default parameters). Ideal for scripts and notebooks.

```python
from cachyDB import set, set_many, get, delete, close
```

| Function                   | Equivalent to                          |
| -------------------------- | -------------------------------------- |
| `set(table, key, value)`   | `_default_db.set(table, key, value)`   |
| `set_many(table, records)` | `_default_db.set_many(table, records)` |
| `get(table, key)`          | `_default_db.get(table, key)`          |
| `delete(table, key)`       | `_default_db.delete(table, key)`       |
| `close()`                  | `_default_db.close()`                  |

> **Caution:** Avoid mixing top-level functions with a custom `CachyDB` instance — they operate on separate storage contexts.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

_© 2026 [ertanturk/cachyDB](https://github.com/ertanturk/cachyDB)_
