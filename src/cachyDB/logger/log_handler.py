"""Module for the LogHandler class in the cachyDB database.

This module provides a process-wide singleton that configures a named
``cachyDB`` logger with a rotating (and gzip-compressed) file handler,
an optional structured JSON file handler, and a Rich console handler.

Environment variables
---------------------
LOG_LEVEL   : Override log level (default: ``INFO``).
              Accepted values: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
LOG_DIR     : Override log directory (default: ``data/logs``).
LOG_JSON    : Enable structured JSON logging (default: ``false``).
              Set to ``true``, ``1``, or ``yes`` to enable.

The singleton guarantees that the handlers are attached exactly once,
regardless of how many times ``LogHandler()`` is called.
"""

from __future__ import annotations

import atexit
import gzip
import json
import logging
import os
import queue
import shutil
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

from rich.logging import RichHandler


class LoggingHandlerError(Exception):
    """Custom exception for logging configuration errors."""


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Each emitted line is a self-contained JSON object with fixed fields,
    making the output directly ingestible by log aggregation systems
    (Elasticsearch, Loki, Datadog, etc.).

    Attributes:
        FIELD_MAP (dict[str, str]): Mapping of output key → LogRecord attribute.

    Example output::

        {"timestamp": "2025-09-28T12:34:56.789Z", "level": "INFO",
         "logger": "cachyDB", "thread": "MainThread", "process": 12345,
         "module": "storage", "func": "flush_wal", "line": 87,
         "message": "WAL successfully flushed."}
    """

    FIELD_MAP: dict[str, str] = {
        "timestamp": "asctime",
        "level": "levelname",
        "logger": "name",
        "thread": "threadName",
        "process": "process",
        "module": "module",
        "func": "funcName",
        "line": "lineno",
        "message": "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Serializes a log record to a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string representing the log record.
        """
        record.asctime = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        record.message = record.getMessage()

        payload: dict[str, Any] = {key: getattr(record, attr, None) for key, attr in self.FIELD_MAP.items()}

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)


class CompressedRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler that gzip-compresses every rotated backup.

    Backups are kept as ``<base>.1.gz`` .. ``<base>.<backupCount>.gz`` instead
    of the uncompressed ``<base>.1`` .. ``<base>.<backupCount>`` files that
    :class:`~logging.handlers.RotatingFileHandler` produces, keeping disk
    usage proportional to the *compressed* size rather than the raw size.

    ``doRollover`` is a full override rather than a thin wrapper around
    ``super().doRollover()``: the parent implementation shifts plain
    ``<base>.N`` files (``.1`` -> ``.2`` -> ``.3`` ...). Since this handler
    deletes the plain ``.1`` file immediately after compressing it, the
    parent's rename chain would find nothing at ``.1`` on the *next*
    rotation and silently fail to preserve older ``.gz`` backups, causing
    every rotation to clobber the single most recent backup instead of
    keeping ``backupCount`` of them.

    Inherits all parameters from :class:`~logging.handlers.RotatingFileHandler`.
    """

    def doRollover(self) -> None:
        """Performs log rotation, shifting and compressing backups.

        Shifts existing ``<base>.N.gz`` backups up by one index (``.4.gz``
        -> ``.5.gz``, ``.3.gz`` -> ``.4.gz``, ...), then renames the active
        log file to ``<base>.1`` and compresses it to ``<base>.1.gz``.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        base_path: Path = Path(self.baseFilename)

        if self.backupCount > 0:
            for index in range(self.backupCount - 1, 0, -1):
                source: Path = base_path.with_name(f"{base_path.name}.{index}.gz")
                destination: Path = base_path.with_name(f"{base_path.name}.{index + 1}.gz")
                if source.exists():
                    if destination.exists():
                        destination.unlink()
                    source.rename(destination)

            first_backup: Path = base_path.with_name(f"{base_path.name}.1")
            first_backup_gz: Path = base_path.with_name(f"{base_path.name}.1.gz")

            if first_backup.exists():
                first_backup.unlink()
            if base_path.exists():
                base_path.rename(first_backup)

            if first_backup.exists():
                try:
                    with first_backup.open("rb") as src, gzip.open(first_backup_gz, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    first_backup.unlink()
                except OSError as exc:
                    # Non-fatal: leave the uncompressed file if compression fails.
                    logging.getLogger("cachyDB").warning(
                        "Failed to compress rotated log file '%s': %s", first_backup, exc
                    )

        if not self.delay:
            self.stream = self._open()


class LogHandler:
    """Singleton logging handler for cachyDB.

    Configures a named ``cachyDB`` logger with:

    - A :class:`CompressedRotatingFileHandler` writing to
      ``<LOG_DIR>/cachyDB.log``, rotating when the file exceeds
      ``MAX_LOG_SIZE`` bytes, keeping ``LOG_BACKUP_COUNT`` backups.
    - An optional :class:`CompressedRotatingFileHandler` writing structured
      JSON to ``<LOG_DIR>/cachyDB.json.log`` (enabled via ``LOG_JSON=true``).
    - A :class:`~rich.logging.RichHandler` for colorized console output.

    All configuration can be overridden via environment variables; see
    module-level docstring for details.

    Records are routed through a :class:`~logging.handlers.QueueHandler` /
    :class:`~logging.handlers.QueueListener` pair so that I/O-bound handler
    work (file writes, gzip compression) happens on a dedicated listener
    thread rather than the caller's thread.

    Attributes:
        log_level (int): Logging level resolved from ``LOG_LEVEL`` env var.
        log_dir (Path): Resolved log directory.
        file_path (Path): Path to the plain-text ``.log`` file.
        json_file_path (Path): Path to the structured JSON ``.json.log`` file.
        json_logging_enabled (bool): Whether JSON logging is active.

    Example:
        >>> handler = LogHandler()
        >>> import logging
        >>> logging.getLogger("cachyDB").info("Hello from cachyDB")
    """

    _instance: LogHandler | None = None
    _initialized: bool = False

    LOG_DIR_DEFAULT: str = "data/logs"
    LOG_BACKUP_COUNT: int = 5
    MAX_LOG_SIZE: int = 10_485_760  # 10 MB
    LOG_QUEUE_MAX_SIZE: int = 0  # 0 = unbounded queue for high-throughput scenarios

    # File handler format — verbose for post-mortem debugging.
    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Console (Rich) format — concise; Rich itself provides time and level columns.
    CONSOLE_LOG_FORMAT: str = "[dim]%(name)s[/dim] | %(funcName)s:%(lineno)d | %(message)s"

    # Log levels accepted from the environment.
    _VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

    def __new__(cls) -> LogHandler:
        """Creates or returns the singleton LogHandler instance.

        Returns:
            The single shared :class:`LogHandler` instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializes the singleton on first call; subsequent calls are no-ops.

        Resolves configuration from environment variables, creates the log
        directory, attaches all handlers to the ``cachyDB`` logger, and
        registers an :func:`atexit` hook to flush handlers on exit.

        Raises:
            LoggingHandlerError: If the log directory or file cannot be
                created, or if handler setup fails for any reason.
        """
        if self._initialized:
            return

        self.log_level: int = self._resolve_log_level()
        self.log_dir: Path = Path(os.environ.get("LOG_DIR", self.LOG_DIR_DEFAULT))
        self.file_path: Path = self.log_dir / "cachyDB.log"
        self.json_file_path: Path = self.log_dir / "cachyDB.json.log"
        self.json_logging_enabled: bool = os.environ.get("LOG_JSON", "false").strip().lower() in {"true", "1", "yes"}

        self._handlers: list[logging.Handler] = []
        self._listener: QueueListener | None = None

        try:
            self._create_log_dir()
            # Use unbounded queue (maxsize=0) to handle high-throughput logging without dropping records
            # This prevents queue.Full exceptions during stress testing or high concurrent load
            self._queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=self.LOG_QUEUE_MAX_SIZE)
            self._setup_logging()
        except Exception as exc:
            raise LoggingHandlerError(f"Failed to configure logging: {exc}") from exc

        atexit.register(self._shutdown)
        LogHandler._initialized = True

    @classmethod
    def _resolve_log_level(cls) -> int:
        """Resolves the logging level from the ``LOG_LEVEL`` environment variable.

        Falls back to ``logging.INFO`` when the variable is absent or invalid.

        Returns:
            An integer logging level constant (e.g., ``logging.DEBUG``).
        """
        raw_level: str = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
        if raw_level not in cls._VALID_LOG_LEVELS:
            logging.warning(
                "Invalid LOG_LEVEL '%s'. Valid values: %s. Defaulting to INFO.",
                raw_level,
                ", ".join(sorted(cls._VALID_LOG_LEVELS)),
            )
            return logging.INFO
        return getattr(logging, raw_level)

    def _create_log_dir(self) -> None:
        """Creates the log directory and log files on disk if they do not exist.

        The log directory receives ``0o700`` permissions (owner
        read/write/execute only). Each log file receives ``0o600``
        permissions (owner read/write only). Files are created with
        :func:`os.open` using an explicit mode (rather than ``touch()``
        followed by a separate ``chmod()``) to avoid a race window where
        the file would briefly exist with default, umask-derived
        permissions before being locked down.

        Raises:
            LoggingHandlerError: If the directory or any log file cannot be created.
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.chmod(0o700)
        except OSError as exc:
            raise LoggingHandlerError(f"Failed to create log directory '{self.log_dir}': {exc}") from exc

        for log_file in (self.file_path, self.json_file_path):
            try:
                if log_file.exists():
                    log_file.chmod(0o600)
                else:
                    fd: int = os.open(log_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    os.close(fd)
            except FileExistsError:
                # Created concurrently by another process/thread; just re-assert permissions.
                log_file.chmod(0o600)
            except OSError as exc:
                raise LoggingHandlerError(f"Failed to create log file '{log_file}': {exc}") from exc

    def _build_file_handler(self) -> CompressedRotatingFileHandler:
        """Constructs the plain-text rotating file handler.

        Returns:
            A configured :class:`CompressedRotatingFileHandler` instance.

        Raises:
            OSError: If the log file cannot be opened.
        """
        file_formatter: logging.Formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.LOG_DATE_FORMAT)
        handler: CompressedRotatingFileHandler = CompressedRotatingFileHandler(
            filename=self.file_path,
            mode="a",
            maxBytes=self.MAX_LOG_SIZE,
            backupCount=self.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(file_formatter)
        handler.setLevel(logging.DEBUG)  # Capture everything; console filters separately.
        return handler

    def _build_json_handler(self) -> CompressedRotatingFileHandler:
        """Constructs the structured JSON rotating file handler.

        Returns:
            A configured :class:`CompressedRotatingFileHandler` using :class:`JsonFormatter`.

        Raises:
            OSError: If the JSON log file cannot be opened.
        """
        handler: CompressedRotatingFileHandler = CompressedRotatingFileHandler(
            filename=self.json_file_path,
            mode="a",
            maxBytes=self.MAX_LOG_SIZE,
            backupCount=self.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.DEBUG)
        return handler

    def _build_console_handler(self) -> RichHandler:
        """Constructs the Rich console handler.

        Returns:
            A configured :class:`~rich.logging.RichHandler` instance.
        """
        console_formatter: logging.Formatter = logging.Formatter(self.CONSOLE_LOG_FORMAT)
        handler: RichHandler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        handler.setFormatter(console_formatter)
        handler.setLevel(self.log_level)  # Respects LOG_LEVEL; file always captures DEBUG.
        return handler

    def _setup_logging(self) -> None:
        """Attaches all configured handlers to the ``cachyDB`` named logger.

        Uses the ``cachyDB`` named logger rather than the root logger to
        avoid interfering with other libraries. ``propagate`` is set to
        ``False`` for the same reason. Any pre-existing handlers are cleared
        before attachment to prevent duplicates on reconfiguration.

        The underlying file/JSON/console handlers are driven by a
        :class:`~logging.handlers.QueueListener` (started with
        ``respect_handler_level=True`` so that each handler's own level,
        e.g. the console handler's ``LOG_LEVEL`` filter, is actually
        honored instead of every record reaching every handler).

        Raises:
            OSError: If any log file cannot be opened by its handler.
        """
        handlers: list[logging.Handler] = [self._build_file_handler(), self._build_console_handler()]
        if self.json_logging_enabled:
            handlers.append(self._build_json_handler())

        self._handlers = handlers

        # Start the QueueListener: it consumes from the queue on a dedicated
        # thread and dispatches each record to the real handlers, so that
        # slow I/O (file writes, gzip compression) never blocks callers.
        self._listener = QueueListener(self._queue, *handlers, respect_handler_level=True)
        self._listener.start()

        logger: logging.Logger = logging.getLogger("cachyDB")
        logger.setLevel(self.log_level)  # Logger controls overall level; handlers filter under it.

        if logger.handlers:
            logger.handlers.clear()

        logger.addHandler(QueueHandler(self._queue))
        logger.propagate = False

        logger.info(
            "CachyDB logging initialised. level=%s json=%s dir=%s",
            logging.getLevelName(self.log_level),
            self.json_logging_enabled,
            self.log_dir,
        )

    def _shutdown(self) -> None:
        """Flushes and closes all logging resources on interpreter exit.

        Stops the :class:`~logging.handlers.QueueListener` first (so the
        listener thread is no longer dispatching to handlers), then
        explicitly flushes and closes every real handler (file, JSON,
        console) followed by the ``QueueHandler`` attached to the logger.
        Each step is isolated so a failure in one handler does not prevent
        cleanup of the others.

        Registered with :func:`atexit` so that buffered records are not
        lost when the interpreter exits normally.
        """
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup, must not raise
                logging.getLogger("cachyDB").debug("Error stopping queue listener: %s", exc)

        for handler in self._handlers:
            try:
                handler.flush()
                handler.close()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup, must not raise
                logging.getLogger("cachyDB").debug("Error closing handler '%s': %s", handler, exc)

        logger: logging.Logger = logging.getLogger("cachyDB")
        for handler in logger.handlers:
            try:
                handler.flush()
                handler.close()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup, must not raise
                logging.getLogger("cachyDB").debug("Error closing logger handler '%s': %s", handler, exc)
