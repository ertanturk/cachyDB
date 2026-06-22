"""Module for the LogHandler class in the cachyDB database.

This module provides a process-wide singleton that configures a named
``cachyDB`` logger with a rotating file handler and a console handler.
The singleton guarantees that the handlers are attached exactly once,
regardless of how many times ``LogHandler()`` is called.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LoggingHandlerError(Exception):
    """Custom exception for logging configuration errors."""


class LogHandler:
    """Singleton logging handler for cachyDB.

    Configures a named ``cachyDB`` logger with:

    - A :class:`~logging.handlers.RotatingFileHandler` writing to
      ``<LOG_DIR>/cachyDB.log``, rotating when the file exceeds
      ``MAX_LOG_SIZE`` bytes.
    - A :class:`~logging.StreamHandler` for console output.

    The singleton guarantees that ``setup_logging`` is executed exactly
    once per process.  Subsequent calls to ``LogHandler()`` return the
    same instance without reconfiguring anything.

    Attributes:
        log_level (int): Logging level applied to all handlers.
        file_path (Path): Path to the ``.log`` file on disk.

    Example:
        >>> handler = LogHandler()
        >>> import logging
        >>> logging.getLogger("cachyDB").info("Hello from cachyDB")
    """

    _instance: LogHandler | None = None
    _initialized: bool = False

    LOG_DIR: str = "data/logs"
    LOG_BACKUP_COUNT: int = 5
    MAX_LOG_SIZE: int = 10_485_760  # 10 MB
    # Example: 2025-09-28 12:34:56 - cachyDB - INFO - WAL successfully flushed
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

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

        Creates the log directory and file if they do not exist, then
        attaches rotating-file and console handlers to the ``cachyDB``
        named logger.

        Raises:
            LoggingHandlerError: If the log directory or file cannot be
                created, or if handler setup fails.
        """
        if self._initialized:
            return

        self.log_level: int = logging.INFO
        self.file_path: Path = Path(self.LOG_DIR) / "cachyDB.log"
        self.create_log_dir()
        try:
            self.setup_logging()
        except Exception as exc:
            raise LoggingHandlerError(f"Failed to configure logging: {exc}") from exc

    def create_log_dir(self) -> None:
        """Creates the log directory and log file on disk if they do not exist.

        The log directory receives ``0o700`` permissions (owner
        read/write/execute only).  The log file receives ``0o600``
        permissions (owner read/write only) when first created.

        Raises:
            LoggingHandlerError: If the directory or file cannot be created.
        """
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.parent.chmod(0o700)
        except OSError as exc:
            raise LoggingHandlerError(f"Failed to create log directory '{self.file_path.parent}': {exc}") from exc

        try:
            if not self.file_path.exists():
                self.file_path.touch()
                self.file_path.chmod(0o600)
        except OSError as exc:
            raise LoggingHandlerError(f"Failed to create log file '{self.file_path}': {exc}") from exc

    def setup_logging(self) -> None:
        """Attaches rotating-file and console handlers to the ``cachyDB`` logger.

        A named ``cachyDB`` logger is used rather than the root logger so
        that this class does not interfere with handlers registered by other
        libraries (pytest, web frameworks, etc.).  ``propagate`` is set to
        ``False`` to prevent records from also reaching root-logger handlers.

        If the logger already has handlers attached (e.g., due to a previous
        call), they are cleared first to avoid duplicates.

        Raises:
            OSError: If the log file cannot be opened by the file handler.
        """
        log_format: logging.Formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.LOG_DATE_FORMAT)
        logger: logging.Logger = logging.getLogger("cachyDB")
        logger.setLevel(self.log_level)

        # Clear any pre-existing handlers to prevent duplicates on reconfiguration
        if logger.handlers:
            logger.handlers.clear()

        file_handler: RotatingFileHandler = RotatingFileHandler(
            filename=self.file_path,
            mode="a",
            maxBytes=self.MAX_LOG_SIZE,
            backupCount=self.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(log_format)
        file_handler.setLevel(self.log_level)

        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        console_handler.setLevel(self.log_level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # Keep cachyDB log records out of the root logger to avoid interfering
        # with other libraries' logging configuration.
        logger.propagate = False

        logger.info("CachyDB logging setup complete.")
        LogHandler._initialized = True
