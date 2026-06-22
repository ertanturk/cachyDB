"""Tests for the LogHandler singleton class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from cachyDB.logger.log_handler import LoggingHandlerError, LogHandler

if TYPE_CHECKING:
    from pathlib import Path


def _reset_singleton() -> None:
    """Reset the LogHandler singleton state between tests."""
    LogHandler._instance = None
    LogHandler._initialized = False
    # Remove any handlers added to the cachyDB logger by a previous test
    cachy_logger = logging.getLogger("cachyDB")
    for handler in list(cachy_logger.handlers):
        handler.close()
        cachy_logger.removeHandler(handler)
    cachy_logger.propagate = True


class TestSingleton:
    """Tests for the singleton contract of LogHandler."""

    def setup_method(self) -> None:
        """Reset singleton state before each test."""
        _reset_singleton()

    def teardown_method(self) -> None:
        """Clean up singleton state after each test."""
        _reset_singleton()

    def test_same_instance_returned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two LogHandler() calls must return the exact same object."""
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        h1 = LogHandler()
        h2 = LogHandler()
        assert h1 is h2

    def test_init_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling LogHandler() three times must not add duplicate handlers."""
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        LogHandler()
        LogHandler()
        logger = logging.getLogger("cachyDB")
        assert len(logger.handlers) == 2  # exactly file + console

    def test_initialized_flag_set_after_first_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_initialized must be True after the first successful LogHandler()."""
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        assert LogHandler._initialized is False
        LogHandler()
        assert LogHandler._initialized is True


class TestCreateLogDir:
    """Tests for LogHandler.create_log_dir."""

    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_creates_log_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_log_dir must create the log directory."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(log_dir))
        LogHandler()
        assert log_dir.is_dir()

    def test_creates_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_log_dir must create the cachyDB.log file."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(log_dir))
        LogHandler()
        assert (log_dir / "cachyDB.log").is_file()

    def test_log_file_is_not_a_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """cachyDB.log must be a regular file, not a directory."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(log_dir))
        LogHandler()
        log_file = log_dir / "cachyDB.log"
        assert log_file.is_file()
        assert not log_file.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_log_dir must not raise if the directory already exists."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(log_dir))
        LogHandler()  # must not raise

    def test_error_message_says_log_not_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """LoggingHandlerError messages must reference 'log', not 'table'."""
        # Make the parent directory unwritable to force an OSError
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(log_dir))
        # Pre-create the parent as a file so mkdir fails
        log_dir.parent.chmod(0o555)
        try:
            h = LogHandler.__new__(LogHandler)
            h.log_level = logging.INFO
            h.file_path = log_dir / "unwritable" / "cachyDB.log"
            with pytest.raises(LoggingHandlerError, match="log directory"):
                h.create_log_dir()
        finally:
            log_dir.parent.chmod(0o755)


class TestSetupLogging:
    """Tests for LogHandler.setup_logging."""

    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_attaches_two_handlers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """setup_logging must attach exactly one file handler and one console handler."""
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        logger = logging.getLogger("cachyDB")
        assert len(logger.handlers) == 2

    def test_file_handler_writes_to_correct_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The RotatingFileHandler must target <LOG_DIR>/cachyDB.log."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(log_dir))
        LogHandler()
        logger = logging.getLogger("cachyDB")
        from logging.handlers import RotatingFileHandler

        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(log_dir / "cachyDB.log")

    def test_does_not_use_root_logger(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """setup_logging must not add handlers to the root logger."""
        root_before = len(logging.getLogger().handlers)
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        assert len(logging.getLogger().handlers) == root_before

    def test_propagate_is_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The cachyDB logger must not propagate to the root logger."""
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        assert logging.getLogger("cachyDB").propagate is False

    def test_does_not_clear_root_logger_handlers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing root-logger handlers must survive setup_logging."""
        sentinel = logging.StreamHandler()
        logging.getLogger().addHandler(sentinel)
        try:
            monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
            LogHandler()
            assert sentinel in logging.getLogger().handlers
        finally:
            logging.getLogger().removeHandler(sentinel)

    def test_log_level_applied_to_handlers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """All handlers must be set to the configured log_level (INFO by default)."""
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        for handler in logging.getLogger("cachyDB").handlers:
            assert handler.level == logging.INFO

    def test_records_are_written_to_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A log record must appear in the log file after setup."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(log_dir))
        LogHandler()
        logger = logging.getLogger("cachyDB")
        logger.info("integration test marker")
        # Flush all handlers to ensure the record is written
        for handler in logger.handlers:
            handler.flush()
        content = (log_dir / "cachyDB.log").read_text(encoding="utf-8")
        assert "integration test marker" in content


class TestNewAndReset:
    """Tests for __new__ and class-level singleton state."""

    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_new_accepts_no_extra_args(self) -> None:
        """__new__ must not accept positional or keyword arguments."""
        with pytest.raises(TypeError):
            LogHandler.__new__(LogHandler, "unexpected")  # ty: ignore[too-many-positional-arguments]

    def test_instance_is_none_before_first_call(self) -> None:
        """_instance must be None before any LogHandler() is created."""
        assert LogHandler._instance is None

    def test_instance_is_set_after_first_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_instance must be the LogHandler object after the first call."""
        monkeypatch.setattr(LogHandler, "LOG_DIR", str(tmp_path / "logs"))
        h = LogHandler()
        assert LogHandler._instance is h
