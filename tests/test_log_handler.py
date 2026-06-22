"""Tests for the LogHandler singleton class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from rich.logging import RichHandler

from cachyDB.logger.log_handler import CompressedRotatingFileHandler, JsonFormatter, LoggingHandlerError, LogHandler

if TYPE_CHECKING:
    from pathlib import Path


def _reset_singleton() -> None:
    """Reset the LogHandler singleton state between tests."""
    LogHandler._instance = None
    LogHandler._initialized = False
    cachy_logger = logging.getLogger("cachyDB")
    for handler in list(cachy_logger.handlers):
        handler.close()
        cachy_logger.removeHandler(handler)
    cachy_logger.propagate = True


class TestSingleton:
    """Tests for the singleton contract of LogHandler."""

    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_same_instance_returned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two LogHandler() calls must return the exact same object."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        h1 = LogHandler()
        h2 = LogHandler()
        assert h1 is h2

    def test_init_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling LogHandler() three times must not add duplicate handlers."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        LogHandler()
        LogHandler()
        logger = logging.getLogger("cachyDB")
        assert len(logger.handlers) == 2  # file + console (JSON off by default)

    def test_initialized_flag_set_after_first_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_initialized must be True after the first successful LogHandler()."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        assert LogHandler._initialized is False
        LogHandler()
        assert LogHandler._initialized is True

    def test_instance_is_none_before_first_call(self) -> None:
        """_instance must be None before any LogHandler() is created."""
        assert LogHandler._instance is None

    def test_instance_is_set_after_first_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_instance must be the LogHandler object after the first call."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        h = LogHandler()
        assert LogHandler._instance is h


class TestCreateLogDir:
    """Tests for LogHandler._create_log_dir."""

    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_creates_log_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_create_log_dir must create the log directory."""
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        LogHandler()
        assert log_dir.is_dir()

    def test_creates_plain_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_create_log_dir must create the cachyDB.log file."""
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        LogHandler()
        assert (log_dir / "cachyDB.log").is_file()

    def test_creates_json_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_create_log_dir must create the cachyDB.json.log file."""
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        LogHandler()
        assert (log_dir / "cachyDB.json.log").is_file()

    def test_log_file_is_not_a_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """cachyDB.log must be a regular file, not a directory."""
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        LogHandler()
        log_file = log_dir / "cachyDB.log"
        assert log_file.is_file()
        assert not log_file.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_create_log_dir must not raise if the directory already exists."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        LogHandler()  # must not raise

    def test_error_on_unwritable_parent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """LoggingHandlerError must be raised when the log directory cannot be created."""
        unwritable = tmp_path / "noperm"
        unwritable.mkdir(mode=0o555)
        monkeypatch.setenv("LOG_DIR", str(unwritable / "logs"))
        try:
            with pytest.raises(LoggingHandlerError, match="log directory"):
                LogHandler()
        finally:
            unwritable.chmod(0o755)


class TestSetupLogging:
    """Tests for LogHandler._setup_logging."""

    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_attaches_two_handlers_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without LOG_JSON, exactly two handlers must be attached (file + console)."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.delenv("LOG_JSON", raising=False)
        LogHandler()
        assert len(logging.getLogger("cachyDB").handlers) == 2

    def test_attaches_three_handlers_with_json_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With LOG_JSON=true, exactly three handlers must be attached."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("LOG_JSON", "true")
        LogHandler()
        assert len(logging.getLogger("cachyDB").handlers) == 3

    def test_file_handler_targets_correct_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CompressedRotatingFileHandler must target <LOG_DIR>/cachyDB.log."""
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        LogHandler()
        logger = logging.getLogger("cachyDB")
        file_handlers = [h for h in logger.handlers if isinstance(h, CompressedRotatingFileHandler)]
        plain_handlers = [h for h in file_handlers if "json" not in h.baseFilename]
        assert len(plain_handlers) == 1
        assert plain_handlers[0].baseFilename == str(log_dir / "cachyDB.log")

    def test_json_handler_targets_correct_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With LOG_JSON=true, the JSON handler must target <LOG_DIR>/cachyDB.json.log."""
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        monkeypatch.setenv("LOG_JSON", "true")
        LogHandler()
        logger = logging.getLogger("cachyDB")
        # Fix: Ensure we are strictly matching the end of the filename
        json_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, CompressedRotatingFileHandler) and h.baseFilename.endswith("cachyDB.json.log")
        ]
        assert len(json_handlers) == 1
        assert json_handlers[0].baseFilename == str(log_dir / "cachyDB.json.log")

    def test_file_handler_level_is_debug(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The file handler must be set to DEBUG to capture all records."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        logger = logging.getLogger("cachyDB")
        file_handlers = [h for h in logger.handlers if isinstance(h, CompressedRotatingFileHandler)]
        for handler in file_handlers:
            assert handler.level == logging.DEBUG

    def test_console_handler_level_matches_log_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The console (Rich) handler must respect LOG_LEVEL (default INFO)."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        LogHandler()
        logger = logging.getLogger("cachyDB")
        rich_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) == 1
        assert rich_handlers[0].level == logging.INFO

    def test_console_handler_level_respects_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The console handler must use the level set in LOG_LEVEL."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        LogHandler()
        logger = logging.getLogger("cachyDB")
        rich_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
        assert rich_handlers[0].level == logging.WARNING

    def test_does_not_add_handlers_to_root_logger(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_setup_logging must not add handlers to the root logger."""
        root_before = len(logging.getLogger().handlers)
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        assert len(logging.getLogger().handlers) == root_before

    def test_propagate_is_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The cachyDB logger must not propagate to the root logger."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        LogHandler()
        assert logging.getLogger("cachyDB").propagate is False

    def test_existing_root_handlers_survive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing root-logger handlers must not be cleared by _setup_logging."""
        sentinel = logging.StreamHandler()
        logging.getLogger().addHandler(sentinel)
        try:
            monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
            LogHandler()
            assert sentinel in logging.getLogger().handlers
        finally:
            logging.getLogger().removeHandler(sentinel)

    def test_records_written_to_plain_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A log record must appear in cachyDB.log after setup."""
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        LogHandler()
        logger = logging.getLogger("cachyDB")
        logger.info("integration test marker")
        for handler in logger.handlers:
            handler.flush()
        content = (log_dir / "cachyDB.log").read_text(encoding="utf-8")
        assert "integration test marker" in content

    def test_records_written_to_json_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With LOG_JSON=true, a log record must appear as valid JSON in cachyDB.json.log."""
        import json as json_mod

        log_dir = tmp_path / "logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        monkeypatch.setenv("LOG_JSON", "true")
        LogHandler()
        logger = logging.getLogger("cachyDB")
        logger.info("json marker")
        for handler in logger.handlers:
            handler.flush()
        lines = [
            line for line in (log_dir / "cachyDB.json.log").read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        parsed = [json_mod.loads(line) for line in lines]
        messages = [r["message"] for r in parsed]
        assert any("json marker" in m for m in messages)


class TestResolveLogLevel:
    """Tests for LogHandler._resolve_log_level."""

    @pytest.mark.parametrize(
        "env_val,expected",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    def test_valid_levels(self, env_val: str, expected: int, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid LOG_LEVEL values must resolve to the correct integer constant."""
        monkeypatch.setenv("LOG_LEVEL", env_val)
        assert LogHandler._resolve_log_level() == expected

    def test_invalid_level_falls_back_to_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An invalid LOG_LEVEL value must fall back to INFO."""
        monkeypatch.setenv("LOG_LEVEL", "NONSENSE")
        assert LogHandler._resolve_log_level() == logging.INFO

    def test_missing_env_var_defaults_to_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent LOG_LEVEL must default to INFO."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        assert LogHandler._resolve_log_level() == logging.INFO


class TestJsonFormatter:
    """Tests for the JsonFormatter class."""

    def test_output_is_valid_json(self) -> None:
        """JsonFormatter.format must produce a valid JSON string."""
        import json as json_mod

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="cachyDB",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json_mod.loads(result)
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "cachyDB"

    def test_exception_included_in_output(self) -> None:
        """JsonFormatter must include exception info when present."""
        import json as json_mod

        formatter = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="cachyDB",
            level=logging.ERROR,
            pathname="",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )
        result = json_mod.loads(formatter.format(record))
        assert "exception" in result
        assert "ValueError" in result["exception"]

    def test_required_fields_present(self) -> None:
        """All FIELD_MAP keys must be present in the formatted output."""
        import json as json_mod

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="cachyDB",
            level=logging.DEBUG,
            pathname="",
            lineno=42,
            msg="fields check",
            args=(),
            exc_info=None,
        )
        result = json_mod.loads(formatter.format(record))
        for key in JsonFormatter.FIELD_MAP:
            assert key in result


class TestNewAndReset:
    """Tests for __new__ and class-level singleton state."""

    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_new_accepts_no_extra_args(self) -> None:
        """__new__ must not accept positional arguments beyond cls."""
        with pytest.raises(TypeError):
            LogHandler.__new__(LogHandler, "unexpected")  # ty: ignore[too-many-positional-arguments]
