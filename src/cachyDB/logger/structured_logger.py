"""Structured logging utilities for enterprise-grade observability in cachyDB.

This module provides professional-grade structured logging with contextual metadata,
performance metrics, error codes, request tracing, and metrics aggregation.
Follows enterprise standards used by Datadog, New Relic, and similar APM platforms.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generator

__all__: list[str] = [
    "OperationType",
    "StatusCode",
    "StructuredLogContext",
    "StructuredLogger",
    "get_structured_logger",
]


class OperationType(Enum):
    """Enumeration of operation types for structured logging."""

    GET = "GET"
    SET = "SET"
    DELETE = "DELETE"
    FLUSH = "FLUSH"
    COMPACT = "COMPACT"
    CHECKPOINT = "CHECKPOINT"
    RECOVERY = "RECOVERY"
    PAGE_READ = "PAGE_READ"
    PAGE_WRITE = "PAGE_WRITE"
    WAL_APPEND = "WAL_APPEND"
    MEMTABLE_INSERT = "MEMTABLE_INSERT"
    CACHE_LOOKUP = "CACHE_LOOKUP"
    DISK_SCAN = "DISK_SCAN"


class StatusCode(Enum):
    """Enumeration of operation status codes."""

    SUCCESS = "success"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    NOT_FOUND = "not_found"
    ERROR = "error"
    TIMEOUT = "timeout"
    DEGRADED = "degraded"


@dataclass
class StructuredLogContext:
    """Structured context for a database operation.

    This dataclass encapsulates all relevant metadata for an operation,
    enabling consistent structured logging across the codebase.

    Attributes:
        trace_id: Unique identifier for request tracing across components.
        operation_type: Type of operation (GET, SET, etc).
        component: Component name (router, wal, pager, etc).
        status: Operation status code.
        latency_ms: Operation duration in milliseconds.
        table_name: Optional target table name.
        key_pattern: Optional key or key pattern being accessed.
        data_size_bytes: Optional size of data involved.
        records_affected: Optional count of records affected.
        error_code: Optional error/result code (0 = success).
        error_message: Optional error or status message.
        cache_hit_rate: Optional cache hit rate percentage.
        throughput_kops: Optional throughput in thousands of operations per second.
        warnings: Optional list of performance warnings.
        context_data: Optional dict of additional contextual details.
    """

    trace_id: str
    operation_type: OperationType
    component: str
    status: StatusCode
    latency_ms: float
    table_name: str | None = None
    key_pattern: str | None = None
    data_size_bytes: int | None = None
    records_affected: int | None = None
    error_code: int = 0
    error_message: str | None = None
    cache_hit_rate: float | None = None
    throughput_kops: float | None = None
    warnings: list[str] | None = None
    context_data: dict[str, Any] | None = None

    def to_log_string(self) -> str:
        """Formats the context as a structured log string.

        Returns:
            Formatted log string suitable for professional logging systems.

        Example:
            [trace_id=abc123][op_type=GET][component=router][status=hit][latency_ms=0.5]
            [table=users][error_code=0][cache_hit_rate=85.5%]
        """
        parts: list[str] = []

        # Core operation info
        parts.append(f"[trace_id={self.trace_id}]")
        parts.append(f"[op_type={self.operation_type.value}]")
        parts.append(f"[component={self.component}]")
        parts.append(f"[status={self.status.value}]")
        parts.append(f"[latency_ms={self.latency_ms:.3f}]")

        # Optional contextual fields
        if self.table_name:
            parts.append(f"[table={self.table_name}]")
        if self.key_pattern:
            parts.append(f"[key_pattern={self.key_pattern}]")
        if self.data_size_bytes is not None:
            parts.append(f"[data_size_bytes={self.data_size_bytes}]")
        if self.records_affected is not None:
            parts.append(f"[records_affected={self.records_affected}]")

        # Error/status info
        parts.append(f"[error_code={self.error_code}]")
        if self.error_message:
            parts.append(f"[error_msg={self.error_message}]")

        # Performance metrics
        if self.cache_hit_rate is not None:
            parts.append(f"[cache_hit_rate={self.cache_hit_rate:.1f}%]")
        if self.throughput_kops is not None:
            parts.append(f"[throughput_kops/s={self.throughput_kops:.2f}]")

        # Warnings (if any)
        if self.warnings:
            warnings_str: str = "; ".join(self.warnings)
            parts.append(f"[warnings={warnings_str}]")

        # Additional context (for debugging/APM integration)
        if self.context_data:
            for key, value in self.context_data.items():
                if value is not None:
                    parts.append(f"[{key}={value}]")

        return "".join(parts)


class StructuredLogger:
    """Professional structured logger for cachyDB components.

    Provides context-aware logging with performance metrics, error tracking,
    and enterprise-grade observability features.

    Attributes:
        logger: Underlying Python logger instance.
        component: Component name for this logger.
        latency_warning_threshold_ms: Threshold for latency warnings.
    """

    # Performance thresholds for warnings
    LATENCY_WARNING_MS: dict[str, float] = {
        "GET": 100.0,
        "SET": 150.0,
        "CACHE_LOOKUP": 250.0,
        "PAGE_READ": 100.0,
        "PAGE_WRITE": 150.0,
        "WAL_APPEND": 200.0,
        "DISK_SCAN": 500.0,
    }

    def __init__(self, component: str, base_logger: logging.Logger | None = None) -> None:
        """Initializes the structured logger for a component.

        Args:
            component: Component name (e.g., 'router', 'wal', 'pager').
            base_logger: Optional underlying logger. If None, creates new logger.
        """
        self.component: str = component
        self.logger: logging.Logger = base_logger or logging.getLogger(f"cachyDB.{component}")

    def _check_latency_warnings(self, context: StructuredLogContext) -> None:
        """Checks if operation latency exceeds thresholds and adds warnings.

        Args:
            context: The structured log context to potentially update with warnings.
        """
        threshold: float = self.LATENCY_WARNING_MS.get(context.operation_type.value, 100.0)
        if context.latency_ms > threshold:
            warning: str = f"latency {context.latency_ms:.2f}ms exceeds threshold {threshold:.2f}ms"
            if context.warnings is None:
                context.warnings = []
            context.warnings.append(warning)

    def log_operation(
        self,
        context: StructuredLogContext,
        level: int = logging.INFO,
    ) -> None:
        """Logs a structured operation with all metadata.

        Args:
            context: The structured log context.
            level: Logging level (default: INFO).
        """
        # Determine the effective level based on errors, warnings, or latency
        threshold: float = self.LATENCY_WARNING_MS.get(context.operation_type.value, 100.0)
        has_warnings: bool = bool(context.warnings) or (context.latency_ms > threshold)
        has_error: bool = context.error_code != 0 or level == logging.ERROR

        target_level: int = level
        if has_error:
            target_level = logging.ERROR
        elif has_warnings:
            target_level = logging.WARNING

        # Early return if the logger is not enabled for this level
        if not self.logger.isEnabledFor(target_level):
            return

        # Ensure we have a trace_id if actually logging
        if not context.trace_id:
            context.trace_id = str(uuid.uuid4())[:8]

        self._check_latency_warnings(context)

        log_string: str = context.to_log_string()

        # Log at the appropriate level
        if target_level == logging.ERROR:
            self.logger.error(log_string)
        elif target_level == logging.WARNING:
            self.logger.warning(log_string)
        elif target_level == logging.INFO:
            self.logger.info(log_string)
        elif target_level == logging.DEBUG:
            self.logger.debug(log_string)
        else:
            self.logger.log(target_level, log_string)

    @contextmanager
    def operation(
        self,
        operation_type: OperationType,
        table_name: str | None = None,
        key_pattern: str | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> Generator[StructuredLogContext]:
        """Context manager for measuring and logging operation performance.

        Automatically captures timing, status, and enables structured logging
        for the duration of the operation.

        Args:
            operation_type: Type of operation.
            table_name: Optional target table.
            key_pattern: Optional key or pattern.
            context_data: Optional additional context.

        Yields:
            StructuredLogContext that can be modified during execution.

        Example:
            with logger.operation(OperationType.GET, table_name="users") as ctx:
                result = perform_get()
                ctx.status = StatusCode.CACHE_HIT if cached else StatusCode.CACHE_MISS
                ctx.data_size_bytes = len(result) if result else 0
        """
        start_time: float = time.perf_counter()

        context: StructuredLogContext = StructuredLogContext(
            trace_id="",
            operation_type=operation_type,
            component=self.component,
            status=StatusCode.SUCCESS,
            latency_ms=0.0,
            table_name=table_name,
            key_pattern=key_pattern,
            context_data=context_data or {},
        )

        try:
            yield context
        except Exception as exc:
            context.status = StatusCode.ERROR
            context.error_code = 1
            context.error_message = str(exc)
            context.latency_ms = (time.perf_counter() - start_time) * 1000
            self.log_operation(context, level=logging.ERROR)
            raise
        else:
            context.latency_ms = (time.perf_counter() - start_time) * 1000
            self.log_operation(context)

    def log_cache_operation(
        self,
        trace_id: str,
        hit: bool,
        key: str,
        table_name: str,
        latency_ms: float,
        hit_rate: float | None = None,
    ) -> None:
        """Logs cache hit/miss operations.

        Args:
            trace_id: Unique trace identifier.
            hit: True if cache hit, False if miss.
            key: The key that was looked up.
            table_name: The table name.
            latency_ms: Operation latency in milliseconds.
            hit_rate: Optional overall cache hit rate percentage.
        """
        status: StatusCode = StatusCode.CACHE_HIT if hit else StatusCode.CACHE_MISS
        context: StructuredLogContext = StructuredLogContext(
            trace_id=trace_id,
            operation_type=OperationType.CACHE_LOOKUP,
            component=self.component,
            status=status,
            latency_ms=latency_ms,
            table_name=table_name,
            key_pattern=key,
            cache_hit_rate=hit_rate,
        )
        self._check_latency_warnings(context)
        self.log_operation(context, level=logging.DEBUG if hit else logging.DEBUG)

    def log_write_operation(
        self,
        trace_id: str,
        operation_type: OperationType,
        table_name: str,
        records_count: int,
        data_size_bytes: int,
        latency_ms: float,
        throughput_kops: float | None = None,
        error_code: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Logs write operations with throughput metrics.

        Args:
            trace_id: Unique trace identifier.
            operation_type: Type of write operation (SET, WAL_APPEND, etc).
            table_name: Target table name.
            records_count: Number of records affected.
            data_size_bytes: Total data size in bytes.
            latency_ms: Operation latency in milliseconds.
            throughput_kops: Optional throughput in thousands of ops/sec.
            error_code: Operation error code (0 = success).
            error_message: Optional error message.
        """
        context: StructuredLogContext = StructuredLogContext(
            trace_id=trace_id,
            operation_type=operation_type,
            component=self.component,
            status=StatusCode.SUCCESS if error_code == 0 else StatusCode.ERROR,
            latency_ms=latency_ms,
            table_name=table_name,
            records_affected=records_count,
            data_size_bytes=data_size_bytes,
            throughput_kops=throughput_kops,
            error_code=error_code,
            error_message=error_message,
        )
        self._check_latency_warnings(context)
        # Low-level high-frequency operations like SET, DELETE, PAGE_WRITE, PAGE_READ, WAL_APPEND should log at DEBUG
        # while system-level/bulk/infrequent operations like FLUSH and COMPACT should log at INFO
        if error_code != 0:
            level: int = logging.ERROR
        elif operation_type in (OperationType.FLUSH, OperationType.COMPACT):
            level = logging.INFO
        else:
            level = logging.DEBUG
        self.log_operation(context, level=level)

    def log_recovery_operation(
        self,
        trace_id: str,
        records_recovered: int,
        latency_ms: float,
        error_code: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Logs recovery/replay operations.

        Args:
            trace_id: Unique trace identifier.
            records_recovered: Number of records successfully recovered.
            latency_ms: Operation latency in milliseconds.
            error_code: Operation error code (0 = success).
            error_message: Optional error message.
        """
        context: StructuredLogContext = StructuredLogContext(
            trace_id=trace_id,
            operation_type=OperationType.RECOVERY,
            component=self.component,
            status=StatusCode.SUCCESS if error_code == 0 else StatusCode.ERROR,
            latency_ms=latency_ms,
            records_affected=records_recovered,
            error_code=error_code,
            error_message=error_message,
        )
        self._check_latency_warnings(context)
        level: int = logging.ERROR if error_code != 0 else logging.INFO
        self.log_operation(context, level=level)


# Global registry of component loggers
_structured_loggers: dict[str, StructuredLogger] = {}


def get_structured_logger(component: str) -> StructuredLogger:
    """Gets or creates a structured logger for the given component.

    Maintains a global registry to ensure one logger per component.

    Args:
        component: Component name (e.g., 'router', 'wal', 'pager').

    Returns:
        The structured logger for the component.
    """
    if component not in _structured_loggers:
        base_logger: logging.Logger = logging.getLogger(f"cachyDB.{component}")
        _structured_loggers[component] = StructuredLogger(component, base_logger)

    return _structured_loggers[component]
