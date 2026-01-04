"""
Logging configuration for ONFRA PFAS application.

Provides structured logging with JSON format for run logs.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        return json.dumps(log_data, ensure_ascii=False, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable log formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname

        if self.use_colors and level in self.COLORS:
            level_str = f"{self.COLORS[level]}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        message = record.getMessage()

        # Add extra data if present
        extra = ""
        if hasattr(record, "extra_data") and record.extra_data:
            extra = f" | {record.extra_data}"

        return f"[{timestamp}] {level_str} {record.name}: {message}{extra}"


class StructuredLogger(logging.Logger):
    """Logger with structured data support."""

    def _log_with_data(
        self,
        level: int,
        msg: str,
        args: tuple = (),
        exc_info: Any = None,
        extra: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Log with extra structured data."""
        if extra is None:
            extra = {}

        if kwargs:
            extra["extra_data"] = kwargs

        super()._log(level, msg, args, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log_with_data(logging.DEBUG, msg, args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log_with_data(logging.INFO, msg, args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log_with_data(logging.WARNING, msg, args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log_with_data(logging.ERROR, msg, args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._log_with_data(logging.CRITICAL, msg, args, **kwargs)


# Set custom logger class
logging.setLoggerClass(StructuredLogger)


def setup_root_logger(
    level: int = logging.INFO,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Setup root logger with console output.

    Args:
        level: Logging level
        use_colors: Use colored output

    Returns:
        Root logger
    """
    logger = logging.getLogger("onfra_pfas")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(HumanFormatter(use_colors=use_colors))
    logger.addHandler(console_handler)

    return logger


def setup_run_logger(
    run_dir: Path,
    log_filename: str = "run.log",
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup logger for a specific run with file output.

    Creates a run.log file with JSON-formatted entries.

    Args:
        run_dir: Directory for run outputs
        log_filename: Name of log file
        level: Logging level

    Returns:
        Logger for the run
    """
    # Ensure log directory exists
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_filename

    # Get or create logger
    logger = logging.getLogger(f"onfra_pfas.run.{run_dir.name}")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler with JSON format
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only INFO and above to console
    console_handler.setFormatter(HumanFormatter(use_colors=True))
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def log_step(
    logger: logging.Logger,
    step_name: str,
    status: str = "started",
    **metrics: Any,
) -> None:
    """
    Log a pipeline step with metrics.

    Args:
        logger: Logger instance
        step_name: Name of the step
        status: Status (started, completed, failed)
        **metrics: Additional metrics to log
    """
    data = {
        "step": step_name,
        "status": status,
        **metrics,
    }

    if status == "started":
        logger.info(f"Step '{step_name}' started", extra_data=data)
    elif status == "completed":
        logger.info(f"Step '{step_name}' completed", extra_data=data)
    elif status == "failed":
        logger.error(f"Step '{step_name}' failed", extra_data=data)
    else:
        logger.info(f"Step '{step_name}' - {status}", extra_data=data)


def log_config_snapshot(logger: logging.Logger, config: dict, path: Path | None = None) -> None:
    """
    Log configuration snapshot.

    Args:
        logger: Logger instance
        config: Configuration dictionary
        path: Optional path to save config JSON
    """
    logger.info("Configuration snapshot", extra_data={"config": config})

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def log_error_with_traceback(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """
    Log an error with full traceback.

    Args:
        logger: Logger instance
        error: Exception to log
        context: Additional context string
    """
    import traceback

    tb = traceback.format_exc()
    logger.error(
        f"Error{' in ' + context if context else ''}: {error}",
        extra_data={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb,
        },
    )
