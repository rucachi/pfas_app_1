"""
Utility functions for ONFRA PFAS application.

Provides resource path resolution, file hashing, timing, and other helpers.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator


@dataclass
class TimingResult:
    """Result of a timed operation."""

    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    elapsed_seconds: float = 0.0

    def stop(self) -> float:
        """Stop timing and return elapsed seconds."""
        self.end_time = time.perf_counter()
        self.elapsed_seconds = self.end_time - self.start_time
        return self.elapsed_seconds

    def __str__(self) -> str:
        if self.elapsed_seconds < 1:
            return f"{self.name}: {self.elapsed_seconds * 1000:.1f}ms"
        elif self.elapsed_seconds < 60:
            return f"{self.name}: {self.elapsed_seconds:.2f}s"
        else:
            minutes = int(self.elapsed_seconds // 60)
            seconds = self.elapsed_seconds % 60
            return f"{self.name}: {minutes}m {seconds:.1f}s"


@contextmanager
def timed_block(name: str) -> Generator[TimingResult, None, None]:
    """
    Context manager for timing a block of code.

    Usage:
        with timed_block("Operation") as timer:
            do_something()
        print(timer)  # "Operation: 1.23s"

    Args:
        name: Name for the operation being timed

    Yields:
        TimingResult with elapsed time after block completes
    """
    result = TimingResult(name=name)
    try:
        yield result
    finally:
        result.stop()


def resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and PyInstaller.

    When running as a PyInstaller bundle (onefile or onedir), resources
    are extracted to a temporary folder referenced by sys._MEIPASS.
    In development, resources are relative to the project root.

    Args:
        relative_path: Path relative to project root or bundle root

    Returns:
        Absolute path to the resource
    """
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        # Development: go up from this file to project root
        # src/onfra_pfas/core/utils.py -> project_root
        base_path = Path(__file__).resolve().parent.parent.parent.parent

    return base_path / relative_path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent.parent.parent


def file_hash(path: Path | str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """
    Calculate hash of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)
        chunk_size: Chunk size for reading file

    Returns:
        Hexadecimal hash string
    """
    path = Path(path)
    h = hashlib.new(algorithm)

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)

    return h.hexdigest()


def file_size_mb(path: Path | str) -> float:
    """
    Get file size in megabytes.

    Args:
        path: Path to file

    Returns:
        File size in MB
    """
    return Path(path).stat().st_size / (1024 * 1024)


def generate_run_id(prefix: str = "run") -> str:
    """
    Generate a unique run ID based on timestamp.

    Args:
        prefix: Prefix for the run ID

    Returns:
        Run ID string (e.g., "run_20240104_095723")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        Path to the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str, max_length: int = 200) -> str:
    """
    Convert string to safe filename.

    Args:
        name: Original name
        max_length: Maximum filename length

    Returns:
        Safe filename string
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")

    # Remove leading/trailing spaces and dots
    name = name.strip(" .")

    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length]

    return name or "unnamed"


def format_number(n: float | int, precision: int = 2) -> str:
    """
    Format number with appropriate units (K, M, etc.).

    Args:
        n: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.{precision}f}G"
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.{precision}f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def format_mass(mass: float, precision: int = 4) -> str:
    """
    Format mass value.

    Args:
        mass: Mass in Da
        precision: Decimal precision

    Returns:
        Formatted mass string
    """
    return f"{mass:.{precision}f}"


def format_rt(rt_seconds: float) -> str:
    """
    Format retention time.

    Args:
        rt_seconds: RT in seconds

    Returns:
        Formatted RT string (e.g., "5.23 min")
    """
    minutes = rt_seconds / 60
    return f"{minutes:.2f} min"


def ppm_difference(theoretical: float, observed: float) -> float:
    """
    Calculate ppm difference between theoretical and observed mass.

    Args:
        theoretical: Theoretical mass
        observed: Observed mass

    Returns:
        Difference in ppm
    """
    if theoretical == 0:
        return float("inf")
    return abs((observed - theoretical) / theoretical) * 1_000_000


def mass_tolerance_da(mass: float, ppm: float) -> float:
    """
    Calculate mass tolerance in Da from ppm.

    Args:
        mass: Mass in Da
        ppm: Tolerance in ppm

    Returns:
        Tolerance in Da
    """
    return mass * ppm / 1_000_000


def is_within_tolerance(
    observed: float,
    theoretical: float,
    tolerance: float,
    use_ppm: bool = True,
) -> bool:
    """
    Check if observed value is within tolerance of theoretical.

    Args:
        observed: Observed value
        theoretical: Theoretical value
        tolerance: Tolerance value (ppm or absolute)
        use_ppm: If True, tolerance is in ppm; else absolute

    Returns:
        True if within tolerance
    """
    if use_ppm:
        return ppm_difference(theoretical, observed) <= tolerance
    else:
        return abs(observed - theoretical) <= tolerance


def get_system_info() -> dict:
    """
    Get system information for logging.

    Returns:
        Dictionary with system info
    """
    import platform

    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.

    Args:
        s: Input string
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix
