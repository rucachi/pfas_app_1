"""
Backend module for array computation.

Provides unified interface for NumPy/CuPy with automatic GPU detection and fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from types import ModuleType
from typing import Any, Callable, TypeVar

import numpy as np

from .config import GPUMode

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Computation backend type."""

    NUMPY = "numpy"
    CUPY = "cupy"


@dataclass
class GPUInfo:
    """GPU device information."""

    available: bool
    device_name: str | None = None
    device_id: int | None = None
    total_memory_gb: float | None = None
    free_memory_gb: float | None = None
    cuda_version: str | None = None
    cupy_version: str | None = None
    error_message: str | None = None


@dataclass
class BackendInfo:
    """Current backend information."""

    backend_type: BackendType
    module: ModuleType
    gpu_info: GPUInfo | None
    reason: str  # Why this backend was selected


# Global state for backend
_current_backend: BackendInfo | None = None
_cupy_module: ModuleType | None = None
_cupy_available: bool | None = None


def _try_import_cupy() -> tuple[ModuleType | None, str | None]:
    """Attempt to import CuPy and return module or error message."""
    global _cupy_module, _cupy_available

    if _cupy_available is not None:
        if _cupy_available:
            return _cupy_module, None
        return None, "CuPy previously failed to import"

    try:
        import cupy as cp

        # Test CUDA availability by allocating a small array
        test_arr = cp.array([1, 2, 3])
        _ = test_arr.sum()  # Force synchronization
        del test_arr

        _cupy_module = cp
        _cupy_available = True
        return cp, None

    except ImportError as e:
        _cupy_available = False
        return None, f"CuPy not installed: {e}"

    except Exception as e:
        _cupy_available = False
        return None, f"CuPy/CUDA error: {e}"


def probe_gpu() -> GPUInfo:
    """
    Probe GPU availability and return device information.

    Returns:
        GPUInfo with details about GPU availability and specs.
    """
    cp, error = _try_import_cupy()

    if cp is None:
        return GPUInfo(available=False, error_message=error)

    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        mem_info = cp.cuda.runtime.memGetInfo()

        return GPUInfo(
            available=True,
            device_name=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            device_id=device.id,
            total_memory_gb=props["totalGlobalMem"] / (1024**3),
            free_memory_gb=mem_info[0] / (1024**3),
            cuda_version=".".join(map(str, cp.cuda.runtime.runtimeGetVersion())),
            cupy_version=cp.__version__,
        )

    except Exception as e:
        return GPUInfo(available=False, error_message=f"Failed to query GPU: {e}")


def get_array_backend(
    mode: GPUMode,
    data_size: int = 0,
    feature_count: int = 0,
    rt_grid_len: int = 0,
    correlation_pairs: int = 0,
    thresholds: dict[str, int] | None = None,
) -> BackendInfo:
    """
    Get the appropriate array backend based on mode and data size.

    Args:
        mode: GPU usage policy
        data_size: General data size metric
        feature_count: Number of features
        rt_grid_len: Length of RT grid
        correlation_pairs: Number of correlation pairs
        thresholds: Custom thresholds dict (overrides defaults)

    Returns:
        BackendInfo with the selected backend module.

    Raises:
        RuntimeError: If FORCE_GPU mode but GPU unavailable.
    """
    global _current_backend

    # Default thresholds from config
    if thresholds is None:
        thresholds = {
            "feature_count": 5000,
            "rt_grid_len": 50000,
            "correlation_pairs": 1000000,
        }

    # Check if we should try GPU
    should_use_gpu = False
    reason = ""

    if mode == GPUMode.FORCE_CPU:
        reason = "FORCE_CPU mode selected"
        should_use_gpu = False

    elif mode == GPUMode.FORCE_GPU:
        should_use_gpu = True
        reason = "FORCE_GPU mode selected"

    elif mode == GPUMode.AUTO:
        # Check thresholds
        exceeds = []
        if feature_count > thresholds["feature_count"]:
            exceeds.append(f"features={feature_count}>{thresholds['feature_count']}")
        if rt_grid_len > thresholds["rt_grid_len"]:
            exceeds.append(f"rt_grid={rt_grid_len}>{thresholds['rt_grid_len']}")
        if correlation_pairs > thresholds["correlation_pairs"]:
            exceeds.append(f"corr_pairs={correlation_pairs}>{thresholds['correlation_pairs']}")

        if exceeds:
            should_use_gpu = True
            reason = f"AUTO mode: thresholds exceeded ({', '.join(exceeds)})"
        else:
            should_use_gpu = False
            reason = "AUTO mode: data size below GPU thresholds"

    # Try to get GPU backend if needed
    if should_use_gpu:
        gpu_info = probe_gpu()

        if gpu_info.available:
            cp, _ = _try_import_cupy()
            _current_backend = BackendInfo(
                backend_type=BackendType.CUPY,
                module=cp,
                gpu_info=gpu_info,
                reason=reason,
            )
            logger.info(
                f"Using CuPy backend: {gpu_info.device_name} "
                f"({gpu_info.free_memory_gb:.1f}GB free)"
            )
            return _current_backend

        elif mode == GPUMode.FORCE_GPU:
            raise RuntimeError(
                f"GPU required but not available: {gpu_info.error_message}\n"
                "Please check CUDA installation or use AUTO/FORCE_CPU mode."
            )
        else:
            # AUTO mode fallback
            reason = f"AUTO mode: GPU unavailable ({gpu_info.error_message}), using CPU"
            logger.warning(reason)

    # Use NumPy backend
    _current_backend = BackendInfo(
        backend_type=BackendType.NUMPY,
        module=np,
        gpu_info=None,
        reason=reason,
    )
    logger.info(f"Using NumPy backend: {reason}")
    return _current_backend


def get_current_backend() -> BackendInfo:
    """Get the current active backend, initializing with CPU if needed."""
    global _current_backend
    if _current_backend is None:
        return get_array_backend(GPUMode.FORCE_CPU)
    return _current_backend


def ensure_numpy(arr: Any) -> np.ndarray:
    """
    Ensure array is a NumPy array (convert from CuPy if needed).

    Args:
        arr: Input array (NumPy or CuPy)

    Returns:
        NumPy array
    """
    if hasattr(arr, "get"):  # CuPy array
        return arr.get()
    return np.asarray(arr)


def to_backend_array(arr: np.ndarray, backend: BackendInfo) -> Any:
    """
    Convert NumPy array to current backend array type.

    Args:
        arr: NumPy array
        backend: Current backend info

    Returns:
        Array in backend format (NumPy or CuPy)
    """
    if backend.backend_type == BackendType.CUPY:
        return backend.module.asarray(arr)
    return arr


# Type for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def handle_oom(
    max_retries: int = 3,
    chunk_reduction_factor: float = 0.5,
    fallback_to_cpu: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to handle CUDA out-of-memory errors.

    On OOM, reduces chunk size and retries. Falls back to CPU if all retries fail.

    Args:
        max_retries: Maximum number of retry attempts
        chunk_reduction_factor: Factor to reduce chunk size (0.5 = halve)
        fallback_to_cpu: Whether to fall back to CPU on failure

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, chunk_size: int | None = None, **kwargs):
            current_chunk_size = chunk_size
            retries = 0

            while True:
                try:
                    if current_chunk_size is not None:
                        kwargs["chunk_size"] = current_chunk_size
                    return func(*args, **kwargs)

                except Exception as e:
                    # Check if it's a CuPy OOM error
                    error_str = str(type(e).__name__) + str(e)
                    is_oom = (
                        "OutOfMemoryError" in error_str
                        or "CUDA out of memory" in error_str
                        or "cudaErrorMemoryAllocation" in error_str
                    )

                    if not is_oom:
                        raise

                    retries += 1
                    if retries >= max_retries:
                        if fallback_to_cpu:
                            logger.warning(
                                f"GPU OOM after {retries} retries, falling back to CPU"
                            )
                            # Force CPU backend for this operation
                            cpu_backend = get_array_backend(GPUMode.FORCE_CPU)
                            kwargs["_force_backend"] = cpu_backend
                            return func(*args, **kwargs)
                        else:
                            raise RuntimeError(
                                f"GPU out of memory after {retries} retries"
                            ) from e

                    # Reduce chunk size
                    if current_chunk_size is not None:
                        current_chunk_size = max(
                            1, int(current_chunk_size * chunk_reduction_factor)
                        )
                        logger.warning(
                            f"GPU OOM, reducing chunk size to {current_chunk_size} "
                            f"(retry {retries}/{max_retries})"
                        )
                    else:
                        logger.warning(
                            f"GPU OOM, retry {retries}/{max_retries}"
                        )

        return wrapper  # type: ignore

    return decorator


def sync_gpu() -> None:
    """Synchronize GPU operations (no-op for CPU)."""
    backend = get_current_backend()
    if backend.backend_type == BackendType.CUPY:
        backend.module.cuda.Stream.null.synchronize()


def clear_gpu_memory() -> None:
    """Clear GPU memory cache (no-op for CPU)."""
    backend = get_current_backend()
    if backend.backend_type == BackendType.CUPY:
        backend.module.get_default_memory_pool().free_all_blocks()
        backend.module.get_default_pinned_memory_pool().free_all_blocks()
