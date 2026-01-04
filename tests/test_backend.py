"""
Tests for backend module (GPU/CPU selection and handling).
"""

import pytest
import numpy as np

from onfra_pfas.core.backend import (
    probe_gpu,
    get_array_backend,
    get_current_backend,
    ensure_numpy,
    to_backend_array,
    handle_oom,
    GPUInfo,
    BackendInfo,
    BackendType,
)
from onfra_pfas.core.config import GPUMode


class TestGPUProbe:
    """Tests for GPU probing."""

    def test_probe_returns_gpu_info(self):
        """Test that probe_gpu returns GPUInfo."""
        info = probe_gpu()

        assert isinstance(info, GPUInfo)
        assert isinstance(info.available, bool)

        if info.available:
            assert info.device_name is not None
            assert info.total_memory_gb is not None
            assert info.free_memory_gb is not None
        else:
            # Should have error message if not available
            assert info.error_message is not None


class TestBackendSelection:
    """Tests for backend selection."""

    def test_force_cpu_backend(self):
        """Test forcing CPU backend."""
        backend = get_array_backend(GPUMode.FORCE_CPU)

        assert isinstance(backend, BackendInfo)
        assert backend.backend_type == BackendType.NUMPY
        assert backend.module is np

    def test_auto_mode_small_data(self):
        """Test AUTO mode with small data uses CPU."""
        backend = get_array_backend(
            GPUMode.AUTO,
            feature_count=100,
            rt_grid_len=1000,
            correlation_pairs=10000,
        )

        # Small data should use CPU
        assert backend.backend_type == BackendType.NUMPY

    def test_auto_mode_large_data(self):
        """Test AUTO mode with large data suggests GPU if available."""
        backend = get_array_backend(
            GPUMode.AUTO,
            feature_count=10000,  # Above threshold
            rt_grid_len=100000,  # Above threshold
            correlation_pairs=10000000,  # Above threshold
        )

        # If GPU available, should use it; otherwise CPU
        gpu_info = probe_gpu()
        if gpu_info.available:
            assert backend.backend_type == BackendType.CUPY
        else:
            assert backend.backend_type == BackendType.NUMPY

    def test_force_gpu_without_gpu(self):
        """Test that FORCE_GPU fails gracefully without GPU."""
        gpu_info = probe_gpu()

        if not gpu_info.available:
            with pytest.raises(RuntimeError) as exc_info:
                get_array_backend(GPUMode.FORCE_GPU)

            assert "not available" in str(exc_info.value).lower()

    def test_custom_thresholds(self):
        """Test using custom thresholds."""
        backend = get_array_backend(
            GPUMode.AUTO,
            feature_count=100,
            thresholds={
                "feature_count": 50,  # Lower threshold
                "rt_grid_len": 50000,
                "correlation_pairs": 1000000,
            },
        )

        # 100 > 50, so should try GPU
        gpu_info = probe_gpu()
        if gpu_info.available:
            assert backend.backend_type == BackendType.CUPY


class TestArrayConversion:
    """Tests for array conversion utilities."""

    def test_ensure_numpy_with_numpy(self):
        """Test ensure_numpy with NumPy array."""
        arr = np.array([1, 2, 3])
        result = ensure_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_ensure_numpy_with_cupy(self):
        """Test ensure_numpy with CuPy array (if available)."""
        gpu_info = probe_gpu()

        if gpu_info.available:
            import cupy as cp

            arr = cp.array([1, 2, 3])
            result = ensure_numpy(arr)

            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, [1, 2, 3])

    def test_to_backend_array_numpy(self):
        """Test converting to backend array (NumPy)."""
        backend = get_array_backend(GPUMode.FORCE_CPU)
        arr = np.array([1, 2, 3])

        result = to_backend_array(arr, backend)

        assert isinstance(result, np.ndarray)

    def test_to_backend_array_cupy(self):
        """Test converting to backend array (CuPy if available)."""
        gpu_info = probe_gpu()

        if gpu_info.available:
            backend = get_array_backend(GPUMode.FORCE_GPU)
            arr = np.array([1, 2, 3])

            result = to_backend_array(arr, backend)

            import cupy as cp
            assert isinstance(result, cp.ndarray)


class TestOOMHandling:
    """Tests for OOM handling decorator."""

    def test_oom_decorator_normal_execution(self):
        """Test OOM decorator doesn't interfere with normal execution."""

        @handle_oom(max_retries=3, fallback_to_cpu=False)
        def normal_function(x, chunk_size=100, **kwargs):
            return x * chunk_size

        result = normal_function(5, chunk_size=10)
        assert result == 50

    def test_oom_decorator_reduces_chunk_size(self):
        """Test OOM decorator reduces chunk size on simulated OOM."""
        attempts = []

        # Create a custom exception that matches the OOM check
        class MockCudaOOM(Exception):
            pass

        @handle_oom(max_retries=5, chunk_reduction_factor=0.5, fallback_to_cpu=False)
        def oom_function(x, chunk_size=1000, **kwargs):
            attempts.append(chunk_size)
            if chunk_size > 200:
                # Simulate OOM - need to match the decorator's check
                raise Exception("CUDA out of memory simulation")
            return chunk_size

        result = oom_function(1, chunk_size=1000)

        # Should have retried with reduced chunk sizes
        assert len(attempts) >= 2
        assert attempts[-1] <= 250  # 1000 * 0.5 * 0.5 = 250

    def test_oom_decorator_max_retries(self):
        """Test OOM decorator respects max retries."""

        @handle_oom(max_retries=2, fallback_to_cpu=False)
        def always_oom(chunk_size=100, **kwargs):
            raise Exception("CUDA out of memory")

        with pytest.raises(RuntimeError) as exc_info:
            always_oom(chunk_size=100)

        assert "after 2 retries" in str(exc_info.value)


class TestCurrentBackend:
    """Tests for current backend tracking."""

    def test_get_current_backend(self):
        """Test getting current backend."""
        # Force a specific backend first
        get_array_backend(GPUMode.FORCE_CPU)

        backend = get_current_backend()

        assert isinstance(backend, BackendInfo)
        assert backend.backend_type == BackendType.NUMPY

    def test_backend_has_reason(self):
        """Test that backend includes selection reason."""
        backend = get_array_backend(GPUMode.FORCE_CPU)

        assert backend.reason is not None
        assert len(backend.reason) > 0
