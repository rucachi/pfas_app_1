"""
Custom error types for ONFRA PFAS application.

Provides user-friendly error messages for common failure scenarios.
"""

from __future__ import annotations


class OnfraError(Exception):
    """Base exception for ONFRA PFAS application."""

    user_message: str = "An error occurred in the PFAS analysis pipeline."
    log_message: str = ""

    def __init__(
        self,
        message: str | None = None,
        user_message: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize error.

        Args:
            message: Technical error message
            user_message: User-friendly message to display
            details: Additional error details for logging
        """
        self.message = message or self.__class__.__doc__ or "Unknown error"
        if user_message:
            self.user_message = user_message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.user_message} ({self.message})"


# ============================================================================
# File I/O Errors
# ============================================================================


class MzMLLoadError(OnfraError):
    """Failed to load mzML file."""

    user_message = "입력 파일을 열 수 없음(경로/권한/손상). 로그 확인."

    def __init__(
        self,
        path: str,
        reason: str | None = None,
        details: dict | None = None,
    ):
        self.path = path
        super().__init__(
            message=f"Cannot load mzML: {path}" + (f" - {reason}" if reason else ""),
            details={"path": path, "reason": reason, **(details or {})},
        )


class MzMLNotIndexedWarning(OnfraError):
    """mzML file is not indexed."""

    user_message = "mzML 파일이 인덱싱되지 않음. 성능이 저하될 수 있습니다."

    def __init__(self, path: str):
        self.path = path
        super().__init__(
            message=f"mzML not indexed: {path}",
            details={"path": path},
        )


class MzMLNotCentroidedError(OnfraError):
    """mzML data is profile mode, not centroided."""

    user_message = (
        "데이터가 centroid 모드가 아닙니다. FeatureFinderMetabo는 centroided LC-MS를 "
        "필요로 합니다. 강제 진행 시 결과 품질이 저하될 수 있습니다."
    )

    def __init__(self, path: str, spectrum_type: str | None = None):
        self.path = path
        self.spectrum_type = spectrum_type
        super().__init__(
            message=f"Profile mode detected: {path}",
            details={"path": path, "spectrum_type": spectrum_type},
        )


class FilePermissionError(OnfraError):
    """File permission denied."""

    user_message = "파일 접근 권한이 없습니다. 파일 권한을 확인하세요."

    def __init__(self, path: str, operation: str = "read"):
        self.path = path
        self.operation = operation
        super().__init__(
            message=f"Permission denied for {operation}: {path}",
            details={"path": path, "operation": operation},
        )


class FileNotFoundError(OnfraError):
    """File not found."""

    user_message = "파일을 찾을 수 없습니다. 경로를 확인하세요."

    def __init__(self, path: str):
        self.path = path
        super().__init__(
            message=f"File not found: {path}",
            details={"path": path},
        )


# ============================================================================
# OpenMS Errors
# ============================================================================


class OpenMSImportError(OnfraError):
    """Failed to import OpenMS/pyOpenMS."""

    user_message = (
        "OpenMS 구성요소를 로드하지 못함(설치/배포 누락). "
        "pyOpenMS가 올바르게 설치되었는지 확인하세요."
    )

    def __init__(self, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(
            message=f"Cannot import pyOpenMS: {original_error}",
            details={"original_error": str(original_error)},
        )


class FeatureFinderError(OnfraError):
    """Error during feature finding."""

    user_message = "Feature 탐색 중 오류가 발생했습니다. 로그를 확인하세요."

    def __init__(self, stage: str, reason: str | None = None):
        self.stage = stage
        super().__init__(
            message=f"FeatureFinder error at {stage}: {reason}",
            details={"stage": stage, "reason": reason},
        )


# ============================================================================
# GPU Errors
# ============================================================================


class GPUNotAvailableError(OnfraError):
    """GPU is not available."""

    user_message = (
        "GPU를 사용할 수 없습니다. CUDA 드라이버 및 CuPy 설치를 확인하세요. "
        "CPU 모드로 계속하려면 설정에서 GPU 모드를 'force_cpu'로 변경하세요."
    )

    def __init__(self, reason: str | None = None):
        super().__init__(
            message=f"GPU not available: {reason}",
            details={"reason": reason},
        )


class GPUOutOfMemoryError(OnfraError):
    """GPU ran out of memory."""

    user_message = (
        "GPU 메모리 부족. 청크 크기를 줄이거나 CPU 모드를 사용하세요."
    )

    def __init__(self, operation: str | None = None, required_mb: float | None = None):
        self.operation = operation
        self.required_mb = required_mb
        super().__init__(
            message=f"GPU OOM during {operation}" + (f", needed ~{required_mb:.0f}MB" if required_mb else ""),
            details={"operation": operation, "required_mb": required_mb},
        )


# ============================================================================
# Pipeline Errors
# ============================================================================


class CheckpointError(OnfraError):
    """Error reading/writing checkpoint."""

    user_message = "체크포인트 저장/불러오기 중 오류가 발생했습니다."

    def __init__(self, operation: str, path: str, reason: str | None = None):
        self.operation = operation
        self.path = path
        super().__init__(
            message=f"Checkpoint {operation} failed: {path} - {reason}",
            details={"operation": operation, "path": path, "reason": reason},
        )


class ConfigurationError(OnfraError):
    """Error in configuration."""

    user_message = "설정 오류가 있습니다. 설정 파일을 확인하세요."

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(
            message=f"Config error in {field}: {message}",
            details={"field": field, "message": message},
        )


class PipelineCancelledError(OnfraError):
    """Pipeline was cancelled by user."""

    user_message = "분석이 사용자에 의해 취소되었습니다."

    def __init__(self, stage: str | None = None):
        self.stage = stage
        super().__init__(
            message=f"Pipeline cancelled at {stage}",
            details={"stage": stage},
        )


# ============================================================================
# Data Validation Errors
# ============================================================================


class InvalidSuspectListError(OnfraError):
    """Suspect list file is invalid."""

    user_message = (
        "의심 물질 목록 파일 형식이 잘못되었습니다. "
        "필수 컬럼(name, exact_mass 또는 formula)이 있는지 확인하세요."
    )

    def __init__(self, path: str, missing_columns: list[str] | None = None):
        self.path = path
        self.missing_columns = missing_columns
        super().__init__(
            message=f"Invalid suspect list: {path}" + (f", missing: {missing_columns}" if missing_columns else ""),
            details={"path": path, "missing_columns": missing_columns},
        )


class InvalidDFRulesError(OnfraError):
    """Diagnostic fragment rules file is invalid."""

    user_message = "진단 조각 규칙 파일 형식이 잘못되었습니다."

    def __init__(self, path: str, reason: str | None = None):
        self.path = path
        super().__init__(
            message=f"Invalid DF rules: {path} - {reason}",
            details={"path": path, "reason": reason},
        )


class NoFeaturesFoundError(OnfraError):
    """No features were found in the data."""

    user_message = (
        "데이터에서 feature를 찾을 수 없습니다. "
        "노이즈 임계값이나 감도 설정을 조정해 보세요."
    )

    def __init__(self, mzml_path: str | None = None):
        self.mzml_path = mzml_path
        super().__init__(
            message=f"No features found in {mzml_path}",
            details={"mzml_path": mzml_path},
        )


# ============================================================================
# Concurrency Errors
# ============================================================================


class ThreadSafetyError(OnfraError):
    """Thread safety violation detected."""

    user_message = (
        "멀티스레드 안전성 위반이 감지되었습니다. "
        "OnDiscMSExperiment는 스레드 간 공유할 수 없습니다."
    )

    def __init__(self, resource: str | None = None):
        self.resource = resource
        super().__init__(
            message=f"Thread safety violation for {resource}",
            details={"resource": resource},
        )
