"""
mzML file I/O for ONFRA PFAS pipeline.

Uses OnDiscMSExperiment for memory-efficient loading of large mzML files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterator, Any

from .errors import (
    MzMLLoadError,
    MzMLNotIndexedWarning,
    MzMLNotCentroidedError,
    OpenMSImportError,
    ThreadSafetyError,
)
from .utils import file_hash, file_size_mb

logger = logging.getLogger(__name__)

# Lazy import of pyOpenMS
_pyopenms = None
_pyopenms_error = None


def _get_pyopenms():
    """Lazy import of pyOpenMS with error handling."""
    global _pyopenms, _pyopenms_error

    if _pyopenms is not None:
        return _pyopenms

    if _pyopenms_error is not None:
        raise _pyopenms_error

    try:
        import pyopenms

        _pyopenms = pyopenms
        return pyopenms
    except ImportError as e:
        _pyopenms_error = OpenMSImportError(e)
        raise _pyopenms_error from e


@dataclass
class ValidationWarning:
    """Warning from mzML validation."""

    code: str
    message: str
    severity: str = "warning"  # "warning", "error", "info"


@dataclass
class MzMLMeta:
    """Metadata extracted from mzML file."""

    path: str
    filename: str
    file_hash: str
    file_size_mb: float

    # Scan information
    total_spectra: int
    ms1_count: int
    ms2_count: int
    msn_count: int  # MS3+

    # RT range
    rt_min: float
    rt_max: float
    rt_range: float

    # m/z range
    mz_min: float
    mz_max: float

    # Data type
    is_indexed: bool
    is_centroided: bool | None  # None if mixed or unknown
    centroid_confidence: float  # 0-1, how confident we are

    # Instrument info
    instrument_model: str | None = None
    instrument_vendor: str | None = None

    # Chromatograms
    chromatogram_count: int = 0

    # Validation
    warnings: list[ValidationWarning] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "file_size_mb": self.file_size_mb,
            "total_spectra": self.total_spectra,
            "ms1_count": self.ms1_count,
            "ms2_count": self.ms2_count,
            "msn_count": self.msn_count,
            "rt_min": self.rt_min,
            "rt_max": self.rt_max,
            "rt_range": self.rt_range,
            "mz_min": self.mz_min,
            "mz_max": self.mz_max,
            "is_indexed": self.is_indexed,
            "is_centroided": self.is_centroided,
            "centroid_confidence": self.centroid_confidence,
            "instrument_model": self.instrument_model,
            "instrument_vendor": self.instrument_vendor,
            "chromatogram_count": self.chromatogram_count,
            "warnings": [{"code": w.code, "message": w.message, "severity": w.severity} for w in self.warnings],
        }


class MzMLLoader:
    """
    Memory-efficient mzML loader using OnDiscMSExperiment.

    WARNING: OnDiscMSExperiment is NOT thread-safe. Each thread must have its own
    instance, or all I/O must be done in a single thread.
    """

    def __init__(self, path: Path | str):
        """
        Initialize mzML loader.

        Args:
            path: Path to mzML file
        """
        self.path = Path(path)
        self._lock = Lock()
        self._ondisc = None
        self._meta: MzMLMeta | None = None
        self._thread_id: int | None = None

        # Validate path
        if not self.path.exists():
            raise MzMLLoadError(str(self.path), "File not found")

        if not self.path.suffix.lower() in (".mzml", ".mzml.gz"):
            raise MzMLLoadError(str(self.path), "Not an mzML file")

        # Open the file
        self._open()

    def _open(self) -> None:
        """Open the mzML file."""
        import threading

        oms = _get_pyopenms()

        try:
            self._ondisc = oms.OnDiscMSExperiment()
            success = self._ondisc.openFile(str(self.path))

            if not success:
                raise MzMLLoadError(str(self.path), "Failed to open file")

            self._thread_id = threading.get_ident()

        except Exception as e:
            if isinstance(e, MzMLLoadError):
                raise
            raise MzMLLoadError(str(self.path), str(e)) from e

    def _check_thread_safety(self) -> None:
        """Check that we're being accessed from the same thread."""
        import threading

        current_thread = threading.get_ident()
        if self._thread_id is not None and current_thread != self._thread_id:
            raise ThreadSafetyError(
                f"OnDiscMSExperiment accessed from different thread "
                f"(created in {self._thread_id}, accessed from {current_thread})"
            )

    def get_meta(self, compute_hash: bool = True) -> MzMLMeta:
        """
        Extract metadata from mzML file.

        Args:
            compute_hash: Whether to compute file hash (slow for large files)

        Returns:
            MzMLMeta with file information
        """
        if self._meta is not None:
            return self._meta

        with self._lock:
            self._check_thread_safety()

            oms = _get_pyopenms()
            warnings = []

            # Basic file info
            file_path = str(self.path.absolute())
            filename = self.path.name
            size_mb = file_size_mb(self.path)

            # Compute hash (optional)
            if compute_hash:
                fhash = file_hash(self.path)
            else:
                fhash = ""

            # Check if indexed - OnDiscMSExperiment requires indexed mzML
            # If we successfully opened the file, it's indexed enough to use
            # We can check by trying to get the number of spectra
            try:
                n_spectra = self._ondisc.getNrSpectra()
                is_indexed = n_spectra > 0
            except Exception:
                is_indexed = False
                n_spectra = 0

            if not is_indexed:
                warnings.append(ValidationWarning(
                    code="NOT_INDEXED",
                    message="mzML file is not indexed. Performance may be reduced.",
                    severity="warning",
                ))

            # Get experiment metadata
            exp_meta = self._ondisc.getExperimentalSettings()

            # Scan through spectra for counts and ranges
            # n_spectra already set above
            ms1_count = 0
            ms2_count = 0
            msn_count = 0

            rt_min = float("inf")
            rt_max = float("-inf")
            mz_min = float("inf")
            mz_max = float("-inf")

            centroid_votes = 0
            profile_votes = 0

            # Sample spectra for metadata (don't load all for large files)
            sample_indices = self._get_sample_indices(n_spectra, max_samples=1000)

            for i in sample_indices:
                spec = self._ondisc.getSpectrum(i)
                ms_level = spec.getMSLevel()

                if ms_level == 1:
                    ms1_count += 1
                elif ms_level == 2:
                    ms2_count += 1
                else:
                    msn_count += 1

                # RT
                rt = spec.getRT()
                rt_min = min(rt_min, rt)
                rt_max = max(rt_max, rt)

                # m/z range (from peaks)
                mzs = spec.get_peaks()[0]
                if len(mzs) > 0:
                    mz_min = min(mz_min, float(mzs.min()))
                    mz_max = max(mz_max, float(mzs.max()))

                # Check centroid vs profile
                if self._is_centroided_spectrum(spec):
                    centroid_votes += 1
                else:
                    profile_votes += 1

            # Extrapolate counts if sampled
            if len(sample_indices) < n_spectra:
                ratio = n_spectra / len(sample_indices)
                ms1_count = int(ms1_count * ratio)
                ms2_count = int(ms2_count * ratio)
                msn_count = int(msn_count * ratio)

            # Handle edge cases
            if rt_min == float("inf"):
                rt_min = 0.0
            if rt_max == float("-inf"):
                rt_max = 0.0
            if mz_min == float("inf"):
                mz_min = 0.0
            if mz_max == float("-inf"):
                mz_max = 0.0

            # Determine centroid status
            total_votes = centroid_votes + profile_votes
            if total_votes > 0:
                centroid_confidence = centroid_votes / total_votes
                is_centroided = centroid_confidence > 0.5
            else:
                centroid_confidence = 0.0
                is_centroided = None

            if is_centroided is False or (is_centroided is None):
                warnings.append(ValidationWarning(
                    code="NOT_CENTROIDED",
                    message="Data appears to be profile mode. FeatureFinderMetabo requires centroided data.",
                    severity="warning",
                ))

            # Chromatogram count
            n_chroms = self._ondisc.getNrChromatograms()

            # Instrument info
            instrument_model = None
            instrument_vendor = None
            # Try to get from experimental settings
            try:
                instruments = exp_meta.getInstrument()
                if instruments:
                    instrument_model = instruments.getModel()
                    instrument_vendor = instruments.getVendor()
            except Exception:
                pass

            self._meta = MzMLMeta(
                path=file_path,
                filename=filename,
                file_hash=fhash,
                file_size_mb=size_mb,
                total_spectra=n_spectra,
                ms1_count=ms1_count,
                ms2_count=ms2_count,
                msn_count=msn_count,
                rt_min=rt_min,
                rt_max=rt_max,
                rt_range=rt_max - rt_min,
                mz_min=mz_min,
                mz_max=mz_max,
                is_indexed=is_indexed,
                is_centroided=is_centroided,
                centroid_confidence=centroid_confidence,
                instrument_model=instrument_model,
                instrument_vendor=instrument_vendor,
                chromatogram_count=n_chroms,
                warnings=warnings,
            )

            return self._meta

    def _get_sample_indices(self, total: int, max_samples: int = 1000) -> list[int]:
        """Get evenly distributed sample indices."""
        if total <= max_samples:
            return list(range(total))

        step = total / max_samples
        return [int(i * step) for i in range(max_samples)]

    def _is_centroided_spectrum(self, spec) -> bool:
        """
        Heuristic to determine if spectrum is centroided.

        Centroided spectra tend to have:
        - Fewer peaks
        - More spacing between peaks
        - Sharper intensity distribution
        """
        mzs, intensities = spec.get_peaks()

        if len(mzs) < 3:
            return True  # Too few peaks to tell, assume centroided

        # Calculate average spacing
        if len(mzs) > 1:
            spacings = mzs[1:] - mzs[:-1]
            avg_spacing = float(spacings.mean())

            # Profile data typically has very small spacing (< 0.01 Da)
            # Centroided data has larger spacing
            if avg_spacing < 0.01:
                return False

        # Check for clusters of peaks (profile mode often has connected peaks)
        # Profile mode: many peaks with similar intensities in a row
        if len(mzs) > 10:
            # Check intensity variation
            intensity_std = float(intensities.std())
            intensity_mean = float(intensities.mean())
            cv = intensity_std / max(intensity_mean, 1e-10)

            # High CV suggests centroided (isolated peaks)
            # Low CV suggests profile (connected peaks)
            if cv < 0.5:
                return False

        return True

    def iter_spectra(
        self,
        ms_level: int | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[Any]:
        """
        Iterate over spectra.

        Args:
            ms_level: Filter by MS level (None = all)
            rt_range: Filter by RT range (min, max) in seconds

        Yields:
            MSSpectrum objects
        """
        with self._lock:
            self._check_thread_safety()

            n_spectra = self._ondisc.getNrSpectra()

            for i in range(n_spectra):
                spec = self._ondisc.getSpectrum(i)

                # Filter by MS level
                if ms_level is not None and spec.getMSLevel() != ms_level:
                    continue

                # Filter by RT
                if rt_range is not None:
                    rt = spec.getRT()
                    if rt < rt_range[0] or rt > rt_range[1]:
                        continue

                yield spec

    def get_spectrum(self, index: int) -> Any:
        """
        Get a single spectrum by index.

        Args:
            index: Spectrum index

        Returns:
            MSSpectrum object
        """
        with self._lock:
            self._check_thread_safety()
            return self._ondisc.getSpectrum(index)

    def get_ms2_spectra(self) -> list[Any]:
        """
        Get all MS2 spectra.

        Returns:
            List of MS2 MSSpectrum objects
        """
        return list(self.iter_spectra(ms_level=2))

    def get_experiment(self) -> Any:
        """
        Load full experiment into memory.

        WARNING: This loads the entire file into memory. Use only for small files.

        Returns:
            MSExperiment object
        """
        with self._lock:
            self._check_thread_safety()

            oms = _get_pyopenms()
            exp = oms.MSExperiment()

            n_spectra = self._ondisc.getNrSpectra()
            for i in range(n_spectra):
                exp.addSpectrum(self._ondisc.getSpectrum(i))

            n_chroms = self._ondisc.getNrChromatograms()
            for i in range(n_chroms):
                exp.addChromatogram(self._ondisc.getChromatogram(i))

            return exp

    def validate(self) -> list[ValidationWarning]:
        """
        Validate mzML file for processing.

        Returns:
            List of validation warnings
        """
        meta = self.get_meta()
        return meta.warnings

    def close(self) -> None:
        """Close the file and release resources."""
        self._ondisc = None
        self._meta = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_mzml_meta(path: Path | str, compute_hash: bool = True) -> MzMLMeta:
    """
    Load mzML metadata without keeping file open.

    Args:
        path: Path to mzML file
        compute_hash: Whether to compute file hash

    Returns:
        MzMLMeta with file information
    """
    with MzMLLoader(path) as loader:
        return loader.get_meta(compute_hash=compute_hash)


def validate_mzml(path: Path | str, require_centroided: bool = True) -> list[ValidationWarning]:
    """
    Validate mzML file.

    Args:
        path: Path to mzML file
        require_centroided: Whether to require centroided data

    Returns:
        List of validation warnings

    Raises:
        MzMLNotCentroidedError: If require_centroided and data is profile mode
    """
    meta = load_mzml_meta(path)

    if require_centroided and meta.is_centroided is False:
        raise MzMLNotCentroidedError(str(path), "profile")

    return meta.warnings
