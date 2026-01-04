"""
Visualization module for ONFRA PFAS pipeline.

Generates EIC, spectrum data, and correlation analysis for UI rendering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .backend import (
    get_array_backend,
    get_current_backend,
    ensure_numpy,
    to_backend_array,
    handle_oom,
    BackendType,
)
from .config import VisualizationConfig, GPUMode
from .checkpoints import RunContext, StepMeta
from .io_mzml import MzMLLoader
from .utils import timed_block

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EICData:
    """Extracted Ion Chromatogram data."""

    feature_id: int
    mz: float
    mz_tolerance: float
    rt_values: np.ndarray
    intensity_values: np.ndarray
    rt_min: float
    rt_max: float
    apex_rt: float
    apex_intensity: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_id": self.feature_id,
            "mz": self.mz,
            "mz_tolerance": self.mz_tolerance,
            "rt": ensure_numpy(self.rt_values).tolist(),
            "intensity": ensure_numpy(self.intensity_values).tolist(),
            "rt_min": self.rt_min,
            "rt_max": self.rt_max,
            "apex_rt": self.apex_rt,
            "apex_intensity": self.apex_intensity,
        }


@dataclass
class SpectrumData:
    """Mass spectrum data."""

    feature_id: int
    spectrum_index: str
    rt: float
    mz_values: np.ndarray
    intensity_values: np.ndarray
    precursor_mz: float | None = None
    ms_level: int = 1

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_id": self.feature_id,
            "spectrum_index": self.spectrum_index,
            "rt": self.rt,
            "mz": ensure_numpy(self.mz_values).tolist(),
            "intensity": ensure_numpy(self.intensity_values).tolist(),
            "precursor_mz": self.precursor_mz,
            "ms_level": self.ms_level,
        }


@dataclass
class CorrelationEdge:
    """Correlation between two features."""

    source_feature_id: int
    target_feature_id: int
    correlation: float  # Pearson r


@dataclass
class VizPayload:
    """Complete visualization payload for UI."""

    eics: list[EICData]
    spectra: list[SpectrumData]
    correlations: list[CorrelationEdge]
    homologous_series: list[dict]
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "eics": [e.to_dict() for e in self.eics],
            "spectra": [s.to_dict() for s in self.spectra],
            "correlations": [
                {
                    "source": c.source_feature_id,
                    "target": c.target_feature_id,
                    "r": c.correlation,
                }
                for c in self.correlations
            ],
            "homologous_series": self.homologous_series,
            "metadata": self.metadata,
        }

    def to_json(self, path: Path | str | None = None) -> str:
        """Convert to JSON string, optionally saving to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str


# =============================================================================
# EIC Extraction
# =============================================================================


def extract_eic(
    loader: MzMLLoader,
    mz: float,
    mz_tolerance: float,
    rt_range: tuple[float, float] | None = None,
    feature_id: int = 0,
) -> EICData:
    """
    Extract ion chromatogram for a given m/z.

    Args:
        loader: mzML loader
        mz: Target m/z value
        mz_tolerance: m/z tolerance in Da
        rt_range: Optional RT range (min, max) in seconds
        feature_id: Feature ID for reference

    Returns:
        EICData with RT and intensity values
    """
    rt_list = []
    int_list = []

    mz_min = mz - mz_tolerance
    mz_max = mz + mz_tolerance

    for spec in loader.iter_spectra(ms_level=1, rt_range=rt_range):
        rt = spec.getRT()
        mzs, ints = spec.get_peaks()

        if len(mzs) == 0:
            rt_list.append(rt)
            int_list.append(0.0)
            continue

        # Sum intensities within m/z range
        mask = (mzs >= mz_min) & (mzs <= mz_max)
        total_int = float(ints[mask].sum()) if mask.any() else 0.0

        rt_list.append(rt)
        int_list.append(total_int)

    rt_values = np.array(rt_list)
    intensity_values = np.array(int_list)

    # Find apex
    if len(intensity_values) > 0 and intensity_values.max() > 0:
        apex_idx = intensity_values.argmax()
        apex_rt = rt_values[apex_idx]
        apex_intensity = intensity_values[apex_idx]
    else:
        apex_rt = 0.0
        apex_intensity = 0.0

    return EICData(
        feature_id=feature_id,
        mz=mz,
        mz_tolerance=mz_tolerance,
        rt_values=rt_values,
        intensity_values=intensity_values,
        rt_min=float(rt_values.min()) if len(rt_values) > 0 else 0.0,
        rt_max=float(rt_values.max()) if len(rt_values) > 0 else 0.0,
        apex_rt=apex_rt,
        apex_intensity=apex_intensity,
    )


def extract_eics_batch(
    mzml_path: Path,
    features_df: pd.DataFrame,
    mz_tolerance: float,
    max_features: int = 1000,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[EICData]:
    """
    Extract EICs for multiple features.

    Args:
        mzml_path: Path to mzML file
        features_df: DataFrame with features (needs 'mz', 'feature_id' columns)
        mz_tolerance: m/z tolerance in Da
        max_features: Maximum number of features to process
        progress_callback: Progress callback

    Returns:
        List of EICData
    """
    # Limit features
    if len(features_df) > max_features:
        # Take top-scored features if score available
        if "pfas_score" in features_df.columns:
            features_df = features_df.nlargest(max_features, "pfas_score")
        else:
            features_df = features_df.head(max_features)

    eics = []

    with MzMLLoader(mzml_path) as loader:
        total = len(features_df)

        for i, (_, row) in enumerate(features_df.iterrows()):
            if progress_callback and i % 10 == 0:
                progress_callback(i / total, f"Extracting EIC {i+1}/{total}")

            eic = extract_eic(
                loader,
                mz=row["mz"],
                mz_tolerance=mz_tolerance,
                feature_id=int(row["feature_id"]),
            )
            eics.append(eic)

    return eics


# =============================================================================
# Correlation Analysis
# =============================================================================


@handle_oom(max_retries=3, fallback_to_cpu=True)
def calculate_eic_correlations(
    eics: list[EICData],
    threshold: float = 0.8,
    chunk_size: int = 1000,
    _force_backend=None,
) -> list[CorrelationEdge]:
    """
    Calculate pairwise correlations between EICs.

    Uses GPU acceleration when available and data is large.

    Args:
        eics: List of EIC data
        threshold: Minimum correlation to include
        chunk_size: Chunk size for GPU processing
        _force_backend: Force specific backend (for OOM fallback)

    Returns:
        List of CorrelationEdge above threshold
    """
    if len(eics) < 2:
        return []

    # Get backend
    if _force_backend:
        backend = _force_backend
    else:
        n_pairs = len(eics) * (len(eics) - 1) // 2
        backend = get_array_backend(
            GPUMode.AUTO,
            correlation_pairs=n_pairs,
        )

    xp = backend.module

    # Build intensity matrix
    # First, align all EICs to common RT grid
    all_rts = set()
    for eic in eics:
        all_rts.update(eic.rt_values.tolist())

    rt_grid = np.array(sorted(all_rts))
    n_features = len(eics)
    n_points = len(rt_grid)

    # Interpolate EICs to common grid
    intensity_matrix = np.zeros((n_features, n_points))

    for i, eic in enumerate(eics):
        if len(eic.rt_values) > 1:
            intensity_matrix[i] = np.interp(rt_grid, eic.rt_values, eic.intensity_values)
        elif len(eic.rt_values) == 1:
            # Single point - find closest
            closest = np.argmin(np.abs(rt_grid - eic.rt_values[0]))
            intensity_matrix[i, closest] = eic.intensity_values[0]

    # Normalize rows (mean-center and scale)
    means = intensity_matrix.mean(axis=1, keepdims=True)
    stds = intensity_matrix.std(axis=1, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero

    normalized = (intensity_matrix - means) / stds

    # Move to GPU if using CuPy
    if backend.backend_type == BackendType.CUPY:
        normalized = xp.asarray(normalized)

    # Calculate correlation matrix
    with timed_block("Correlation calculation") as timer:
        # Correlation = dot product of normalized vectors / (n-1)
        corr_matrix = xp.dot(normalized, normalized.T) / (n_points - 1)

    # Move back to CPU
    corr_matrix = ensure_numpy(corr_matrix)

    # Extract edges above threshold
    edges = []
    feature_ids = [eic.feature_id for eic in eics]

    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = corr_matrix[i, j]
            if corr >= threshold:
                edges.append(CorrelationEdge(
                    source_feature_id=feature_ids[i],
                    target_feature_id=feature_ids[j],
                    correlation=float(corr),
                ))

    logger.info(f"Found {len(edges)} correlation edges above {threshold} threshold")
    return edges


def find_coeluting_ions(
    eics: list[EICData],
    feature_id: int,
    threshold: float = 0.8,
) -> list[tuple[int, float]]:
    """
    Find ions that coelute with a given feature.

    Args:
        eics: List of all EICs
        feature_id: Target feature ID
        threshold: Correlation threshold

    Returns:
        List of (feature_id, correlation) tuples
    """
    # Find target EIC
    target_eic = None
    for eic in eics:
        if eic.feature_id == feature_id:
            target_eic = eic
            break

    if target_eic is None:
        return []

    coeluters = []

    for eic in eics:
        if eic.feature_id == feature_id:
            continue

        # Calculate correlation
        if len(target_eic.rt_values) < 3 or len(eic.rt_values) < 3:
            continue

        # Interpolate to common grid
        common_rt = np.union1d(target_eic.rt_values, eic.rt_values)
        target_int = np.interp(common_rt, target_eic.rt_values, target_eic.intensity_values)
        other_int = np.interp(common_rt, eic.rt_values, eic.intensity_values)

        # Pearson correlation
        if target_int.std() > 0 and other_int.std() > 0:
            corr = np.corrcoef(target_int, other_int)[0, 1]
            if corr >= threshold:
                coeluters.append((eic.feature_id, float(corr)))

    # Sort by correlation
    coeluters.sort(key=lambda x: x[1], reverse=True)
    return coeluters


# =============================================================================
# Spectrum Data
# =============================================================================


def extract_spectrum_data(
    loader: MzMLLoader,
    feature_id: int,
    rt: float,
    ms_level: int = 2,
    rt_tolerance: float = 5.0,
) -> list[SpectrumData]:
    """
    Extract spectrum data near a given RT.

    Args:
        loader: mzML loader
        feature_id: Feature ID
        rt: Target RT
        ms_level: MS level (1 or 2)
        rt_tolerance: RT tolerance in seconds

    Returns:
        List of SpectrumData
    """
    spectra = []
    rt_range = (rt - rt_tolerance, rt + rt_tolerance)

    for spec in loader.iter_spectra(ms_level=ms_level, rt_range=rt_range):
        mzs, ints = spec.get_peaks()

        precursor_mz = None
        if ms_level >= 2:
            precursors = spec.getPrecursors()
            if precursors:
                precursor_mz = precursors[0].getMZ()

        spectra.append(SpectrumData(
            feature_id=feature_id,
            spectrum_index=spec.getNativeID(),
            rt=spec.getRT(),
            mz_values=mzs,
            intensity_values=ints,
            precursor_mz=precursor_mz,
            ms_level=ms_level,
        ))

    return spectra


# =============================================================================
# Main Visualization Pipeline
# =============================================================================


def generate_viz_payload(
    mzml_path: Path,
    features_df: pd.DataFrame,
    config: VisualizationConfig,
    homologous_series: list[dict] | None = None,
    max_features: int = 500,
    progress_callback: Callable[[float, str], None] | None = None,
) -> VizPayload:
    """
    Generate complete visualization payload.

    Args:
        mzml_path: Path to mzML file
        features_df: Prioritized features DataFrame
        config: Visualization config
        homologous_series: Optional series data
        max_features: Maximum features to include
        progress_callback: Progress callback

    Returns:
        VizPayload ready for UI rendering
    """
    if len(features_df) == 0:
        return VizPayload(
            eics=[],
            spectra=[],
            correlations=[],
            homologous_series=[],
            metadata={"feature_count": 0},
        )

    def report(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct:.0%}] {msg}")

    # Limit features
    if len(features_df) > max_features:
        if "pfas_score" in features_df.columns:
            features_df = features_df.nlargest(max_features, "pfas_score")
        else:
            features_df = features_df.head(max_features)

    report(0.0, "Extracting EICs...")

    # Extract EICs
    eics = extract_eics_batch(
        mzml_path,
        features_df,
        config.eic_mz_tolerance,
        max_features=max_features,
        progress_callback=lambda p, m: report(p * 0.5, m),
    )

    report(0.5, "Calculating correlations...")

    # Calculate correlations
    correlations = []
    if config.correlation_enabled and len(eics) >= 2:
        correlations = calculate_eic_correlations(
            eics,
            threshold=config.correlation_threshold,
            chunk_size=config.correlation_chunk_size,
        )

    report(0.8, "Extracting spectra...")

    # Extract MS2 spectra for top features
    spectra = []
    with MzMLLoader(mzml_path) as loader:
        top_features = features_df.head(50)  # Limit spectra extraction

        for _, row in top_features.iterrows():
            feature_spectra = extract_spectrum_data(
                loader,
                feature_id=int(row["feature_id"]),
                rt=row["rt"],
                ms_level=2,
            )
            spectra.extend(feature_spectra[:3])  # Max 3 spectra per feature

    report(1.0, "Visualization payload complete")

    # Build metadata
    metadata = {
        "feature_count": len(features_df),
        "eic_count": len(eics),
        "correlation_edge_count": len(correlations),
        "spectrum_count": len(spectra),
        "mzml_path": str(mzml_path),
    }

    return VizPayload(
        eics=eics,
        spectra=spectra,
        correlations=correlations,
        homologous_series=homologous_series or [],
        metadata=metadata,
    )


def run_visualization_step(
    run_context: RunContext,
    mzml_path: Path,
    features_df: pd.DataFrame,
    config: VisualizationConfig,
    homologous_series: list[dict] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> VizPayload:
    """
    Run visualization step with checkpointing.

    Args:
        run_context: Run context
        mzml_path: Path to mzML
        features_df: Features DataFrame
        config: Visualization config
        homologous_series: Series data
        progress_callback: Progress callback

    Returns:
        VizPayload
    """
    step_meta = StepMeta(
        step_name="visualization",
        step_number=3,
        status="running",
        started_at=datetime.now().isoformat(),
        input_feature_count=len(features_df),
    )

    try:
        payload = generate_viz_payload(
            mzml_path,
            features_df,
            config,
            homologous_series,
            progress_callback=progress_callback,
        )

        step_meta.status = "completed"
        step_meta.completed_at = datetime.now().isoformat()
        step_meta.extra = payload.metadata

        # Save as JSON checkpoint
        run_context.save_checkpoint("visualization", payload.to_dict(), step_meta)

        return payload

    except Exception as e:
        step_meta.status = "failed"
        step_meta.completed_at = datetime.now().isoformat()
        step_meta.extra = {"error": str(e)}
        run_context.save_checkpoint("visualization", {}, step_meta)
        raise
