"""
Feature finding module for ONFRA PFAS pipeline.

Uses pyOpenMS FeatureFinderMetabo for centroided LC-MS data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import FeatureFinderConfig, BlankCorrectionConfig, BlankCorrectionPolicy
from .errors import FeatureFinderError, NoFeaturesFoundError, OpenMSImportError
from .io_mzml import MzMLLoader, load_mzml_meta
from .checkpoints import RunContext, StepMeta
from .utils import timed_block, ppm_difference

logger = logging.getLogger(__name__)

# Lazy import pyOpenMS
_pyopenms = None


def _get_pyopenms():
    """Lazy import of pyOpenMS."""
    global _pyopenms
    if _pyopenms is None:
        try:
            import pyopenms

            _pyopenms = pyopenms
        except ImportError as e:
            raise OpenMSImportError(e) from e
    return _pyopenms


@dataclass
class FeatureResult:
    """Result from feature finding."""

    features_df: pd.DataFrame
    feature_map: Any  # pyOpenMS FeatureMap
    ms2_mappings: pd.DataFrame  # MS2 spectrum to feature mappings
    stats: dict


def run_feature_finder_metabo(
    mzml_path: Path,
    config: FeatureFinderConfig,
    progress_callback: Callable[[float, str], None] | None = None,
) -> FeatureResult:
    """
    Run FeatureFinderMetabo on mzML file.

    Args:
        mzml_path: Path to input mzML file
        config: Feature finder configuration
        progress_callback: Optional callback(progress, message)

    Returns:
        FeatureResult with features DataFrame and FeatureMap
    """
    oms = _get_pyopenms()

    def report_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct:.0%}] {msg}")

    report_progress(0.0, "Loading mzML file...")

    # Load the mzML file
    with timed_block("Load mzML") as timer:
        loader = MzMLLoader(mzml_path)
        exp = loader.get_experiment()  # Load into memory for FFM
        meta = loader.get_meta()
        loader.close()

    logger.info(f"Loaded {meta.total_spectra} spectra in {timer}")

    report_progress(0.1, "Configuring FeatureFinderMetabo...")

    # Setup feature map for output
    feature_map = oms.FeatureMap()

    report_progress(0.2, "Running mass trace detection...")

    # Run feature finding using MassTraceDetection -> ElutionPeakDetection -> FeatureFindingMetabo
    with timed_block("FeatureFinderMetabo") as timer:
        try:
            # Setup mass traces
            mass_traces = []
            mtd = oms.MassTraceDetection()
            mtd_params = mtd.getDefaults()
            mtd_params.setValue("mass_error_ppm", config.mass_trace_mz_tolerance * 1e6 / 500)  # Convert to ppm at m/z 500
            mtd_params.setValue("noise_threshold_int", config.noise_threshold_int)
            mtd.setParameters(mtd_params)
            mtd.run(exp, mass_traces, max(1, int(exp.size() / 100)))

            logger.info(f"Found {len(mass_traces)} mass traces")

            # Elution peak detection
            epd = oms.ElutionPeakDetection()
            epd_params = epd.getDefaults()
            epd_params.setValue("chrom_peak_snr", config.chrom_peak_snr)
            epd_params.setValue("chrom_fwhm", config.chrom_fwhm)
            epd.setParameters(epd_params)

            mass_traces_filtered = []
            epd.detectPeaks(mass_traces, mass_traces_filtered)

            logger.info(f"After peak detection: {len(mass_traces_filtered)} mass traces")

            if len(mass_traces_filtered) == 0:
                logger.warning("No mass traces after filtering")
                # Return empty result
                return FeatureResult(
                    features_df=pd.DataFrame(),
                    feature_map=feature_map,
                    ms2_mappings=pd.DataFrame(),
                    stats={"feature_count": 0, "processing_time_s": timer.elapsed_seconds},
                )

            # Feature linking (isotope grouping)
            feat_map_tmp = []
            ffm_link = oms.FeatureFindingMetabo()
            ffm_link_params = ffm_link.getDefaults()
            ffm_link_params.setValue("isotope_filtering_model", config.isotope_filtering_model)
            ffm_link_params.setValue("mz_scoring_13C", "true" if config.mz_scoring_13c else "false")
            ffm_link_params.setValue("report_summed_ints", "true")
            ffm_link_params.setValue("remove_single_traces", "true")
            ffm_link.setParameters(ffm_link_params)

            ffm_link.run(mass_traces_filtered, feature_map, feat_map_tmp)

        except Exception as e:
            logger.error(f"FeatureFinderMetabo failed: {e}")
            raise FeatureFinderError("run", str(e)) from e

    logger.info(f"Feature finding completed in {timer}: {feature_map.size()} features")

    report_progress(0.6, f"Found {feature_map.size()} features")

    if feature_map.size() == 0:
        raise NoFeaturesFoundError(str(mzml_path))

    report_progress(0.7, "Converting features to DataFrame...")

    # Convert to DataFrame
    features_df = _feature_map_to_dataframe(feature_map)

    report_progress(0.8, "Mapping MS2 spectra to features...")

    # Map MS2 spectra (loader was closed, pass None to create new one)
    ms2_mappings = _map_ms2_to_features(
        features_df,
        None,  # loader was closed, will create new one
        mzml_path,
        config.ms2_mz_tolerance,
        config.ms2_rt_tolerance,
    )

    report_progress(1.0, "Feature finding complete")

    stats = {
        "feature_count": len(features_df),
        "ms2_mapped_count": int(ms2_mappings["feature_id"].nunique()) if len(ms2_mappings) > 0 else 0,
        "processing_time_s": timer.elapsed_seconds,
        "input_spectra": meta.total_spectra,
        "input_ms1": meta.ms1_count,
        "input_ms2": meta.ms2_count,
    }

    return FeatureResult(
        features_df=features_df,
        feature_map=feature_map,
        ms2_mappings=ms2_mappings,
        stats=stats,
    )


def _feature_map_to_dataframe(feature_map) -> pd.DataFrame:
    """
    Convert FeatureMap to pandas DataFrame.

    Args:
        feature_map: pyOpenMS FeatureMap

    Returns:
        DataFrame with feature information
    """
    features = []

    for i, feature in enumerate(feature_map):
        # Get basic properties
        mz = feature.getMZ()
        rt = feature.getRT()
        intensity = feature.getIntensity()
        charge = feature.getCharge()
        quality = feature.getOverallQuality()

        # Get convex hull for RT/MZ ranges
        hull = feature.getConvexHull()
        hull_points = hull.getHullPoints()

        if len(hull_points) > 0:
            hull_arr = np.array([(p.getX(), p.getY()) for p in hull_points])
            rt_min = hull_arr[:, 0].min()
            rt_max = hull_arr[:, 0].max()
            mz_min = hull_arr[:, 1].min()
            mz_max = hull_arr[:, 1].max()
        else:
            rt_min = rt_max = rt
            mz_min = mz_max = mz

        # Get subordinate features (isotopes)
        subordinates = feature.getSubordinates()
        isotope_count = len(subordinates)

        features.append({
            "feature_id": i,
            "mz": mz,
            "rt": rt,
            "rt_min": rt_min,
            "rt_max": rt_max,
            "mz_min": mz_min,
            "mz_max": mz_max,
            "intensity": intensity,
            "charge": charge,
            "quality": quality,
            "isotope_count": isotope_count,
            "width_rt": rt_max - rt_min,
            "width_mz": mz_max - mz_min,
        })

    return pd.DataFrame(features)


def _map_ms2_to_features(
    features_df: pd.DataFrame,
    loader: MzMLLoader | None,
    mzml_path: Path,
    mz_tolerance: float,
    rt_tolerance: float,
) -> pd.DataFrame:
    """
    Map MS2 spectra to features based on precursor m/z and RT.

    Args:
        features_df: DataFrame with features
        loader: Optional existing loader
        mzml_path: Path to mzML file
        mz_tolerance: m/z tolerance in Da
        rt_tolerance: RT tolerance in seconds

    Returns:
        DataFrame with MS2-to-feature mappings
    """
    if len(features_df) == 0:
        return pd.DataFrame(columns=["feature_id", "spectrum_index", "precursor_mz", "rt", "peaks_count"])

    # Open new loader if needed (thread safety)
    if loader is None:
        loader = MzMLLoader(mzml_path)

    mappings = []

    # Get MS2 spectra
    for spec in loader.iter_spectra(ms_level=2):
        # Get precursor info
        precursors = spec.getPrecursors()
        if not precursors:
            continue

        precursor = precursors[0]
        precursor_mz = precursor.getMZ()
        spec_rt = spec.getRT()

        # Find matching features
        mz_match = (features_df["mz"] - precursor_mz).abs() <= mz_tolerance
        rt_match = (features_df["rt"] - spec_rt).abs() <= rt_tolerance

        matches = features_df[mz_match & rt_match]

        for _, feat in matches.iterrows():
            mzs, ints = spec.get_peaks()
            mappings.append({
                "feature_id": int(feat["feature_id"]),
                "spectrum_index": spec.getNativeID(),
                "precursor_mz": precursor_mz,
                "rt": spec_rt,
                "peaks_count": len(mzs),
                "mz_error": abs(feat["mz"] - precursor_mz),
                "rt_error": abs(feat["rt"] - spec_rt),
            })

    return pd.DataFrame(mappings)


def apply_blank_correction(
    features_df: pd.DataFrame,
    blank_features_df: pd.DataFrame,
    config: BlankCorrectionConfig,
    mz_tolerance: float = 0.01,
    rt_tolerance: float = 30.0,
) -> pd.DataFrame:
    """
    Apply blank correction to features.

    Args:
        features_df: Sample features DataFrame
        blank_features_df: Blank features DataFrame
        config: Blank correction configuration
        mz_tolerance: m/z tolerance for matching (Da)
        rt_tolerance: RT tolerance for matching (seconds)

    Returns:
        Corrected features DataFrame
    """
    if config.policy == BlankCorrectionPolicy.NONE:
        return features_df.copy()

    if len(blank_features_df) == 0:
        logger.warning("No blank features provided, skipping blank correction")
        return features_df.copy()

    result = features_df.copy()
    result["blank_matched"] = False
    result["blank_intensity"] = 0.0
    result["blank_ratio"] = float("inf")

    # Match sample features to blank features
    for idx, sample_feat in features_df.iterrows():
        # Find matching blank features
        mz_match = (blank_features_df["mz"] - sample_feat["mz"]).abs() <= mz_tolerance
        rt_match = (blank_features_df["rt"] - sample_feat["rt"]).abs() <= rt_tolerance

        blank_matches = blank_features_df[mz_match & rt_match]

        if len(blank_matches) > 0:
            # Use highest intensity blank match
            best_blank = blank_matches.loc[blank_matches["intensity"].idxmax()]
            result.loc[idx, "blank_matched"] = True
            result.loc[idx, "blank_intensity"] = best_blank["intensity"]

            if best_blank["intensity"] > 0:
                result.loc[idx, "blank_ratio"] = sample_feat["intensity"] / best_blank["intensity"]

    # Apply correction policy
    if config.policy == BlankCorrectionPolicy.PRESENCE:
        # Remove features present in blank
        result = result[~result["blank_matched"]]

    elif config.policy == BlankCorrectionPolicy.FOLD_CHANGE:
        # Keep only if sample/blank ratio exceeds threshold
        threshold = config.fold_change_threshold
        result = result[(~result["blank_matched"]) | (result["blank_ratio"] >= threshold)]

    elif config.policy == BlankCorrectionPolicy.SUBTRACT:
        # Subtract blank intensity
        result["intensity"] = result["intensity"] - result["blank_intensity"]
        result = result[result["intensity"] > 0]

    # Clean up helper columns
    result = result.drop(columns=["blank_matched", "blank_intensity", "blank_ratio"])

    logger.info(f"Blank correction: {len(features_df)} -> {len(result)} features")

    return result.reset_index(drop=True)


def run_feature_finding_step(
    run_context: RunContext,
    mzml_path: Path,
    config: FeatureFinderConfig,
    blank_config: BlankCorrectionConfig | None = None,
    blank_mzml_paths: list[Path] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """
    Run complete feature finding step with checkpointing.

    Args:
        run_context: Run context for checkpoints
        mzml_path: Path to sample mzML
        config: Feature finder config
        blank_config: Blank correction config
        blank_mzml_paths: Paths to blank mzML files
        progress_callback: Progress callback

    Returns:
        Features DataFrame
    """
    step_meta = StepMeta(
        step_name="featurefinding",
        step_number=1,
        status="running",
        started_at=datetime.now().isoformat(),
    )

    try:
        # Run feature finding on sample
        result = run_feature_finder_metabo(mzml_path, config, progress_callback)
        features_df = result.features_df
        step_meta.input_feature_count = len(features_df)

        # Process blanks if provided
        if blank_config and blank_mzml_paths and blank_config.policy != BlankCorrectionPolicy.NONE:
            blank_features = []
            for blank_path in blank_mzml_paths:
                try:
                    blank_result = run_feature_finder_metabo(blank_path, config)
                    blank_features.append(blank_result.features_df)
                except Exception as e:
                    logger.warning(f"Failed to process blank {blank_path}: {e}")

            if blank_features:
                # Combine blank features
                combined_blank = pd.concat(blank_features, ignore_index=True)

                # Apply blank correction
                features_df = apply_blank_correction(
                    features_df,
                    combined_blank,
                    blank_config,
                )

        step_meta.output_feature_count = len(features_df)
        step_meta.status = "completed"
        step_meta.completed_at = datetime.now().isoformat()
        step_meta.extra = result.stats

        # Add feature_id as index if not present
        if "feature_id" not in features_df.columns:
            features_df["feature_id"] = range(len(features_df))

        # Save checkpoint
        run_context.save_checkpoint("featurefinding", features_df, step_meta)

        return features_df

    except Exception as e:
        step_meta.status = "failed"
        step_meta.completed_at = datetime.now().isoformat()
        step_meta.extra = {"error": str(e)}
        run_context.save_checkpoint("featurefinding", pd.DataFrame(), step_meta)
        raise
