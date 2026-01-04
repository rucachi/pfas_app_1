"""
PFAS prioritization module for ONFRA PFAS pipeline.

Implements MD/C, KMD analysis, diagnostic fragment matching, suspect screening, and scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import (
    PrioritizationConfig,
    KMDConfig,
    MDCConfig,
    DiagnosticFragmentConfig,
    SuspectScreeningConfig,
    ScoringConfig,
)
from .checkpoints import RunContext, StepMeta
from .backend import get_current_backend, to_backend_array, ensure_numpy
from .utils import ppm_difference, is_within_tolerance, timed_block

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Evidence:
    """Evidence for PFAS identification."""

    type: str  # e.g., "kmd_series", "df_match", "suspect_match"
    score: float
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class HomologousSeries:
    """A group of features forming a homologous series."""

    series_id: int
    repeat_unit: str
    members: list[int]  # feature_ids
    kmd_value: float
    delta_m: float
    confidence: float


@dataclass
class DFMatch:
    """Diagnostic fragment match."""

    feature_id: int
    fragment_name: str
    theoretical_mz: float
    observed_mz: float
    error_ppm: float
    intensity: float


@dataclass
class SuspectMatch:
    """Suspect list match."""

    feature_id: int
    suspect_name: str
    suspect_formula: str | None
    theoretical_mass: float
    observed_mz: float
    adduct: str
    error_ppm: float


# =============================================================================
# Default Rules
# =============================================================================

# Default diagnostic fragments for PFAS (negative mode)
DEFAULT_DF_RULES = {
    "CF3-": 68.9952,
    "C2F5-": 118.9920,
    "C3F7-": 168.9888,
    "C4F9-": 218.9856,
    "C5F11-": 268.9824,
    "C6F13-": 318.9792,
    "C7F15-": 368.9760,
    "C8F17-": 418.9728,
    "FSO3-": 98.9552,
    "HSO4-": 96.9601,
    "SO3-": 79.9568,
    "PO3-": 78.9585,
    "H2PO4-": 96.9696,
    "C2F4H-": 98.9952,
    "C3F6H-": 148.9920,
}

# Default delta-m rules (neutral losses)
DEFAULT_DELTA_M_RULES = {
    "CF2": 49.9968,
    "HF": 20.0063,
    "CO2": 43.9898,
    "SO3": 79.9568,
    "H2O": 18.0106,
    "C2F4": 99.9936,
    "CF2O": 65.9917,
}


# =============================================================================
# Kendrick Mass Defect Analysis
# =============================================================================


def calculate_kendrick_mass(mz: np.ndarray, repeat_unit_mass: float) -> np.ndarray:
    """
    Calculate Kendrick mass.

    Args:
        mz: Array of m/z values
        repeat_unit_mass: Exact mass of repeat unit (e.g., CF2 = 49.9968)

    Returns:
        Array of Kendrick masses
    """
    # Kendrick mass = observed mass × (nominal mass of repeat unit / exact mass of repeat unit)
    # For CF2: nominal = 50, exact = 49.9968473
    nominal_mass = round(repeat_unit_mass)
    return mz * (nominal_mass / repeat_unit_mass)


def calculate_kmd(mz: np.ndarray, repeat_unit_mass: float) -> np.ndarray:
    """
    Calculate Kendrick Mass Defect (KMD).

    Args:
        mz: Array of m/z values
        repeat_unit_mass: Exact mass of repeat unit

    Returns:
        Array of KMD values
    """
    kendrick_mass = calculate_kendrick_mass(mz, repeat_unit_mass)
    # KMD = nominal Kendrick mass - Kendrick mass
    return np.floor(kendrick_mass) - kendrick_mass


def find_homologous_series(
    features_df: pd.DataFrame,
    config: KMDConfig,
) -> list[HomologousSeries]:
    """
    Find homologous series based on KMD analysis.

    Features with similar KMD and m/z differing by repeat unit mass
    are grouped into homologous series.

    Args:
        features_df: DataFrame with features (must have 'mz' column)
        config: KMD configuration

    Returns:
        List of HomologousSeries
    """
    if not config.enabled or len(features_df) == 0:
        return []

    all_series = []
    series_id = 0

    for repeat_unit in config.repeat_units:
        if repeat_unit not in config.repeat_unit_masses:
            logger.warning(f"Unknown repeat unit: {repeat_unit}")
            continue

        repeat_mass = config.repeat_unit_masses[repeat_unit]

        # Calculate KMD for all features
        mz_values = features_df["mz"].values
        kmd_values = calculate_kmd(mz_values, repeat_mass)

        # Add KMD to dataframe temporarily
        features_df[f"kmd_{repeat_unit}"] = kmd_values

        # Group features by similar KMD
        kmd_tolerance = config.kmd_tolerance

        # Sort by KMD for efficient grouping
        sorted_idx = np.argsort(kmd_values)
        used = set()

        for i in sorted_idx:
            if i in used:
                continue

            kmd_i = kmd_values[i]
            mz_i = mz_values[i]

            # Find features with similar KMD
            kmd_diff = np.abs(kmd_values - kmd_i)
            candidates = np.where(kmd_diff <= kmd_tolerance)[0]

            if len(candidates) < config.min_series_length:
                continue

            # Check for m/z spacing consistent with repeat unit
            series_members = [i]
            used.add(i)

            for j in candidates:
                if j in used or j == i:
                    continue

                mz_j = mz_values[j]
                # Check if m/z difference is multiple of repeat unit mass
                mz_diff = abs(mz_j - mz_i)
                n_units = round(mz_diff / repeat_mass)

                if n_units > 0:
                    expected_diff = n_units * repeat_mass
                    error = abs(mz_diff - expected_diff)

                    if error < 0.02:  # 20 mDa tolerance for series membership
                        series_members.append(j)
                        used.add(j)

            if len(series_members) >= config.min_series_length:
                # Valid series found
                feature_ids = [int(features_df.iloc[idx]["feature_id"]) for idx in series_members]

                series = HomologousSeries(
                    series_id=series_id,
                    repeat_unit=repeat_unit,
                    members=feature_ids,
                    kmd_value=float(kmd_i),
                    delta_m=repeat_mass,
                    confidence=min(1.0, len(feature_ids) / 5),  # Higher confidence with more members
                )
                all_series.append(series)
                series_id += 1

        # Clean up temporary column
        features_df.drop(columns=[f"kmd_{repeat_unit}"], inplace=True, errors="ignore")

    logger.info(f"Found {len(all_series)} homologous series")
    return all_series


# =============================================================================
# Mass Defect / Carbon Analysis
# =============================================================================


def calculate_mass_defect(mz: np.ndarray) -> np.ndarray:
    """
    Calculate mass defect.

    Args:
        mz: Array of m/z values

    Returns:
        Array of mass defects (fractional part of mass)
    """
    return mz - np.floor(mz)


def estimate_carbon_count(mz: np.ndarray, mode: str = "negative") -> np.ndarray:
    """
    Estimate carbon count from m/z.

    This is a rough estimate based on typical PFAS composition.
    For PFAS, the average mass per carbon is ~50 Da (considering F atoms).

    Args:
        mz: Array of m/z values
        mode: Ionization mode ("negative" or "positive")

    Returns:
        Estimated carbon count
    """
    # Adjust for adduct
    if mode == "negative":
        neutral = mz + 1.007276  # [M-H]- -> M
    else:
        neutral = mz - 1.007276  # [M+H]+ -> M

    # PFAS rough estimate: ~50 Da per carbon equivalent
    # This is a simplification; actual depends on F/C ratio
    return np.round(neutral / 50).astype(int)


def calculate_mdc(mz: np.ndarray, carbon_count: np.ndarray | None = None) -> np.ndarray:
    """
    Calculate MD/C (mass defect per carbon).

    Args:
        mz: Array of m/z values
        carbon_count: Optional carbon counts (estimated if not provided)

    Returns:
        Array of MD/C values
    """
    if carbon_count is None:
        carbon_count = estimate_carbon_count(mz)

    mass_defect = calculate_mass_defect(mz)

    # Avoid division by zero
    carbon_count = np.maximum(carbon_count, 1)

    return mass_defect / carbon_count


def filter_by_mdc(
    features_df: pd.DataFrame,
    config: MDCConfig,
) -> pd.DataFrame:
    """
    Filter features by MD/C region typical for PFAS.

    Args:
        features_df: DataFrame with features
        config: MD/C configuration

    Returns:
        Boolean array indicating PFAS region membership
    """
    if not config.enabled:
        return np.ones(len(features_df), dtype=bool)

    mz_values = features_df["mz"].values

    # Filter by mass range first
    in_mass_range = (mz_values >= config.mass_min) & (mz_values <= config.mass_max)

    # Calculate MD/C
    carbon_count = estimate_carbon_count(mz_values)
    mdc_values = calculate_mdc(mz_values, carbon_count)

    # Filter by MD/C region
    in_mdc_region = (mdc_values >= config.mdc_min) & (mdc_values <= config.mdc_max)

    return in_mass_range & in_mdc_region


# =============================================================================
# Diagnostic Fragment Matching
# =============================================================================


def load_df_rules(rules_file: str | None = None) -> dict[str, float]:
    """
    Load diagnostic fragment rules.

    Args:
        rules_file: Path to YAML file with rules, or None for defaults

    Returns:
        Dictionary of fragment_name -> m/z
    """
    if rules_file is None:
        return DEFAULT_DF_RULES.copy()

    try:
        with open(rules_file, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
        return rules.get("diagnostic_fragments", DEFAULT_DF_RULES)
    except Exception as e:
        logger.warning(f"Failed to load DF rules from {rules_file}: {e}")
        return DEFAULT_DF_RULES.copy()


def match_diagnostic_fragments(
    feature_id: int,
    ms2_peaks: np.ndarray,  # Shape: (n, 2) with columns [mz, intensity]
    rules: dict[str, float],
    config: DiagnosticFragmentConfig,
) -> list[DFMatch]:
    """
    Match MS2 peaks against diagnostic fragment rules.

    Args:
        feature_id: Feature ID
        ms2_peaks: MS2 peaks array (mz, intensity)
        rules: Diagnostic fragment rules
        config: DF configuration

    Returns:
        List of DFMatch objects
    """
    if not config.enabled or len(ms2_peaks) == 0:
        return []

    matches = []
    observed_mz = ms2_peaks[:, 0]
    observed_int = ms2_peaks[:, 1]

    for frag_name, theoretical_mz in rules.items():
        # Calculate tolerance
        if config.use_ppm:
            tolerance = theoretical_mz * config.tolerance_ppm / 1e6
        else:
            tolerance = config.tolerance_da

        # Find matching peaks
        mz_diff = np.abs(observed_mz - theoretical_mz)
        match_idx = np.where(mz_diff <= tolerance)[0]

        for idx in match_idx:
            error_ppm = ppm_difference(theoretical_mz, observed_mz[idx])
            matches.append(DFMatch(
                feature_id=feature_id,
                fragment_name=frag_name,
                theoretical_mz=theoretical_mz,
                observed_mz=float(observed_mz[idx]),
                error_ppm=error_ppm,
                intensity=float(observed_int[idx]),
            ))

    return matches


def match_delta_m_rules(
    precursor_mz: float,
    ms2_peaks: np.ndarray,
    rules: dict[str, float] | None = None,
    tolerance_da: float = 0.02,
) -> list[tuple[str, float, float]]:
    """
    Check for neutral losses (delta-m rules).

    Args:
        precursor_mz: Precursor m/z
        ms2_peaks: MS2 peaks array (mz, intensity)
        rules: Delta-m rules (name -> mass)
        tolerance_da: Mass tolerance

    Returns:
        List of (rule_name, theoretical_loss, observed_loss, error)
    """
    if rules is None:
        rules = DEFAULT_DELTA_M_RULES

    matches = []
    observed_mz = ms2_peaks[:, 0]

    for rule_name, loss_mass in rules.items():
        expected_product = precursor_mz - loss_mass

        if expected_product < 50:  # Too small to be meaningful
            continue

        # Find matching peaks
        mz_diff = np.abs(observed_mz - expected_product)
        match_idx = np.where(mz_diff <= tolerance_da)[0]

        if len(match_idx) > 0:
            best_idx = match_idx[np.argmin(mz_diff[match_idx])]
            observed_loss = precursor_mz - observed_mz[best_idx]
            error = abs(observed_loss - loss_mass)
            matches.append((rule_name, loss_mass, float(mz_diff[best_idx])))

    return matches


# =============================================================================
# Suspect Screening
# =============================================================================


def load_suspect_list(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load suspect list from CSV/TSV.

    Required columns: name, exact_mass OR formula
    Optional columns: formula, cas, class, adducts

    Args:
        path: Path to suspect list file

    Returns:
        DataFrame with suspect information
    """
    path = Path(path)

    # Detect delimiter
    if path.suffix.lower() == ".tsv":
        sep = "\t"
    else:
        sep = ","

    df = pd.read_csv(path, sep=sep)

    # Validate required columns
    if "name" not in df.columns:
        raise ValueError("Suspect list must have 'name' column")

    if "exact_mass" not in df.columns and "formula" not in df.columns:
        raise ValueError("Suspect list must have 'exact_mass' or 'formula' column")

    # Calculate exact mass from formula if needed
    if "exact_mass" not in df.columns:
        # TODO: Implement formula to mass conversion
        # For now, skip entries without exact_mass
        logger.warning("Formula-to-mass conversion not implemented, skipping entries without exact_mass")
        df = df[df["exact_mass"].notna()]

    return df


def match_suspects(
    features_df: pd.DataFrame,
    suspects_df: pd.DataFrame,
    config: SuspectScreeningConfig,
) -> list[SuspectMatch]:
    """
    Match features against suspect list.

    Args:
        features_df: DataFrame with features
        suspects_df: DataFrame with suspects
        config: Suspect screening configuration

    Returns:
        List of SuspectMatch objects
    """
    if not config.enabled or len(features_df) == 0 or len(suspects_df) == 0:
        return []

    matches = []

    for _, suspect in suspects_df.iterrows():
        exact_mass = suspect["exact_mass"]
        name = suspect["name"]
        formula = suspect.get("formula", None)

        # Check each adduct
        for adduct in config.adducts:
            if adduct not in config.adduct_masses:
                continue

            # Calculate expected m/z
            adduct_mass = config.adduct_masses[adduct]
            expected_mz = exact_mass + adduct_mass

            # Find matching features
            for _, feature in features_df.iterrows():
                observed_mz = feature["mz"]
                error_ppm = ppm_difference(expected_mz, observed_mz)

                if error_ppm <= config.ppm_tolerance:
                    matches.append(SuspectMatch(
                        feature_id=int(feature["feature_id"]),
                        suspect_name=name,
                        suspect_formula=formula,
                        theoretical_mass=exact_mass,
                        observed_mz=observed_mz,
                        adduct=adduct,
                        error_ppm=error_ppm,
                    ))

    logger.info(f"Found {len(matches)} suspect matches")
    return matches


# =============================================================================
# Scoring
# =============================================================================


def calculate_pfas_scores(
    features_df: pd.DataFrame,
    homologous_series: list[HomologousSeries],
    df_matches: dict[int, list[DFMatch]],
    delta_m_matches: dict[int, list[tuple]],
    suspect_matches: list[SuspectMatch],
    mdc_in_region: np.ndarray,
    config: ScoringConfig,
) -> pd.DataFrame:
    """
    Calculate PFAS scores for all features.

    Args:
        features_df: DataFrame with features
        homologous_series: List of homologous series
        df_matches: Feature ID -> DF matches
        delta_m_matches: Feature ID -> delta-m matches
        suspect_matches: List of suspect matches
        mdc_in_region: Boolean array for MD/C region
        config: Scoring configuration

    Returns:
        DataFrame with scores and evidence
    """
    weights = config.weights

    # Initialize score columns
    results = features_df.copy()
    results["pfas_score"] = 0.0
    results["evidence_count"] = 0
    results["evidence_types"] = ""
    results["evidence_details"] = ""

    # Build feature -> series mapping
    feature_series = {}
    for series in homologous_series:
        for fid in series.members:
            if fid not in feature_series:
                feature_series[fid] = []
            feature_series[fid].append(series)

    # Build feature -> suspect mapping
    feature_suspects = {}
    for match in suspect_matches:
        if match.feature_id not in feature_suspects:
            feature_suspects[match.feature_id] = []
        feature_suspects[match.feature_id].append(match)

    # Calculate scores for each feature
    for idx, row in results.iterrows():
        fid = int(row["feature_id"])
        score = 0.0
        evidences = []
        evidence_details = []

        # MD/C region
        if mdc_in_region[idx]:
            score += weights.get("mdc_region", 0)
            evidences.append("mdc")
            evidence_details.append("MD/C in PFAS region")

        # Homologous series
        if fid in feature_series:
            series_list = feature_series[fid]
            score += weights.get("kmd_series", 0) * len(series_list)
            evidences.append("kmd")
            for s in series_list:
                evidence_details.append(f"Series {s.series_id} ({s.repeat_unit}, {len(s.members)} members)")

        # Diagnostic fragments
        if fid in df_matches and df_matches[fid]:
            matches = df_matches[fid]
            score += weights.get("df_match", 0) * min(len(matches), 3)  # Cap at 3
            evidences.append("df")
            for m in matches[:3]:
                evidence_details.append(f"DF: {m.fragment_name} ({m.error_ppm:.1f} ppm)")

        # Delta-m matches
        if fid in delta_m_matches and delta_m_matches[fid]:
            matches = delta_m_matches[fid]
            score += weights.get("delta_m_match", 0) * min(len(matches), 3)
            evidences.append("delta_m")
            for m in matches[:3]:
                evidence_details.append(f"ΔM: {m[0]} ({m[2]:.3f} Da error)")

        # Suspect matches
        if fid in feature_suspects:
            matches = feature_suspects[fid]
            score += weights.get("suspect_match", 0)
            evidences.append("suspect")
            for m in matches[:3]:
                evidence_details.append(f"Suspect: {m.suspect_name} ({m.adduct}, {m.error_ppm:.1f} ppm)")

        results.at[idx, "pfas_score"] = score
        results.at[idx, "evidence_count"] = len(evidences)
        results.at[idx, "evidence_types"] = ",".join(evidences)
        results.at[idx, "evidence_details"] = "; ".join(evidence_details)

    # Sort by score
    results = results.sort_values("pfas_score", ascending=False)

    # Filter by minimum score
    if config.min_score_threshold > 0:
        results = results[results["pfas_score"] >= config.min_score_threshold]

    return results.reset_index(drop=True)


# =============================================================================
# Main Prioritization Pipeline
# =============================================================================


def run_prioritization(
    features_df: pd.DataFrame,
    ms2_data: dict[int, np.ndarray] | None,  # feature_id -> MS2 peaks
    config: PrioritizationConfig,
    suspect_list_path: str | None = None,
    df_rules_path: str | None = None,
) -> pd.DataFrame:
    """
    Run complete PFAS prioritization pipeline.

    Args:
        features_df: DataFrame with features from feature finding
        ms2_data: Dictionary mapping feature_id to MS2 peaks
        config: Prioritization configuration
        suspect_list_path: Path to suspect list
        df_rules_path: Path to DF rules YAML

    Returns:
        DataFrame with prioritized features
    """
    if len(features_df) == 0:
        logger.warning("No features to prioritize")
        return features_df

    ms2_data = ms2_data or {}

    with timed_block("Prioritization") as timer:
        # Step 1: MD/C filtering
        logger.info("Calculating MD/C region membership...")
        mdc_in_region = filter_by_mdc(features_df, config.mdc)

        # Step 2: KMD analysis
        logger.info("Finding homologous series...")
        homologous_series = find_homologous_series(features_df, config.kmd)

        # Step 3: Diagnostic fragment matching
        logger.info("Matching diagnostic fragments...")
        df_rules = load_df_rules(df_rules_path or config.diagnostic_fragments.rules_file)
        df_matches = {}
        delta_m_matches = {}

        for fid, peaks in ms2_data.items():
            if len(peaks) > 0:
                # DF matches
                df_matches[fid] = match_diagnostic_fragments(
                    fid, peaks, df_rules, config.diagnostic_fragments
                )

                # Delta-m matches
                feature_row = features_df[features_df["feature_id"] == fid]
                if len(feature_row) > 0:
                    precursor_mz = feature_row.iloc[0]["mz"]
                    delta_m_matches[fid] = match_delta_m_rules(precursor_mz, peaks)

        # Step 4: Suspect matching
        logger.info("Matching against suspect list...")
        suspect_matches = []
        if suspect_list_path:
            try:
                suspects_df = load_suspect_list(suspect_list_path)
                suspect_matches = match_suspects(features_df, suspects_df, config.suspect_screening)
            except Exception as e:
                logger.warning(f"Failed to load/match suspect list: {e}")

        # Step 5: Scoring
        logger.info("Calculating PFAS scores...")
        results = calculate_pfas_scores(
            features_df,
            homologous_series,
            df_matches,
            delta_m_matches,
            suspect_matches,
            mdc_in_region,
            config.scoring,
        )

        # Add additional columns
        results["mdc_in_region"] = mdc_in_region[results.index] if len(results) > 0 else []

    logger.info(f"Prioritization completed in {timer}: {len(results)} features scored")
    return results


def run_prioritization_step(
    run_context: RunContext,
    features_df: pd.DataFrame,
    ms2_data: dict[int, np.ndarray] | None,
    config: PrioritizationConfig,
    suspect_list_path: str | None = None,
    df_rules_path: str | None = None,
) -> pd.DataFrame:
    """
    Run prioritization step with checkpointing.

    Args:
        run_context: Run context for checkpointing
        features_df: Features DataFrame
        ms2_data: MS2 data
        config: Prioritization config
        suspect_list_path: Suspect list path
        df_rules_path: DF rules path

    Returns:
        Prioritized features DataFrame
    """
    step_meta = StepMeta(
        step_name="prioritization",
        step_number=2,
        status="running",
        started_at=datetime.now().isoformat(),
        input_feature_count=len(features_df),
    )

    try:
        result = run_prioritization(
            features_df,
            ms2_data,
            config,
            suspect_list_path,
            df_rules_path,
        )

        step_meta.status = "completed"
        step_meta.completed_at = datetime.now().isoformat()
        step_meta.output_feature_count = len(result)

        run_context.save_checkpoint("prioritization", result, step_meta)

        return result

    except Exception as e:
        step_meta.status = "failed"
        step_meta.completed_at = datetime.now().isoformat()
        step_meta.extra = {"error": str(e)}
        run_context.save_checkpoint("prioritization", pd.DataFrame(), step_meta)
        raise
