"""
Configuration management for ONFRA PFAS pipeline.

Pydantic-based configuration schema with load/save capabilities.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GPUMode(str, Enum):
    """GPU usage policy."""

    AUTO = "auto"  # Choose GPU based on data size thresholds
    FORCE_GPU = "force_gpu"  # Always use GPU, fail if unavailable
    FORCE_CPU = "force_cpu"  # Never use GPU


class BlankCorrectionPolicy(str, Enum):
    """Blank correction policies for feature filtering."""

    NONE = "none"  # No blank correction
    SUBTRACT = "subtract"  # Subtract blank intensity
    FOLD_CHANGE = "fold_change"  # Filter by sample/blank ratio
    PRESENCE = "presence"  # Remove features present in blank


class FeatureFinderConfig(BaseModel):
    """Configuration for FeatureFinderMetabo."""

    # Mass trace detection
    mass_trace_mz_tolerance: float = Field(
        default=0.005, ge=0.0001, le=0.1, description="m/z tolerance for mass trace detection (Da)"
    )
    noise_threshold_int: float = Field(
        default=1000.0, ge=0.0, description="Intensity threshold for noise filtering"
    )
    chrom_peak_snr: float = Field(
        default=3.0, ge=1.0, description="Signal-to-noise ratio for chromatographic peaks"
    )
    chrom_fwhm: float = Field(
        default=10.0, ge=1.0, le=300.0, description="Expected chromatographic peak width (seconds)"
    )

    # Isotope filtering
    isotope_filtering_model: str = Field(
        default="metabolites (5% RMS)",
        description="Isotope pattern model for filtering",
    )
    mz_scoring_13c: bool = Field(
        default=True, description="Use 13C isotope pattern for scoring"
    )

    # Feature linking
    rt_tolerance: float = Field(
        default=5.0, ge=0.1, le=60.0, description="RT tolerance for feature linking (seconds)"
    )

    # MS2 mapping
    ms2_mz_tolerance: float = Field(
        default=0.01, ge=0.001, le=0.1, description="m/z tolerance for MS2 mapping (Da)"
    )
    ms2_rt_tolerance: float = Field(
        default=10.0, ge=1.0, le=60.0, description="RT tolerance for MS2 mapping (seconds)"
    )


class KMDConfig(BaseModel):
    """Configuration for Kendrick Mass Defect analysis."""

    enabled: bool = Field(default=True, description="Enable KMD analysis")
    repeat_units: list[str] = Field(
        default=["CF2", "CF2O", "C2F4", "C2F4O"],
        description="Repeat units for KMD calculation",
    )
    # Exact masses for repeat units
    repeat_unit_masses: dict[str, float] = Field(
        default={
            "CF2": 49.9968473,
            "CF2O": 65.9917565,
            "C2F4": 99.9936946,
            "C2F4O": 115.9886038,
            "CHF": 32.0062284,
            "CH2": 14.0156500,
        },
        description="Exact masses for repeat units",
    )
    kmd_tolerance: float = Field(
        default=0.005, ge=0.001, le=0.05, description="KMD tolerance for series grouping"
    )
    min_series_length: int = Field(
        default=3, ge=2, le=20, description="Minimum members in a homologous series"
    )


class MDCConfig(BaseModel):
    """Configuration for Mass Defect / Carbon (MD/C - m/C) analysis."""

    enabled: bool = Field(default=True, description="Enable MD/C analysis")
    # PFAS typically has MD/C between -0.08 and 0.04
    mdc_min: float = Field(default=-0.10, description="Minimum MD/C value for PFAS region")
    mdc_max: float = Field(default=0.05, description="Maximum MD/C value for PFAS region")
    # Mass range constraints
    mass_min: float = Field(default=100.0, description="Minimum mass for analysis")
    mass_max: float = Field(default=1500.0, description="Maximum mass for analysis")
    # Fallback to simple mass-based filtering if carbon count unavailable
    use_mass_fallback: bool = Field(
        default=True, description="Use mass-based fallback if carbon count unavailable"
    )


class DiagnosticFragmentConfig(BaseModel):
    """Configuration for diagnostic fragment matching."""

    enabled: bool = Field(default=True, description="Enable DF matching")
    tolerance_ppm: float = Field(
        default=10.0, ge=1.0, le=200.0, description="Mass tolerance for DF matching (ppm)"
    )
    tolerance_da: float = Field(
        default=0.01, ge=0.001, le=0.1, description="Mass tolerance for DF matching (Da)"
    )
    use_ppm: bool = Field(default=True, description="Use ppm tolerance (else Da)")
    rules_file: str | None = Field(
        default=None, description="Path to custom DF rules YAML file"
    )


class SuspectScreeningConfig(BaseModel):
    """Configuration for suspect list matching."""

    enabled: bool = Field(default=True, description="Enable suspect screening")
    ppm_tolerance: float = Field(
        default=5.0, ge=1.0, le=20.0, description="Mass tolerance for suspect matching (ppm)"
    )
    adducts: list[str] = Field(
        default=["[M-H]-", "[M+H]+", "[M+FA-H]-", "[M+Na]+"],
        description="Adduct forms to consider",
    )
    # Adduct mass shifts (negative mode first, then positive)
    adduct_masses: dict[str, float] = Field(
        default={
            "[M-H]-": -1.007276,
            "[M+Cl]-": 34.969402,
            "[M+FA-H]-": 44.998201,
            "[M+H]+": 1.007276,
            "[M+Na]+": 22.989218,
            "[M+NH4]+": 18.034374,
        },
        description="Mass shifts for adducts",
    )


class ScoringConfig(BaseModel):
    """Configuration for PFAS scoring/ranking."""

    # Evidence weights (sum to calculate final score)
    weights: dict[str, float] = Field(
        default={
            "kmd_series": 2.0,  # Member of homologous series
            "mdc_region": 1.5,  # Falls in PFAS MD/C region
            "df_match": 3.0,  # Diagnostic fragment match
            "delta_m_match": 2.5,  # Δm rule match
            "suspect_match": 5.0,  # Suspect list match
            "ms2_available": 1.0,  # Has MS2 spectrum
            "isotope_pattern": 1.0,  # Good isotope pattern
        },
        description="Weights for evidence types",
    )
    min_score_threshold: float = Field(
        default=2.0, description="Minimum score to include in results"
    )


class BlankCorrectionConfig(BaseModel):
    """Configuration for blank correction."""

    policy: BlankCorrectionPolicy = Field(
        default=BlankCorrectionPolicy.FOLD_CHANGE,
        description="Blank correction policy",
    )
    fold_change_threshold: float = Field(
        default=3.0, ge=1.0, description="Minimum sample/blank ratio for fold_change policy"
    )
    blank_files: list[str] = Field(
        default=[], description="Paths to blank mzML files"
    )


class VisualizationConfig(BaseModel):
    """Configuration for visualization data generation."""

    eic_mz_tolerance: float = Field(
        default=0.01, ge=0.001, le=0.1, description="m/z tolerance for EIC extraction (Da)"
    )
    correlation_enabled: bool = Field(
        default=True, description="Enable EIC correlation analysis"
    )
    correlation_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum correlation coefficient"
    )
    correlation_chunk_size: int = Field(
        default=1000, ge=100, le=10000, description="Chunk size for correlation calculation"
    )


class GPUConfig(BaseModel):
    """Configuration for GPU acceleration."""

    mode: GPUMode = Field(default=GPUMode.AUTO, description="GPU usage policy")
    # Thresholds for AUTO mode
    feature_count_threshold: int = Field(
        default=5000, description="Use GPU if feature count exceeds this"
    )
    rt_grid_len_threshold: int = Field(
        default=50000, description="Use GPU if RT grid length exceeds this"
    )
    correlation_pairs_threshold: int = Field(
        default=1000000, description="Use GPU if correlation pairs exceed this"
    )
    # OOM handling
    oom_max_retries: int = Field(
        default=3, ge=1, le=10, description="Max retries on VRAM OOM"
    )
    oom_chunk_reduction_factor: float = Field(
        default=0.5, ge=0.1, le=0.9, description="Factor to reduce chunk size on OOM"
    )


class PrioritizationConfig(BaseModel):
    """Configuration for PFAS prioritization."""

    mdc: MDCConfig = Field(default_factory=MDCConfig)
    kmd: KMDConfig = Field(default_factory=KMDConfig)
    diagnostic_fragments: DiagnosticFragmentConfig = Field(
        default_factory=DiagnosticFragmentConfig
    )
    suspect_screening: SuspectScreeningConfig = Field(
        default_factory=SuspectScreeningConfig
    )
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


class PipelineConfig(BaseModel):
    """Main configuration for the PFAS analysis pipeline."""

    # General settings
    project_name: str = Field(default="ONFRA_PFAS_Run", description="Project/run name")
    output_dir: str = Field(default="runs", description="Output directory for runs")

    # GPU settings
    gpu: GPUConfig = Field(default_factory=GPUConfig)

    # Pipeline stage configs
    feature_finder: FeatureFinderConfig = Field(default_factory=FeatureFinderConfig)
    blank_correction: BlankCorrectionConfig = Field(default_factory=BlankCorrectionConfig)
    prioritization: PrioritizationConfig = Field(default_factory=PrioritizationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    # Input files
    input_files: list[str] = Field(default=[], description="Input mzML file paths")
    suspect_list_file: str | None = Field(
        default=None, description="Path to suspect list CSV/TSV"
    )
    df_rules_file: str | None = Field(
        default=None, description="Path to diagnostic fragment rules YAML"
    )

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create from dictionary."""
        return cls.model_validate(data)


# Default presets for different use cases
def get_default_config() -> PipelineConfig:
    """Get default configuration."""
    return PipelineConfig()


def get_high_sensitivity_config() -> PipelineConfig:
    """Get configuration optimized for high sensitivity (more false positives)."""
    config = PipelineConfig()
    config.feature_finder.noise_threshold_int = 500.0
    config.feature_finder.chrom_peak_snr = 2.0
    config.prioritization.scoring.min_score_threshold = 1.0
    return config


def get_high_specificity_config() -> PipelineConfig:
    """Get configuration optimized for high specificity (fewer false positives)."""
    config = PipelineConfig()
    config.feature_finder.noise_threshold_int = 2000.0
    config.feature_finder.chrom_peak_snr = 5.0
    config.prioritization.scoring.min_score_threshold = 4.0
    config.prioritization.kmd.min_series_length = 4
    return config
