"""
Tests for configuration module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from onfra_pfas.core.config import (
    PipelineConfig,
    FeatureFinderConfig,
    PrioritizationConfig,
    GPUMode,
    BlankCorrectionPolicy,
    get_default_config,
    get_high_sensitivity_config,
    get_high_specificity_config,
)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = PipelineConfig()

        assert config.gpu.mode == GPUMode.AUTO
        assert config.feature_finder.mass_trace_mz_tolerance == 0.005
        assert config.prioritization.kmd.enabled is True

    def test_config_validation(self):
        """Test config validation with Pydantic."""
        # Valid config
        config = PipelineConfig(
            project_name="Test Project",
            output_dir="test_runs",
        )
        assert config.project_name == "Test Project"

        # Invalid value should raise
        with pytest.raises(ValueError):
            FeatureFinderConfig(mass_trace_mz_tolerance=-1)

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = PipelineConfig(
            project_name="SaveLoadTest",
            gpu={"mode": "force_cpu"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Save
            config.save(config_path)
            assert config_path.exists()

            # Load
            loaded = PipelineConfig.load(config_path)
            assert loaded.project_name == "SaveLoadTest"
            assert loaded.gpu.mode == GPUMode.FORCE_CPU

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = PipelineConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "gpu" in d
        assert "feature_finder" in d
        assert "prioritization" in d

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "project_name": "FromDict",
            "gpu": {"mode": "force_gpu"},
            "feature_finder": {"noise_threshold_int": 500},
        }

        config = PipelineConfig.from_dict(d)
        assert config.project_name == "FromDict"
        assert config.gpu.mode == GPUMode.FORCE_GPU
        assert config.feature_finder.noise_threshold_int == 500


class TestPresets:
    """Tests for configuration presets."""

    def test_default_preset(self):
        """Test default preset."""
        config = get_default_config()
        assert config.feature_finder.noise_threshold_int == 1000

    def test_high_sensitivity_preset(self):
        """Test high sensitivity preset."""
        config = get_high_sensitivity_config()
        assert config.feature_finder.noise_threshold_int == 500
        assert config.feature_finder.chrom_peak_snr == 2.0

    def test_high_specificity_preset(self):
        """Test high specificity preset."""
        config = get_high_specificity_config()
        assert config.feature_finder.noise_threshold_int == 2000
        assert config.prioritization.scoring.min_score_threshold == 4.0


class TestBlankCorrectionConfig:
    """Tests for blank correction configuration."""

    def test_blank_correction_policy_enum(self):
        """Test blank correction policy enum values."""
        assert BlankCorrectionPolicy.NONE == "none"
        assert BlankCorrectionPolicy.SUBTRACT == "subtract"
        assert BlankCorrectionPolicy.FOLD_CHANGE == "fold_change"
        assert BlankCorrectionPolicy.PRESENCE == "presence"

    def test_default_blank_config(self):
        """Test default blank correction settings."""
        config = PipelineConfig()
        assert config.blank_correction.policy == BlankCorrectionPolicy.FOLD_CHANGE
        assert config.blank_correction.fold_change_threshold == 3.0


class TestGPUConfig:
    """Tests for GPU configuration."""

    def test_gpu_mode_enum(self):
        """Test GPU mode enum values."""
        assert GPUMode.AUTO == "auto"
        assert GPUMode.FORCE_GPU == "force_gpu"
        assert GPUMode.FORCE_CPU == "force_cpu"

    def test_gpu_thresholds(self):
        """Test GPU threshold defaults."""
        config = PipelineConfig()
        assert config.gpu.feature_count_threshold == 5000
        assert config.gpu.rt_grid_len_threshold == 50000
        assert config.gpu.correlation_pairs_threshold == 1000000

    def test_oom_settings(self):
        """Test OOM handling settings."""
        config = PipelineConfig()
        assert config.gpu.oom_max_retries == 3
        assert config.gpu.oom_chunk_reduction_factor == 0.5


class TestKMDConfig:
    """Tests for KMD configuration."""

    def test_default_repeat_units(self):
        """Test default repeat units."""
        config = PipelineConfig()
        assert "CF2" in config.prioritization.kmd.repeat_units
        assert "CF2O" in config.prioritization.kmd.repeat_units

    def test_repeat_unit_masses(self):
        """Test repeat unit mass values."""
        config = PipelineConfig()
        masses = config.prioritization.kmd.repeat_unit_masses

        # Check CF2 mass is approximately correct
        assert 49.99 < masses["CF2"] < 50.00

        # Check CF2O mass
        assert 65.99 < masses["CF2O"] < 66.00
