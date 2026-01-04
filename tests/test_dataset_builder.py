"""
Tests for dataset builder module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from onfra_pfas.core.dataset_builder import (
    DatasetConfig,
    DatasetStats,
    DatasetBuilder,
    build_dataset_from_results,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_creation(self):
        """Test creating a DatasetConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir)
            
            assert config.output_dir == Path(tmpdir)
            assert config.eic_length == 256
            assert config.ms2_bins == 2000
            assert config.labeling_strategy == "auto"

    def test_path_conversion(self):
        """Test that string path is converted to Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir)
            assert isinstance(config.output_dir, Path)

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                output_dir=tmpdir,
                positive_score_threshold=7.0,
                negative_score_threshold=0.5,
            )
            assert config.positive_score_threshold == 7.0
            assert config.negative_score_threshold == 0.5


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = DatasetStats()
        
        assert stats.total_features == 0
        assert stats.positive_count == 0
        assert stats.negative_count == 0
        assert stats.unknown_count == 0
        assert stats.with_ms2_count == 0
        assert stats.with_eic_count == 0


class TestDatasetBuilder:
    """Tests for DatasetBuilder class."""

    def test_build_empty_dataframe(self):
        """Test building from empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir)
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame()
            output = builder.build(df)
            
            assert output == Path(tmpdir)
            assert (output / "features.parquet").exists()
            assert (output / "labels.json").exists()
            assert (output / "manifest.json").exists()

    def test_build_with_features(self):
        """Test building with sample features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir)
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame([
                {"feature_id": 1, "mz": 413.0, "rt": 120.0, "intensity": 50000, "pfas_score": 6.0},
                {"feature_id": 2, "mz": 463.0, "rt": 150.0, "intensity": 30000, "pfas_score": 0.5},
            ])
            
            output = builder.build(df)
            
            # Check parquet
            features_out = pd.read_parquet(output / "features.parquet")
            assert len(features_out) == 2
            
            # Check labels
            with open(output / "labels.json") as f:
                labels = json.load(f)
            assert len(labels["labels"]) == 2

    def test_labeling_positive(self):
        """Test automatic positive labeling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir, positive_score_threshold=5.0)
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame([
                {"feature_id": 1, "mz": 413.0, "rt": 120.0, "pfas_score": 6.0},
            ])
            
            builder.build(df)
            
            assert builder.stats.positive_count == 1
            assert builder.stats.negative_count == 0

    def test_labeling_negative(self):
        """Test automatic negative labeling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                output_dir=tmpdir,
                negative_score_threshold=1.0,
                blank_ratio_threshold=3.0,
            )
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame([
                {"feature_id": 1, "mz": 100.0, "rt": 60.0, "pfas_score": 0.5, "blank_ratio": 5.0},
            ])
            
            builder.build(df)
            
            assert builder.stats.negative_count == 1
            assert builder.stats.positive_count == 0

    def test_labeling_suspect_match(self):
        """Test suspect match triggers positive label."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir)
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame([
                {"feature_id": 1, "mz": 413.0, "rt": 120.0, "pfas_score": 2.0, "evidence_types": "suspect"},
            ])
            
            builder.build(df)
            
            assert builder.stats.positive_count == 1

    def test_manual_labels(self):
        """Test manual label override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir)
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame([
                {"feature_id": 1, "mz": 413.0, "rt": 120.0, "pfas_score": 6.0},
                {"feature_id": 2, "mz": 463.0, "rt": 150.0, "pfas_score": 6.0},
            ])
            
            # Override feature 1 as negative
            manual_labels = {1: "negative"}
            
            builder.build(df, manual_labels=manual_labels)
            
            assert builder.stats.negative_count == 1
            assert builder.stats.positive_count == 1

    def test_eic_generation(self):
        """Test EIC synthetic generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir, eic_length=128)
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame([
                {"feature_id": 1, "mz": 413.0, "rt": 120.0, "rt_min": 100.0, "rt_max": 140.0, "intensity": 50000},
            ])
            
            builder.build(df)
            
            # Check EIC file
            eic_path = Path(tmpdir) / "eic" / "feature_1.npy"
            assert eic_path.exists()
            
            eic = np.load(eic_path)
            assert len(eic) == 128
            assert eic.max() > 0

    def test_ms2_binning(self):
        """Test MS2 binning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                output_dir=tmpdir,
                ms2_bins=100,
                ms2_min_mz=50.0,
                ms2_max_mz=150.0,
            )
            builder = DatasetBuilder(config)
            
            # Create MS2 peaks
            peaks = np.array([[100.0, 1000.0], [120.0, 500.0]])
            
            binned = builder._bin_ms2(peaks)
            
            assert len(binned) == 100
            assert binned.max() == 100.0  # Normalized to 100

    def test_manifest_creation(self):
        """Test manifest.json creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(output_dir=tmpdir)
            builder = DatasetBuilder(config)
            
            df = pd.DataFrame([{"feature_id": 1, "mz": 413.0, "rt": 120.0}])
            builder.build(df)
            
            with open(Path(tmpdir) / "manifest.json") as f:
                manifest = json.load(f)
            
            assert manifest["version"] == "1.0"
            assert "created_at" in manifest
            assert manifest["stats"]["total_features"] == 1


class TestBuildDatasetFromResults:
    """Tests for build_dataset_from_results convenience function."""

    def test_convenience_function(self):
        """Test the convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame([
                {"feature_id": 1, "mz": 413.0, "rt": 120.0, "pfas_score": 6.0},
            ])
            
            output = build_dataset_from_results(
                features_df=df,
                output_dir=tmpdir,
                eic_length=64,
            )
            
            assert output == Path(tmpdir)
            assert (output / "features.parquet").exists()
