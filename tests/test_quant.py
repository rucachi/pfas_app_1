"""
Tests for quantification module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from onfra_pfas.core.quant import (
    QuantMode,
    QuantResult,
    CalibrationPoint,
    CalibrationCurve,
    ResponseFactorPredictor,
    QuantEstimator,
    add_quant_to_dataframe,
)


class TestQuantMode:
    """Tests for QuantMode enum."""

    def test_mode_values(self):
        """Test enum values."""
        assert QuantMode.NONE.value == "none"
        assert QuantMode.QUANTITATIVE.value == "quantitative"
        assert QuantMode.SEMI_QUANTITATIVE.value == "semi-quantitative"


class TestQuantResult:
    """Tests for QuantResult dataclass."""

    def test_creation(self):
        """Test creating QuantResult."""
        result = QuantResult(
            mode=QuantMode.QUANTITATIVE,
            value=150.0,
            unit="ng/L",
            uncertainty=20.0,
        )
        
        assert result.mode == QuantMode.QUANTITATIVE
        assert result.value == 150.0
        assert result.unit == "ng/L"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = QuantResult(
            mode=QuantMode.SEMI_QUANTITATIVE,
            value=100.0,
            fold_error="3x",
            confidence_interval=(33.0, 300.0),
        )
        d = result.to_dict()
        
        assert d["quant_mode"] == "semi-quantitative"
        assert d["quant_value"] == 100.0
        assert d["quant_fold_error"] == "3x"
        assert d["quant_ci_low"] == 33.0
        assert d["quant_ci_high"] == 300.0


class TestCalibrationPoint:
    """Tests for CalibrationPoint dataclass."""

    def test_creation(self):
        """Test creating CalibrationPoint."""
        point = CalibrationPoint(concentration=100.0, response=500000.0)
        
        assert point.concentration == 100.0
        assert point.response == 500000.0


class TestCalibrationCurve:
    """Tests for CalibrationCurve class."""

    def test_creation(self):
        """Test creating CalibrationCurve."""
        curve = CalibrationCurve(compound_name="PFOA")
        
        assert curve.compound_name == "PFOA"
        assert curve.slope == 0.0
        assert len(curve.points) == 0

    def test_fit_linear(self):
        """Test fitting calibration curve."""
        curve = CalibrationCurve(compound_name="PFOA")
        curve.points = [
            CalibrationPoint(10, 50000),
            CalibrationPoint(100, 500000),
            CalibrationPoint(1000, 5000000),
        ]
        
        curve.fit()
        
        assert curve.slope > 0
        assert curve.r_squared > 0.99

    def test_fit_too_few_points(self):
        """Test fitting with too few points."""
        curve = CalibrationCurve(compound_name="PFOA")
        curve.points = [CalibrationPoint(10, 50000)]
        
        curve.fit()
        
        assert curve.slope == 0.0

    def test_predict(self):
        """Test predicting concentration."""
        curve = CalibrationCurve(compound_name="PFOA")
        curve.points = [
            CalibrationPoint(10, 50000),
            CalibrationPoint(100, 500000),
            CalibrationPoint(1000, 5000000),
        ]
        curve.fit()
        
        conc, uncertainty = curve.predict(250000)
        
        assert conc > 0
        assert 30 < conc < 70  # Should be around 50

    def test_predict_no_fit(self):
        """Test predicting without fitting."""
        curve = CalibrationCurve(compound_name="PFOA")
        
        conc, uncertainty = curve.predict(100000)
        
        assert conc == 0.0
        assert uncertainty == float("inf")


class TestResponseFactorPredictor:
    """Tests for ResponseFactorPredictor class."""

    def test_creation(self):
        """Test creating predictor."""
        predictor = ResponseFactorPredictor()
        
        assert len(predictor.DEFAULT_RF) > 0

    def test_predict_pfca(self):
        """Test predicting PFCA response factor."""
        predictor = ResponseFactorPredictor()
        
        rf, fold_error = predictor.predict_rf(413.0, compound_class="PFCA")
        
        assert rf == 1.0
        assert fold_error == 2.0

    def test_predict_pfsa(self):
        """Test predicting PFSA response factor."""
        predictor = ResponseFactorPredictor()
        
        rf, fold_error = predictor.predict_rf(499.0, compound_class="PFSA")
        
        assert rf == 1.2
        assert fold_error == 2.0

    def test_infer_class_from_formula(self):
        """Test inferring class from formula."""
        predictor = ResponseFactorPredictor()
        
        # Sulfonic acid
        assert predictor._infer_class(499.0, "C8HF17SO3") == "PFSA"
        # Carboxylic acid
        assert predictor._infer_class(413.0, "C8HF15O2") == "PFCA"
        # Phosphate
        assert predictor._infer_class(500.0, "C10H8F17O4P") == "diPAP"

    def test_high_mz_adjustment(self):
        """Test adjustment for high m/z."""
        predictor = ResponseFactorPredictor()
        
        rf_low, _ = predictor.predict_rf(300.0, compound_class="PFCA")
        rf_high, _ = predictor.predict_rf(600.0, compound_class="PFCA")
        
        assert rf_high < rf_low


class TestQuantEstimator:
    """Tests for QuantEstimator class."""

    def test_creation(self):
        """Test creating estimator."""
        estimator = QuantEstimator()
        
        assert estimator.mode == QuantMode.NONE

    def test_set_mode(self):
        """Test setting mode."""
        estimator = QuantEstimator()
        estimator.set_mode(QuantMode.SEMI_QUANTITATIVE)
        
        assert estimator.mode == QuantMode.SEMI_QUANTITATIVE

    def test_estimate_none_mode(self):
        """Test estimation in NONE mode."""
        estimator = QuantEstimator(QuantMode.NONE)
        
        result = estimator.estimate(100000, 413.0)
        
        assert result.mode == QuantMode.NONE
        assert result.value is None

    def test_estimate_semi_quantitative(self):
        """Test semi-quantitative estimation."""
        estimator = QuantEstimator(QuantMode.SEMI_QUANTITATIVE)
        estimator.set_reference(1e6, 100.0)
        
        result = estimator.estimate(5e5, 413.0, formula="C8HF15O2")
        
        assert result.mode == QuantMode.SEMI_QUANTITATIVE
        assert result.value is not None
        assert result.value > 0
        assert result.fold_error is not None

    def test_estimate_quantitative_no_calibration(self):
        """Test quantitative estimation without calibration."""
        estimator = QuantEstimator(QuantMode.QUANTITATIVE)
        
        result = estimator.estimate(100000, 413.0, compound_name="PFOA")
        
        assert result.mode == QuantMode.QUANTITATIVE
        assert result.value is None

    def test_load_calibration(self):
        """Test loading calibration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create calibration CSV
            cal_path = Path(tmpdir) / "calibration.csv"
            cal_df = pd.DataFrame([
                {"compound": "PFOA", "concentration": 10, "response": 50000},
                {"compound": "PFOA", "concentration": 100, "response": 500000},
                {"compound": "PFOA", "concentration": 1000, "response": 5000000},
            ])
            cal_df.to_csv(cal_path, index=False)
            
            estimator = QuantEstimator(QuantMode.QUANTITATIVE)
            count = estimator.load_calibration(cal_path)
            
            assert count == 1
            assert "PFOA" in estimator._calibrations

    def test_estimate_with_calibration(self):
        """Test quantitative estimation with calibration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = Path(tmpdir) / "calibration.csv"
            cal_df = pd.DataFrame([
                {"compound": "PFOA", "concentration": 10, "response": 50000},
                {"compound": "PFOA", "concentration": 100, "response": 500000},
                {"compound": "PFOA", "concentration": 1000, "response": 5000000},
            ])
            cal_df.to_csv(cal_path, index=False)
            
            estimator = QuantEstimator(QuantMode.QUANTITATIVE)
            estimator.load_calibration(cal_path)
            
            result = estimator.estimate(250000, 413.0, compound_name="PFOA")
            
            assert result.mode == QuantMode.QUANTITATIVE
            assert result.value is not None
            assert 30 < result.value < 70

    def test_set_reference(self):
        """Test setting reference values."""
        estimator = QuantEstimator(QuantMode.SEMI_QUANTITATIVE)
        estimator.set_reference(2e6, 200.0)
        
        assert estimator._reference_intensity == 2e6
        assert estimator._reference_concentration == 200.0

    def test_calc_fold_error(self):
        """Test fold error calculation."""
        assert QuantEstimator._calc_fold_error(10, 100) == "2x"
        assert QuantEstimator._calc_fold_error(100, 100) == "5x"
        assert QuantEstimator._calc_fold_error(500, 100) == ">10x"


class TestAddQuantToDataframe:
    """Tests for add_quant_to_dataframe function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = add_quant_to_dataframe(df)
        
        assert "quant_mode" in result.columns
        assert "quant_value" in result.columns

    def test_none_mode(self):
        """Test with NONE mode."""
        df = pd.DataFrame([
            {"mz": 413.0, "intensity": 100000},
        ])
        
        result = add_quant_to_dataframe(df, mode=QuantMode.NONE)
        
        assert result["quant_mode"].iloc[0] == "none"
        assert pd.isna(result["quant_value"].iloc[0]) or result["quant_value"].iloc[0] is None

    def test_semi_quantitative_mode(self):
        """Test with semi-quantitative mode."""
        df = pd.DataFrame([
            {"mz": 413.0, "intensity": 500000},
            {"mz": 499.0, "intensity": 300000},
        ])
        
        estimator = QuantEstimator(QuantMode.SEMI_QUANTITATIVE)
        estimator.set_reference(1e6, 100.0)
        
        result = add_quant_to_dataframe(df, estimator=estimator)
        
        assert all(result["quant_mode"] == "semi-quantitative")
        assert all(result["quant_value"] > 0)

    def test_preserves_original(self):
        """Test that original columns are preserved."""
        df = pd.DataFrame([
            {"mz": 413.0, "intensity": 100000, "custom": "value"},
        ])
        
        result = add_quant_to_dataframe(df, mode=QuantMode.NONE)
        
        assert "custom" in result.columns
        assert result["custom"].iloc[0] == "value"
