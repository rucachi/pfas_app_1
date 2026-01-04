"""
Quantification module for ONFRA PFAS.

Provides quantitative and semi-quantitative concentration estimation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QuantMode(Enum):
    """Quantification mode."""
    
    NONE = "none"
    QUANTITATIVE = "quantitative"        # Calibration curve based
    SEMI_QUANTITATIVE = "semi-quantitative"  # Response factor estimation


@dataclass
class QuantResult:
    """Quantification result for a feature."""
    
    mode: QuantMode
    value: float | None = None          # Concentration value
    unit: str = "ng/L"                   # Concentration unit
    uncertainty: float | None = None    # Uncertainty (std)
    fold_error: str | None = None       # "2x", "5x"
    confidence_interval: tuple[float, float] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "quant_mode": self.mode.value,
            "quant_value": self.value,
            "quant_unit": self.unit,
            "quant_uncertainty": self.uncertainty,
            "quant_fold_error": self.fold_error,
            "quant_ci_low": self.confidence_interval[0] if self.confidence_interval else None,
            "quant_ci_high": self.confidence_interval[1] if self.confidence_interval else None,
        }


@dataclass
class CalibrationPoint:
    """Calibration curve data point."""
    
    concentration: float    # Known concentration
    response: float         # Measured response (area/intensity)
    name: str = ""          # Compound name
    

@dataclass
class CalibrationCurve:
    """Calibration curve for quantification."""
    
    compound_name: str
    points: list[CalibrationPoint] = field(default_factory=list)
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    unit: str = "ng/L"
    
    def fit(self) -> None:
        """Fit linear regression to calibration points."""
        if len(self.points) < 2:
            logger.warning(f"Not enough points for {self.compound_name}")
            return
        
        # Log-log regression for better linearity
        x = np.array([np.log10(p.concentration + 1e-10) for p in self.points])
        y = np.array([np.log10(p.response + 1e-10) for p in self.points])
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return
        
        self.slope = (n * sum_xy - sum_x * sum_y) / denom
        self.intercept = (sum_y - self.slope * sum_x) / n
        
        # Calculate R²
        y_pred = self.slope * x + self.intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        logger.info(
            f"Calibration {self.compound_name}: slope={self.slope:.3f}, "
            f"intercept={self.intercept:.3f}, R²={self.r_squared:.4f}"
        )
    
    def predict(self, response: float) -> tuple[float, float]:
        """
        Predict concentration from response.
        
        Returns:
            (concentration, uncertainty)
        """
        if self.slope == 0:
            return 0.0, float("inf")
        
        log_response = np.log10(response + 1e-10)
        log_conc = (log_response - self.intercept) / self.slope
        concentration = 10 ** log_conc
        
        # Simple uncertainty based on R²
        uncertainty = concentration * (1 - self.r_squared) * 0.5
        
        return float(concentration), float(uncertainty)


class ResponseFactorPredictor:
    """
    Predict response factors for semi-quantitative analysis.
    
    Uses molecular properties to estimate ionization efficiency.
    """
    
    # Default response factors for PFAS classes (relative to PFOA)
    DEFAULT_RF = {
        "PFCA": 1.0,    # Carboxylic acids
        "PFSA": 1.2,    # Sulfonic acids
        "FTOH": 0.5,    # Fluorotelomer alcohols
        "FOSA": 0.8,    # Sulfonamides
        "diPAP": 0.3,   # Phosphates
        "other": 0.7,   # Unknown PFAS
    }
    
    # Uncertainty factors (fold-error)
    RF_UNCERTAINTY = {
        "PFCA": 2.0,
        "PFSA": 2.0,
        "FTOH": 3.0,
        "FOSA": 2.5,
        "diPAP": 5.0,
        "other": 5.0,
    }
    
    def __init__(self):
        self._custom_rf: dict[str, float] = {}
    
    def predict_rf(
        self,
        mz: float,
        formula: str | None = None,
        compound_class: str | None = None,
    ) -> tuple[float, float]:
        """
        Predict response factor and uncertainty.
        
        Args:
            mz: Precursor m/z
            formula: Molecular formula (optional)
            compound_class: PFAS class (optional)
            
        Returns:
            (response_factor, fold_error)
        """
        # Determine class from formula or name
        if compound_class is None:
            compound_class = self._infer_class(mz, formula)
        
        rf = self.DEFAULT_RF.get(compound_class, self.DEFAULT_RF["other"])
        fold_error = self.RF_UNCERTAINTY.get(compound_class, 5.0)
        
        # Adjust for m/z (larger molecules tend to have lower response)
        if mz > 500:
            rf *= 0.8
            fold_error *= 1.2
        elif mz < 300:
            rf *= 1.1
        
        return float(rf), float(fold_error)
    
    def _infer_class(self, mz: float, formula: str | None) -> str:
        """Infer PFAS class from properties."""
        if formula:
            formula_upper = formula.upper()
            if "S" in formula_upper and "N" in formula_upper:
                return "FOSA"
            elif "S" in formula_upper:
                return "PFSA"
            elif "P" in formula_upper:
                return "diPAP"
            else:
                return "PFCA"
        
        # Infer from m/z patterns
        if mz in [413.0, 463.0, 513.0, 563.0]:  # PFOA, PFNA, PFDA, PFUnDA
            return "PFCA"
        elif mz in [499.0, 399.0, 299.0]:  # PFOS, PFHxS, PFBS
            return "PFSA"
        
        return "other"
    
    def set_custom_rf(self, compound: str, rf: float) -> None:
        """Set custom response factor for a compound."""
        self._custom_rf[compound] = rf


class QuantEstimator:
    """
    Concentration estimator supporting both quantitative and semi-quantitative modes.
    """
    
    def __init__(self, mode: QuantMode = QuantMode.NONE):
        self.mode = mode
        self._calibrations: dict[str, CalibrationCurve] = {}
        self._default_calibration: CalibrationCurve | None = None
        self._rf_predictor = ResponseFactorPredictor()
        self._reference_intensity: float = 1e6  # Reference intensity for semi-quant
        self._reference_concentration: float = 100.0  # ng/L
    
    def set_mode(self, mode: QuantMode) -> None:
        """Set quantification mode."""
        self.mode = mode
    
    def load_calibration(self, path: str | Path) -> int:
        """
        Load calibration curves from file.
        
        Supports CSV format:
        compound,concentration,response
        PFOA,10,50000
        PFOA,100,500000
        ...
        
        Returns:
            Number of curves loaded
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"Calibration file not found: {path}")
            return 0
        
        df = pd.read_csv(path)
        
        # Group by compound
        for compound, group in df.groupby("compound"):
            curve = CalibrationCurve(compound_name=str(compound))
            
            for _, row in group.iterrows():
                point = CalibrationPoint(
                    concentration=float(row["concentration"]),
                    response=float(row["response"]),
                    name=str(compound),
                )
                curve.points.append(point)
            
            curve.fit()
            self._calibrations[str(compound)] = curve
        
        # Set first curve as default
        if self._calibrations:
            self._default_calibration = list(self._calibrations.values())[0]
        
        logger.info(f"Loaded {len(self._calibrations)} calibration curves")
        return len(self._calibrations)
    
    def estimate(
        self,
        intensity: float,
        mz: float,
        compound_name: str | None = None,
        formula: str | None = None,
    ) -> QuantResult:
        """
        Estimate concentration.
        
        Args:
            intensity: Peak intensity/area
            mz: Precursor m/z
            compound_name: Optional matched compound name
            formula: Optional molecular formula
            
        Returns:
            QuantResult
        """
        if self.mode == QuantMode.NONE:
            return QuantResult(mode=QuantMode.NONE)
        
        if self.mode == QuantMode.QUANTITATIVE:
            return self._quantitative_estimate(intensity, compound_name)
        else:
            return self._semi_quantitative_estimate(intensity, mz, formula)
    
    def _quantitative_estimate(
        self,
        intensity: float,
        compound_name: str | None,
    ) -> QuantResult:
        """Quantitative estimation using calibration curve."""
        # Find matching calibration
        curve = None
        if compound_name and compound_name in self._calibrations:
            curve = self._calibrations[compound_name]
        elif self._default_calibration:
            curve = self._default_calibration
        
        if curve is None:
            return QuantResult(
                mode=QuantMode.QUANTITATIVE,
                value=None,
                fold_error="N/A",
            )
        
        conc, uncertainty = curve.predict(intensity)
        
        # Calculate confidence interval
        ci_low = max(0, conc - 2 * uncertainty)
        ci_high = conc + 2 * uncertainty
        
        return QuantResult(
            mode=QuantMode.QUANTITATIVE,
            value=conc,
            unit=curve.unit,
            uncertainty=uncertainty,
            fold_error=self._calc_fold_error(uncertainty, conc),
            confidence_interval=(ci_low, ci_high),
        )
    
    def _semi_quantitative_estimate(
        self,
        intensity: float,
        mz: float,
        formula: str | None,
    ) -> QuantResult:
        """Semi-quantitative estimation using response factor prediction."""
        rf, fold_error = self._rf_predictor.predict_rf(mz, formula)
        
        # Estimate concentration relative to reference
        # C = (I / I_ref) * C_ref / RF
        conc = (intensity / self._reference_intensity) * self._reference_concentration / rf
        
        # Calculate uncertainty from fold error
        uncertainty = conc * (fold_error - 1) / 2
        
        # Confidence interval based on fold error
        ci_low = conc / fold_error
        ci_high = conc * fold_error
        
        return QuantResult(
            mode=QuantMode.SEMI_QUANTITATIVE,
            value=conc,
            unit="ng/L",
            uncertainty=uncertainty,
            fold_error=f"{fold_error:.0f}x",
            confidence_interval=(ci_low, ci_high),
        )
    
    @staticmethod
    def _calc_fold_error(uncertainty: float, value: float) -> str:
        """Calculate fold error string."""
        if value == 0:
            return "N/A"
        
        ratio = 1 + (2 * uncertainty / value)
        if ratio <= 2:
            return "2x"
        elif ratio <= 5:
            return "5x"
        elif ratio <= 10:
            return "10x"
        else:
            return ">10x"
    
    def set_reference(self, intensity: float, concentration: float) -> None:
        """Set reference values for semi-quantitative mode."""
        self._reference_intensity = intensity
        self._reference_concentration = concentration


def add_quant_to_dataframe(
    df: pd.DataFrame,
    estimator: QuantEstimator | None = None,
    mode: QuantMode = QuantMode.NONE,
) -> pd.DataFrame:
    """
    Add quantification columns to DataFrame.
    
    Args:
        df: Prioritization result DataFrame
        estimator: Optional QuantEstimator instance
        mode: Quantification mode if no estimator provided
        
    Returns:
        DataFrame with added quant columns
    """
    if len(df) == 0:
        for col in ["quant_mode", "quant_value", "quant_unit", "quant_uncertainty", "quant_fold_error"]:
            df[col] = pd.Series(dtype=object if col == "quant_mode" else float)
        return df
    
    if estimator is None:
        estimator = QuantEstimator(mode)
    
    df = df.copy()
    
    quant_results = []
    for _, row in df.iterrows():
        result = estimator.estimate(
            intensity=row.get("intensity", 0),
            mz=row.get("mz", 0),
            compound_name=row.get("suspect_name"),
            formula=row.get("formula"),
        )
        quant_results.append(result.to_dict())
    
    quant_df = pd.DataFrame(quant_results)
    for col in quant_df.columns:
        df[col] = quant_df[col]
    
    return df
