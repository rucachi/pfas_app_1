"""
Dataset builder for ONFRA PFAS ML training.

Generates training datasets from prioritization results in formats suitable
for deep learning: Parquet (features), NPY/NPZ (EIC/MS2), MGF (spectra).
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    
    output_dir: Path
    eic_length: int = 256           # EIC time series length (resampled)
    ms2_bins: int = 2000            # MS2 binned vector size
    ms2_max_mz: float = 1000.0      # Max m/z for MS2 binning
    ms2_min_mz: float = 50.0        # Min m/z for MS2 binning
    include_mgf: bool = True        # Output MGF format
    include_binned: bool = True     # Output binned NPZ format
    labeling_strategy: str = "auto" # auto | manual
    positive_score_threshold: float = 5.0   # pfas_score for positive label
    negative_score_threshold: float = 1.0   # pfas_score for negative label
    blank_ratio_threshold: float = 3.0      # blank ratio for negative
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class DatasetStats:
    """Statistics for generated dataset."""
    
    total_features: int = 0
    positive_count: int = 0
    negative_count: int = 0
    unknown_count: int = 0
    with_ms2_count: int = 0
    with_eic_count: int = 0


class DatasetBuilder:
    """
    Build ML training datasets from ONFRA PFAS results.
    
    Generates:
    - features.parquet: Feature metadata
    - eic/feature_*.npy: EIC time series
    - ms2/feature_*.mgf: MS2 spectra (MGF format)
    - ms2/feature_*.npz: MS2 binned vectors
    - labels.json: Classification/regression labels
    - manifest.json: Dataset metadata
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.stats = DatasetStats()
        self._labels: list[dict] = []
    
    def build(
        self,
        features_df: pd.DataFrame,
        viz_payload: dict | None = None,
        mzml_path: Path | None = None,
        manual_labels: dict[int, str] | None = None,
    ) -> Path:
        """
        Build dataset from prioritization results.
        
        Args:
            features_df: Prioritization result DataFrame
            viz_payload: Optional visualization payload with EIC/MS2 data
            mzml_path: Optional path to original mzML file
            manual_labels: Optional manual labels {feature_id: "positive"|"negative"}
            
        Returns:
            Path to output directory
        """
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "eic").mkdir(exist_ok=True)
        (output_dir / "ms2").mkdir(exist_ok=True)
        
        logger.info(f"Building dataset with {len(features_df)} features")
        
        # Reset stats
        self.stats = DatasetStats(total_features=len(features_df))
        self._labels = []
        
        # Process features
        feature_records = []
        
        for idx, row in features_df.iterrows():
            feature_id = int(row.get("feature_id", idx))
            record = self._process_feature(
                feature_id=feature_id,
                row=row,
                viz_payload=viz_payload,
                manual_labels=manual_labels,
            )
            feature_records.append(record)
        
        # Save features.parquet
        features_out_df = pd.DataFrame(feature_records)
        parquet_path = output_dir / "features.parquet"
        features_out_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved {len(features_out_df)} features to {parquet_path}")
        
        # Save labels.json
        labels_data = {
            "version": "1.0",
            "labeling_strategy": self.config.labeling_strategy,
            "labels": self._labels,
        }
        labels_path = output_dir / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, indent=2, ensure_ascii=False)
        
        # Generate manifest
        self._save_manifest(output_dir, features_df, mzml_path)
        
        logger.info(
            f"Dataset complete: {self.stats.positive_count} positive, "
            f"{self.stats.negative_count} negative, {self.stats.unknown_count} unknown"
        )
        
        return output_dir
    
    def _process_feature(
        self,
        feature_id: int,
        row: pd.Series,
        viz_payload: dict | None,
        manual_labels: dict[int, str] | None,
    ) -> dict:
        """Process single feature."""
        record = {
            "feature_id": int(feature_id),
            "mz": float(row.get("mz", 0)),
            "rt": float(row.get("rt", 0)),
            "rt_min": float(row.get("rt_min", row.get("rt", 0) - 30)),
            "rt_max": float(row.get("rt_max", row.get("rt", 0) + 30)),
            "intensity": float(row.get("intensity", 0)),
            "charge": int(row.get("charge", 1)),
            "isotope_count": int(row.get("isotope_count", 0)),
            "blank_ratio": float(row.get("blank_ratio", 0)),
            "pfas_score": float(row.get("pfas_score", 0)),
            "evidence_types": str(row.get("evidence_types", "")),
            "kmd_series_id": row.get("kmd_series_id"),
            "has_ms2": False,
            "ms2_peaks_count": 0,
            "eic_path": "",
            "ms2_path": "",
        }
        
        # Extract and save EIC
        eic_data = self._extract_eic(feature_id, row, viz_payload)
        if eic_data is not None:
            eic_path = self.config.output_dir / "eic" / f"feature_{feature_id}.npy"
            np.save(eic_path, eic_data)
            record["eic_path"] = f"eic/feature_{feature_id}.npy"
            self.stats.with_eic_count += 1
        
        # Extract and save MS2
        ms2_data = self._extract_ms2(feature_id, row, viz_payload)
        if ms2_data is not None and len(ms2_data) > 0:
            record["has_ms2"] = True
            record["ms2_peaks_count"] = len(ms2_data)
            self.stats.with_ms2_count += 1
            
            # Save MGF format
            if self.config.include_mgf:
                mgf_path = self.config.output_dir / "ms2" / f"feature_{feature_id}.mgf"
                self._save_mgf(mgf_path, feature_id, row, ms2_data)
                record["ms2_path"] = f"ms2/feature_{feature_id}.mgf"
            
            # Save binned format
            if self.config.include_binned:
                binned = self._bin_ms2(ms2_data)
                npz_path = self.config.output_dir / "ms2" / f"feature_{feature_id}.npz"
                np.savez_compressed(npz_path, spectrum=binned, peaks=ms2_data)
        
        # Generate label
        label = self._generate_label(feature_id, row, manual_labels)
        self._labels.append(label)
        
        # Update stats
        if label["classification"] == "positive":
            self.stats.positive_count += 1
        elif label["classification"] == "negative":
            self.stats.negative_count += 1
        else:
            self.stats.unknown_count += 1
        
        return record
    
    def _extract_eic(
        self,
        feature_id: int,
        row: pd.Series,
        viz_payload: dict | None,
    ) -> np.ndarray | None:
        """Extract EIC time series for feature."""
        eic_data = None
        
        # Try to get EIC from viz_payload if available
        if viz_payload is not None:
            eic_data = viz_payload.get("eic", {}).get(str(feature_id))
            if eic_data is None:
                eic_data = viz_payload.get("eic", {}).get(feature_id)
        
        if eic_data is None:
            # Generate synthetic EIC (Gaussian peak shape)
            rt = row.get("rt", 300)
            rt_min = row.get("rt_min", rt - 30)
            rt_max = row.get("rt_max", rt + 30)
            intensity = row.get("intensity", 10000)
            
            times = np.linspace(rt_min, rt_max, self.config.eic_length)
            sigma = (rt_max - rt_min) / 6  # ~3 sigma coverage
            eic = intensity * np.exp(-0.5 * ((times - rt) / sigma) ** 2)
            return eic.astype(np.float32)
        
        # Resample EIC to fixed length
        if isinstance(eic_data, dict):
            times = np.array(eic_data.get("times", []))
            intensities = np.array(eic_data.get("intensities", []))
        elif isinstance(eic_data, (list, np.ndarray)):
            intensities = np.array(eic_data)
            times = np.arange(len(intensities))
        else:
            return None
        
        if len(intensities) == 0:
            return None
        
        # Resample to fixed length
        if len(intensities) != self.config.eic_length:
            x_old = np.linspace(0, 1, len(intensities))
            x_new = np.linspace(0, 1, self.config.eic_length)
            intensities = np.interp(x_new, x_old, intensities)
        
        return intensities.astype(np.float32)
    
    def _extract_ms2(
        self,
        feature_id: int,
        row: pd.Series,
        viz_payload: dict | None,
    ) -> np.ndarray | None:
        """Extract MS2 peak list for feature."""
        # Try from row first
        ms2_mz = row.get("ms2_mz") or row.get("ms2_peaks_mz")
        ms2_int = row.get("ms2_intensity") or row.get("ms2_peaks_intensity")
        
        if ms2_mz is not None and ms2_int is not None:
            if isinstance(ms2_mz, str):
                ms2_mz = [float(x) for x in ms2_mz.split(",") if x.strip()]
                ms2_int = [float(x) for x in str(ms2_int).split(",") if x.strip()]
            
            if len(ms2_mz) > 0:
                return np.column_stack([ms2_mz, ms2_int]).astype(np.float32)
        
        # Try from viz_payload
        if viz_payload is not None:
            ms2_data = viz_payload.get("ms2", {}).get(str(feature_id))
            if ms2_data is None:
                ms2_data = viz_payload.get("ms2", {}).get(feature_id)
            
            if ms2_data is not None:
                if isinstance(ms2_data, dict):
                    mzs = np.array(ms2_data.get("mz", []))
                    ints = np.array(ms2_data.get("intensity", []))
                    if len(mzs) > 0:
                        return np.column_stack([mzs, ints]).astype(np.float32)
                elif isinstance(ms2_data, np.ndarray):
                    return ms2_data.astype(np.float32)
        
        return None
    
    def _bin_ms2(self, peaks: np.ndarray) -> np.ndarray:
        """Convert MS2 peaks to binned vector."""
        binned = np.zeros(self.config.ms2_bins, dtype=np.float32)
        
        if len(peaks) == 0:
            return binned
        
        mzs = peaks[:, 0]
        ints = peaks[:, 1]
        
        # Normalize intensities
        max_int = ints.max()
        if max_int > 0:
            ints = ints / max_int * 100
        
        # Calculate bin indices
        bin_width = (self.config.ms2_max_mz - self.config.ms2_min_mz) / self.config.ms2_bins
        indices = ((mzs - self.config.ms2_min_mz) / bin_width).astype(int)
        
        # Filter valid indices
        valid = (indices >= 0) & (indices < self.config.ms2_bins)
        
        # Assign to bins (max pooling for overlaps)
        for idx, intensity in zip(indices[valid], ints[valid]):
            binned[idx] = max(binned[idx], intensity)
        
        return binned
    
    def _save_mgf(
        self,
        path: Path,
        feature_id: int,
        row: pd.Series,
        peaks: np.ndarray,
    ) -> None:
        """Save MS2 spectrum in MGF format."""
        mz = row.get("mz", 0)
        rt = row.get("rt", 0)
        charge = row.get("charge", 1)
        
        with open(path, "w") as f:
            f.write("BEGIN IONS\n")
            f.write(f"TITLE=Feature_{feature_id}\n")
            f.write(f"PEPMASS={mz:.6f}\n")
            f.write(f"RTINSECONDS={rt:.2f}\n")
            f.write(f"CHARGE={charge}+\n")
            
            for peak_mz, peak_int in peaks:
                f.write(f"{peak_mz:.6f} {peak_int:.2f}\n")
            
            f.write("END IONS\n")
    
    def _generate_label(
        self,
        feature_id: int,
        row: pd.Series,
        manual_labels: dict[int, str] | None,
    ) -> dict:
        """Generate label for feature."""
        label = {
            "feature_id": int(feature_id),
            "classification": "unknown",
            "confidence": "low",
            "rationale": "",
        }
        
        # Check manual labels first
        if manual_labels and feature_id in manual_labels:
            label["classification"] = manual_labels[feature_id]
            label["confidence"] = "high"
            label["rationale"] = "manual_label"
            return label
        
        # Auto-labeling
        pfas_score = row.get("pfas_score", 0)
        blank_ratio = row.get("blank_ratio", 0)
        evidence_types = str(row.get("evidence_types", ""))
        suspect_match = "suspect" in evidence_types.lower()
        
        # Strong positive: high score + suspect match
        if pfas_score >= self.config.positive_score_threshold or suspect_match:
            label["classification"] = "positive"
            label["confidence"] = "high" if suspect_match else "medium"
            rationale_parts = []
            if suspect_match:
                rationale_parts.append("suspect_match")
            if pfas_score >= self.config.positive_score_threshold:
                rationale_parts.append(f"pfas_score:{pfas_score:.1f}")
            label["rationale"] = ", ".join(rationale_parts)
        
        # Strong negative: low score + blank dominant
        elif pfas_score <= self.config.negative_score_threshold:
            if blank_ratio >= self.config.blank_ratio_threshold:
                label["classification"] = "negative"
                label["confidence"] = "high"
                label["rationale"] = f"blank_dominant:ratio={blank_ratio:.1f}"
            elif pfas_score == 0 and not evidence_types:
                label["classification"] = "negative"
                label["confidence"] = "medium"
                label["rationale"] = "no_pfas_evidence"
        
        # Unknown: gray zone (useful for contrastive learning)
        else:
            label["classification"] = "unknown"
            label["confidence"] = "low"
            label["rationale"] = f"gray_zone:score={pfas_score:.1f}"
        
        return label
    
    def _save_manifest(
        self,
        output_dir: Path,
        features_df: pd.DataFrame,
        mzml_path: Path | None,
    ) -> None:
        """Save dataset manifest."""
        # Calculate data hash
        parquet_path = output_dir / "features.parquet"
        if parquet_path.exists():
            with open(parquet_path, "rb") as f:
                data_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        else:
            data_hash = "unknown"
        
        manifest = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_files": [str(mzml_path)] if mzml_path else [],
            "config": {
                "eic_length": self.config.eic_length,
                "ms2_bins": self.config.ms2_bins,
                "ms2_max_mz": self.config.ms2_max_mz,
                "labeling_strategy": self.config.labeling_strategy,
                "positive_threshold": self.config.positive_score_threshold,
                "negative_threshold": self.config.negative_score_threshold,
            },
            "stats": {
                "total_features": self.stats.total_features,
                "positive_count": self.stats.positive_count,
                "negative_count": self.stats.negative_count,
                "unknown_count": self.stats.unknown_count,
                "with_ms2_count": self.stats.with_ms2_count,
                "with_eic_count": self.stats.with_eic_count,
            },
            "data_hash": f"sha256:{data_hash}",
            "model_compatible": ["ONFRA-PFAS-v1"],
        }
        
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved manifest to {manifest_path}")


def build_dataset_from_results(
    features_df: pd.DataFrame,
    output_dir: str | Path,
    viz_payload: dict | None = None,
    mzml_path: str | Path | None = None,
    **config_kwargs,
) -> Path:
    """
    Convenience function to build dataset.
    
    Args:
        features_df: Prioritization result DataFrame
        output_dir: Output directory path
        viz_payload: Optional visualization payload
        mzml_path: Optional mzML file path
        **config_kwargs: Additional DatasetConfig parameters
        
    Returns:
        Path to output directory
    """
    config = DatasetConfig(output_dir=Path(output_dir), **config_kwargs)
    builder = DatasetBuilder(config)
    
    return builder.build(
        features_df=features_df,
        viz_payload=viz_payload,
        mzml_path=Path(mzml_path) if mzml_path else None,
    )
