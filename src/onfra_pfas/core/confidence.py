"""
Confidence level calculator for ONFRA PFAS.

Implements Schymanski-based 5-level confidence system for compound identification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class ConfidenceLevel(IntEnum):
    """Schymanski-based identification confidence levels."""
    
    LEVEL_1 = 1  # Confirmed: Reference standard RT + MS2 match
    LEVEL_2 = 2  # Probable: MS2 library match (≥0.8 similarity)
    LEVEL_3 = 3  # Tentative: Diagnostic fragments + homologous series
    LEVEL_4 = 4  # Unequivocal Formula: KMD/MDC + suspect match
    LEVEL_5 = 5  # Exact Mass: m/z only

    @property
    def description(self) -> str:
        """Get human-readable description."""
        descriptions = {
            1: "Confirmed (표준물질 RT + MS2 일치)",
            2: "Probable (MS2 라이브러리 매칭)",
            3: "Tentative (진단조각 + 동족체)",
            4: "Unequivocal Formula (KMD/MDC + Suspect)",
            5: "Exact Mass (m/z만 일치)",
        }
        return descriptions.get(self.value, "Unknown")
    
    @property
    def description_en(self) -> str:
        """Get English description."""
        descriptions = {
            1: "Confirmed structure by reference standard",
            2: "Probable structure by MS2 library match",
            3: "Tentative candidates by diagnostic evidence",
            4: "Unequivocal molecular formula",
            5: "Exact mass of interest",
        }
        return descriptions.get(self.value, "Unknown")


@dataclass
class ConfidenceResult:
    """Result of confidence level calculation."""
    
    level: ConfidenceLevel
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "confidence_level": self.level.value,
            "confidence_level_name": self.level.name,
            "confidence_description": self.level.description,
            "confidence_rationale": self.rationale,
            "confidence_evidence": self.evidence,
        }


def calculate_confidence(
    feature: dict[str, Any],
    suspect_match: bool = False,
    suspect_name: str | None = None,
    ms2_similarity: float | None = None,
    ms2_match_count: int = 0,
    rt_match: bool = False,
    rt_error: float | None = None,
    kmd_series: bool = False,
    kmd_series_size: int = 0,
    mdc_region: bool = False,
    df_matches: list[str] | None = None,
    delta_m_matches: list[str] | None = None,
    has_reference_standard: bool = False,
) -> ConfidenceResult:
    """
    Calculate Schymanski-based confidence level.
    
    Args:
        feature: Feature dictionary with mz, rt, intensity, etc.
        suspect_match: Whether feature matched a suspect compound
        suspect_name: Name of matched suspect compound
        ms2_similarity: MS2 spectral similarity score (0-1)
        ms2_match_count: Number of matched MS2 peaks
        rt_match: Whether RT matches reference (within tolerance)
        rt_error: RT error in seconds
        kmd_series: Whether feature is part of KMD homologous series
        kmd_series_size: Number of members in KMD series
        mdc_region: Whether feature falls in PFAS MD/C region
        df_matches: List of matched diagnostic fragments
        delta_m_matches: List of matched neutral losses
        has_reference_standard: Whether reference standard was used
        
    Returns:
        ConfidenceResult with level, rationale, and evidence
    """
    df_matches = df_matches or []
    delta_m_matches = delta_m_matches or []
    
    evidence = {
        "suspect_match": suspect_match,
        "suspect_name": suspect_name,
        "ms2_similarity": ms2_similarity,
        "ms2_match_count": ms2_match_count,
        "rt_match": rt_match,
        "rt_error": rt_error,
        "kmd_series": kmd_series,
        "kmd_series_size": kmd_series_size,
        "mdc_region": mdc_region,
        "df_matches": df_matches,
        "delta_m_matches": delta_m_matches,
        "has_reference_standard": has_reference_standard,
    }
    
    rationale_parts = []
    
    # Level 1: Confirmed by reference standard
    if has_reference_standard and rt_match and ms2_similarity is not None and ms2_similarity >= 0.9:
        rationale_parts.append("표준물질 확인")
        if rt_error is not None:
            rationale_parts.append(f"RT 오차: {rt_error:.1f}s")
        rationale_parts.append(f"MS2 유사도: {ms2_similarity:.2f}")
        
        return ConfidenceResult(
            level=ConfidenceLevel.LEVEL_1,
            rationale=", ".join(rationale_parts),
            evidence=evidence,
        )
    
    # Level 2: Probable by MS2 library match
    if ms2_similarity is not None and ms2_similarity >= 0.8:
        rationale_parts.append(f"MS2 라이브러리 매칭 (유사도: {ms2_similarity:.2f})")
        if suspect_match and suspect_name:
            rationale_parts.append(f"Suspect: {suspect_name}")
        if ms2_match_count > 0:
            rationale_parts.append(f"매칭 피크: {ms2_match_count}개")
        
        return ConfidenceResult(
            level=ConfidenceLevel.LEVEL_2,
            rationale=", ".join(rationale_parts),
            evidence=evidence,
        )
    
    # Level 3: Tentative by diagnostic evidence
    has_df_evidence = len(df_matches) >= 2
    has_series_evidence = kmd_series and kmd_series_size >= 3
    has_ms2_partial = ms2_similarity is not None and ms2_similarity >= 0.5
    
    if has_df_evidence or (has_series_evidence and (has_ms2_partial or len(df_matches) >= 1)):
        if df_matches:
            rationale_parts.append(f"진단조각: {', '.join(df_matches[:3])}")
        if kmd_series:
            rationale_parts.append(f"동족체 시리즈 ({kmd_series_size}개)")
        if has_ms2_partial:
            rationale_parts.append(f"MS2 부분 매칭 ({ms2_similarity:.2f})")
        if delta_m_matches:
            rationale_parts.append(f"중성손실: {', '.join(delta_m_matches[:2])}")
        
        return ConfidenceResult(
            level=ConfidenceLevel.LEVEL_3,
            rationale=", ".join(rationale_parts),
            evidence=evidence,
        )
    
    # Level 4: Unequivocal formula by mass defect patterns
    if suspect_match or (mdc_region and kmd_series):
        if suspect_match and suspect_name:
            rationale_parts.append(f"Suspect 매칭: {suspect_name}")
        if mdc_region:
            rationale_parts.append("PFAS MD/C 영역")
        if kmd_series:
            rationale_parts.append("KMD 시리즈")
        if df_matches:
            rationale_parts.append(f"진단조각: {len(df_matches)}개")
        
        return ConfidenceResult(
            level=ConfidenceLevel.LEVEL_4,
            rationale=", ".join(rationale_parts),
            evidence=evidence,
        )
    
    # Level 5: Exact mass only
    mz = feature.get("mz", 0)
    rationale_parts.append(f"m/z {mz:.4f}")
    if mdc_region:
        rationale_parts.append("MD/C 영역 내")
    
    return ConfidenceResult(
        level=ConfidenceLevel.LEVEL_5,
        rationale=", ".join(rationale_parts),
        evidence=evidence,
    )


def confidence_from_pfas_result(
    feature_row: dict[str, Any],
) -> ConfidenceResult:
    """
    Calculate confidence from PFAS prioritization result.
    
    This is a convenience function that extracts evidence from
    the standard prioritization result format.
    
    Args:
        feature_row: Row from prioritization result DataFrame as dict
        
    Returns:
        ConfidenceResult
    """
    # Parse evidence_types
    evidence_types = feature_row.get("evidence_types", "")
    if isinstance(evidence_types, str):
        evidence_list = [e.strip() for e in evidence_types.split(",") if e.strip()]
    else:
        evidence_list = []
    
    # Extract evidence flags
    has_suspect = "suspect" in evidence_list or feature_row.get("suspect_match", False)
    has_kmd = "kmd" in evidence_list or "kmd_series" in evidence_list
    has_mdc = "mdc" in evidence_list or "mdc_region" in evidence_list
    has_df = "df_match" in evidence_list or "df" in evidence_list
    has_delta_m = "delta_m" in evidence_list or "delta_m_match" in evidence_list
    
    # Get suspect name if available
    suspect_name = feature_row.get("suspect_name") or feature_row.get("matched_suspect")
    
    # Get MS2 info if available
    ms2_similarity = feature_row.get("ms2_similarity") or feature_row.get("ms2_score")
    ms2_count = feature_row.get("ms2_count", 0)
    
    # Get series info
    kmd_series_size = feature_row.get("kmd_series_size", 0)
    if has_kmd and kmd_series_size == 0:
        kmd_series_size = 2  # Assume at least 2 if in series
    
    # Collect diagnostic fragments
    df_matches = []
    if has_df:
        df_list = feature_row.get("diagnostic_fragments", [])
        if isinstance(df_list, list):
            df_matches = df_list[:5]
        elif has_df:
            df_matches = ["CF3-", "C2F5-"]  # Default PFAS fragments
    
    # Collect neutral losses
    delta_m_matches = []
    if has_delta_m:
        dm_list = feature_row.get("neutral_losses", [])
        if isinstance(dm_list, list):
            delta_m_matches = dm_list[:3]
        elif has_delta_m:
            delta_m_matches = ["HF", "CF2"]  # Default PFAS losses
    
    return calculate_confidence(
        feature=feature_row,
        suspect_match=has_suspect,
        suspect_name=suspect_name,
        ms2_similarity=ms2_similarity,
        ms2_match_count=ms2_count,
        rt_match=False,  # No RT reference by default
        kmd_series=has_kmd,
        kmd_series_size=kmd_series_size,
        mdc_region=has_mdc,
        df_matches=df_matches,
        delta_m_matches=delta_m_matches,
        has_reference_standard=False,  # No standard by default
    )


def add_confidence_to_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add confidence level columns to prioritization result DataFrame.
    
    Args:
        df: Prioritization result DataFrame
        
    Returns:
        DataFrame with added confidence columns
    """
    import pandas as pd
    
    if len(df) == 0:
        df["confidence_level"] = pd.Series(dtype=int)
        df["confidence_rationale"] = pd.Series(dtype=str)
        return df
    
    levels = []
    rationales = []
    
    for _, row in df.iterrows():
        result = confidence_from_pfas_result(row.to_dict())
        levels.append(result.level.value)
        rationales.append(result.rationale)
    
    df = df.copy()
    df["confidence_level"] = levels
    df["confidence_rationale"] = rationales
    
    return df
