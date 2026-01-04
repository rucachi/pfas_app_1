"""
Tests for confidence level module.
"""

import pytest

from onfra_pfas.core.confidence import (
    ConfidenceLevel,
    ConfidenceResult,
    calculate_confidence,
    confidence_from_pfas_result,
    add_confidence_to_dataframe,
)


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_level_values(self):
        """Test confidence level integer values."""
        assert ConfidenceLevel.LEVEL_1 == 1
        assert ConfidenceLevel.LEVEL_2 == 2
        assert ConfidenceLevel.LEVEL_3 == 3
        assert ConfidenceLevel.LEVEL_4 == 4
        assert ConfidenceLevel.LEVEL_5 == 5

    def test_level_ordering(self):
        """Test that lower levels are more confident."""
        assert ConfidenceLevel.LEVEL_1 < ConfidenceLevel.LEVEL_2
        assert ConfidenceLevel.LEVEL_2 < ConfidenceLevel.LEVEL_5

    def test_description(self):
        """Test level descriptions."""
        assert "Confirmed" in ConfidenceLevel.LEVEL_1.description
        assert "Probable" in ConfidenceLevel.LEVEL_2.description

    def test_description_en(self):
        """Test English descriptions."""
        assert "reference standard" in ConfidenceLevel.LEVEL_1.description_en
        assert "library" in ConfidenceLevel.LEVEL_2.description_en


class TestConfidenceResult:
    """Tests for ConfidenceResult dataclass."""

    def test_creation(self):
        """Test creating a ConfidenceResult."""
        result = ConfidenceResult(
            level=ConfidenceLevel.LEVEL_3,
            rationale="Test rationale",
            evidence={"key": "value"},
        )
        assert result.level == ConfidenceLevel.LEVEL_3
        assert result.rationale == "Test rationale"
        assert result.evidence["key"] == "value"

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ConfidenceResult(
            level=ConfidenceLevel.LEVEL_2,
            rationale="MS2 match",
        )
        d = result.to_dict()

        assert d["confidence_level"] == 2
        assert d["confidence_level_name"] == "LEVEL_2"
        assert "MS2 match" in d["confidence_rationale"]


class TestCalculateConfidence:
    """Tests for calculate_confidence function."""

    def test_level_1_confirmed(self):
        """Test Level 1: Confirmed by reference standard."""
        result = calculate_confidence(
            feature={"mz": 413.0},
            has_reference_standard=True,
            rt_match=True,
            rt_error=5.0,
            ms2_similarity=0.95,
        )
        assert result.level == ConfidenceLevel.LEVEL_1
        assert "표준물질" in result.rationale

    def test_level_2_ms2_library(self):
        """Test Level 2: MS2 library match."""
        result = calculate_confidence(
            feature={"mz": 413.0},
            ms2_similarity=0.85,
            suspect_match=True,
            suspect_name="PFOA",
        )
        assert result.level == ConfidenceLevel.LEVEL_2
        assert "라이브러리" in result.rationale

    def test_level_3_diagnostic_fragments(self):
        """Test Level 3: Diagnostic fragments."""
        result = calculate_confidence(
            feature={"mz": 413.0},
            df_matches=["CF3-", "C2F5-", "C3F7-"],
        )
        assert result.level == ConfidenceLevel.LEVEL_3
        assert "진단조각" in result.rationale

    def test_level_3_series_with_fragments(self):
        """Test Level 3: KMD series with fragments."""
        result = calculate_confidence(
            feature={"mz": 413.0},
            kmd_series=True,
            kmd_series_size=5,
            ms2_similarity=0.6,
        )
        assert result.level == ConfidenceLevel.LEVEL_3

    def test_level_4_suspect_match(self):
        """Test Level 4: Suspect match."""
        result = calculate_confidence(
            feature={"mz": 413.0},
            suspect_match=True,
            suspect_name="PFOA",
        )
        assert result.level == ConfidenceLevel.LEVEL_4
        assert "Suspect" in result.rationale

    def test_level_4_mdc_with_kmd(self):
        """Test Level 4: MD/C region + KMD series."""
        result = calculate_confidence(
            feature={"mz": 413.0},
            mdc_region=True,
            kmd_series=True,
        )
        assert result.level == ConfidenceLevel.LEVEL_4

    def test_level_5_exact_mass_only(self):
        """Test Level 5: Only exact mass."""
        result = calculate_confidence(
            feature={"mz": 413.0659},
        )
        assert result.level == ConfidenceLevel.LEVEL_5
        assert "413.0659" in result.rationale

    def test_evidence_in_result(self):
        """Test that evidence is captured in result."""
        result = calculate_confidence(
            feature={"mz": 413.0},
            suspect_match=True,
            suspect_name="PFOA",
            ms2_similarity=0.75,
        )
        assert result.evidence["suspect_match"] is True
        assert result.evidence["suspect_name"] == "PFOA"
        assert result.evidence["ms2_similarity"] == 0.75


class TestConfidenceFromPfasResult:
    """Tests for confidence_from_pfas_result function."""

    def test_with_suspect_evidence(self):
        """Test with suspect evidence type."""
        row = {
            "mz": 413.0659,
            "evidence_types": "suspect, kmd",
            "suspect_name": "PFOA",
        }
        result = confidence_from_pfas_result(row)
        assert result.level == ConfidenceLevel.LEVEL_4

    def test_with_ms2_similarity(self):
        """Test with MS2 similarity score."""
        row = {
            "mz": 413.0659,
            "evidence_types": "suspect",
            "ms2_similarity": 0.82,
        }
        result = confidence_from_pfas_result(row)
        assert result.level == ConfidenceLevel.LEVEL_2

    def test_empty_evidence(self):
        """Test with no evidence."""
        row = {"mz": 413.0659}
        result = confidence_from_pfas_result(row)
        assert result.level == ConfidenceLevel.LEVEL_5


class TestAddConfidenceToDataframe:
    """Tests for add_confidence_to_dataframe function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame()
        result = add_confidence_to_dataframe(df)
        assert "confidence_level" in result.columns
        assert len(result) == 0

    def test_adds_columns(self):
        """Test that confidence columns are added."""
        import pandas as pd
        
        df = pd.DataFrame([
            {"mz": 413.0, "evidence_types": "suspect", "suspect_name": "PFOA"},
            {"mz": 463.0, "evidence_types": "kmd, mdc"},
        ])
        
        result = add_confidence_to_dataframe(df)
        
        assert "confidence_level" in result.columns
        assert "confidence_rationale" in result.columns
        assert len(result) == 2
        assert result.iloc[0]["confidence_level"] == 4
        assert result.iloc[1]["confidence_level"] == 4

    def test_preserves_original_columns(self):
        """Test that original columns are preserved."""
        import pandas as pd
        
        df = pd.DataFrame([
            {"mz": 413.0, "rt": 120.5, "intensity": 50000},
        ])
        
        result = add_confidence_to_dataframe(df)
        
        assert "mz" in result.columns
        assert "rt" in result.columns
        assert "intensity" in result.columns
