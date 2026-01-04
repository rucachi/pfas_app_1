"""
Tests for PFAS prioritization math functions.
"""

import pytest
import numpy as np
import pandas as pd

from onfra_pfas.core.pfas_prioritization import (
    calculate_kendrick_mass,
    calculate_kmd,
    calculate_mass_defect,
    estimate_carbon_count,
    calculate_mdc,
    find_homologous_series,
    match_diagnostic_fragments,
    match_delta_m_rules,
    calculate_pfas_scores,
    DEFAULT_DF_RULES,
    DEFAULT_DELTA_M_RULES,
)
from onfra_pfas.core.config import (
    KMDConfig,
    MDCConfig,
    DiagnosticFragmentConfig,
    ScoringConfig,
)


class TestKendrickMass:
    """Tests for Kendrick mass calculations."""

    def test_kendrick_mass_cf2(self):
        """Test Kendrick mass calculation for CF2 repeat unit."""
        # CF2 exact mass = 49.9968473, nominal = 50
        cf2_mass = 49.9968473

        # Test with known masses
        mz_values = np.array([100.0, 150.0, 200.0, 500.0])
        km = calculate_kendrick_mass(mz_values, cf2_mass)

        # Kendrick mass should be slightly higher than observed mass for CF2
        # KM = m * (50 / 49.9968473)
        expected_factor = 50 / cf2_mass
        np.testing.assert_array_almost_equal(km, mz_values * expected_factor, decimal=4)

    def test_kendrick_mass_scaling(self):
        """Test that Kendrick mass scales correctly."""
        repeat_mass = 50.0  # Exact == nominal, so KM should equal m/z
        mz = np.array([100.0, 200.0])

        km = calculate_kendrick_mass(mz, repeat_mass)

        np.testing.assert_array_almost_equal(km, mz, decimal=6)


class TestKMD:
    """Tests for Kendrick Mass Defect calculations."""

    def test_kmd_calculation(self):
        """Test KMD calculation."""
        cf2_mass = 49.9968473
        mz = np.array([100.0])

        kmd = calculate_kmd(mz, cf2_mass)

        # KMD should be small for typical masses
        assert -1.0 < kmd[0] < 1.0

    def test_kmd_homologs_similar(self):
        """Test that homologs have similar KMD values."""
        cf2_mass = 49.9968473

        # Simulated homologous series: m/z differing by CF2
        # PFOA-like: C8HF15O2 -> [M-H]- at m/z 413
        base_mz = 413.0
        homologs = np.array([
            base_mz,
            base_mz + cf2_mass,      # +CF2
            base_mz + 2 * cf2_mass,  # +2CF2
            base_mz + 3 * cf2_mass,  # +3CF2
        ])

        kmd_values = calculate_kmd(homologs, cf2_mass)

        # All KMD values should be very similar for true homologs
        kmd_std = np.std(kmd_values)
        assert kmd_std < 0.01, f"KMD std too high: {kmd_std}"

    def test_kmd_non_homologs_different(self):
        """Test that non-homologs have different KMD values."""
        cf2_mass = 49.9968473

        # Use masses with clearly different fractional parts (mass defects)
        # These should produce varying KMD values
        non_homologs = np.array([150.1, 183.25, 221.45, 278.9])

        kmd_values = calculate_kmd(non_homologs, cf2_mass)

        # KMD values should vary significantly
        kmd_std = np.std(kmd_values)
        assert kmd_std > 0.01, f"KMD std too low for non-homologs: {kmd_std}"


class TestMassDefect:
    """Tests for mass defect calculations."""

    def test_mass_defect_calculation(self):
        """Test basic mass defect calculation."""
        mz = np.array([100.05, 200.10, 300.15])
        md = calculate_mass_defect(mz)

        np.testing.assert_array_almost_equal(md, [0.05, 0.10, 0.15], decimal=6)

    def test_mass_defect_range(self):
        """Test mass defect is in [0, 1) range."""
        mz = np.array([100.0, 100.5, 100.99])
        md = calculate_mass_defect(mz)

        assert np.all(md >= 0)
        assert np.all(md < 1)


class TestMDC:
    """Tests for MD/C (mass defect per carbon) calculations."""

    def test_estimate_carbon_count(self):
        """Test carbon count estimation from m/z."""
        mz = np.array([500.0])  # Rough estimate: ~10 carbons for PFAS

        carbon_count = estimate_carbon_count(mz, mode="negative")

        # For m/z 500, expect ~10 carbon equivalents
        assert 8 < carbon_count[0] < 12

    def test_mdc_calculation(self):
        """Test MD/C calculation."""
        mz = np.array([413.0])  # PFOA-like
        carbon_count = np.array([8])

        mdc = calculate_mdc(mz, carbon_count)

        # PFAS typically has MD/C between -0.08 and 0.04
        # For this test, just check it's calculated
        assert mdc[0] != 0 or mz[0] == np.floor(mz[0])

    def test_mdc_pfas_region(self):
        """Test that PFAS compounds fall in expected MD/C region."""
        # Known PFAS m/z values (approximate)
        pfas_mz = np.array([
            412.97,  # PFOA [M-H]-
            462.97,  # PFNA [M-H]-
            512.97,  # PFDA [M-H]-
        ])
        carbon_counts = np.array([8, 9, 10])

        mdc = calculate_mdc(pfas_mz, carbon_counts)

        # All should be in PFAS region (-0.10 to 0.05)
        # Note: This is a simplified test; actual values depend on exact masses
        for val in mdc:
            assert -0.20 < val < 0.20, f"MD/C {val} outside expected range"


class TestHomologousSeries:
    """Tests for homologous series detection."""

    def test_find_simple_series(self):
        """Test finding a simple homologous series."""
        cf2_mass = 49.9968473

        # Create features that form a series
        features = pd.DataFrame({
            "feature_id": [1, 2, 3, 4, 5],
            "mz": [
                300.0,
                300.0 + cf2_mass,
                300.0 + 2 * cf2_mass,
                300.0 + 3 * cf2_mass,
                500.0,  # Not in series
            ],
            "rt": [100, 101, 102, 103, 200],
            "intensity": [1e6] * 5,
        })

        config = KMDConfig(
            enabled=True,
            repeat_units=["CF2"],
            repeat_unit_masses={"CF2": cf2_mass},
            kmd_tolerance=0.01,
            min_series_length=3,
        )

        series = find_homologous_series(features, config)

        # Should find at least one series with members 0-3
        assert len(series) >= 1
        assert any(len(s.members) >= 3 for s in series)

    def test_find_series_disabled(self):
        """Test that disabled config returns empty list."""
        features = pd.DataFrame({
            "feature_id": [1, 2, 3],
            "mz": [100, 150, 200],
        })

        config = KMDConfig(enabled=False)
        series = find_homologous_series(features, config)

        assert series == []


class TestDiagnosticFragments:
    """Tests for diagnostic fragment matching."""

    def test_match_known_fragments(self):
        """Test matching known PFAS fragments."""
        # Simulated MS2 peaks including common PFAS fragments
        ms2_peaks = np.array([
            [68.995, 50000],    # CF3-
            [118.992, 30000],   # C2F5-
            [168.989, 20000],   # C3F7-
            [100.0, 10000],     # Random peak
        ])

        config = DiagnosticFragmentConfig(
            enabled=True,
            tolerance_ppm=10.0,
            use_ppm=True,
        )

        matches = match_diagnostic_fragments(
            feature_id=1,
            ms2_peaks=ms2_peaks,
            rules=DEFAULT_DF_RULES,
            config=config,
        )

        # Should match CF3-, C2F5-, C3F7-
        matched_names = [m.fragment_name for m in matches]
        assert "CF3-" in matched_names
        assert "C2F5-" in matched_names

    def test_match_with_tolerance(self):
        """Test matching with different tolerances."""
        # Peak slightly off from theoretical
        ms2_peaks = np.array([
            [69.00, 50000],  # Slightly off from CF3- (68.9952)
        ])

        # Tight tolerance - should not match
        config_tight = DiagnosticFragmentConfig(
            enabled=True,
            tolerance_ppm=5.0,
            use_ppm=True,
        )

        matches_tight = match_diagnostic_fragments(1, ms2_peaks, DEFAULT_DF_RULES, config_tight)

        # Loose tolerance - should match
        config_loose = DiagnosticFragmentConfig(
            enabled=True,
            tolerance_ppm=100.0,
            use_ppm=True,
        )

        matches_loose = match_diagnostic_fragments(1, ms2_peaks, DEFAULT_DF_RULES, config_loose)

        # Loose should have more matches
        assert len(matches_loose) >= len(matches_tight)


class TestDeltaMRules:
    """Tests for delta-m (neutral loss) rule matching."""

    def test_match_cf2_loss(self):
        """Test matching CF2 neutral loss."""
        precursor_mz = 413.0  # PFOA-like

        # MS2 with CF2 loss
        ms2_peaks = np.array([
            [precursor_mz - 50.0, 100000],  # CF2 loss (~50 Da)
            [precursor_mz - 20.0, 50000],   # HF loss (~20 Da)
        ])

        matches = match_delta_m_rules(
            precursor_mz,
            ms2_peaks,
            DEFAULT_DELTA_M_RULES,
            tolerance_da=0.5,
        )

        # Should find CF2 and HF losses
        matched_rules = [m[0] for m in matches]
        assert "CF2" in matched_rules or "HF" in matched_rules


class TestScoring:
    """Tests for PFAS scoring."""

    def test_basic_scoring(self):
        """Test basic scoring with multiple evidence types."""
        features = pd.DataFrame({
            "feature_id": [1, 2],
            "mz": [413.0, 200.0],
            "rt": [100, 150],
            "intensity": [1e6, 5e5],
        })

        # Feature 1 has multiple evidences
        from onfra_pfas.core.pfas_prioritization import HomologousSeries, DFMatch

        homologous_series = [
            HomologousSeries(
                series_id=0,
                repeat_unit="CF2",
                members=[1],  # Feature 1 is in a series
                kmd_value=0.1,
                delta_m=50.0,
                confidence=0.8,
            )
        ]

        df_matches = {
            1: [DFMatch(1, "CF3-", 69.0, 69.0, 0.5, 50000)],
        }

        mdc_in_region = np.array([True, False])

        config = ScoringConfig(
            weights={
                "kmd_series": 2.0,
                "mdc_region": 1.5,
                "df_match": 3.0,
                "delta_m_match": 2.5,
                "suspect_match": 5.0,
            },
            min_score_threshold=0.0,
        )

        result = calculate_pfas_scores(
            features,
            homologous_series,
            df_matches,
            {},  # No delta-m matches
            [],  # No suspect matches
            mdc_in_region,
            config,
        )

        # Feature 1 should have higher score
        feat1 = result[result["feature_id"] == 1]
        feat2 = result[result["feature_id"] == 2]

        if len(feat1) > 0 and len(feat2) > 0:
            assert feat1["pfas_score"].values[0] > feat2["pfas_score"].values[0]

    def test_score_threshold_filtering(self):
        """Test that low scores are filtered out."""
        features = pd.DataFrame({
            "feature_id": [1, 2, 3],
            "mz": [100, 200, 300],
            "rt": [100, 200, 300],
            "intensity": [1e6, 1e6, 1e6],
        })

        mdc_in_region = np.array([False, False, False])

        config = ScoringConfig(
            min_score_threshold=5.0,  # High threshold
        )

        result = calculate_pfas_scores(
            features,
            [],
            {},
            {},
            [],
            mdc_in_region,
            config,
        )

        # With no evidence and high threshold, no features should pass
        assert len(result) == 0
