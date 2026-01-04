"""
Tests for ML inference module.
"""

import numpy as np
import pytest

from onfra_pfas.core.ml_inference import (
    MLScoreResult,
    SimilarHit,
    PFASClassifier,
    MS2Embedder,
    compute_cosine_similarity,
    add_ml_scores_to_dataframe,
)


class TestMLScoreResult:
    """Tests for MLScoreResult dataclass."""

    def test_creation(self):
        """Test creating MLScoreResult."""
        result = MLScoreResult(score=0.85, uncertainty=0.1)
        
        assert result.score == 0.85
        assert result.uncertainty == 0.1
        assert "v1" in result.model_version

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MLScoreResult(score=0.75, uncertainty=0.15)
        d = result.to_dict()
        
        assert d["ml_score"] == 0.75
        assert d["ml_uncertainty"] == 0.15
        assert "model_version" in d


class TestSimilarHit:
    """Tests for SimilarHit dataclass."""

    def test_creation(self):
        """Test creating SimilarHit."""
        hit = SimilarHit(name="PFOA", score=0.92, formula="C8HF15O2")
        
        assert hit.name == "PFOA"
        assert hit.score == 0.92
        assert hit.formula == "C8HF15O2"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hit = SimilarHit(name="PFOS", score=0.88, mz=499.0, source="nist")
        d = hit.to_dict()
        
        assert d["name"] == "PFOS"
        assert d["score"] == 0.88
        assert d["mz"] == 499.0
        assert d["source"] == "nist"


class TestPFASClassifier:
    """Tests for PFASClassifier class."""

    def test_init_no_model(self):
        """Test initialization without model."""
        classifier = PFASClassifier()
        assert classifier._model is None

    def test_predict_rule_based(self):
        """Test rule-based prediction fallback."""
        classifier = PFASClassifier()
        
        features = np.array([
            [413.0, 120.0, 50000, 1, 3, 0.1, 0.0, 0.0, 2, 1, 6.0, 3, 1, 10, 5],  # High score
            [200.0, 60.0, 10000, 1, 0, 5.0, 0.0, 0.0, 0, 0, 0.5, 0, 0, 0, 0],   # Low score
        ])
        
        results = classifier.predict(features)
        
        assert len(results) == 2
        assert results[0].score > results[1].score

    def test_predict_from_dict(self):
        """Test prediction from dictionary features."""
        classifier = PFASClassifier()
        
        features = [
            {"mz": 413.0, "rt": 120.0, "pfas_score": 6.0, "has_ms2": True},
            {"mz": 200.0, "rt": 60.0, "pfas_score": 0.5, "has_ms2": False},
        ]
        
        results = classifier.predict(features)
        
        assert len(results) == 2
        assert isinstance(results[0], MLScoreResult)

    def test_dict_to_array(self):
        """Test feature dict to array conversion."""
        classifier = PFASClassifier()
        
        features = [{"mz": 413.0, "rt": 120.0, "intensity": 50000}]
        array = classifier._dict_to_array(features)
        
        assert array.shape == (1, len(classifier.META_FEATURES))
        assert array[0, 0] == 413.0
        assert array[0, 1] == 120.0

    def test_sigmoid(self):
        """Test sigmoid function."""
        assert PFASClassifier._sigmoid(0) == pytest.approx(0.5, abs=0.01)
        assert PFASClassifier._sigmoid(10) > 0.99
        assert PFASClassifier._sigmoid(-10) < 0.01

    def test_uncertainty_with_ms2(self):
        """Test that MS2 reduces uncertainty."""
        classifier = PFASClassifier()
        
        features_with_ms2 = np.zeros((1, 15))
        features_with_ms2[0, 12] = 1  # has_ms2
        
        features_no_ms2 = np.zeros((1, 15))
        features_no_ms2[0, 12] = 0
        
        result_with = classifier._rule_based_predict(features_with_ms2, None, None)
        result_without = classifier._rule_based_predict(features_no_ms2, None, None)
        
        assert result_with[0].uncertainty < result_without[0].uncertainty


class TestMS2Embedder:
    """Tests for MS2Embedder class."""

    def test_init(self):
        """Test initialization."""
        embedder = MS2Embedder()
        
        assert embedder.EMBED_DIM == 128
        assert len(embedder._library) == 0

    def test_embed_single(self):
        """Test embedding a single spectrum."""
        embedder = MS2Embedder()
        
        spectrum = np.array([[100.0, 1000.0], [150.0, 500.0], [200.0, 250.0]])
        embeddings = embedder.embed([spectrum])
        
        assert embeddings.shape == (1, 128)
        assert embeddings[0].max() > 0

    def test_embed_batch(self):
        """Test embedding multiple spectra."""
        embedder = MS2Embedder()
        
        spectra = [
            np.array([[100.0, 1000.0], [150.0, 500.0]]),
            np.array([[200.0, 800.0], [300.0, 400.0]]),
        ]
        
        embeddings = embedder.embed(spectra)
        
        assert embeddings.shape == (2, 128)

    def test_embed_normalize(self):
        """Test L2 normalization."""
        embedder = MS2Embedder()
        
        spectrum = np.array([[100.0, 1000.0]])
        embeddings = embedder.embed([spectrum], normalize=True)
        
        norm = np.linalg.norm(embeddings[0])
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_embed_empty(self):
        """Test embedding empty spectrum."""
        embedder = MS2Embedder()
        
        embeddings = embedder.embed([np.array([])])
        
        assert embeddings.shape == (1, 128)
        assert np.all(embeddings[0] == 0)

    def test_add_to_library(self):
        """Test adding to library."""
        embedder = MS2Embedder()
        
        spectrum = np.array([[100.0, 1000.0], [200.0, 500.0]])
        embedder.add_to_library("PFOA", spectrum, formula="C8HF15O2", mz=413.0)
        
        assert len(embedder._library) == 1
        assert embedder._library[0]["name"] == "PFOA"

    def test_search_similar(self):
        """Test similarity search."""
        embedder = MS2Embedder()
        
        # Add reference spectra
        embedder.add_to_library("PFOA", np.array([[100.0, 1000.0], [200.0, 500.0]]))
        embedder.add_to_library("PFOS", np.array([[150.0, 800.0], [250.0, 400.0]]))
        
        # Query with similar spectrum
        query = np.array([[100.0, 900.0], [200.0, 600.0]])
        query_emb = embedder.embed([query])[0]
        
        results = embedder.search_similar(query_emb, top_k=2)
        
        assert len(results) == 2
        assert results[0].name == "PFOA"  # Should be most similar

    def test_search_empty_library(self):
        """Test search with empty library."""
        embedder = MS2Embedder()
        
        query_emb = np.random.randn(128)
        results = embedder.search_similar(query_emb)
        
        assert len(results) == 0


class TestComputeCosineSimilarity:
    """Tests for compute_cosine_similarity function."""

    def test_identical_spectra(self):
        """Test similarity of identical spectra."""
        spectrum = np.array([[100.0, 1000.0], [200.0, 500.0]])
        sim = compute_cosine_similarity(spectrum, spectrum)
        
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_different_spectra(self):
        """Test similarity of different spectra."""
        spectrum1 = np.array([[100.0, 1000.0]])
        spectrum2 = np.array([[500.0, 1000.0]])  # No matching peaks
        
        sim = compute_cosine_similarity(spectrum1, spectrum2)
        
        assert sim == 0.0

    def test_partial_match(self):
        """Test partial matching."""
        spectrum1 = np.array([[100.0, 1000.0], [200.0, 500.0]])
        spectrum2 = np.array([[100.0, 800.0], [300.0, 500.0]])  # One match
        
        sim = compute_cosine_similarity(spectrum1, spectrum2)
        
        assert 0 < sim < 1

    def test_empty_spectrum(self):
        """Test with empty spectrum."""
        spectrum = np.array([[100.0, 1000.0]])
        empty = np.array([])
        
        assert compute_cosine_similarity(spectrum, empty) == 0.0
        assert compute_cosine_similarity(empty, spectrum) == 0.0

    def test_tolerance(self):
        """Test m/z tolerance matching."""
        spectrum1 = np.array([[100.0, 1000.0]])
        spectrum2 = np.array([[100.005, 1000.0]])  # Within 0.01 tolerance
        
        sim = compute_cosine_similarity(spectrum1, spectrum2, tolerance=0.01)
        assert sim > 0.9


class TestAddMLScoresToDataframe:
    """Tests for add_ml_scores_to_dataframe function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame()
        result = add_ml_scores_to_dataframe(df)
        
        assert "ml_score" in result.columns
        assert "ml_uncertainty" in result.columns

    def test_adds_scores(self):
        """Test adding ML scores."""
        import pandas as pd
        
        df = pd.DataFrame([
            {"mz": 413.0, "rt": 120.0, "pfas_score": 6.0},
            {"mz": 463.0, "rt": 150.0, "pfas_score": 2.0},
        ])
        
        result = add_ml_scores_to_dataframe(df)
        
        assert "ml_score" in result.columns
        assert len(result["ml_score"]) == 2
        assert result["ml_score"].iloc[0] > result["ml_score"].iloc[1]

    def test_preserves_original(self):
        """Test that original columns are preserved."""
        import pandas as pd
        
        df = pd.DataFrame([{"mz": 413.0, "rt": 120.0, "custom_col": "value"}])
        result = add_ml_scores_to_dataframe(df)
        
        assert "custom_col" in result.columns
        assert result["custom_col"].iloc[0] == "value"
