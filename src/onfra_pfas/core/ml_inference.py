"""
ML inference module for ONFRA PFAS.

Provides ML-based PFAS scoring, MS2 embedding, and similarity search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MLScoreResult:
    """Result of ML-based PFAS scoring."""
    
    score: float                    # 0-1 probability
    uncertainty: float              # Standard deviation (MC Dropout)
    model_version: str = "ONFRA-PFAS-v1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ml_score": self.score,
            "ml_uncertainty": self.uncertainty,
            "model_version": self.model_version,
        }


@dataclass
class SimilarHit:
    """Similar spectrum hit."""
    
    name: str
    score: float                    # Similarity (0-1)
    formula: str | None = None
    mz: float = 0.0
    source: str = "library"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "formula": self.formula,
            "mz": self.mz,
            "source": self.source,
        }


class PFASClassifier:
    """
    PFAS classification model.
    
    Uses meta features + optional EIC + optional MS2 to predict
    PFAS probability with uncertainty estimation.
    """
    
    MODEL_VERSION = "ONFRA-PFAS-v1.0.0"
    
    # Feature names for meta features
    META_FEATURES = [
        "mz", "rt", "intensity", "charge",
        "isotope_count", "blank_ratio",
        "kmd_value", "mdc_value",
        "df_match_count", "delta_m_count",
        "pfas_score",  # For knowledge distillation
        "evidence_count",
        "has_ms2", "ms2_peaks_count",
        "kmd_series_size",
    ]
    
    def __init__(self, model_path: str | Path | None = None):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model weights (ONNX or pickle)
        """
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._scaler = None
        
        # Load model if path provided
        if self.model_path and self.model_path.exists():
            self._load_model()
        else:
            logger.info("No model loaded - using rule-based fallback")
    
    def _load_model(self) -> None:
        """Load trained model from file."""
        # TODO: Implement actual model loading (ONNX/pickle)
        logger.info(f"Loading model from {self.model_path}")
        pass
    
    def predict(
        self,
        features: np.ndarray | list[dict],
        eic: np.ndarray | None = None,
        ms2: list[np.ndarray] | None = None,
        return_uncertainty: bool = True,
        n_mc_samples: int = 10,
    ) -> list[MLScoreResult]:
        """
        Predict PFAS probability for features.
        
        Args:
            features: Feature array (batch, feat_dim) or list of dicts
            eic: Optional EIC arrays (batch, time_steps)
            ms2: Optional list of MS2 peak arrays
            return_uncertainty: Whether to compute uncertainty
            n_mc_samples: Number of MC Dropout samples
            
        Returns:
            List of MLScoreResult
        """
        # Convert dict features to array
        if isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
            features = self._dict_to_array(features)
        
        batch_size = len(features)
        
        # If no model loaded, use rule-based fallback
        if self._model is None:
            return self._rule_based_predict(features, eic, ms2)
        
        # TODO: Implement actual model inference with MC Dropout
        results = []
        for i in range(batch_size):
            # Placeholder: use pfas_score as base
            pfas_score = features[i, self.META_FEATURES.index("pfas_score")] if features.shape[1] > 10 else 0
            score = self._sigmoid(pfas_score - 3)  # Center around score 3
            
            results.append(MLScoreResult(
                score=float(score),
                uncertainty=0.1,
                model_version=self.MODEL_VERSION,
            ))
        
        return results
    
    def _rule_based_predict(
        self,
        features: np.ndarray,
        eic: np.ndarray | None,
        ms2: list[np.ndarray] | None,
    ) -> list[MLScoreResult]:
        """
        Rule-based fallback when no ML model is available.
        
        Uses pfas_score to estimate probability.
        """
        results = []
        
        for i in range(len(features)):
            # Get pfas_score from features
            pfas_score = 0.0
            if features.shape[1] > 10:
                pfas_score = features[i, 10]  # Assuming pfas_score is at index 10
            
            # Convert score to probability (sigmoid-like)
            score = self._sigmoid(pfas_score - 3)  # Center around 3
            
            # Increase uncertainty if no MS2
            has_ms2 = features[i, 12] if features.shape[1] > 12 else 0
            uncertainty = 0.15 if has_ms2 else 0.25
            
            results.append(MLScoreResult(
                score=float(score),
                uncertainty=float(uncertainty),
                model_version=f"{self.MODEL_VERSION}-fallback",
            ))
        
        return results
    
    def _dict_to_array(self, features: list[dict]) -> np.ndarray:
        """Convert list of feature dicts to numpy array."""
        array = np.zeros((len(features), len(self.META_FEATURES)), dtype=np.float32)
        
        for i, feat in enumerate(features):
            for j, name in enumerate(self.META_FEATURES):
                value = feat.get(name, 0)
                if isinstance(value, bool):
                    value = float(value)
                elif value is None:
                    value = 0.0
                array[i, j] = float(value)
        
        return array
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class MS2Embedder:
    """
    MS2 spectrum embedder for similarity search.
    
    Converts MS2 spectra to dense embeddings for nearest neighbor search.
    """
    
    EMBED_DIM = 128
    MODEL_VERSION = "MS2Embed-v1.0.0"
    
    def __init__(self, model_path: str | Path | None = None):
        """
        Initialize embedder.
        
        Args:
            model_path: Path to trained embedding model
        """
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._index = None  # For KNN search
        self._library: list[dict] = []  # Reference library
        
        if self.model_path and self.model_path.exists():
            self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model."""
        logger.info(f"Loading MS2 embedder from {self.model_path}")
        pass
    
    def embed(
        self,
        ms2_spectra: list[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for MS2 spectra.
        
        Args:
            ms2_spectra: List of (n_peaks, 2) arrays with [mz, intensity]
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            (batch, embed_dim) array
        """
        batch_size = len(ms2_spectra)
        embeddings = np.zeros((batch_size, self.EMBED_DIM), dtype=np.float32)
        
        for i, spectrum in enumerate(ms2_spectra):
            if spectrum is None or len(spectrum) == 0:
                continue
            
            # Simple bag-of-peaks embedding (fallback)
            emb = self._simple_embed(spectrum)
            embeddings[i] = emb
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms
        
        return embeddings
    
    def _simple_embed(self, spectrum: np.ndarray) -> np.ndarray:
        """Simple binned embedding (fallback)."""
        embedding = np.zeros(self.EMBED_DIM, dtype=np.float32)
        
        if len(spectrum) == 0:
            return embedding
        
        mzs = spectrum[:, 0]
        ints = spectrum[:, 1]
        
        # Normalize intensities
        max_int = ints.max()
        if max_int > 0:
            ints = ints / max_int
        
        # Bin into embedding dimensions (50-1000 m/z range)
        bin_width = 950.0 / self.EMBED_DIM
        indices = ((mzs - 50) / bin_width).astype(int)
        valid = (indices >= 0) & (indices < self.EMBED_DIM)
        
        for idx, intensity in zip(indices[valid], ints[valid]):
            embedding[idx] = max(embedding[idx], intensity)
        
        return embedding
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SimilarHit]:
        """
        Search for similar spectra in library.
        
        Args:
            query_embedding: (embed_dim,) query vector
            top_k: Number of results to return
            
        Returns:
            List of SimilarHit
        """
        if len(self._library) == 0:
            return []
        
        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        similarities = []
        for entry in self._library:
            lib_emb = entry.get("embedding")
            if lib_emb is None:
                continue
            
            lib_norm = lib_emb / (np.linalg.norm(lib_emb) + 1e-8)
            sim = float(np.dot(query_norm, lib_norm))
            similarities.append((sim, entry))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-K
        results = []
        for sim, entry in similarities[:top_k]:
            results.append(SimilarHit(
                name=entry.get("name", "Unknown"),
                score=sim,
                formula=entry.get("formula"),
                mz=entry.get("mz", 0),
                source=entry.get("source", "library"),
            ))
        
        return results
    
    def add_to_library(
        self,
        name: str,
        spectrum: np.ndarray,
        formula: str | None = None,
        mz: float = 0,
        source: str = "user",
    ) -> None:
        """Add spectrum to library."""
        embedding = self.embed([spectrum])[0]
        self._library.append({
            "name": name,
            "formula": formula,
            "mz": mz,
            "source": source,
            "embedding": embedding,
        })
    
    def load_library(self, library_path: Path) -> int:
        """Load reference library from file."""
        # TODO: Implement library loading (MGF, JSON, etc.)
        logger.info(f"Loading library from {library_path}")
        return len(self._library)


def compute_cosine_similarity(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    tolerance: float = 0.01,
) -> float:
    """
    Compute cosine similarity between two MS2 spectra.
    
    Args:
        spectrum1: (n_peaks, 2) array [mz, intensity]
        spectrum2: (m_peaks, 2) array [mz, intensity]
        tolerance: m/z matching tolerance in Da
        
    Returns:
        Cosine similarity (0-1)
    """
    if len(spectrum1) == 0 or len(spectrum2) == 0:
        return 0.0
    
    mz1, int1 = spectrum1[:, 0], spectrum1[:, 1]
    mz2, int2 = spectrum2[:, 0], spectrum2[:, 1]
    
    # Normalize intensities
    int1 = int1 / (np.max(int1) + 1e-8)
    int2 = int2 / (np.max(int2) + 1e-8)
    
    # Match peaks
    matched_product = 0.0
    
    for i, m1 in enumerate(mz1):
        diffs = np.abs(mz2 - m1)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tolerance:
            matched_product += int1[i] * int2[min_idx]
    
    # Compute norms
    norm1 = np.sqrt(np.sum(int1 ** 2))
    norm2 = np.sqrt(np.sum(int2 ** 2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return matched_product / (norm1 * norm2)


def add_ml_scores_to_dataframe(
    df: "pd.DataFrame",
    classifier: PFASClassifier | None = None,
    embedder: MS2Embedder | None = None,
    top_k: int = 5,
) -> "pd.DataFrame":
    """
    Add ML scores and similar hits to DataFrame.
    
    Args:
        df: Prioritization result DataFrame
        classifier: Optional PFASClassifier instance
        embedder: Optional MS2Embedder for similarity search
        top_k: Number of similar hits to return
        
    Returns:
        DataFrame with added ML columns
    """
    import pandas as pd
    
    if len(df) == 0:
        df["ml_score"] = pd.Series(dtype=float)
        df["ml_uncertainty"] = pd.Series(dtype=float)
        df["similar_hits"] = pd.Series(dtype=object)
        return df
    
    # Use default classifier if not provided
    if classifier is None:
        classifier = PFASClassifier()
    
    # Predict ML scores
    features = df.to_dict("records")
    results = classifier.predict(features)
    
    df = df.copy()
    df["ml_score"] = [r.score for r in results]
    df["ml_uncertainty"] = [r.uncertainty for r in results]
    df["model_version"] = [r.model_version for r in results]
    
    # Add similar hits if embedder provided
    if embedder is not None:
        similar_hits = []
        for _, row in df.iterrows():
            # TODO: Extract MS2 and search
            similar_hits.append([])
        df["similar_hits"] = similar_hits
    else:
        df["similar_hits"] = [[] for _ in range(len(df))]
    
    return df
