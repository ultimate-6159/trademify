"""
Similarity Engine Package
"""
from .faiss_engine import (
    FAISSEngine,
    PatternMatcher,
    calculate_correlation,
    find_similar_patterns
)
from .dtw_matcher import (
    DTWMatcher,
    HybridMatcher,
    dtw_distance,
    dtw_similarity,
    find_similar_dtw
)

__all__ = [
    "FAISSEngine",
    "PatternMatcher",
    "calculate_correlation",
    "find_similar_patterns",
    "DTWMatcher",
    "HybridMatcher",
    "dtw_distance",
    "dtw_similarity",
    "find_similar_dtw",
]
