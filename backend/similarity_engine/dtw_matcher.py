"""
DTW (Dynamic Time Warping) Similarity Module
เสริม FAISS ด้วย DTW สำหรับการเทียบ Pattern ที่ "ยืด-หด" ได้

FAISS ใช้ Euclidean Distance = เทียบจุดต่อจุด
DTW = ยอมให้ Pattern shift ได้เล็กน้อย (เหมาะกับกราฟที่หน้าตาเหมือนแต่ช้า/เร็วต่างกัน)
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import euclidean
import logging

# Try to import fastdtw for speed, fallback to pure Python
try:
    from dtw import dtw as dtw_func
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    logging.warning("dtw-python not installed. Using fallback DTW implementation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dtw_distance(
    series1: np.ndarray,
    series2: np.ndarray,
    window: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Calculate DTW distance between two time series
    
    Args:
        series1: First time series
        series2: Second time series
        window: Sakoe-Chiba band width (None = no constraint)
    
    Returns:
        Tuple of (distance, alignment_path)
    """
    if DTW_AVAILABLE:
        # Use dtw-python library
        alignment = dtw_func(
            series1,
            series2,
            keep_internals=True,
            step_pattern="symmetric2"
        )
        return alignment.distance, alignment.index1
    else:
        # Fallback to pure Python implementation
        return _dtw_pure_python(series1, series2, window)


def _dtw_pure_python(
    series1: np.ndarray,
    series2: np.ndarray,
    window: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Pure Python DTW implementation (slower but always available)
    
    Args:
        series1: First time series
        series2: Second time series
        window: Sakoe-Chiba band width
    
    Returns:
        Tuple of (distance, path)
    """
    n, m = len(series1), len(series2)
    
    # Initialize cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill the matrix
    for i in range(1, n + 1):
        # Apply window constraint
        if window is not None:
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
        else:
            j_start, j_end = 1, m + 1
        
        for j in range(j_start, j_end):
            cost = abs(series1[i - 1] - series2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )
    
    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dtw_matrix[i - 1, j], (i - 1, j)),
            (dtw_matrix[i, j - 1], (i, j - 1)),
            (dtw_matrix[i - 1, j - 1], (i - 1, j - 1))
        ]
        _, (i, j) = min(candidates, key=lambda x: x[0])
    
    path.reverse()
    
    return dtw_matrix[n, m], np.array([p[0] for p in path])


def dtw_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Calculate DTW-based similarity (0 to 1 scale)
    
    Args:
        series1: First time series
        series2: Second time series
        normalize: Whether to normalize to 0-1 range
    
    Returns:
        Similarity score (higher = more similar)
    """
    distance, _ = dtw_distance(series1, series2)
    
    if normalize:
        # Normalize by path length
        path_length = max(len(series1), len(series2))
        normalized_distance = distance / path_length
        # Convert to similarity
        similarity = 1 / (1 + normalized_distance)
        return similarity
    else:
        return 1 / (1 + distance)


def find_similar_dtw(
    query: np.ndarray,
    database: np.ndarray,
    k: int = 10,
    min_similarity: float = 0.7
) -> List[Tuple[int, float]]:
    """
    Find k most similar patterns using DTW
    ช้ากว่า FAISS แต่แม่นกว่าสำหรับ shifted patterns
    
    Args:
        query: Query pattern
        database: Database of patterns
        k: Number of results
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    logger.info(f"DTW search over {len(database)} patterns...")
    
    similarities = []
    
    for i, pattern in enumerate(database):
        sim = dtw_similarity(query, pattern)
        if sim >= min_similarity:
            similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:k]


class DTWMatcher:
    """
    DTW-based Pattern Matcher
    ใช้เมื่อต้องการความแม่นยำสูงสุด (ยอมช้า)
    """
    
    def __init__(
        self,
        window_constraint: Optional[int] = None,
        min_similarity: float = 0.7
    ):
        """
        Initialize DTW Matcher
        
        Args:
            window_constraint: Sakoe-Chiba band width
            min_similarity: Minimum similarity threshold
        """
        self.window_constraint = window_constraint
        self.min_similarity = min_similarity
        
        self.patterns: Optional[np.ndarray] = None
        self.metadata: List = []
    
    def fit(self, patterns: np.ndarray, metadata: Optional[List] = None) -> None:
        """
        Store patterns for matching
        
        Args:
            patterns: Array of patterns
            metadata: Optional metadata
        """
        self.patterns = patterns
        self.metadata = metadata if metadata else list(range(len(patterns)))
        logger.info(f"DTW Matcher fitted with {len(patterns)} patterns")
    
    def find_matches(self, query: np.ndarray, k: int = 10) -> List[dict]:
        """
        Find matching patterns using DTW
        
        Args:
            query: Query pattern
            k: Number of matches
        
        Returns:
            List of match dicts with index, similarity, metadata
        """
        if self.patterns is None:
            raise ValueError("Matcher not fitted. Call fit first.")
        
        matches = find_similar_dtw(
            query,
            self.patterns,
            k=k,
            min_similarity=self.min_similarity
        )
        
        results = []
        for idx, similarity in matches:
            results.append({
                "index": idx,
                "similarity": similarity,
                "pattern": self.patterns[idx],
                "metadata": self.metadata[idx] if idx < len(self.metadata) else None
            })
        
        return results


class HybridMatcher:
    """
    Hybrid Matcher: FAISS + DTW
    ใช้ FAISS กรองคร่าวๆ ก่อน แล้วใช้ DTW refine
    
    เร็วและแม่น!
    """
    
    def __init__(
        self,
        window_size: int,
        faiss_k: int = 50,
        final_k: int = 10,
        min_dtw_similarity: float = 0.7
    ):
        """
        Initialize Hybrid Matcher
        
        Args:
            window_size: Pattern window size
            faiss_k: Number of candidates from FAISS
            final_k: Final number of results
            min_dtw_similarity: Minimum DTW similarity
        """
        from .faiss_engine import FAISSEngine
        
        self.window_size = window_size
        self.faiss_k = faiss_k
        self.final_k = final_k
        self.min_dtw_similarity = min_dtw_similarity
        
        self.faiss_engine = FAISSEngine(dimension=window_size, index_type="IVF")
        self.patterns: Optional[np.ndarray] = None
        self.futures: Optional[np.ndarray] = None
        self.metadata: List = []
    
    def fit(
        self,
        patterns: np.ndarray,
        futures: Optional[np.ndarray] = None,
        metadata: Optional[List] = None
    ) -> None:
        """
        Build hybrid index
        
        Args:
            patterns: Array of patterns
            futures: Optional future prices
            metadata: Optional metadata
        """
        self.patterns = patterns
        self.futures = futures
        self.metadata = metadata if metadata else list(range(len(patterns)))
        
        self.faiss_engine.build_index(patterns, self.metadata)
        
        logger.info(f"Hybrid Matcher fitted with {len(patterns)} patterns")
    
    def find_matches(self, query: np.ndarray, k: Optional[int] = None) -> List[dict]:
        """
        Find matches using FAISS + DTW refinement
        
        Args:
            query: Query pattern
            k: Number of final results (default: self.final_k)
        
        Returns:
            List of match dicts
        """
        if self.patterns is None:
            raise ValueError("Matcher not fitted. Call fit first.")
        
        k = k or self.final_k
        
        # Step 1: FAISS rough search
        distances, indices, faiss_meta = self.faiss_engine.search(
            query.astype(np.float32),
            k=self.faiss_k
        )
        
        # Step 2: DTW refinement
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.patterns):
                continue
            
            # Calculate DTW similarity
            dtw_sim = dtw_similarity(query, self.patterns[idx])
            
            if dtw_sim >= self.min_dtw_similarity:
                candidates.append({
                    "index": int(idx),
                    "faiss_distance": float(dist),
                    "dtw_similarity": dtw_sim,
                    "pattern": self.patterns[idx],
                    "future": self.futures[idx] if self.futures is not None else None,
                    "metadata": self.metadata[idx]
                })
        
        # Sort by DTW similarity
        candidates.sort(key=lambda x: x["dtw_similarity"], reverse=True)
        
        return candidates[:k]


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("Trademify DTW Module - Example Usage")
    print("=" * 50)
    
    # Create sample patterns
    np.random.seed(42)
    
    # Base pattern: sine wave
    base = np.sin(np.linspace(0, 4 * np.pi, 60))
    
    # Similar pattern: shifted sine wave
    shifted = np.sin(np.linspace(0.5, 4.5 * np.pi, 60))
    
    # Stretched pattern
    stretched = np.sin(np.linspace(0, 3 * np.pi, 60))
    
    # Random pattern
    random_pattern = np.random.randn(60)
    
    # Calculate DTW distances
    print("\nDTW Similarities with base pattern:")
    
    sim_shifted = dtw_similarity(base, shifted)
    print(f"  Shifted sine: {sim_shifted:.4f}")
    
    sim_stretched = dtw_similarity(base, stretched)
    print(f"  Stretched sine: {sim_stretched:.4f}")
    
    sim_random = dtw_similarity(base, random_pattern)
    print(f"  Random: {sim_random:.4f}")
    
    # Euclidean comparison
    from numpy.linalg import norm
    print("\nEuclidean Distances (for comparison):")
    print(f"  Shifted sine: {norm(base - shifted):.4f}")
    print(f"  Stretched sine: {norm(base - stretched):.4f}")
    print(f"  Random: {norm(base - random_pattern):.4f}")
    
    print("\nNote: DTW handles time shifts better than Euclidean distance!")
