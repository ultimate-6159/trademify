"""
FAISS Similarity Engine - Phase 2: The Similarity Engine
เครื่องยนต์ค้นหา Pattern ที่เหมือนกันที่สุด

FAISS (Facebook AI Similarity Search) = เครื่องมือที่เร็วที่สุดในโลก
สำหรับหา Vector ที่เหมือนกัน (หา Pattern นับล้านได้ในเสี้ยววินาที)

Index Types:
- Flat: Brute force, แม่นที่สุด แต่ช้า
- IVF: Inverted File Index, เร็วขึ้น ยอมเสียความแม่นนิดหน่อย
- HNSW: Hierarchical Navigable Small World, เร็วมากสำหรับ dataset ใหญ่
"""
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import pickle

from config import PatternConfig, INDEX_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSEngine:
    """
    FAISS-based Similarity Search Engine
    ค้นหา Pattern ที่เหมือนที่สุดจากฐานข้อมูลนับล้าน
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "IVF",
        n_clusters: int = 100,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS Engine
        
        Args:
            dimension: Vector dimension (= window_size for 1D patterns)
            index_type: Type of index (Flat, IVF, HNSW)
            n_clusters: Number of clusters for IVF
            use_gpu: Whether to use GPU acceleration
        """
        self.dimension = dimension
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        self.n_vectors = 0
        
        # Metadata storage
        self.metadata: List[Any] = []
        
    def _create_index(self, n_vectors: int = 10000) -> faiss.Index:
        """
        Create FAISS index based on configuration
        
        Args:
            n_vectors: Expected number of vectors (for IVF training)
        
        Returns:
            FAISS Index
        """
        if self.index_type == "Flat":
            # Brute force - exact nearest neighbor
            # แม่นที่สุด แต่ช้าสำหรับ dataset ใหญ่
            index = faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "IVF":
            # Inverted File Index - balance between speed and accuracy
            # ดีสำหรับ dataset 100K - 10M vectors
            n_clusters = min(self.n_clusters, n_vectors // 39 + 1)  # Rule of thumb
            n_clusters = max(1, n_clusters)
            
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World
            # เร็วที่สุดสำหรับ dataset ใหญ่มาก (>10M)
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 = M parameter
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Use GPU if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                index
            )
            logger.info("Using GPU acceleration")
        
        return index
    
    def build_index(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Any]] = None
    ) -> None:
        """
        Build FAISS index from vectors
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
            metadata: Optional list of metadata for each vector
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Ensure float32 (FAISS requirement)
        vectors = vectors.astype(np.float32)
        
        n_vectors, dim = vectors.shape
        
        if dim != self.dimension:
            raise ValueError(f"Vector dimension {dim} doesn't match index dimension {self.dimension}")
        
        logger.info(f"Building {self.index_type} index for {n_vectors} vectors of dimension {dim}")
        
        # Create index
        self.index = self._create_index(n_vectors)
        
        # Train index (required for IVF)
        if self.index_type == "IVF":
            logger.info("Training IVF index...")
            self.index.train(vectors)
        
        # Add vectors
        logger.info("Adding vectors to index...")
        self.index.add(vectors)
        
        self.is_trained = True
        self.n_vectors = n_vectors
        
        # Store metadata
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = list(range(n_vectors))
        
        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        n_probe: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """
        Search for k most similar vectors
        
        Args:
            query_vector: Query vector(s) of shape (dimension,) or (n_queries, dimension)
            k: Number of neighbors to return
            n_probe: Number of cells to visit for IVF (higher = more accurate, slower)
        
        Returns:
            Tuple of (distances, indices, metadata)
            - distances: shape (n_queries, k) - ค่ายิ่งต่ำยิ่งเหมือน (0 = เหมือนเป๊ะ)
            - indices: shape (n_queries, k)
            - metadata: List of metadata for each result
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Reshape if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Ensure float32
        query_vector = query_vector.astype(np.float32)
        
        # Set number of probes for IVF
        if self.index_type == "IVF":
            self.index.nprobe = n_probe
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Get metadata for results
        result_metadata = []
        for idx_row in indices:
            row_meta = []
            for idx in idx_row:
                if idx >= 0 and idx < len(self.metadata):
                    row_meta.append(self.metadata[idx])
                else:
                    row_meta.append(None)
            result_metadata.append(row_meta)
        
        return distances, indices, result_metadata
    
    def search_with_threshold(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        max_distance: float = float('inf'),
        min_correlation: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """
        Search with quality threshold
        กรองผลลัพธ์ที่ไม่ดีพอออก
        
        Args:
            query_vector: Query vector
            k: Number of neighbors
            max_distance: Maximum allowed distance (ยิ่งต่ำยิ่งเข้มงวด)
            min_correlation: Minimum correlation (0-1)
        
        Returns:
            Filtered (distances, indices, metadata)
        """
        distances, indices, metadata = self.search(query_vector, k)
        
        # Filter by distance
        mask = distances[0] <= max_distance
        
        filtered_distances = distances[0][mask]
        filtered_indices = indices[0][mask]
        filtered_metadata = [m for m, keep in zip(metadata[0], mask) if keep]
        
        # Convert distance to similarity/correlation (approximate)
        # L2 distance -> similarity: sim = 1 / (1 + dist)
        if min_correlation > 0 and len(filtered_distances) > 0:
            similarities = 1 / (1 + filtered_distances)
            corr_mask = similarities >= min_correlation
            
            filtered_distances = filtered_distances[corr_mask]
            filtered_indices = filtered_indices[corr_mask]
            filtered_metadata = [m for m, keep in zip(filtered_metadata, corr_mask) if keep]
        
        return (
            filtered_distances.reshape(1, -1),
            filtered_indices.reshape(1, -1),
            [filtered_metadata]
        )
    
    def save(self, filepath: Optional[Path] = None, name: str = "pattern_index") -> Path:
        """
        Save index to disk
        
        Args:
            filepath: Path to save (uses default if None)
            name: Name of the index
        
        Returns:
            Path to saved file
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        if filepath is None:
            filepath = INDEX_DIR / f"{name}.faiss"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(filepath))
        
        # Save metadata separately
        meta_path = filepath.with_suffix(".meta")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "n_vectors": self.n_vectors
            }, f)
        
        logger.info(f"Saved index to {filepath}")
        return filepath
    
    def load(self, filepath: Optional[Path] = None, name: str = "pattern_index") -> None:
        """
        Load index from disk
        
        Args:
            filepath: Path to load (uses default if None)
            name: Name of the index
        """
        if filepath is None:
            filepath = INDEX_DIR / f"{name}.faiss"
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        # Load index
        self.index = faiss.read_index(str(filepath))
        
        # Load metadata
        meta_path = filepath.with_suffix(".meta")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta_data = pickle.load(f)
                self.metadata = meta_data.get("metadata", [])
                self.dimension = meta_data.get("dimension", self.dimension)
                self.index_type = meta_data.get("index_type", self.index_type)
                self.n_vectors = meta_data.get("n_vectors", 0)
        
        self.is_trained = True
        logger.info(f"Loaded index from {filepath}. Total vectors: {self.index.ntotal}")


def calculate_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Pearson correlation between two vectors
    ค่าความเหมือนที่แท้จริง (0-1 scale)
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Correlation coefficient (-1 to 1, where 1 = identical)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    
    # Handle edge cases
    if np.std(vec1) == 0 or np.std(vec2) == 0:
        return 0.0
    
    correlation = np.corrcoef(vec1, vec2)[0, 1]
    
    return float(correlation) if not np.isnan(correlation) else 0.0


def find_similar_patterns(
    query: np.ndarray,
    database: np.ndarray,
    k: int = 10,
    min_correlation: float = 0.85
) -> Tuple[List[int], List[float], List[float]]:
    """
    Find similar patterns with correlation check
    
    Args:
        query: Query pattern
        database: Database of patterns
        k: Number of results
        min_correlation: Minimum correlation threshold
    
    Returns:
        Tuple of (indices, distances, correlations)
    """
    # Build temporary index
    engine = FAISSEngine(dimension=len(query), index_type="Flat")
    engine.build_index(database)
    
    # Search
    distances, indices, _ = engine.search(query, k=k * 2)  # Get more to filter
    
    # Calculate actual correlations and filter
    results_indices = []
    results_distances = []
    results_correlations = []
    
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        
        corr = calculate_correlation(query, database[idx])
        
        if corr >= min_correlation:
            results_indices.append(int(idx))
            results_distances.append(float(dist))
            results_correlations.append(corr)
            
            if len(results_indices) >= k:
                break
    
    return results_indices, results_distances, results_correlations


class PatternMatcher:
    """
    High-level Pattern Matcher
    รวม FAISS + Correlation Check + Session Filter
    """
    
    def __init__(
        self,
        window_size: int = 60,
        index_type: str = "IVF",
        min_correlation: float = 0.85
    ):
        """
        Initialize Pattern Matcher
        
        Args:
            window_size: Pattern window size
            index_type: FAISS index type
            min_correlation: Minimum correlation threshold
        """
        self.window_size = window_size
        self.min_correlation = min_correlation
        
        self.engine = FAISSEngine(
            dimension=window_size,
            index_type=index_type
        )
        
        # Store original patterns for correlation check
        self.patterns: Optional[np.ndarray] = None
        self.futures: Optional[np.ndarray] = None
    
    def fit(
        self,
        patterns: np.ndarray,
        futures: np.ndarray,
        metadata: Optional[List[Any]] = None
    ) -> None:
        """
        Build pattern database
        
        Args:
            patterns: Array of patterns shape (n_patterns, window_size)
            futures: Array of future prices shape (n_patterns, future_candles)
            metadata: Optional metadata for each pattern
        """
        self.patterns = patterns
        self.futures = futures
        
        self.engine.build_index(patterns, metadata)
        
        logger.info(f"Pattern Matcher fitted with {len(patterns)} patterns")
    
    def find_matches(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Find matching patterns with full validation
        
        Args:
            query: Query pattern (normalized)
            k: Number of matches to find
        
        Returns:
            Dict with matches, correlations, futures
        """
        if self.patterns is None:
            raise ValueError("Pattern Matcher not fitted. Call fit first.")
        
        # Search FAISS
        distances, indices, metadata = self.engine.search(query, k=k * 2)
        
        # Validate with correlation
        matches = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.patterns):
                continue
            
            # Calculate correlation
            corr = calculate_correlation(query.flatten(), self.patterns[idx].flatten())
            
            if corr >= self.min_correlation:
                matches.append({
                    "index": int(idx),
                    "distance": float(dist),
                    "correlation": corr,
                    "pattern": self.patterns[idx],
                    "future": self.futures[idx] if self.futures is not None else None,
                    "metadata": metadata[0][indices[0].tolist().index(idx)] if metadata else None
                })
            
            if len(matches) >= k:
                break
        
        return {
            "n_matches": len(matches),
            "matches": matches,
            "query": query,
            "valid": len(matches) >= k // 2  # At least half the requested matches
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("Trademify FAISS Engine - Example Usage")
    print("=" * 50)
    
    # Create sample patterns
    np.random.seed(42)
    n_patterns = 10000
    window_size = 60
    
    # Generate random patterns
    print(f"\nGenerating {n_patterns} random patterns...")
    patterns = np.random.randn(n_patterns, window_size).astype(np.float32)
    
    # Add some similar patterns
    base_pattern = np.sin(np.linspace(0, 4 * np.pi, window_size))
    for i in range(100):
        noise = np.random.randn(window_size) * 0.1
        patterns[i] = base_pattern + noise
    
    # Build index
    print("\n" + "=" * 50)
    print("Building FAISS Index")
    print("=" * 50)
    
    engine = FAISSEngine(dimension=window_size, index_type="IVF")
    engine.build_index(patterns)
    
    # Search for similar patterns
    print("\n" + "=" * 50)
    print("Searching for Similar Patterns")
    print("=" * 50)
    
    query = base_pattern + np.random.randn(window_size) * 0.05
    
    distances, indices, metadata = engine.search(query.astype(np.float32), k=10)
    
    print(f"\nTop 10 similar patterns:")
    print(f"Indices: {indices[0]}")
    print(f"Distances: {distances[0]}")
    
    # Calculate correlations
    print("\nCorrelations with query:")
    for i, idx in enumerate(indices[0][:5]):
        corr = calculate_correlation(query, patterns[idx])
        print(f"  Pattern {idx}: distance={distances[0][i]:.4f}, correlation={corr:.4f}")
    
    # Save and load
    print("\n" + "=" * 50)
    print("Save and Load Index")
    print("=" * 50)
    
    engine.save(name="test_index")
    
    engine2 = FAISSEngine(dimension=window_size)
    engine2.load(name="test_index")
    
    distances2, indices2, _ = engine2.search(query.astype(np.float32), k=5)
    print(f"Loaded index search results: {indices2[0]}")
