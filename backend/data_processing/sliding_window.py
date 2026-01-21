"""
Sliding Window Module - Phase 1: Data Preparation
ตัดข้อมูลเป็นท่อนๆ เพื่อเตรียมเป็นฐานข้อมูลในการค้นหา

แต่ละ Window คือ Pattern หนึ่งอัน ที่พร้อมจะถูก Index และค้นหา
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from config import DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WindowMetadata:
    """Metadata for each window"""
    index: int
    start_time: datetime
    end_time: datetime
    start_idx: int
    end_idx: int
    symbol: str
    timeframe: str
    
    # Original price info for reconstruction
    first_price: float
    last_price: float
    high: float
    low: float
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "first_price": self.first_price,
            "last_price": self.last_price,
            "high": self.high,
            "low": self.low,
        }


class SlidingWindowGenerator:
    """
    สร้าง Sliding Windows จากข้อมูล OHLC
    เพื่อเตรียมเป็นฐานข้อมูลสำหรับ FAISS
    """
    
    def __init__(
        self,
        window_size: int = 60,
        stride: int = 1,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize Sliding Window Generator
        
        Args:
            window_size: จำนวนแท่งเทียนในแต่ละ window
            stride: ระยะห่างระหว่าง window (1 = ทับกันมาก, window_size = ไม่ทับกัน)
            feature_columns: Columns to use (default: close only)
        """
        self.window_size = window_size
        self.stride = stride
        self.feature_columns = feature_columns or ["close"]
        
        self.metadata: List[WindowMetadata] = []
    
    def create_windows(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "H1",
        include_future: bool = True,
        future_candles: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[WindowMetadata]]:
        """
        Create sliding windows from OHLC DataFrame
        
        Args:
            df: DataFrame with OHLC data (must have datetime index)
            symbol: Symbol name for metadata
            timeframe: Timeframe for metadata
            include_future: Whether to include future candles for voting
            future_candles: Number of future candles to include
        
        Returns:
            Tuple of (windows, futures, metadata)
            - windows: shape (n_windows, window_size, n_features)
            - futures: shape (n_windows, future_candles) - future close prices
            - metadata: list of WindowMetadata
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Extract features
        features = []
        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")
            features.append(df[col].values)
        
        # Stack features
        data = np.column_stack(features) if len(features) > 1 else features[0]
        
        n_samples = len(data)
        n_features = len(self.feature_columns)
        
        # Calculate number of windows
        if include_future:
            max_idx = n_samples - self.window_size - future_candles
        else:
            max_idx = n_samples - self.window_size
        
        n_windows = (max_idx // self.stride) + 1
        
        if n_windows <= 0:
            raise ValueError(f"Not enough data. Need at least {self.window_size + future_candles} samples, got {n_samples}")
        
        logger.info(f"Creating {n_windows} windows from {n_samples} samples")
        
        # Pre-allocate arrays
        if n_features == 1:
            windows = np.zeros((n_windows, self.window_size))
        else:
            windows = np.zeros((n_windows, self.window_size, n_features))
        
        futures = np.zeros((n_windows, future_candles)) if include_future else None
        metadata = []
        
        # Get datetime index
        timestamps = df.index.to_list() if hasattr(df.index, 'to_list') else list(range(len(df)))
        close_prices = df["close"].values
        high_prices = df["high"].values
        low_prices = df["low"].values
        
        # Create windows
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            # Extract window
            windows[i] = data[start_idx:end_idx]
            
            # Extract future prices
            if include_future:
                future_start = end_idx
                future_end = end_idx + future_candles
                futures[i] = close_prices[future_start:future_end]
            
            # Create metadata
            meta = WindowMetadata(
                index=i,
                start_time=timestamps[start_idx] if isinstance(timestamps[0], datetime) else None,
                end_time=timestamps[end_idx - 1] if isinstance(timestamps[0], datetime) else None,
                start_idx=start_idx,
                end_idx=end_idx,
                symbol=symbol,
                timeframe=timeframe,
                first_price=float(close_prices[start_idx]),
                last_price=float(close_prices[end_idx - 1]),
                high=float(high_prices[start_idx:end_idx].max()),
                low=float(low_prices[start_idx:end_idx].min()),
            )
            metadata.append(meta)
        
        self.metadata = metadata
        
        logger.info(f"Created windows shape: {windows.shape}")
        if futures is not None:
            logger.info(f"Created futures shape: {futures.shape}")
        
        return windows, futures, metadata
    
    def create_normalized_windows(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "H1",
        norm_method: str = "zscore",
        include_future: bool = True,
        future_candles: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[WindowMetadata]]:
        """
        Create sliding windows with per-window normalization
        แต่ละ window จะถูก normalize ด้วยค่าเฉลี่ยและส่วนเบี่ยงเบนของตัวเอง
        
        Args:
            df: DataFrame with OHLC data
            symbol: Symbol name
            timeframe: Timeframe
            norm_method: Normalization method (zscore, minmax, log_return)
            include_future: Include future candles
            future_candles: Number of future candles
        
        Returns:
            Tuple of (normalized_windows, futures, metadata)
        """
        # First create raw windows
        windows, futures, metadata = self.create_windows(
            df, symbol, timeframe, include_future, future_candles
        )
        
        # Normalize each window
        normalized_windows = np.zeros_like(windows)
        
        for i in range(len(windows)):
            window = windows[i]
            
            if norm_method == "zscore":
                mean = np.mean(window)
                std = np.std(window)
                std = std if std > 0 else 1.0
                normalized_windows[i] = (window - mean) / std
                
            elif norm_method == "minmax":
                min_val = np.min(window)
                max_val = np.max(window)
                range_val = max_val - min_val
                range_val = range_val if range_val > 0 else 1.0
                normalized_windows[i] = (window - min_val) / range_val
                
            elif norm_method == "log_return":
                # Log return normalized
                window = np.maximum(window, 1e-10)
                log_returns = np.log(window[1:] / window[:-1])
                normalized_windows[i] = np.concatenate([[0], log_returns])
                
            else:
                normalized_windows[i] = window
        
        logger.info(f"Normalized windows using {norm_method} method")
        
        return normalized_windows, futures, metadata


class SessionFilter:
    """
    Filter windows by trading session
    แยก Pattern ตามช่วงเวลาการเทรด
    """
    
    SESSIONS = {
        "ASIA": {"start_hour": 0, "end_hour": 8},      # 00:00 - 08:00 UTC
        "LONDON": {"start_hour": 8, "end_hour": 16},   # 08:00 - 16:00 UTC
        "NEW_YORK": {"start_hour": 13, "end_hour": 22}, # 13:00 - 22:00 UTC
        "OVERLAP": {"start_hour": 13, "end_hour": 16},  # London-NY overlap
    }
    
    @classmethod
    def filter_by_session(
        cls,
        windows: np.ndarray,
        metadata: List[WindowMetadata],
        session: str
    ) -> Tuple[np.ndarray, List[WindowMetadata]]:
        """
        Filter windows by trading session
        
        Args:
            windows: Array of windows
            metadata: List of metadata
            session: Session name (ASIA, LONDON, NEW_YORK, OVERLAP)
        
        Returns:
            Filtered (windows, metadata)
        """
        if session not in cls.SESSIONS:
            raise ValueError(f"Unknown session: {session}. Available: {list(cls.SESSIONS.keys())}")
        
        session_config = cls.SESSIONS[session]
        start_hour = session_config["start_hour"]
        end_hour = session_config["end_hour"]
        
        filtered_indices = []
        filtered_metadata = []
        
        for i, meta in enumerate(metadata):
            if meta.start_time is None:
                continue
            
            hour = meta.start_time.hour
            
            if start_hour <= hour < end_hour:
                filtered_indices.append(i)
                filtered_metadata.append(meta)
        
        if not filtered_indices:
            logger.warning(f"No windows found for session {session}")
            return np.array([]), []
        
        filtered_windows = windows[filtered_indices]
        
        logger.info(f"Filtered {len(filtered_indices)} windows for {session} session")
        
        return filtered_windows, filtered_metadata


def prepare_database(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    window_size: int = 60,
    future_candles: int = 10,
    norm_method: str = "zscore"
) -> Dict[str, Any]:
    """
    Prepare complete database for FAISS indexing
    เตรียมฐานข้อมูลสำเร็จรูปสำหรับ FAISS
    
    Args:
        df: OHLC DataFrame
        symbol: Symbol name
        timeframe: Timeframe
        window_size: Window size
        future_candles: Future candles for voting
        norm_method: Normalization method
    
    Returns:
        Dict with windows, futures, metadata
    """
    generator = SlidingWindowGenerator(
        window_size=window_size,
        stride=1,
        feature_columns=["close"]
    )
    
    windows, futures, metadata = generator.create_normalized_windows(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        norm_method=norm_method,
        include_future=True,
        future_candles=future_candles
    )
    
    return {
        "windows": windows,
        "futures": futures,
        "metadata": metadata,
        "config": {
            "symbol": symbol,
            "timeframe": timeframe,
            "window_size": window_size,
            "future_candles": future_candles,
            "norm_method": norm_method,
            "n_windows": len(windows)
        }
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("Trademify Sliding Window - Example Usage")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="1h")
    
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    
    df = pd.DataFrame({
        "open": prices + np.random.randn(1000) * 0.1,
        "high": prices + np.abs(np.random.randn(1000) * 0.3),
        "low": prices - np.abs(np.random.randn(1000) * 0.3),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    print(f"\nSample data shape: {df.shape}")
    print(df.head())
    
    # Create windows
    print("\n" + "=" * 50)
    print("Creating Sliding Windows")
    print("=" * 50)
    
    generator = SlidingWindowGenerator(
        window_size=60,
        stride=1,
        feature_columns=["close"]
    )
    
    windows, futures, metadata = generator.create_normalized_windows(
        df=df,
        symbol="EURUSD",
        timeframe="H1",
        norm_method="zscore",
        include_future=True,
        future_candles=10
    )
    
    print(f"\nWindows shape: {windows.shape}")
    print(f"Futures shape: {futures.shape}")
    print(f"Number of metadata records: {len(metadata)}")
    
    # Show sample metadata
    print(f"\nSample metadata:")
    print(metadata[0].to_dict())
    
    # Filter by session
    print("\n" + "=" * 50)
    print("Filtering by Session")
    print("=" * 50)
    
    for session in ["ASIA", "LONDON", "NEW_YORK"]:
        filtered_windows, filtered_meta = SessionFilter.filter_by_session(
            windows, metadata, session
        )
        print(f"{session}: {len(filtered_windows)} windows")
