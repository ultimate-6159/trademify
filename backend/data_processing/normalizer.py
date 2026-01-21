"""
Normalizer Module - Phase 1: Data Preparation
โมดูล Normalization เพื่อให้กราฟที่มี "ทรงเดียวกัน" แต่ "คนละราคา" สามารถจับคู่กันได้

หลักการ: Garbage In, Garbage Out
ถ้า Normalize ไม่ดี Pattern Matching จะพัง!

Methods:
- Z-Score: (x - mean) / std
- Min-Max: (x - min) / (max - min)
- Log Return: log(price_t / price_t-1)
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal
from enum import Enum


class NormMethod(str, Enum):
    """Normalization methods"""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    LOG_RETURN = "log_return"
    PERCENT_CHANGE = "percent_change"


class Normalizer:
    """
    Normalizer สำหรับแปลงราคาให้เป็นค่ากลาง
    ทำให้กราฟต่างราคาสามารถเทียบกันได้
    """
    
    def __init__(self, method: str = "zscore", window: int = 20):
        """
        Initialize Normalizer
        
        Args:
            method: normalization method (zscore, minmax, log_return, percent_change)
            window: rolling window size for zscore calculation
        """
        self.method = NormMethod(method.lower())
        self.window = window
        
        # Store parameters for inverse transform
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None
    
    def normalize(
        self,
        data: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize data
        
        Args:
            data: Input array (1D or 2D)
            fit: Whether to fit parameters (False for transform only)
        
        Returns:
            Normalized array
        """
        if self.method == NormMethod.ZSCORE:
            return self._zscore_normalize(data, fit)
        elif self.method == NormMethod.MINMAX:
            return self._minmax_normalize(data, fit)
        elif self.method == NormMethod.LOG_RETURN:
            return self._log_return(data)
        elif self.method == NormMethod.PERCENT_CHANGE:
            return self._percent_change(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _zscore_normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Z-Score Normalization: (x - mean) / std
        ทำให้ค่าเฉลี่ย = 0, ส่วนเบี่ยงเบน = 1
        """
        if fit:
            self._mean = np.mean(data)
            self._std = np.std(data)
            if self._std == 0:
                self._std = 1.0  # Prevent division by zero
        
        return (data - self._mean) / self._std
    
    def _minmax_normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Min-Max Normalization: (x - min) / (max - min)
        ทำให้ค่าอยู่ระหว่าง 0 ถึง 1
        """
        if fit:
            self._min = np.min(data)
            self._max = np.max(data)
            if self._max == self._min:
                self._max = self._min + 1.0  # Prevent division by zero
        
        return (data - self._min) / (self._max - self._min)
    
    def _log_return(self, data: np.ndarray) -> np.ndarray:
        """
        Log Return: log(price_t / price_t-1)
        ดีที่สุดสำหรับการเทียบ pattern ข้ามเวลา
        """
        # Ensure positive values
        data = np.maximum(data, 1e-10)
        
        if data.ndim == 1:
            returns = np.log(data[1:] / data[:-1])
            # Pad with 0 at the beginning
            return np.concatenate([[0], returns])
        else:
            # For 2D array (multiple features)
            returns = np.log(data[1:] / data[:-1])
            return np.vstack([np.zeros(data.shape[1]), returns])
    
    def _percent_change(self, data: np.ndarray) -> np.ndarray:
        """
        Percent Change: (price_t - price_t-1) / price_t-1
        ง่ายต่อการเข้าใจ
        """
        if data.ndim == 1:
            pct = (data[1:] - data[:-1]) / np.maximum(data[:-1], 1e-10)
            return np.concatenate([[0], pct])
        else:
            pct = (data[1:] - data[:-1]) / np.maximum(data[:-1], 1e-10)
            return np.vstack([np.zeros(data.shape[1]), pct])
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        แปลงกลับเป็นค่าจริง (สำหรับ zscore และ minmax เท่านั้น)
        
        Args:
            normalized_data: Normalized array
        
        Returns:
            Original scale array
        """
        if self.method == NormMethod.ZSCORE:
            if self._mean is None or self._std is None:
                raise ValueError("Normalizer not fitted. Call normalize with fit=True first.")
            return normalized_data * self._std + self._mean
        
        elif self.method == NormMethod.MINMAX:
            if self._min is None or self._max is None:
                raise ValueError("Normalizer not fitted. Call normalize with fit=True first.")
            return normalized_data * (self._max - self._min) + self._min
        
        else:
            raise ValueError(f"Inverse transform not supported for {self.method}")


def normalize_ohlc(
    df: pd.DataFrame,
    method: str = "zscore",
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Normalize OHLC DataFrame
    
    Args:
        df: DataFrame with OHLC data
        method: Normalization method
        columns: Columns to normalize (default: open, high, low, close)
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = ["open", "high", "low", "close"]
    
    df_norm = df.copy()
    normalizer = Normalizer(method=method)
    
    for col in columns:
        if col in df_norm.columns:
            df_norm[col] = normalizer.normalize(df_norm[col].values)
    
    return df_norm


def normalize_window(
    window: np.ndarray,
    method: str = "zscore"
) -> Tuple[np.ndarray, dict]:
    """
    Normalize a single window and return parameters for inverse transform
    
    Args:
        window: Array of shape (window_size,) or (window_size, n_features)
        method: Normalization method
    
    Returns:
        Tuple of (normalized_window, parameters)
    """
    normalizer = Normalizer(method=method)
    normalized = normalizer.normalize(window)
    
    params = {
        "mean": normalizer._mean,
        "std": normalizer._std,
        "min": normalizer._min,
        "max": normalizer._max,
        "method": method
    }
    
    return normalized, params


def create_normalized_sliding_windows(
    data: np.ndarray,
    window_size: int = 60,
    stride: int = 1,
    method: str = "zscore"
) -> Tuple[np.ndarray, list]:
    """
    Create normalized sliding windows from data
    ตัดข้อมูลเป็นท่อนๆ และ Normalize แต่ละท่อน
    
    Args:
        data: 1D array of prices
        window_size: Size of each window (จำนวนแท่งเทียน)
        stride: Step size between windows
        method: Normalization method
    
    Returns:
        Tuple of (windows_array, parameters_list)
    """
    n = len(data)
    n_windows = (n - window_size) // stride + 1
    
    windows = []
    params_list = []
    
    for i in range(0, n - window_size + 1, stride):
        window = data[i:i + window_size]
        normalized, params = normalize_window(window, method)
        
        params["start_idx"] = i
        params["end_idx"] = i + window_size
        
        windows.append(normalized)
        params_list.append(params)
    
    return np.array(windows), params_list


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("Trademify Normalizer - Example Usage")
    print("=" * 50)
    
    # Create sample price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    print(f"\nOriginal prices (first 10): {prices[:10]}")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Z-Score normalization
    normalizer = Normalizer(method="zscore")
    normalized = normalizer.normalize(prices)
    print(f"\nZ-Score normalized (first 10): {normalized[:10]}")
    print(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
    
    # Inverse transform
    recovered = normalizer.inverse_transform(normalized)
    print(f"\nRecovered prices (first 10): {recovered[:10]}")
    print(f"Recovery error: {np.abs(prices - recovered).max():.10f}")
    
    # Log return
    normalizer_log = Normalizer(method="log_return")
    log_returns = normalizer_log.normalize(prices)
    print(f"\nLog returns (first 10): {log_returns[:10]}")
    
    # Create sliding windows
    print("\n" + "=" * 50)
    print("Creating Sliding Windows")
    print("=" * 50)
    
    windows, params = create_normalized_sliding_windows(
        prices,
        window_size=20,
        stride=5,
        method="zscore"
    )
    
    print(f"Created {len(windows)} windows of shape {windows[0].shape}")
    print(f"Each window is normalized with its own parameters")
    print(f"Window 0 params: mean={params[0]['mean']:.2f}, std={params[0]['std']:.2f}")
