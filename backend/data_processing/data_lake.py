"""
Data Lake Module - Phase 1: Data Preparation & Engineering
โมดูลจัดการข้อมูลกราฟย้อนหลัง

Features:
- ดึงข้อมูล OHLC ย้อนหลัง 5-10 ปี
- บันทึกในรูปแบบ .parquet หรือ .npy (เร็วกว่า Database 100 เท่า)
- รองรับหลาย Timeframe (M5, M15, H1)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Union
import yfinance as yf
import logging
from tqdm import tqdm

from config import DataConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLake:
    """
    คลังข้อมูลกราฟย้อนหลัง
    เก็บข้อมูลในรูปแบบที่อ่านเร็วที่สุด
    """
    
    def __init__(self, symbol: str, timeframe: str = "H1"):
        """
        Initialize DataLake
        
        Args:
            symbol: ชื่อสินทรัพย์ เช่น EURUSD, BTCUSDT
            timeframe: M5, M15, H1
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.raw_dir = RAW_DATA_DIR / symbol
        self.processed_dir = PROCESSED_DATA_DIR / symbol
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self._data: Optional[pd.DataFrame] = None
    
    @property
    def parquet_path(self) -> Path:
        """Path to parquet file"""
        return self.raw_dir / f"{self.timeframe}.parquet"
    
    @property
    def numpy_path(self) -> Path:
        """Path to numpy file"""
        return self.processed_dir / f"{self.timeframe}_processed.npy"
    
    def download_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = "yfinance"
    ) -> pd.DataFrame:
        """
        ดึงข้อมูลจาก Data Provider
        
        Args:
            start_date: วันเริ่มต้น (YYYY-MM-DD)
            end_date: วันสิ้นสุด (YYYY-MM-DD)
            source: แหล่งข้อมูล (yfinance, ccxt)
        
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * DataConfig.HISTORY_YEARS)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Downloading {self.symbol} data from {start_date} to {end_date}")
        
        # Map timeframe to yfinance interval
        tf_map = {
            "M5": "5m",
            "M15": "15m",
            "H1": "1h",
            "D1": "1d"
        }
        interval = tf_map.get(self.timeframe, "1h")
        
        # Map symbol for yfinance
        yf_symbol = self._convert_symbol_for_yfinance(self.symbol)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            # yfinance has limitations on historical data for intraday
            if self.timeframe in ["M5", "M15"]:
                # For intraday, we can only get 60 days max
                df = ticker.history(period="60d", interval=interval)
            else:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data downloaded for {self.symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Keep only OHLCV
            df = df[["open", "high", "low", "close", "volume"]]
            df.index.name = "datetime"
            
            self._data = df
            logger.info(f"Downloaded {len(df)} candles for {self.symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return pd.DataFrame()
    
    def _convert_symbol_for_yfinance(self, symbol: str) -> str:
        """Convert symbol format for yfinance"""
        # Forex pairs need =X suffix
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD", "USDCAD"]
        if symbol in forex_pairs:
            return f"{symbol}=X"
        
        # Gold
        if symbol == "XAUUSD":
            return "GC=F"
        
        # Crypto
        crypto_pairs = ["BTCUSDT", "ETHUSDT", "BTCUSD", "ETHUSD"]
        if symbol in crypto_pairs:
            return symbol.replace("USDT", "-USD").replace("USD", "-USD")
        
        return symbol
    
    def save_to_parquet(self, df: Optional[pd.DataFrame] = None) -> Path:
        """
        บันทึกข้อมูลเป็น Parquet (เร็วกว่า CSV 10-100 เท่า)
        
        Args:
            df: DataFrame to save (uses internal data if None)
        
        Returns:
            Path to saved file
        """
        if df is None:
            df = self._data
        
        if df is None or df.empty:
            raise ValueError("No data to save")
        
        df.to_parquet(self.parquet_path, compression="snappy")
        logger.info(f"Saved {len(df)} rows to {self.parquet_path}")
        
        return self.parquet_path
    
    def save_to_numpy(self, data: np.ndarray, metadata: Optional[dict] = None) -> Path:
        """
        บันทึกข้อมูลเป็น NumPy Binary (เร็วที่สุดสำหรับการอ่าน)
        
        Args:
            data: NumPy array to save
            metadata: Optional metadata dict
        
        Returns:
            Path to saved file
        """
        np.save(self.numpy_path, data)
        
        if metadata:
            meta_path = self.numpy_path.with_suffix(".meta.npy")
            np.save(meta_path, metadata, allow_pickle=True)
        
        logger.info(f"Saved numpy array shape {data.shape} to {self.numpy_path}")
        return self.numpy_path
    
    def load_from_parquet(self) -> pd.DataFrame:
        """โหลดข้อมูลจาก Parquet file"""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
        
        df = pd.read_parquet(self.parquet_path)
        self._data = df
        logger.info(f"Loaded {len(df)} rows from {self.parquet_path}")
        
        return df
    
    def load_from_numpy(self) -> np.ndarray:
        """โหลดข้อมูลจาก NumPy file"""
        if not self.numpy_path.exists():
            raise FileNotFoundError(f"NumPy file not found: {self.numpy_path}")
        
        data = np.load(self.numpy_path)
        logger.info(f"Loaded numpy array shape {data.shape}")
        
        return data
    
    def get_data(self) -> pd.DataFrame:
        """
        Get data (load from cache if available)
        """
        if self._data is not None:
            return self._data
        
        if self.parquet_path.exists():
            return self.load_from_parquet()
        
        # Download if not cached
        return self.download_data()


class DataGenerator:
    """
    Generate sample data for testing when real data is not available
    สร้างข้อมูลตัวอย่างสำหรับทดสอบระบบ
    """
    
    @staticmethod
    def generate_sample_ohlcv(
        n_candles: int = 10000,
        start_price: float = 1.0,
        volatility: float = 0.002,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate realistic OHLCV data using Geometric Brownian Motion
        
        Args:
            n_candles: Number of candles to generate
            start_price: Starting price
            volatility: Price volatility
            seed: Random seed for reproducibility
        
        Returns:
            DataFrame with OHLCV data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate returns using normal distribution
        returns = np.random.normal(0, volatility, n_candles)
        
        # Calculate close prices
        close = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close
        high_factor = np.abs(np.random.normal(0, volatility/2, n_candles))
        low_factor = np.abs(np.random.normal(0, volatility/2, n_candles))
        open_factor = np.random.normal(0, volatility/3, n_candles)
        
        high = close * (1 + high_factor)
        low = close * (1 - low_factor)
        open_prices = np.roll(close, 1) * (1 + open_factor)
        open_prices[0] = start_price
        
        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_prices, close))
        low = np.minimum(low, np.minimum(open_prices, close))
        
        # Generate volume
        volume = np.abs(np.random.normal(1000000, 200000, n_candles))
        
        # Create DataFrame
        dates = pd.date_range(end=datetime.now(), periods=n_candles, freq="1h")
        
        df = pd.DataFrame({
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }, index=dates)
        
        df.index.name = "datetime"
        
        return df
    
    @staticmethod
    def inject_patterns(
        df: pd.DataFrame,
        pattern_type: str = "double_bottom",
        n_patterns: int = 10
    ) -> pd.DataFrame:
        """
        Inject known patterns into data for testing
        ใส่ Pattern ที่รู้จักลงในข้อมูลเพื่อทดสอบระบบ
        
        Args:
            df: DataFrame to inject patterns into
            pattern_type: Type of pattern to inject
            n_patterns: Number of patterns to inject
        
        Returns:
            Modified DataFrame
        """
        df = df.copy()
        n = len(df)
        
        for _ in range(n_patterns):
            # Random position (not too close to edges)
            pos = np.random.randint(100, n - 100)
            
            if pattern_type == "double_bottom":
                # Create a W pattern (double bottom)
                pattern_length = 20
                pattern = np.array([0, -1, -2, -1.5, -2, -1, 0, 0.5, 1, 1.5])
                pattern = np.interp(
                    np.linspace(0, len(pattern)-1, pattern_length),
                    np.arange(len(pattern)),
                    pattern
                )
                
                # Scale pattern to current price level
                base_price = df.iloc[pos]["close"]
                scale = base_price * 0.01  # 1% of price
                
                for i, adj in enumerate(pattern):
                    if pos + i < n:
                        df.iloc[pos + i, df.columns.get_loc("close")] += adj * scale
            
            elif pattern_type == "head_shoulders":
                # Create an inverted head and shoulders pattern
                pattern_length = 25
                pattern = np.array([0, 1, 2, 1, 0, 1, 3, 1, 0, 1, 2, 1, 0])
                pattern = np.interp(
                    np.linspace(0, len(pattern)-1, pattern_length),
                    np.arange(len(pattern)),
                    pattern
                )
                
                base_price = df.iloc[pos]["close"]
                scale = base_price * 0.015
                
                for i, adj in enumerate(pattern):
                    if pos + i < n:
                        df.iloc[pos + i, df.columns.get_loc("close")] += adj * scale
        
        # Recalculate OHLC consistency
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        
        return df


def download_all_symbols(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None
) -> dict:
    """
    Download data for all configured symbols and timeframes
    ดาวน์โหลดข้อมูลทั้งหมดตามที่ตั้งค่าไว้
    
    Args:
        symbols: List of symbols (uses default if None)
        timeframes: List of timeframes (uses default if None)
    
    Returns:
        Dict with download status
    """
    if symbols is None:
        symbols = DataConfig.DEFAULT_SYMBOLS
    if timeframes is None:
        timeframes = DataConfig.TIMEFRAMES
    
    results = {}
    
    for symbol in tqdm(symbols, desc="Downloading symbols"):
        results[symbol] = {}
        
        for tf in timeframes:
            try:
                lake = DataLake(symbol, tf)
                df = lake.download_data()
                
                if not df.empty:
                    lake.save_to_parquet(df)
                    results[symbol][tf] = {"status": "success", "rows": len(df)}
                else:
                    results[symbol][tf] = {"status": "empty", "rows": 0}
                    
            except Exception as e:
                results[symbol][tf] = {"status": "error", "error": str(e)}
                logger.error(f"Failed to download {symbol} {tf}: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("Trademify Data Lake - Example Usage")
    print("=" * 50)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    sample_df = DataGenerator.generate_sample_ohlcv(n_candles=5000, seed=42)
    print(f"Generated {len(sample_df)} candles")
    print(sample_df.head())
    
    # Inject patterns for testing
    print("\n2. Injecting test patterns...")
    sample_df = DataGenerator.inject_patterns(sample_df, "double_bottom", 5)
    
    # Save to Data Lake
    print("\n3. Saving to Data Lake...")
    lake = DataLake("SAMPLE", "H1")
    lake._data = sample_df
    lake.save_to_parquet()
    
    # Load back
    print("\n4. Loading from Data Lake...")
    loaded_df = lake.load_from_parquet()
    print(f"Loaded {len(loaded_df)} candles")
