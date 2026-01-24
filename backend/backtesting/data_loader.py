"""
Historical Data Loader for Backtesting
โหลดข้อมูลย้อนหลังจากหลายแหล่ง:
- MT5 (ถ้าเชื่อมต่อได้)
- Binance API
- Local CSV files
- Yahoo Finance (backup)
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """
    โหลดข้อมูลราคาย้อนหลังสำหรับ Backtest
    รองรับหลาย source และ cache ข้อมูล
    """
    
    # Symbol mapping for different sources
    SYMBOL_MAP = {
        # MT5/Exness symbols -> Other sources
        "EURUSDm": {"binance": None, "yahoo": "EURUSD=X", "mt5": "EURUSDm"},
        "GBPUSDm": {"binance": None, "yahoo": "GBPUSD=X", "mt5": "GBPUSDm"},
        "XAUUSDm": {"binance": None, "yahoo": "GC=F", "mt5": "XAUUSDm"},
        "EURUSD": {"binance": None, "yahoo": "EURUSD=X", "mt5": "EURUSD"},
        "GBPUSD": {"binance": None, "yahoo": "GBPUSD=X", "mt5": "GBPUSD"},
        "XAUUSD": {"binance": None, "yahoo": "GC=F", "mt5": "XAUUSD"},
        
        # Crypto symbols
        "BTCUSDT": {"binance": "BTCUSDT", "yahoo": "BTC-USD", "mt5": None},
        "ETHUSDT": {"binance": "ETHUSDT", "yahoo": "ETH-USD", "mt5": None},
        "BNBUSDT": {"binance": "BNBUSDT", "yahoo": "BNB-USD", "mt5": None},
    }
    
    # Timeframe mapping
    TF_MAP = {
        "M1": {"mt5": "TIMEFRAME_M1", "period": 1, "yf_interval": "1m"},
        "M5": {"mt5": "TIMEFRAME_M5", "period": 5, "yf_interval": "5m"},
        "M15": {"mt5": "TIMEFRAME_M15", "period": 15, "yf_interval": "15m"},
        "M30": {"mt5": "TIMEFRAME_M30", "period": 30, "yf_interval": "30m"},
        "H1": {"mt5": "TIMEFRAME_H1", "period": 60, "yf_interval": "1h"},
        "H4": {"mt5": "TIMEFRAME_H4", "period": 240, "yf_interval": "1h"},
        "D1": {"mt5": "TIMEFRAME_D1", "period": 1440, "yf_interval": "1d"},
    }
    
    def __init__(self, cache_dir: str = "data/backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._mt5 = None
        self._mt5_connected = False
        
    async def connect_mt5(self) -> bool:
        """เชื่อมต่อ MT5"""
        if self._mt5_connected:
            return True
            
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            
            if not mt5.initialize():
                return False
                
            # Login
            mt5_login = int(os.getenv("MT5_LOGIN", "0"))
            mt5_password = os.getenv("MT5_PASSWORD", "")
            mt5_server = os.getenv("MT5_SERVER", "")
            
            if mt5_login > 0 and mt5_password:
                if not mt5.login(mt5_login, mt5_password, mt5_server, timeout=60000):
                    return False
                    
            self._mt5_connected = True
            logger.info("✅ MT5 connected for backtest data")
            return True
            
        except ImportError:
            logger.warning("MetaTrader5 not installed")
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def _get_cache_path(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
        """สร้าง path สำหรับ cache file"""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        filename = f"{symbol}_{timeframe}_{start_str}_{end_str}.parquet"
        return self.cache_dir / filename
    
    def _load_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """โหลดข้อมูลจาก cache"""
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"📦 Loaded from cache: {cache_path.name} ({len(df)} candles)")
                return df
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
    
    def _save_cache(self, df: pd.DataFrame, cache_path: Path):
        """บันทึกข้อมูลลง cache"""
        try:
            df.to_parquet(cache_path)
            logger.info(f"💾 Saved to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def load_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        years: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
        force_source: Optional[str] = None  # "mt5", "binance", "yahoo"
    ) -> pd.DataFrame:
        """
        โหลดข้อมูลราคาย้อนหลัง
        
        Args:
            symbol: สัญลักษณ์ (เช่น EURUSDm, BTCUSDT)
            timeframe: ช่วงเวลา (M1, M5, M15, M30, H1, H4, D1)
            years: จำนวนปีย้อนหลัง (default: 10)
            start_date: วันเริ่มต้น (ถ้าไม่ระบุ จะคำนวณจาก years)
            end_date: วันสิ้นสุด (ถ้าไม่ระบุ ใช้วันปัจจุบัน)
            use_cache: ใช้ cache หรือไม่
            force_source: บังคับใช้แหล่งข้อมูลเฉพาะ
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=years * 365)
            
        logger.info(f"📊 Loading {symbol} {timeframe} from {start_date.date()} to {end_date.date()} ({years} years)")
        
        # Check cache
        if use_cache:
            cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached
        
        # Determine best data source
        df = pd.DataFrame()
        
        if force_source:
            sources = [force_source]
        else:
            # Try multiple sources
            sources = self._get_source_priority(symbol)
        
        for source in sources:
            try:
                if source == "mt5":
                    df = await self._load_from_mt5(symbol, timeframe, start_date, end_date)
                elif source == "binance":
                    df = await self._load_from_binance(symbol, timeframe, start_date, end_date)
                elif source == "yahoo":
                    df = await self._load_from_yahoo(symbol, timeframe, start_date, end_date)
                elif source == "local":
                    df = self._load_from_local(symbol, timeframe, start_date, end_date)
                    
                if df is not None and not df.empty:
                    logger.info(f"✅ Loaded {len(df)} candles from {source}")
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load from {source}: {e}")
                continue
        
        if df.empty:
            logger.error(f"❌ Could not load data for {symbol} from any source")
            return df
        
        # Validate and clean data
        df = self._clean_data(df)
        
        # Save to cache
        if use_cache and not df.empty:
            self._save_cache(df, cache_path)
        
        return df
    
    def _get_source_priority(self, symbol: str) -> List[str]:
        """กำหนดลำดับความสำคัญของแหล่งข้อมูล"""
        symbol_info = self.SYMBOL_MAP.get(symbol, {})
        sources = []
        
        # Forex/CFD → MT5 first
        if symbol_info.get("mt5"):
            sources.append("mt5")
            
        # Crypto → Binance first
        if symbol_info.get("binance"):
            sources.insert(0, "binance")
            
        # Yahoo as backup for everything
        if symbol_info.get("yahoo"):
            sources.append("yahoo")
            
        # Local files as last resort
        sources.append("local")
        
        return sources if sources else ["yahoo", "local"]
    
    async def _load_from_mt5(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """โหลดข้อมูลจาก MT5"""
        if not await self.connect_mt5():
            return pd.DataFrame()
            
        try:
            import MetaTrader5 as mt5
            
            # Get MT5 timeframe constant
            tf_name = self.TF_MAP.get(timeframe, {}).get("mt5", "TIMEFRAME_H1")
            mt5_tf = getattr(mt5, tf_name, mt5.TIMEFRAME_H1)
            
            # Enable symbol
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Cannot select symbol {symbol}")
                return pd.DataFrame()
            
            # Get historical data
            rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('datetime')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"MT5 data error: {e}")
            return pd.DataFrame()
    
    async def _load_from_binance(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """โหลดข้อมูลจาก Binance API"""
        try:
            from data_processing.binance_data import BinanceDataProvider
            
            binance_symbol = self.SYMBOL_MAP.get(symbol, {}).get("binance", symbol)
            if not binance_symbol:
                return pd.DataFrame()
            
            provider = BinanceDataProvider()
            
            # Calculate days
            days = (end_date - start_date).days
            
            try:
                # Get historical data
                df = await provider.get_historical_klines(
                    symbol=binance_symbol,
                    timeframe=timeframe,
                    days=min(days, 365 * 5)  # Binance limit ~5 years for most pairs
                )
                return df
            finally:
                await provider.close()
                
        except Exception as e:
            logger.error(f"Binance data error: {e}")
            return pd.DataFrame()
    
    async def _load_from_yahoo(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """โหลดข้อมูลจาก Yahoo Finance (10+ years supported)"""
        try:
            import yfinance as yf
            
            yahoo_symbol = self.SYMBOL_MAP.get(symbol, {}).get("yahoo", symbol)
            if not yahoo_symbol:
                # Try to guess Yahoo symbol
                if "USD" in symbol:
                    yahoo_symbol = symbol.replace("m", "") + "=X"
                else:
                    yahoo_symbol = symbol
            
            logger.info(f"📥 Fetching from Yahoo Finance: {yahoo_symbol}")
            
            # Get interval
            interval = self.TF_MAP.get(timeframe, {}).get("yf_interval", "1h")
            
            # Yahoo limits for intraday data
            if interval in ["1m", "5m", "15m", "30m"]:
                # Yahoo only supports 7-60 days for intraday
                max_days = 60 if interval in ["15m", "30m", "1h"] else 7
                if (end_date - start_date).days > max_days:
                    logger.warning(f"Yahoo intraday limited to {max_days} days, switching to 1h")
                    interval = "1h"
            
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                return df
            
            # Rename columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except ImportError:
            logger.warning("yfinance not installed: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Yahoo data error: {e}")
            return pd.DataFrame()
    
    def _load_from_local(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """โหลดข้อมูลจากไฟล์ CSV/Parquet ใน local"""
        # Check common file patterns
        patterns = [
            f"data/{symbol}_{timeframe}.csv",
            f"data/{symbol}_{timeframe}.parquet",
            f"data/historical/{symbol}_{timeframe}.csv",
            f"data/backtest/{symbol}.csv",
        ]
        
        for pattern in patterns:
            path = Path(pattern)
            if path.exists():
                try:
                    if path.suffix == ".parquet":
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path, index_col=0, parse_dates=True)
                    
                    # Filter date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if not df.empty:
                        logger.info(f"📂 Loaded from local: {path}")
                        return df
                        
                except Exception as e:
                    logger.warning(f"Error reading {path}: {e}")
        
        return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ทำความสะอาดและตรวจสอบข้อมูล"""
        if df.empty:
            return df
            
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by datetime
        df = df.sort_index()
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Remove rows with zero/negative values
        for col in ['open', 'high', 'low', 'close']:
            df = df[df[col] > 0]
        
        # Validate OHLC logic
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['open']]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['open']]
        df = df[df['low'] <= df['close']]
        
        return df
    
    def get_available_data_range(self, symbol: str) -> Dict[str, Any]:
        """ตรวจสอบช่วงข้อมูลที่มีอยู่"""
        # Check cache
        cache_files = list(self.cache_dir.glob(f"{symbol}_*.parquet"))
        
        if cache_files:
            start_dates = []
            end_dates = []
            total_candles = 0
            
            for f in cache_files:
                try:
                    df = pd.read_parquet(f)
                    start_dates.append(df.index.min())
                    end_dates.append(df.index.max())
                    total_candles += len(df)
                except:
                    pass
            
            if start_dates:
                return {
                    "symbol": symbol,
                    "earliest": min(start_dates),
                    "latest": max(end_dates),
                    "cached_files": len(cache_files),
                    "total_candles": total_candles
                }
        
        return {"symbol": symbol, "cached_files": 0}
    
    async def download_all_symbols(
        self,
        symbols: List[str],
        timeframe: str = "H1",
        years: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """ดาวน์โหลดข้อมูลหลาย symbols พร้อมกัน"""
        results = {}
        
        for symbol in symbols:
            logger.info(f"📥 Downloading {symbol}...")
            df = await self.load_data(symbol, timeframe, years)
            if not df.empty:
                results[symbol] = df
            else:
                logger.warning(f"⚠️ No data for {symbol}")
        
        return results


# Synchronous wrapper
def load_backtest_data(
    symbol: str,
    timeframe: str = "H1",
    years: int = 10
) -> pd.DataFrame:
    """
    Synchronous wrapper สำหรับโหลดข้อมูล backtest
    
    Example:
        df = load_backtest_data("EURUSDm", "H1", 10)
    """
    async def _load():
        loader = HistoricalDataLoader()
        return await loader.load_data(symbol, timeframe, years)
    
    return asyncio.run(_load())


if __name__ == "__main__":
    # Test data loading
    import asyncio
    
    async def test():
        loader = HistoricalDataLoader()
        
        # Test loading EURUSD
        df = await loader.load_data(
            symbol="EURUSD",
            timeframe="H1",
            years=1  # Test with 1 year first
        )
        
        print(f"\nLoaded {len(df)} candles")
        if not df.empty:
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"\nSample data:")
            print(df.head())
    
    asyncio.run(test())
