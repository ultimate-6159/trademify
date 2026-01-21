"""
Binance Data Provider
ดึงข้อมูลราคา OHLCV จาก Binance โดยตรง (Real-time & Historical)
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Timeframe mapping
TIMEFRAME_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1w",
}

# Binance API limits
MAX_CANDLES_PER_REQUEST = 1000


class BinanceDataProvider:
    """
    ดึงข้อมูลจาก Binance API
    - รองรับทั้ง Spot และ Futures
    - ดึงข้อมูลย้อนหลังได้หลายปี
    - Real-time WebSocket (optional)
    """
    
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self._session = None
        
        if testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
    
    async def _ensure_session(self):
        """Create aiohttp session if needed"""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
    
    async def close(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def get_klines(
        self,
        symbol: str,
        timeframe: str = "H1",
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        ดึงข้อมูล Klines (OHLCV) จาก Binance
        
        Args:
            symbol: เช่น BTCUSDT, ETHUSDT
            timeframe: M1, M5, M15, M30, H1, H4, D1
            limit: จำนวนแท่งเทียน (max 1000)
            start_time: เวลาเริ่มต้น
            end_time: เวลาสิ้นสุด
            
        Returns:
            DataFrame with OHLCV
        """
        await self._ensure_session()
        
        interval = TIMEFRAME_MAP.get(timeframe, "1h")
        
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, MAX_CANDLES_PER_REQUEST)
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            url = f"{self.base_url}/api/v3/klines"
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Binance API error: {error}")
                    return pd.DataFrame()
                
                data = await resp.json()
                
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            
            # Process data
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.set_index("datetime")
            
            # Convert to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            
            # Keep only OHLCV
            df = df[["open", "high", "low", "close", "volume"]]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()
    
    async def get_historical_klines(
        self,
        symbol: str,
        timeframe: str = "H1",
        days: int = 365
    ) -> pd.DataFrame:
        """
        ดึงข้อมูลย้อนหลังหลายวัน (ดึงทีละ 1000 แท่งเทียน)
        
        Args:
            symbol: เช่น BTCUSDT
            timeframe: Timeframe
            days: จำนวนวันย้อนหลัง
            
        Returns:
            DataFrame with historical OHLCV
        """
        await self._ensure_session()
        
        all_data = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Timeframe to milliseconds
        tf_ms = {
            "M1": 60000,
            "M5": 300000,
            "M15": 900000,
            "M30": 1800000,
            "H1": 3600000,
            "H4": 14400000,
            "D1": 86400000,
        }
        
        interval_ms = tf_ms.get(timeframe, 3600000)
        current_end = end_time
        
        logger.info(f"Downloading {symbol} {timeframe} from {start_time} to {end_time}")
        
        while current_end > start_time:
            df = await self.get_klines(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000,
                end_time=current_end
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Move end time back
            current_end = df.index[0].to_pydatetime() - timedelta(milliseconds=interval_ms)
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(all_data)
        result = result.sort_index()
        result = result[~result.index.duplicated(keep='first')]
        
        logger.info(f"Downloaded {len(result)} candles for {symbol}")
        
        return result
    
    async def get_current_price(self, symbol: str) -> float:
        """ดึงราคาปัจจุบัน"""
        await self._ensure_session()
        
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol.upper()}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data["price"])
                    
        except Exception as e:
            logger.error(f"Error getting price: {e}")
        
        return 0.0
    
    async def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        """ดึงสถิติ 24 ชั่วโมง"""
        await self._ensure_session()
        
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {"symbol": symbol.upper()}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                    
        except Exception as e:
            logger.error(f"Error getting 24h stats: {e}")
        
        return {}


class RealTimeDataStream:
    """
    WebSocket stream สำหรับข้อมูล real-time
    """
    
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self._ws = None
        self._running = False
        self._callbacks: Dict[str, List] = {}
        
        if testnet:
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            self.ws_url = "wss://stream.binance.com:9443/ws"
    
    async def subscribe_klines(
        self,
        symbol: str,
        timeframe: str = "M1",
        callback=None
    ):
        """
        Subscribe to real-time kline updates
        
        Args:
            symbol: เช่น BTCUSDT
            timeframe: Timeframe
            callback: Function to call when new kline received
        """
        import websockets
        import json
        
        interval = TIMEFRAME_MAP.get(timeframe, "1m")
        stream_name = f"{symbol.lower()}@kline_{interval}"
        
        if callback:
            if stream_name not in self._callbacks:
                self._callbacks[stream_name] = []
            self._callbacks[stream_name].append(callback)
        
        ws_url = f"{self.ws_url}/{stream_name}"
        
        self._running = True
        
        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    logger.info(f"Connected to {stream_name}")
                    
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            
                            # Parse kline data
                            kline = data.get("k", {})
                            candle = {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "datetime": datetime.fromtimestamp(kline["t"] / 1000),
                                "open": float(kline["o"]),
                                "high": float(kline["h"]),
                                "low": float(kline["l"]),
                                "close": float(kline["c"]),
                                "volume": float(kline["v"]),
                                "is_closed": kline["x"]
                            }
                            
                            # Call callbacks
                            for cb in self._callbacks.get(stream_name, []):
                                try:
                                    await cb(candle) if asyncio.iscoroutinefunction(cb) else cb(candle)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                                    
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            await ws.ping()
                            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    await asyncio.sleep(5)  # Reconnect after 5 seconds
    
    async def stop(self):
        """Stop streaming"""
        self._running = False
        if self._ws:
            await self._ws.close()


# Synchronous wrapper for easy use
def download_binance_data(
    symbol: str,
    timeframe: str = "H1",
    days: int = 365
) -> pd.DataFrame:
    """
    Synchronous wrapper to download Binance data
    
    Example:
        df = download_binance_data("BTCUSDT", "H1", 365)
    """
    async def _download():
        provider = BinanceDataProvider()
        try:
            return await provider.get_historical_klines(symbol, timeframe, days)
        finally:
            await provider.close()
    
    return asyncio.run(_download())


def get_current_price(symbol: str) -> float:
    """
    Get current price synchronously
    
    Example:
        price = get_current_price("BTCUSDT")
    """
    async def _get():
        provider = BinanceDataProvider()
        try:
            return await provider.get_current_price(symbol)
        finally:
            await provider.close()
    
    return asyncio.run(_get())


if __name__ == "__main__":
    # Test
    print("Testing Binance Data Provider...")
    
    # Get current price
    price = get_current_price("BTCUSDT")
    print(f"BTC Price: ${price:,.2f}")
    
    # Download historical data
    df = download_binance_data("BTCUSDT", "H1", 7)
    print(f"\nDownloaded {len(df)} candles")
    print(df.tail())
