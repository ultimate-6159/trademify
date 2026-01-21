"""
MT5 Real-Time Service
บริการเชื่อมต่อ MT5 จริงสำหรับดึงข้อมูลราคาและตลาด

🚨 PRODUCTION ONLY - NO MOCK MODE
Runs on Windows VPS (66.42.50.149) with Exness MT5
Symbols: EURUSDm, GBPUSDm, XAUUSDm
"""
import os
import sys
import asyncio
import logging
import random
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ตรวจสอบ OS - MT5 รองรับ Windows เท่านั้น
IS_WINDOWS = sys.platform == 'win32'


class MarketStatus(str, Enum):
    """สถานะตลาด"""
    OPEN = "OPEN"           # ตลาดเปิด - เทรดได้
    CLOSED = "CLOSED"       # ตลาดปิด - ช่วงวันหยุด
    WEEKEND = "WEEKEND"     # วันเสาร์-อาทิตย์
    PRE_MARKET = "PRE_MARKET"   # ก่อนตลาดเปิด
    POST_MARKET = "POST_MARKET"  # หลังตลาดปิด
    MAINTENANCE = "MAINTENANCE"  # ช่วงปิดระบบ
    UNKNOWN = "UNKNOWN"     # ไม่ทราบสถานะ


@dataclass
class MarketInfo:
    """ข้อมูลสถานะตลาด"""
    status: MarketStatus
    message: str
    message_th: str
    next_open: Optional[datetime] = None
    time_until_open: Optional[str] = None
    server_time: Optional[datetime] = None
    is_tradeable: bool = False
    color: str = "gray"  # สีสำหรับแสดงผล


@dataclass
class SymbolPrice:
    """ข้อมูลราคา Symbol"""
    symbol: str
    bid: float
    ask: float
    spread: float
    price: float  # mid price
    high: float
    low: float
    volume: float
    time: datetime
    market_status: MarketStatus
    source: str = "mt5"
    is_live: bool = True
    last_update: Optional[datetime] = None
    error: Optional[str] = None


class MT5Service:
    """
    MT5 Real-Time Service - Production Only
    เชื่อมต่อ Exness MT5 บน Windows VPS
    
    Broker: Exness-MT5Real39
    Account: 267643655
    Symbols: ต้องใช้ชื่อตรงกับ broker (เช่น EURUSDm, GBPUSDm, XAUUSDm)
    """
    
    def __init__(self):
        self._mt5 = None
        self._connected = False
        self._last_prices: Dict[str, SymbolPrice] = {}
        self._market_status = MarketStatus.UNKNOWN
        self._config = {
            "login": int(os.getenv("MT5_LOGIN", 0)),
            "password": os.getenv("MT5_PASSWORD", ""),
            "server": os.getenv("MT5_SERVER", ""),
            "path": os.getenv("MT5_PATH", ""),
        }
        
        # ช่วงเวลาตลาด Forex (UTC)
        self._forex_schedule = {
            "sydney_open": time(22, 0),  # Sydney เปิด
            "tokyo_open": time(0, 0),    # Tokyo เปิด
            "london_open": time(8, 0),   # London เปิด
            "newyork_open": time(13, 0), # New York เปิด
            "newyork_close": time(22, 0), # New York ปิด
        }
    
    async def connect(self) -> bool:
        """เชื่อมต่อ MT5 Terminal - Windows VPS only, no mock mode"""
        
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            
            # Initialize MT5
            init_path = self._config["path"] if self._config["path"] else None
            if not mt5.initialize(path=init_path):
                error = mt5.last_error()
                logger.error(f"MT5 initialize failed: {error}")
                return False
            
            # Login ถ้ามี credentials
            if self._config["login"] > 0 and self._config["password"]:
                authorized = mt5.login(
                    login=self._config["login"],
                    password=self._config["password"],
                    server=self._config["server"],
                    timeout=60000
                )
                
                if not authorized:
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    return False
                
                logger.info(f"✅ MT5 logged in: {self._config['login']}@{self._config['server']}")
            else:
                logger.info("✅ MT5 initialized (using terminal's active account)")
            
            self._connected = True
            
            # Log available symbols for debugging
            symbols = mt5.symbols_get()
            if symbols:
                forex_symbols = [s.name for s in symbols if 'USD' in s.name or 'EUR' in s.name or 'XAU' in s.name]
                logger.info(f"📋 Available Forex/Gold symbols: {forex_symbols[:20]}...")
            
            return True
            
        except ImportError:
            logger.error("❌ MetaTrader5 package not installed!")
            logger.error("📦 This bot requires Windows with MT5 - Install: pip install MetaTrader5")
            self._connected = False
            return False
            return True
        except Exception as e:
            logger.error(f"❌ MT5 connection error: {e}")
            return False
    
    async def disconnect(self) -> None:
        """ตัดการเชื่อมต่อ"""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("🔌 MT5 disconnected")
    
    def is_connected(self) -> bool:
        """ตรวจสอบสถานะการเชื่อมต่อ"""
        if not self._mt5 or not self._connected:
            return False
        
        try:
            terminal_info = self._mt5.terminal_info()
            return terminal_info is not None
        except:
            return False
    
    def get_market_status(self, symbol: str = "EURUSD") -> MarketInfo:
        """
        ตรวจสอบสถานะตลาดแบบละเอียด
        
        Forex Market Hours (UTC):
        - ตลาดเปิด 24/5 (วันอาทิตย์ 22:00 - วันศุกร์ 22:00 UTC)
        - Weekend: วันเสาร์และวันอาทิตย์ก่อน 22:00 UTC
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        current_time = now.time()
        
        # วันเสาร์ = ปิดทั้งวัน
        if weekday == 5:  # Saturday
            next_open = now + timedelta(days=1)
            next_open = next_open.replace(hour=22, minute=0, second=0, microsecond=0)
            time_until = self._format_time_until(next_open - now)
            
            return MarketInfo(
                status=MarketStatus.WEEKEND,
                message="Weekend - Forex market closed",
                message_th="🔴 วันหยุดสุดสัปดาห์ - ตลาด Forex ปิด",
                next_open=next_open,
                time_until_open=time_until,
                server_time=now,
                is_tradeable=False,
                color="red"
            )
        
        # วันอาทิตย์ ก่อน 22:00 UTC = ยังปิด
        if weekday == 6 and current_time < time(22, 0):
            next_open = now.replace(hour=22, minute=0, second=0, microsecond=0)
            time_until = self._format_time_until(next_open - now)
            
            return MarketInfo(
                status=MarketStatus.WEEKEND,
                message="Weekend - Forex market opens at 22:00 UTC",
                message_th="🔴 ตลาดจะเปิดเวลา 05:00 น. (UTC+7)",
                next_open=next_open,
                time_until_open=time_until,
                server_time=now,
                is_tradeable=False,
                color="red"
            )
        
        # วันศุกร์ หลัง 22:00 UTC = ปิด
        if weekday == 4 and current_time >= time(22, 0):
            next_open = now + timedelta(days=2)
            next_open = next_open.replace(hour=22, minute=0, second=0, microsecond=0)
            time_until = self._format_time_until(next_open - now)
            
            return MarketInfo(
                status=MarketStatus.CLOSED,
                message="Forex market closed for the weekend",
                message_th="🔴 ตลาดปิดวันศุกร์ - เปิดวันจันทร์",
                next_open=next_open,
                time_until_open=time_until,
                server_time=now,
                is_tradeable=False,
                color="red"
            )
        
        # ตลาดเปิด - ตรวจสอบ session
        session = self._get_current_session(current_time)
        
        return MarketInfo(
            status=MarketStatus.OPEN,
            message=f"Market OPEN - {session} session active",
            message_th=f"🟢 ตลาดเปิด - Session: {session}",
            server_time=now,
            is_tradeable=True,
            color="green"
        )
    
    def _get_current_session(self, current_time: time) -> str:
        """ระบุ session ที่กำลังเทรดอยู่"""
        hour = current_time.hour
        
        if 22 <= hour or hour < 8:
            if 22 <= hour or hour < 3:
                return "Sydney"
            else:
                return "Tokyo"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 17:
            return "London/New York"
        else:
            return "New York"
    
    def _format_time_until(self, delta: timedelta) -> str:
        """แปลง timedelta เป็น string อ่านง่าย"""
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"

    async def get_price(self, symbol: str) -> SymbolPrice:
        """
        ดึงราคาจาก MT5 แบบ real-time
        
        Args:
            symbol: เช่น EURUSDm, GBPUSDm, XAUUSDm
            
        Returns:
            SymbolPrice พร้อมข้อมูลราคาและสถานะตลาด
        """
        market_info = self.get_market_status(symbol)
        
        # ถ้า MT5 ไม่ได้เชื่อมต่อ
        if not self._connected or not self._mt5:
            # ส่งคืนราคาล่าสุดที่ cache ไว้ (ถ้ามี)
            if symbol in self._last_prices:
                cached = self._last_prices[symbol]
                cached.is_live = False
                cached.market_status = market_info.status
                cached.error = "MT5 not connected"
                return cached
            
            # ไม่มี cache - ส่ง error
            return SymbolPrice(
                symbol=symbol,
                bid=0, ask=0, spread=0, price=0,
                high=0, low=0, volume=0,
                time=datetime.now(),
                market_status=market_info.status,
                source="error",
                is_live=False,
                error="MT5 not connected"
            )
        
        try:
            # ดึง tick ล่าสุด
            tick = self._mt5.symbol_info_tick(symbol)
            
            if tick is None:
                # Symbol อาจไม่ถูกต้อง หรือไม่ได้ subscribe
                # ลองเปิด symbol ในตลาด
                if not self._mt5.symbol_select(symbol, True):
                    return SymbolPrice(
                        symbol=symbol,
                        bid=0, ask=0, spread=0, price=0,
                        high=0, low=0, volume=0,
                        time=datetime.now(),
                        market_status=market_info.status,
                        source="error",
                        is_live=False,
                        error=f"Symbol {symbol} not found in MT5"
                    )
                
                # ลองดึงอีกครั้ง
                tick = self._mt5.symbol_info_tick(symbol)
                if tick is None:
                    return SymbolPrice(
                        symbol=symbol,
                        bid=0, ask=0, spread=0, price=0,
                        high=0, low=0, volume=0,
                        time=datetime.now(),
                        market_status=market_info.status,
                        source="error",
                        is_live=False,
                        error=f"Could not get tick for {symbol}"
                    )
            
            # ดึงข้อมูล symbol สำหรับ digits
            symbol_info = self._mt5.symbol_info(symbol)
            digits = symbol_info.digits if symbol_info else 5
            
            # คำนวณ spread
            spread = round(tick.ask - tick.bid, digits)
            mid_price = round((tick.ask + tick.bid) / 2, digits)
            
            # สร้าง SymbolPrice
            price_data = SymbolPrice(
                symbol=symbol,
                bid=round(tick.bid, digits),
                ask=round(tick.ask, digits),
                spread=spread,
                price=mid_price,
                high=tick.last if hasattr(tick, 'last') else mid_price,
                low=tick.last if hasattr(tick, 'last') else mid_price,
                volume=tick.volume if hasattr(tick, 'volume') else 0,
                time=datetime.fromtimestamp(tick.time),
                market_status=market_info.status,
                source="mt5_live",
                is_live=market_info.is_tradeable,
                last_update=datetime.now()
            )
            
            # Cache ราคาล่าสุด
            self._last_prices[symbol] = price_data
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            
            # ส่งคืน cached price ถ้ามี
            if symbol in self._last_prices:
                cached = self._last_prices[symbol]
                cached.is_live = False
                cached.error = str(e)
                return cached
            
            return SymbolPrice(
                symbol=symbol,
                bid=0, ask=0, spread=0, price=0,
                high=0, low=0, volume=0,
                time=datetime.now(),
                market_status=market_info.status,
                source="error",
                is_live=False,
                error=str(e)
            )
    
    async def get_prices_bulk(self, symbols: List[str]) -> Dict[str, SymbolPrice]:
        """ดึงราคาหลาย symbol พร้อมกัน"""
        result = {}
        for symbol in symbols:
            result[symbol] = await self.get_price(symbol)
        return result
    
    async def get_account_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลบัญชี MT5"""
        if not self._connected or not self._mt5:
            return {
                "connected": False,
                "error": "MT5 not connected"
            }
        
        try:
            account = self._mt5.account_info()
            if account is None:
                return {
                    "connected": True,
                    "error": "Could not get account info"
                }
            
            return {
                "connected": True,
                "login": account.login,
                "name": account.name,
                "server": account.server,
                "currency": account.currency,
                "balance": account.balance,
                "equity": account.equity,
                "margin": account.margin,
                "margin_free": account.margin_free,
                "margin_level": account.margin_level if account.margin_level else 0,
                "profit": account.profit,
                "leverage": account.leverage,
                "trade_allowed": account.trade_allowed,
            }
        except Exception as e:
            return {
                "connected": True,
                "error": str(e)
            }
    
    def _find_symbol(self, base_symbol: str) -> Optional[str]:
        """
        หา symbol ที่ถูกต้องจาก broker
        Exness ใช้ชื่อเช่น EURUSDm, GBPUSDm, XAUUSDm
        """
        if not self._mt5:
            return None
        
        # ลองใช้ชื่อตรงก่อน
        symbol_info = self._mt5.symbol_info(base_symbol)
        if symbol_info:
            if not symbol_info.visible:
                self._mt5.symbol_select(base_symbol, True)
            return base_symbol
        
        # ลองหา variants ทั่วไป
        # Remove 'm' if exists, then try with/without
        base_clean = base_symbol.rstrip('mM')
        
        variants = [
            base_symbol,           # EURUSDm
            f"{base_clean}m",      # EURUSDm
            base_clean,            # EURUSD
            f"{base_clean}.",      # EURUSD.
            f"{base_clean}.i",     # EURUSD.i
            f"{base_clean}micro",  # EURUSDmicro
            f"{base_clean}c",      # EURUSDc (cent)
        ]
        
        for variant in variants:
            symbol_info = self._mt5.symbol_info(variant)
            if symbol_info:
                if not symbol_info.visible:
                    self._mt5.symbol_select(variant, True)
                logger.info(f"📍 Symbol found: {base_symbol} -> {variant}")
                return variant
        
        # ค้นหาใน all symbols ที่มี base name
        all_symbols = self._mt5.symbols_get()
        if all_symbols:
            # ค้นหา exact match ก่อน
            for s in all_symbols:
                if s.name.upper().startswith(base_clean.upper()):
                    if not s.visible:
                        self._mt5.symbol_select(s.name, True)
                    logger.info(f"📍 Found symbol: {base_symbol} -> {s.name}")
                    return s.name
        
        logger.warning(f"⚠️ Symbol not found: {base_symbol}")
        return None

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        ดึงข้อมูล OHLCV จาก MT5
        
        Args:
            symbol: เช่น EURUSDm, GBPUSDm, XAUUSDm
            timeframe: M1, M5, M15, M30, H1, H4, D1
            count: จำนวนแท่งเทียน
        """
        if not self._connected or not self._mt5:
            logger.warning(f"MT5 not connected, cannot get OHLCV for {symbol}")
            return []
        
        try:
            # หา symbol ที่ถูกต้อง
            actual_symbol = self._find_symbol(symbol)
            if not actual_symbol:
                logger.error(f"❌ Symbol {symbol} not found in MT5. Use symbols_get() to list available symbols.")
                # Log some available symbols
                all_symbols = self._mt5.symbols_get()
                if all_symbols:
                    names = [s.name for s in all_symbols[:30]]
                    logger.info(f"📋 Sample available symbols: {names}")
                return []
            
            # Select symbol เพื่อ enable data
            if not self._mt5.symbol_select(actual_symbol, True):
                logger.warning(f"⚠️ Could not select symbol {actual_symbol}")
            
            # แปลง timeframe เป็น MT5 format
            tf_map = {
                "M1": self._mt5.TIMEFRAME_M1,
                "M5": self._mt5.TIMEFRAME_M5,
                "M15": self._mt5.TIMEFRAME_M15,
                "M30": self._mt5.TIMEFRAME_M30,
                "H1": self._mt5.TIMEFRAME_H1,
                "H4": self._mt5.TIMEFRAME_H4,
                "D1": self._mt5.TIMEFRAME_D1,
                "W1": self._mt5.TIMEFRAME_W1,
            }
            
            mt5_tf = tf_map.get(timeframe.upper(), self._mt5.TIMEFRAME_H1)
            
            logger.debug(f"📊 Getting {count} {timeframe} candles for {actual_symbol}")
            
            # ดึงข้อมูล
            rates = self._mt5.copy_rates_from_pos(actual_symbol, mt5_tf, 0, count)
            
            if rates is None or len(rates) == 0:
                error = self._mt5.last_error()
                logger.warning(f"⚠️ No rates for {actual_symbol}: MT5 error = {error}")
                return []
            
            logger.info(f"✅ Got {len(rates)} candles for {actual_symbol}")
            
            # แปลงเป็น list of dict
            result = []
            for rate in rates:
                result.append({
                    "time": datetime.fromtimestamp(rate['time']).isoformat(),
                    "open": rate['open'],
                    "high": rate['high'],
                    "low": rate['low'],
                    "close": rate['close'],
                    "volume": rate['tick_volume'],
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting OHLCV for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_available_symbols(self, filter: str = "") -> Dict[str, Any]:
        """
        Get available symbols from MT5
        
        Returns:
            dict with symbols list and metadata
        """
        if not self._connected or not self._mt5:
            return {
                "symbols": [],
                "count": 0,
                "source": "error",
                "message": "MT5 not connected"
            }
        
        try:
            all_symbols = self._mt5.symbols_get()
            if not all_symbols:
                return {"symbols": [], "count": 0, "source": "mt5", "message": "No symbols found"}
            
            names = [s.name for s in all_symbols]
            
            if filter:
                names = [s for s in names if filter.upper() in s.upper()]
            
            return {
                "symbols": names[:100],  # Limit to 100
                "count": len(names[:100]),
                "total": len(all_symbols),
                "source": "mt5",
                "message": f"Found {len(all_symbols)} symbols total"
            }
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return {
                "symbols": [],
                "count": 0,
                "source": "error",
                "message": str(e)
            }


# Global MT5 Service instance
_mt5_service: Optional[MT5Service] = None


def get_mt5_service() -> MT5Service:
    """Get or create global MT5 service instance"""
    global _mt5_service
    if _mt5_service is None:
        _mt5_service = MT5Service()
    return _mt5_service


async def init_mt5_service() -> bool:
    """Initialize and connect MT5 service"""
    service = get_mt5_service()
    connected = await service.connect()
    
    if connected:
        logger.info("✅ MT5 Service initialized successfully")
        account = await service.get_account_info()
        if account.get("connected"):
            logger.info(f"📊 Account: {account.get('login')} | Balance: ${account.get('balance', 0):,.2f}")
    else:
        logger.warning("⚠️ MT5 Service connection failed - check VPS connection")
    
    return connected
