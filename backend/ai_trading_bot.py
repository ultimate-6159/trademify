"""
Trademify AI Trading Bot - Expert Pattern Recognition System
ระบบเทรดอัตโนมัติด้วย AI เพียงหนึ่งเดียว - แม่นยำ เสถียร ฉลาด

🚨 PRODUCTION ONLY - Windows VPS at 66.42.50.149
   Broker: Exness MT5 (Exness-MT5Real39)
   Account: 267643655
   Symbols: EURUSDm, GBPUSDm, XAUUSDm

🎯 Core Features:
- FAISS Pattern Recognition (millions of patterns in milliseconds)
- AI Multi-factor Analysis (RSI, MACD, Volume, MTF)
- Quality-based Signal Filtering (PREMIUM/HIGH/MEDIUM/LOW)
- Adaptive Risk Management with Position Sizing
- MT5 Broker for Forex/CFD Trading
- Real-time Signal Broadcasting (SSE/Firebase)

🔧 Usage:
    # Forex (MT5) - Production
    python ai_trading_bot.py --broker MT5 --symbols EURUSDm,GBPUSDm,XAUUSDm --quality HIGH

📊 Signal Quality Levels:
    PREMIUM - 85%+ confidence (safest, fewer trades)
    HIGH    - 75%+ confidence (recommended)
    MEDIUM  - 65%+ confidence (more trades, higher risk)
    LOW     - 50%+ confidence (aggressive, high risk)
"""
import asyncio
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing.binance_data import BinanceDataProvider
from data_processing import Normalizer, prepare_database
from similarity_engine import PatternMatcher
from analysis import VotingSystem, PatternAnalyzer, Signal
from analysis import EnhancedAnalyzer, SignalQuality, get_enhanced_analyzer
from trading.engine import TradingEngine, RiskManager, Order, OrderSide, OrderType
from trading.binance_connector import BinanceBroker, BinanceConfig
from trading.settings import TradingConfig, BrokerType
from config import PatternConfig, DataConfig
from services import get_firebase_service
from services.mt5_service import get_mt5_service, MT5Service
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Create logs directory
import os
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class MT5DataProvider:
    """
    Data Provider สำหรับ MT5 - ดึงข้อมูล OHLCV จาก MT5
    มี interface เหมือน BinanceDataProvider
    
    ใช้ MetaTrader5 โดยตรง (singleton connection)
    """
    
    def __init__(self):
        self._mt5 = None
        self._connected = False
        logger.info("MT5DataProvider initialized")
    
    async def connect(self):
        """เชื่อมต่อ MT5 โดยตรง"""
        if self._connected:
            return True
            
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            
            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"❌ MT5 initialize failed: {error}")
                return False
            
            # Login ถ้ามี credentials
            mt5_login = int(os.getenv("MT5_LOGIN", "0"))
            mt5_password = os.getenv("MT5_PASSWORD", "")
            mt5_server = os.getenv("MT5_SERVER", "")
            
            if mt5_login > 0 and mt5_password:
                if not mt5.login(mt5_login, mt5_password, mt5_server, timeout=60000):
                    error = mt5.last_error()
                    logger.error(f"❌ MT5 login failed: {error}")
                    return False
                logger.info(f"✅ MT5 logged in: {mt5_login}@{mt5_server}")
            
            # Enable symbols
            for symbol in ["EURUSDm", "GBPUSDm", "XAUUSDm"]:
                info = mt5.symbol_info(symbol)
                if info and not info.visible:
                    mt5.symbol_select(symbol, True)
            
            self._connected = True
            logger.info("✅ MT5DataProvider connected")
            return True
            
        except ImportError:
            logger.error("❌ MetaTrader5 package not installed!")
            return False
        except Exception as e:
            logger.error(f"❌ MT5 connection error: {e}")
            return False
    
    async def close(self):
        """ปิดการเชื่อมต่อ MT5"""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("🔌 MT5DataProvider disconnected")
    
    async def get_klines(
        self,
        symbol: str,
        timeframe: str = "H1",
        limit: int = 100
    ) -> pd.DataFrame:
        """ดึงข้อมูล OHLCV จาก MT5"""
        try:
            # ตรวจสอบว่าเชื่อมต่อแล้วหรือยัง
            if not self._connected:
                await self.connect()
            
            if not self._connected or not self._mt5:
                logger.warning(f"MT5 not connected, returning empty data for {symbol}")
                return pd.DataFrame()
            
            # แปลง timeframe เป็น MT5 format
            tf_map = {
                "M1": self._mt5.TIMEFRAME_M1,
                "M5": self._mt5.TIMEFRAME_M5,
                "M15": self._mt5.TIMEFRAME_M15,
                "M30": self._mt5.TIMEFRAME_M30,
                "H1": self._mt5.TIMEFRAME_H1,
                "H4": self._mt5.TIMEFRAME_H4,
                "D1": self._mt5.TIMEFRAME_D1,
            }
            mt5_tf = tf_map.get(timeframe.upper(), self._mt5.TIMEFRAME_H1)
            
            # Enable symbol
            info = self._mt5.symbol_info(symbol)
            if info and not info.visible:
                self._mt5.symbol_select(symbol, True)
            
            # ดึงข้อมูล
            rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, limit)
            
            if rates is None or len(rates) == 0:
                error = self._mt5.last_error()
                logger.warning(f"No OHLCV data for {symbol} from MT5: {error}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'tick_volume': 'volume'
            })
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error getting klines from MT5: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    async def get_historical_klines(
        self,
        symbol: str,
        timeframe: str = "H1",
        days: int = 90
    ) -> pd.DataFrame:
        """ดึงข้อมูลย้อนหลัง"""
        # คำนวณจำนวนแท่งเทียนที่ต้องการ
        tf_minutes = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440
        }
        minutes = tf_minutes.get(timeframe.upper(), 60)
        candles_per_day = (24 * 60) // minutes
        total_candles = min(days * candles_per_day, 10000)  # MT5 limit
        
        return await self.get_klines(symbol, timeframe, total_candles)


class AITradingBot:
    """
    🤖 Trademify AI Trading Bot - Expert Pattern Recognition System
    
    ระบบเทรดอัตโนมัติแบบ AI เพียงหนึ่งเดียว
    ใช้ Pattern Recognition + Multi-factor Analysis เพื่อ Win Rate สูง
    
    รองรับ:
    - MT5: Forex (EURUSD, GBPUSD) และ Gold (XAUUSD)
    - Binance: Crypto (BTCUSDT, ETHUSDT)
    
    Quality Levels:
    - PREMIUM: 85%+ confidence (safest)
    - HIGH: 75%+ confidence (recommended)
    - MEDIUM: 65%+ confidence (moderate)
    - LOW: 50%+ confidence (aggressive)
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = "H1",
        htf_timeframe: str = "H4",
        window_size: int = 60,
        min_quality: SignalQuality = SignalQuality.MEDIUM,
        max_risk_percent: float = 2.0,
        broker_type: str = "MT5",  # MT5 only - Exness broker
        broadcast_to_firebase: bool = True,
        allowed_signals: List[str] = None,  # Allow specific signals only
    ):
        # Default to Exness MT5 symbols (with 'm' suffix)
        if symbols is None:
            self.symbols = ["EURUSDm", "GBPUSDm", "XAUUSDm"]
        else:
            self.symbols = symbols
            
        self.timeframe = timeframe
        self.htf_timeframe = htf_timeframe
        self.window_size = window_size
        self.min_quality = min_quality
        self.max_risk_percent = max_risk_percent
        self.broker_type = broker_type
        self.broadcast_to_firebase = broadcast_to_firebase
        
        # Allowed signals - default includes all trading signals
        self.allowed_signals = allowed_signals or ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]
        
        # Get confidence threshold based on quality setting
        self._min_confidence = self._get_confidence_for_quality(min_quality)
        
        # Components
        self.data_provider: Optional[BinanceDataProvider] = None
        self.pattern_matchers: Dict[str, PatternMatcher] = {}
        self.trading_engine: Optional[TradingEngine] = None
        self.enhanced_analyzer: Optional[EnhancedAnalyzer] = None
        self.firebase_service = None
        
        # State
        self._running = False
        self._last_signals: Dict[str, Any] = {}
        self._trade_history: List[Dict] = []
        self._daily_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "date": datetime.now().date().isoformat()
        }
        
        # Subscribers for real-time updates (SSE)
        self._subscribers: List[asyncio.Queue] = []
    
    def _get_confidence_for_quality(self, quality: SignalQuality) -> float:
        """
        Convert quality level to minimum confidence threshold
        Quality thresholds from EnhancedAnalyzer.QUALITY_THRESHOLDS
        """
        quality_to_confidence = {
            SignalQuality.PREMIUM: 85.0,  # >= 85%
            SignalQuality.HIGH: 75.0,     # >= 75%
            SignalQuality.MEDIUM: 65.0,   # >= 65%
            SignalQuality.LOW: 50.0,      # >= 50%
            SignalQuality.SKIP: 0.0,      # any
        }
        return quality_to_confidence.get(quality, 70.0)
    
    def add_subscriber(self, queue: asyncio.Queue):
        """Add SSE subscriber"""
        self._subscribers.append(queue)
    
    def remove_subscriber(self, queue: asyncio.Queue):
        """Remove SSE subscriber"""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
    
    def _build_factor_details(self, enhanced_result, ohlcv_data: dict) -> list:
        """Build detailed factor breakdown for UI display"""
        factors = []
        
        # 1. Pattern Score Factor
        pattern_score = enhanced_result.pattern_score
        factors.append({
            "name": "Pattern Match",
            "score": round(pattern_score, 1),
            "weight": 25,
            "status": "STRONG_BULLISH" if pattern_score >= 80 else "BULLISH" if pattern_score >= 60 else "NEUTRAL" if pattern_score >= 40 else "WEAK",
            "details": f"Pattern matching confidence: {pattern_score:.1f}%. {'Strong historical pattern match found' if pattern_score >= 80 else 'Good pattern similarity' if pattern_score >= 60 else 'Moderate pattern match' if pattern_score >= 40 else 'Weak pattern match'}",
            "passed": pattern_score >= 60,
        })
        
        # 2. Technical Score Factor
        technical_score = enhanced_result.technical_score
        indicators = enhanced_result.indicators
        tech_details = ""
        if indicators:
            rsi_status = "Overbought" if indicators.rsi > 70 else "Oversold" if indicators.rsi < 30 else "Neutral"
            tech_details = f"RSI: {indicators.rsi:.1f} ({rsi_status}), MACD: {indicators.macd_trend}"
        factors.append({
            "name": "Technical Indicators",
            "score": round(technical_score, 1),
            "weight": 20,
            "status": "STRONG_BULLISH" if technical_score >= 80 else "BULLISH" if technical_score >= 60 else "NEUTRAL" if technical_score >= 40 else "WEAK",
            "details": f"Technical score: {technical_score:.1f}%. {tech_details}",
            "passed": technical_score >= 50,
        })
        
        # 3. Volume Score Factor  
        volume_score = enhanced_result.volume_score
        vol_analysis = enhanced_result.volume_analysis
        vol_details = ""
        if vol_analysis:
            vol_details = f"Volume ratio: {vol_analysis.volume_ratio:.2f}x average. {'Volume spike detected!' if vol_analysis.volume_spike else 'Volume confirmed' if vol_analysis.volume_confirmation else 'Normal volume'}"
        factors.append({
            "name": "Volume Confirmation",
            "score": round(volume_score, 1),
            "weight": 15,
            "status": "STRONG_BULLISH" if volume_score >= 80 else "BULLISH" if volume_score >= 60 else "NEUTRAL" if volume_score >= 40 else "WEAK",
            "details": f"Volume analysis: {volume_score:.1f}%. {vol_details}",
            "passed": volume_score >= 40,
        })
        
        # 4. Multi-Timeframe Score Factor
        mtf_score = enhanced_result.mtf_score
        mtf_analysis = enhanced_result.mtf_analysis
        mtf_details = ""
        if mtf_analysis:
            mtf_details = f"HTF Trend: {mtf_analysis.htf_trend}. {'Trend aligned across timeframes' if mtf_analysis.trend_alignment else 'Timeframe divergence detected'}"
        factors.append({
            "name": "Multi-Timeframe",
            "score": round(mtf_score, 1),
            "weight": 15,
            "status": "STRONG_BULLISH" if mtf_score >= 80 else "BULLISH" if mtf_score >= 60 else "NEUTRAL" if mtf_score >= 40 else "WEAK",
            "details": f"MTF confluence: {mtf_score:.1f}%. {mtf_details}",
            "passed": mtf_score >= 50,
        })
        
        # 5. Market Regime Score Factor
        regime_score = enhanced_result.regime_score
        market_regime = enhanced_result.market_regime.value
        factors.append({
            "name": "Market Regime",
            "score": round(regime_score, 1),
            "weight": 10,
            "status": "STRONG_BULLISH" if regime_score >= 80 else "BULLISH" if regime_score >= 60 else "NEUTRAL" if regime_score >= 40 else "WEAK",
            "details": f"Regime: {market_regime}. {'Trending market - good for signals' if 'TREND' in market_regime else 'Ranging market - caution advised' if market_regime == 'RANGING' else 'Volatile market - high risk'}",
            "passed": regime_score >= 50,
        })
        
        # 6. Session Timing Score Factor
        timing_score = enhanced_result.timing_score
        factors.append({
            "name": "Session Timing",
            "score": round(timing_score, 1),
            "weight": 10,
            "status": "STRONG_BULLISH" if timing_score >= 80 else "BULLISH" if timing_score >= 60 else "NEUTRAL" if timing_score >= 40 else "WEAK",
            "details": f"Session score: {timing_score:.1f}%. {'Prime trading session' if timing_score >= 80 else 'Good session' if timing_score >= 60 else 'Average session' if timing_score >= 40 else 'Poor session timing'}",
            "passed": timing_score >= 40,
        })
        
        # 7. Momentum Score Factor
        momentum_score = enhanced_result.momentum_score
        momentum_details = ""
        if indicators:
            momentum_details = f"RSI at {indicators.rsi:.1f}, MACD Histogram {'positive' if indicators.macd_histogram > 0 else 'negative'}"
        factors.append({
            "name": "Momentum",
            "score": round(momentum_score, 1),
            "weight": 5,
            "status": "STRONG_BULLISH" if momentum_score >= 80 else "BULLISH" if momentum_score >= 60 else "NEUTRAL" if momentum_score >= 40 else "WEAK",
            "details": f"Momentum score: {momentum_score:.1f}%. {momentum_details}",
            "passed": momentum_score >= 50,
        })
        
        return factors

    async def _broadcast_update(self, event_type: str, data: dict):
        """Broadcast update to all subscribers"""
        # Convert numpy types for JSON
        clean_data = self._convert_for_json(data) if hasattr(self, '_convert_for_json') else data
        
        message = {
            "event": event_type,
            "data": clean_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to SSE subscribers
        for queue in self._subscribers:
            try:
                await queue.put(message)
            except:
                pass
        
        # Broadcast to Firebase
        if self.broadcast_to_firebase and self.firebase_service:
            try:
                if event_type == "signal":
                    self.firebase_service.update_current_signal(
                        clean_data.get("symbol", "UNKNOWN"),
                        self.timeframe,
                        clean_data
                    )
                elif event_type == "trade":
                    self.firebase_service.add_trade_history(clean_data)
            except Exception as e:
                logger.warning(f"Firebase broadcast failed: {e}")
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("🚀 Initializing Enhanced Trading Bot")
        logger.info(f"📊 Broker: {self.broker_type}")
        logger.info(f"📈 Symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # 1. Data Provider - เลือกตาม broker type
        if self.broker_type == "MT5":
            # ใช้ MT5 สำหรับ Forex - Production on Windows VPS
            self.data_provider = MT5DataProvider()
            connected = await self.data_provider.connect()
            if connected:
                logger.info("✓ MT5 Data provider connected (Forex)")
            else:
                logger.error("❌ MT5 Data provider NOT connected - check VPS/MT5 terminal")
        else:
            # ใช้ Binance สำหรับ Crypto
            self.data_provider = BinanceDataProvider()
            logger.info("✓ Binance Data provider initialized (Crypto)")
        
        # 2. Build Pattern Indices for all symbols
        for symbol in self.symbols:
            await self._build_index(symbol)
        
        # 3. Initialize Trading Engine (MT5 or Binance)
        await self._init_trading_engine()
        
        # 4. Enhanced Analyzer
        self.enhanced_analyzer = EnhancedAnalyzer(
            min_quality=self.min_quality,
            enable_volume_filter=True,
            enable_mtf_filter=True,
            enable_regime_filter=True,
        )
        logger.info(f"✓ Enhanced analyzer initialized (Min Quality: {self.min_quality.value})")
        
        # 5. Firebase (optional)
        if self.broadcast_to_firebase:
            try:
                self.firebase_service = get_firebase_service()
                logger.info("✓ Firebase service initialized")
            except Exception as e:
                logger.warning(f"Firebase not available: {e}")
                self.firebase_service = None
        
        logger.info("=" * 60)
        logger.info("✓ Bot initialization complete!")
        logger.info("=" * 60)
    
    async def _build_index(self, symbol: str):
        """Build pattern index for a symbol"""
        logger.info(f"📊 Building index for {symbol}...")
        
        try:
            df = await self.data_provider.get_historical_klines(
                symbol=symbol,
                timeframe=self.timeframe,
                days=90
            )
            
            logger.info(f"   Got {len(df)} historical candles for {symbol}")
            
            if len(df) < self.window_size + 50:
                logger.warning(f"   ❌ Not enough data for {symbol}: {len(df)} candles (need {self.window_size + 50})")
                return
            
            logger.info(f"   Preparing database for {symbol}...")
            database = prepare_database(
                df=df,
                symbol=symbol,
                timeframe=self.timeframe,
                window_size=self.window_size,
                future_candles=10,
                norm_method="zscore"
            )
            
            n_patterns = len(database["windows"])
            logger.info(f"   Creating pattern matcher with {n_patterns} patterns...")
            matcher = PatternMatcher(
                window_size=self.window_size,
                index_type="IVF" if n_patterns > 500 else "Flat",
                min_correlation=PatternConfig.MIN_CORRELATION
            )
            
            matcher.fit(
                patterns=database["windows"],
                futures=database["futures"],
                metadata=[m.to_dict() for m in database["metadata"]]
            )
            
            self.pattern_matchers[symbol] = matcher
            logger.info(f"   ✅ {symbol}: Index built with {n_patterns} patterns")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to build index for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _init_trading_engine(self):
        """Initialize trading engine - MT5 only (Production)"""
        from trading.mt5_connector import MT5Broker, MT5Config
        
        if self.broker_type == "MT5":
            # MT5 Broker for Forex/CFD - PRODUCTION ONLY
            mt5_login = int(os.getenv("MT5_LOGIN", "0"))
            mt5_password = os.getenv("MT5_PASSWORD", "")
            mt5_server = os.getenv("MT5_SERVER", "")
            
            if not mt5_login:
                raise ValueError("MT5_LOGIN not set - Cannot run without MT5 credentials")
            
            logger.info(f"💰 Using MT5 REAL trading (Server: {mt5_server})")
            broker = MT5Broker(MT5Config(
                login=mt5_login,
                password=mt5_password,
                server=mt5_server,
            ))
        else:
            # Binance Broker for Crypto
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")
            
            if not api_key:
                raise ValueError("BINANCE_API_KEY not set - Cannot run without Binance credentials")
            
            logger.info("💰 Using Binance REAL trading")
            broker = BinanceBroker(BinanceConfig(
                api_key=api_key,
                api_secret=api_secret,
                testnet=False
            ))
        
        risk_manager = RiskManager(
            max_risk_per_trade=self.max_risk_percent,
            max_daily_loss=5.0,
            max_positions=5,
            min_confidence=self._min_confidence,  # Use quality-based threshold
            max_drawdown=10.0
        )
        logger.info(f"✓ Risk manager: min_confidence={self._min_confidence}% (based on {self.min_quality.value} quality)")
        
        self.trading_engine = TradingEngine(
            broker=broker,
            risk_manager=risk_manager,
            max_positions=5,
            enabled=True
        )
        
        await self.trading_engine.start()
        logger.info("✓ Trading engine started")
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a symbol with enhanced AI factors"""
        logger.info(f"📊 Analyzing {symbol}...")
        
        # Default response structure with scores
        default_response = {
            "symbol": symbol,
            "signal": "WAIT",
            "current_price": 0,
            "enhanced_confidence": 0,
            "quality": "SKIP",
            "scores": {
                "pattern": 0,
                "trend": 0,
                "volume": 0,
                "momentum": 0,
                "session": 0,
                "volatility": 0,
                "recency": 0
            },
            "indicators": None,
            "factors": {
                "bullish": [],
                "bearish": [],
                "skip_reasons": []
            },
            "factor_details": [],
            "market_regime": "UNKNOWN",
            "timestamp": datetime.now().isoformat()
        }
        
        if symbol not in self.pattern_matchers:
            logger.warning(f"⚠️ {symbol}: No pattern index")
            default_response["reason"] = "No index"
            default_response["factors"]["skip_reasons"] = ["Pattern index not built"]
            return default_response
        
        # Get current timeframe data
        logger.info(f"   Fetching {self.timeframe} data for {symbol}...")
        df = await self.data_provider.get_klines(
            symbol=symbol,
            timeframe=self.timeframe,
            limit=self.window_size + 100
        )
        logger.info(f"   Got {len(df)} candles for {symbol}")
        
        # Get higher timeframe data
        htf_df = await self.data_provider.get_klines(
            symbol=symbol,
            timeframe=self.htf_timeframe,
            limit=100
        )
        
        if len(df) < self.window_size:
            logger.warning(f"⚠️ {symbol}: Insufficient data - need {self.window_size}, got {len(df)}")
            default_response["reason"] = "Insufficient data"
            default_response["factors"]["skip_reasons"] = [f"Need {self.window_size} candles, got {len(df)}"]
            return default_response
        
        current_price = float(df['close'].iloc[-1])
        logger.info(f"   {symbol} current price: {current_price}")
        
        # Prepare OHLCV arrays
        ohlcv_data = {
            "open": df['open'].values[-100:].astype(np.float32),
            "high": df['high'].values[-100:].astype(np.float32),
            "low": df['low'].values[-100:].astype(np.float32),
            "close": df['close'].values[-100:].astype(np.float32),
            "volume": df['volume'].values[-100:].astype(np.float32),
        }
        
        htf_data = None
        if not htf_df.empty:
            htf_data = {
                "open": htf_df['open'].values.astype(np.float32),
                "high": htf_df['high'].values.astype(np.float32),
                "low": htf_df['low'].values.astype(np.float32),
                "close": htf_df['close'].values.astype(np.float32),
                "volume": htf_df['volume'].values.astype(np.float32),
            }
        
        # Normalize pattern
        normalizer = Normalizer(method="zscore")
        normalized = normalizer.normalize(df['close'].values[-self.window_size:])
        
        # Get base signal from pattern matching
        matcher = self.pattern_matchers[symbol]
        voting_system = VotingSystem(
            confidence_threshold=70.0, 
            strong_signal_threshold=80.0,
            timeframe=self.timeframe  # Add timeframe for duration estimation
        )
        analyzer = PatternAnalyzer(
            similarity_engine=matcher,
            voting_system=voting_system,
            min_correlation=PatternConfig.MIN_CORRELATION
        )
        
        base_result = analyzer.analyze(
            query_pattern=normalized.astype(np.float32),
            current_price=current_price,
            k=10
        )
        
        base_signal = base_result.get("signal", "WAIT")
        base_confidence = base_result.get("confidence", 0.0)
        price_projection = base_result.get("price_projection", {})
        
        # Enhanced Analysis
        enhanced_result = self.enhanced_analyzer.analyze(
            base_signal=base_signal,
            base_confidence=base_confidence,
            ohlcv_data=ohlcv_data,
            current_price=current_price,
            stop_loss=price_projection.get("stop_loss"),
            take_profit=price_projection.get("take_profit"),
            htf_data=htf_data,
            current_time=datetime.now(),
        )
        
        result = {
            "symbol": symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "signal": enhanced_result.signal,
            "base_confidence": base_confidence,
            "enhanced_confidence": enhanced_result.enhanced_confidence,
            "quality": enhanced_result.quality.value,
            "scores": {
                "pattern": enhanced_result.pattern_score,
                "trend": enhanced_result.technical_score,  # Technical includes trend
                "volume": enhanced_result.volume_score,
                "momentum": enhanced_result.momentum_score,
                "session": enhanced_result.timing_score,
                "volatility": enhanced_result.regime_score,  # Regime includes volatility
                "recency": enhanced_result.mtf_score,  # MTF as recency proxy
            },
            "market_regime": enhanced_result.market_regime.value,
            "indicators": enhanced_result.indicators.to_dict() if enhanced_result.indicators else None,
            "risk_management": {
                "stop_loss": enhanced_result.adjusted_stop_loss,
                "take_profit": enhanced_result.adjusted_take_profit,
                "risk_reward": enhanced_result.risk_reward_ratio,
                "position_size": enhanced_result.recommended_position_size,
                "entry_timing": enhanced_result.entry_timing,
            },
            "factors": {
                "bullish": enhanced_result.bullish_factors,
                "bearish": enhanced_result.bearish_factors,
                "skip_reasons": enhanced_result.skip_reasons,
            },
            # Detailed factor breakdown for UI display
            "factor_details": self._build_factor_details(enhanced_result, ohlcv_data),
            "vote_details": base_result.get("vote_details"),
            "n_matches": base_result.get("n_matches", 0),
            # Signal duration estimation
            "duration": base_result.get("duration"),
            "market_data": {
                "open": float(df['open'].iloc[-1]),
                "high": float(df['high'].iloc[-1]),
                "low": float(df['low'].iloc[-1]),
                "close": current_price,
                "volume": float(df['volume'].iloc[-1]),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Log analysis result
        logger.info(f"✅ {symbol}: Signal={enhanced_result.signal} | Confidence={enhanced_result.enhanced_confidence:.1f}% | Quality={enhanced_result.quality.value}")
        logger.info(f"   Scores: Pattern={enhanced_result.pattern_score:.0f} Tech={enhanced_result.technical_score:.0f} Vol={enhanced_result.volume_score:.0f} Mom={enhanced_result.momentum_score:.0f}")
        
        return result
    
    async def execute_trade(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade based on enhanced analysis
        
        SECURITY: Mandatory Stop Loss Enforcement
        - All trades MUST have a Stop Loss
        - If no SL provided, auto-calculate from ATR or use 2% default
        """
        symbol = analysis.get("symbol")
        signal = analysis.get("signal", "WAIT")
        quality = analysis.get("quality", "SKIP")
        current_price = analysis.get("current_price", 0)
        risk_mgmt = analysis.get("risk_management", {})
        
        logger.info(f"🔍 execute_trade() called for {symbol}")
        logger.info(f"   Signal: {signal}, Quality: {quality}, Price: {current_price}")
        
        # Skip if quality below threshold
        quality_order = ["SKIP", "LOW", "MEDIUM", "HIGH", "PREMIUM"]
        min_quality_idx = quality_order.index(self.min_quality.value)
        current_quality_idx = quality_order.index(quality)
        
        logger.info(f"   Quality check: {quality}({current_quality_idx}) >= {self.min_quality.value}({min_quality_idx})")
        
        if current_quality_idx < min_quality_idx:
            logger.info(f"   ❌ SKIP: Quality below threshold")
            return {
                "action": "SKIP",
                "reason": f"Quality {quality} below {self.min_quality.value}"
            }
        
        if signal == "WAIT":
            logger.info(f"   ❌ SKIP: Signal is WAIT")
            return {"action": "SKIP", "reason": "Signal is WAIT"}
        
        # Check entry timing - but allow STRONG signals to trade immediately
        entry_timing = risk_mgmt.get("entry_timing", "NOW")
        logger.info(f"   Entry timing: {entry_timing}, Signal: {signal}")
        
        if entry_timing != "NOW" and signal not in ["STRONG_BUY", "STRONG_SELL"]:
            logger.info(f"   ❌ SKIP: Entry timing not NOW and signal not STRONG")
            return {"action": "SKIP", "reason": f"Entry timing: {entry_timing}"}
        
        logger.info(f"   ✅ Entry timing check passed (STRONG signal or NOW)")
        
        # Check if signal is in allowed_signals list
        if signal not in self.allowed_signals:
            logger.info(f"   ❌ SKIP: Signal {signal} not in {self.allowed_signals}")
            return {"action": "SKIP", "reason": f"Signal {signal} not in allowed: {self.allowed_signals}"}
        
        logger.info(f"   ✅ Signal in allowed list")
        
        # Check existing positions
        for pos in self.trading_engine.positions.values():
            if pos.symbol == symbol:
                logger.info(f"   ❌ SKIP: Already have position for {symbol}")
                return {"action": "SKIP", "reason": "Already have position"}
        
        logger.info(f"   ✅ No existing position for {symbol}")
        
        # Determine side
        if signal in ["STRONG_BUY", "BUY"]:
            side = OrderSide.BUY
        elif signal in ["STRONG_SELL", "SELL"]:
            side = OrderSide.SELL
        else:
            return {"action": "SKIP", "reason": f"Unknown signal: {signal}"}
        
        # Get SL/TP from analysis
        stop_loss = risk_mgmt.get("stop_loss")
        take_profit = risk_mgmt.get("take_profit")
        position_multiplier = risk_mgmt.get("position_size", 1.0)
        
        # 🔒 MANDATORY STOP LOSS ENFORCEMENT
        if not stop_loss or stop_loss <= 0:
            # Auto-calculate Stop Loss (2% from current price)
            default_stop_percent = 0.02
            if side == OrderSide.BUY:
                stop_loss = current_price * (1 - default_stop_percent)
            else:
                stop_loss = current_price * (1 + default_stop_percent)
            logger.warning(f"⚠️ No Stop Loss provided for {symbol}. Auto-set to {stop_loss:.5f} (2%)")
        
        # Validate Stop Loss direction
        if side == OrderSide.BUY and stop_loss >= current_price:
            logger.error(f"❌ Invalid SL for BUY: SL ({stop_loss}) must be below price ({current_price})")
            return {"action": "SKIP", "reason": "Invalid SL direction for BUY"}
        if side == OrderSide.SELL and stop_loss <= current_price:
            logger.error(f"❌ Invalid SL for SELL: SL ({stop_loss}) must be above price ({current_price})")
            return {"action": "SKIP", "reason": "Invalid SL direction for SELL"}
        
        # Calculate position size
        balance = await self.trading_engine.broker.get_balance()
        risk_amount = balance * (self.max_risk_percent / 100) * position_multiplier
        
        if stop_loss:
            stop_distance = abs(current_price - stop_loss)
            quantity = risk_amount / stop_distance if stop_distance > 0 else 0.001
        else:
            quantity = risk_amount / (current_price * 0.02)  # 2% default stop
        
        quantity = round(max(0.001, quantity), 4)
        
        # Create order
        order = Order(
            id=f"ENH-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        logger.info(f"📈 Executing {side.value} {symbol}")
        logger.info(f"   Quality: {quality} | Position Size: {position_multiplier}x")
        sl_str = f"${stop_loss:,.5f}" if stop_loss else "N/A"
        tp_str = f"${take_profit:,.5f}" if take_profit else "N/A"
        logger.info(f"   Entry: ${current_price:,.5f} | SL: {sl_str} | TP: {tp_str}")
        
        # Debug: Check trading engine state
        logger.info(f"   🔍 TradingEngine enabled: {self.trading_engine.enabled if self.trading_engine else 'N/A'}")
        logger.info(f"   🔍 TradingEngine running: {self.trading_engine._running if self.trading_engine else 'N/A'}")
        
        # Execute
        result = await self.trading_engine.execute_order(order)
        
        logger.info(f"   🔍 Execute result: {result}")
        
        if result and result.success:
            self._daily_stats["trades"] += 1
            trade_record = {
                "order_id": order.id,
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "entry_price": result.order.filled_price if result.order else current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "quality": quality,
                "timestamp": datetime.now().isoformat()
            }
            self._trade_history.append(trade_record)
            
            # Broadcast trade event
            await self._broadcast_update("trade", trade_record)
            
            logger.info(f"✅ Trade executed!")
            return {"action": "EXECUTED", "order": order.to_dict(), "result": str(result)}
        elif result:
            # Result exists but not success
            logger.warning(f"❌ Trade failed: {result.error if result.error else result.message}")
            return {"action": "FAILED", "reason": result.error or result.message or "Unknown error"}
        else:
            # Result is None - trading engine might be disabled
            logger.warning("❌ Trade failed: execute_order returned None (trading engine disabled?)")
            return {"action": "SKIP", "reason": "Trading engine returned None"}
    
    async def run(self, interval_seconds: int = 60):
        """Run the enhanced trading bot"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤖 Starting Enhanced Trading Bot")
        logger.info("=" * 60)
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Timeframe: {self.timeframe} (HTF: {self.htf_timeframe})")
        logger.info(f"   Min Quality: {self.min_quality.value}")
        logger.info(f"   Check Interval: {interval_seconds}s")
        logger.info(f"   Mode: PRODUCTION (MT5 Real Trading)")
        logger.info("=" * 60)
        logger.info("")
        
        self._running = True
        
        # Broadcast bot status
        await self._broadcast_update("bot_status", {
            "status": "running",
            "symbols": self.symbols,
            "min_quality": self.min_quality.value,
        })
        
        while self._running:
            try:
                # Reset daily stats at midnight
                today = datetime.now().date().isoformat()
                if self._daily_stats["date"] != today:
                    self._daily_stats = {
                        "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "date": today
                    }
                
                # Analyze all symbols
                for symbol in self.symbols:
                    logger.info(f"📊 Analyzing {symbol}...")
                    
                    analysis = await self.analyze_symbol(symbol)
                    self._last_signals[symbol] = analysis
                    
                    # Log result
                    signal = analysis.get("signal", "WAIT")
                    quality = analysis.get("quality", "SKIP")
                    confidence = analysis.get("enhanced_confidence", 0)
                    price = analysis.get("current_price", 0)
                    regime = analysis.get("market_regime", "UNKNOWN")
                    
                    signal_emoji = {
                        "STRONG_BUY": "🟢🟢", "BUY": "🟢",
                        "WAIT": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"
                    }
                    
                    quality_emoji = {
                        "PREMIUM": "⭐⭐⭐", "HIGH": "⭐⭐",
                        "MEDIUM": "⭐", "LOW": "⚠️", "SKIP": "❌"
                    }
                    
                    logger.info(f"   {signal_emoji.get(signal, '❓')} Signal: {signal}")
                    logger.info(f"   {quality_emoji.get(quality, '')} Quality: {quality}")
                    logger.info(f"   💰 Price: ${price:,.2f}")
                    logger.info(f"   📈 Confidence: {confidence:.1f}%")
                    logger.info(f"   🌊 Regime: {regime}")
                    
                    # Broadcast signal update
                    await self._broadcast_update("signal", analysis)
                    
                    # Execute trade if conditions met
                    # Check against min_quality setting
                    quality_order = ["SKIP", "LOW", "MEDIUM", "HIGH", "PREMIUM"]
                    min_quality_idx = quality_order.index(self.min_quality.value)
                    current_quality_idx = quality_order.index(quality) if quality in quality_order else 0
                    
                    # Log trade decision
                    logger.info(f"   📋 Trade Check: signal={signal}, quality={quality}({current_quality_idx}) >= min_quality={self.min_quality.value}({min_quality_idx})")
                    
                    if signal != "WAIT" and current_quality_idx >= min_quality_idx:
                        logger.info(f"   ✅ Conditions met! Executing trade...")
                        trade_result = await self.execute_trade(analysis)
                        logger.info(f"   🎯 Trade Result: {trade_result}")
                        
                        # Broadcast trade result to frontend
                        if trade_result.get('action') == 'EXECUTED':
                            await self._broadcast_update("trade_executed", {
                                "symbol": symbol,
                                "signal": signal,
                                "result": trade_result,
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            await self._broadcast_update("trade_skipped", {
                                "symbol": symbol,
                                "signal": signal,
                                "reason": trade_result.get('reason'),
                                "timestamp": datetime.now().isoformat()
                            })
                    elif signal != "WAIT":
                        logger.info(f"   ⏭️ Skipped: Quality {quality} < Min {self.min_quality.value}")
                    
                    logger.info("")
                
                # Show positions
                positions = self.trading_engine.positions
                if positions:
                    logger.info(f"📋 Open Positions: {len(positions)}")
                    for pos in positions.values():
                        logger.info(f"   - {pos.symbol}: {pos.side.value} @ ${pos.entry_price:,.2f}")
                
                # Show daily stats
                logger.info(f"📊 Today: {self._daily_stats['trades']} trades | W:{self._daily_stats['wins']} L:{self._daily_stats['losses']}")
                logger.info(f"⏰ Next check in {interval_seconds}s")
                logger.info("-" * 40)
                
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                await asyncio.sleep(10)
        
        await self._broadcast_update("bot_status", {"status": "stopped"})
    
    async def stop(self):
        """Stop the bot"""
        self._running = False
        if self.trading_engine:
            await self.trading_engine.stop()
        if self.data_provider:
            await self.data_provider.close()
        logger.info("🛑 Bot stopped")
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        return obj
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        status = {
            "running": self._running,
            "broker_type": self.broker_type,
            "symbols": self.symbols,
            "min_quality": self.min_quality.value if hasattr(self.min_quality, 'value') else str(self.min_quality),
            "allowed_signals": self.allowed_signals,
            "mode": "PRODUCTION",
            "last_signals": self._convert_for_json(self._last_signals),
            "daily_stats": self._convert_for_json(self._daily_stats),
            "open_positions": len(self.trading_engine.positions) if self.trading_engine else 0,
        }
        return self._convert_for_json(status)


# Alias for backward compatibility
EnhancedTradingBot = AITradingBot

# Global bot instance for API access
_bot_instance: Optional[AITradingBot] = None


def get_bot() -> Optional[AITradingBot]:
    """Get global bot instance"""
    return _bot_instance


async def main():
    global _bot_instance
    
    parser = argparse.ArgumentParser(
        description='🤖 Trademify AI Trading Bot - Expert Pattern Recognition System (Production)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forex (MT5) - Production Trading on Windows VPS
  python ai_trading_bot.py --broker MT5 --symbols EURUSDm,GBPUSDm,XAUUSDm --quality HIGH
  
  # High Quality Only
  python ai_trading_bot.py --broker MT5 --symbols EURUSDm,XAUUSDm --quality PREMIUM
        """
    )
    parser.add_argument('--symbols', default='EURUSDm,GBPUSDm,XAUUSDm', help='Comma-separated symbols (Exness format)')
    parser.add_argument('--timeframe', default='H1', help='Timeframe (M5, M15, M30, H1, H4, D1)')
    parser.add_argument('--htf', default='H4', help='Higher timeframe for MTF analysis')
    parser.add_argument('--interval', type=int, default=60, help='Analysis interval (seconds)')
    parser.add_argument('--quality', default='HIGH', choices=['PREMIUM', 'HIGH', 'MEDIUM', 'LOW'], 
                       help='Signal quality filter (PREMIUM=safest, LOW=aggressive)')
    parser.add_argument('--risk', type=float, default=2.0, help='Max risk per trade (%%)')
    parser.add_argument('--broker', default='MT5', choices=['MT5', 'BINANCE'], help='Broker type')
    
    args = parser.parse_args()
    
    quality_map = {
        "PREMIUM": SignalQuality.PREMIUM,
        "HIGH": SignalQuality.HIGH,
        "MEDIUM": SignalQuality.MEDIUM,
        "LOW": SignalQuality.LOW,
    }
    
    # Show startup banner
    print("=" * 60)
    print("🤖 TRADEMIFY AI TRADING BOT - PRODUCTION")
    print("=" * 60)
    print(f"   Broker:    {args.broker} (Exness MT5)")
    print(f"   Symbols:   {args.symbols}")
    print(f"   Timeframe: {args.timeframe} (HTF: {args.htf})")
    print(f"   Quality:   {args.quality}")
    print(f"   Risk:      {args.risk}% per trade")
    print(f"   Mode:      🔴 LIVE TRADING")
    print("=" * 60)
    
    print("\n⚠️  PRODUCTION MODE - REAL MONEY AT RISK!")
    print("    Press Ctrl+C within 5 seconds to cancel...\n")
    await asyncio.sleep(5)
    
    bot = AITradingBot(
        symbols=args.symbols.split(','),
        timeframe=args.timeframe,
        htf_timeframe=args.htf,
        min_quality=quality_map[args.quality],
        max_risk_percent=args.risk,
        broker_type=args.broker,
    )
    
    _bot_instance = bot
    
    try:
        await bot.initialize()
        await bot.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
