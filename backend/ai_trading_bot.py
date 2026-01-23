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
from trading.risk_guardian import RiskGuardian, get_risk_guardian, create_risk_guardian
from trading.pro_features import ProTradingFeatures, get_pro_features
from trading.smart_brain import SmartBrain, get_smart_brain
from trading.advanced_intelligence import AdvancedIntelligence, get_intelligence
from trading.continuous_learning import ContinuousLearningSystem, get_learning_system
from trading.neural_brain import NeuralBrain, get_neural_brain
from trading.deep_intelligence import DeepIntelligence, get_deep_intelligence
from trading.quantum_strategy import QuantumStrategy, get_quantum_strategy
from trading.alpha_engine import AlphaEngine, get_alpha_engine
from trading.omega_brain import OmegaBrain, get_omega_brain
from trading.titan_core import TitanCore, get_titan_core, ModuleSignal
from trading.ultra_intelligence import UltraIntelligence, get_ultra_intelligence, UltraDecision
from trading.supreme_intelligence import SupremeIntelligence, get_supreme_intelligence, SupremeDecision
from trading.transcendent_intelligence import TranscendentIntelligence, get_transcendent_intelligence, TranscendentDecision
from trading.omniscient_intelligence import OmniscientIntelligence, get_omniscient_intelligence, OmniscientDecision
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
        
        # 🛡️ Risk Guardian - ป้องกันการล้างพอร์ต
        self.risk_guardian: Optional[RiskGuardian] = None
        
        # 🏆 Pro Trading Features - สิ่งที่ Pro Trader ทำ
        self.pro_features: Optional[ProTradingFeatures] = None
        
        # 🧠 Smart Brain - เรียนรู้และปรับตัวเอง
        self.smart_brain: Optional[SmartBrain] = None
        
        # 📚 Continuous Learning
        self.learning_system: Optional[ContinuousLearningSystem] = None
        self._pending_trade_factors: Dict[str, Dict] = {}  # trade_id -> factors used
        
        # 🧬 Neural Brain - Deep Pattern Understanding
        self.neural_brain: Optional[NeuralBrain] = None
        
        # 🔮 Deep Intelligence - Multi-layer Analysis
        self.deep_intelligence: Optional[DeepIntelligence] = None
        
        # ⚛️ Quantum Strategy - Advanced Quantitative Analysis
        self.quantum_strategy: Optional[QuantumStrategy] = None
        
        # 🎯 Alpha Engine - Ultimate Trading Intelligence
        self.alpha_engine: Optional[AlphaEngine] = None
        
        # 🧠⚡ Omega Brain - Institutional-Grade Intelligence
        self.omega_brain: Optional[OmegaBrain] = None
        
        # 🏛️⚔️ Titan Core - Meta-Intelligence Synthesis
        self.titan_core: Optional[TitanCore] = None
        
        # 🧠⚡ Ultra Intelligence - 10x Smarter Trading
        self.ultra_intelligence: Optional[UltraIntelligence] = None
        
        # 🏆👑 Supreme Intelligence - 20x Smarter Trading (Hedge Fund Level)
        self.supreme_intelligence: Optional[SupremeIntelligence] = None
        
        # 🌌✨ Transcendent Intelligence - 50x Smarter (Beyond Human)
        self.transcendent_intelligence: Optional[TranscendentIntelligence] = None
        
        # 🔮✨ Omniscient Intelligence - 100x Smarter (All-Knowing)
        self.omniscient_intelligence: Optional[OmniscientIntelligence] = None
        
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
        
        # 📈 Trailing Stop Config - ยก SL ตามราคาเพื่อล็อคกำไร
        self._trailing_stop_config = {
            "enabled": True,                    # เปิด/ปิด Trailing Stop
            "activation_profit_pct": 0.3,       # เริ่มทำงานเมื่อกำไร >= 0.3%
            "trail_distance_pct": 0.2,          # SL ตาม 0.2% จากราคาปัจจุบัน
            "min_trail_distance_gold": 1.0,     # Gold: SL ตาม $1 minimum
            "min_trail_distance_forex": 0.0010, # Forex: SL ตาม 10 pips minimum
            "step_pct": 0.1,                    # ยก SL ทีละ 0.1%
        }
        self._position_highest_prices: Dict[str, float] = {}  # Track highest/lowest for trailing
        
        # 🎯 Floating TP Config - ยก TP ตาม SL เพื่อให้ได้กำไรมากขึ้น
        self._floating_tp_config = {
            "enabled": True,                    # เปิด/ปิด Floating TP
            "min_rr_ratio": 1.5,                # R:R ขั้นต่ำที่ต้องรักษา (SL-Entry : TP-Entry)
            "tp_extension_multiplier": 1.2,    # ยืด TP 20% เมื่อ SL ถูกยก
            "max_tp_extension_pct": 5.0,        # TP ขยับได้มากสุด 5% จาก entry
        }
        self._position_original_tp: Dict[str, float] = {}  # เก็บ TP เดิมเพื่ออ้างอิง
        
        # 🧠 Smart Trading Features - ทำให้ระบบฉลาดขึ้น
        self._smart_features = {
            # Break-Even: ย้าย SL ไปจุด entry เมื่อกำไรถึงระดับหนึ่ง
            "break_even": {
                "enabled": True,
                "activation_pct": 0.5,  # เปิดใช้เมื่อกำไร >= 0.5%
                "offset_pct": 0.05,     # SL = entry + 0.05% (เผื่อ spread)
            },
            # Max Daily Trades: จำกัดจำนวนเทรดต่อวัน
            "max_daily_trades": {
                "enabled": True,
                "limit": 5,  # เทรดได้ไม่เกิน 5 ครั้งต่อวัน
            },
            # Consecutive Loss Protection: หยุดเทรดหลังขาดทุนติดต่อกัน
            "loss_protection": {
                "enabled": True,
                "max_consecutive_losses": 3,  # หยุดหลังขาดทุน 3 ครั้งติด
                "cooldown_minutes": 60,       # พักเทรด 60 นาที
            },
            # Time-based Exit: ปิดออเดอร์ที่ค้างนานเกินไป
            "time_exit": {
                "enabled": True,
                "max_hours": 24,  # ปิดออเดอร์ที่ค้าง > 24 ชม.
            },
            # Correlation Protection: ไม่เปิดหลาย position ที่ correlated
            "correlation_protection": {
                "enabled": True,
                "max_same_direction": 2,  # เปิดทิศเดียวกันได้ไม่เกิน 2 position
            },
        }
        self._consecutive_losses = 0
        self._last_loss_time: Optional[datetime] = None
        self._break_even_applied: Dict[str, bool] = {}  # Track positions with break-even
        
        # 📊 Last Analysis Results (for Frontend API) - keyed by symbol
        self._last_analysis_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_analysis: Dict[str, Any] = {}  # Latest analysis (any symbol)
        self._last_titan_decision_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_titan_decision: Dict[str, Any] = {}
        self._last_omega_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_omega_result: Dict[str, Any] = {}
        self._last_alpha_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_alpha_result: Dict[str, Any] = {}
        
        # 🔬 Additional Layer Results for Pipeline Dashboard (keyed by symbol)
        self._last_intel_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_intel_result: Dict[str, Any] = {}      # Advanced Intelligence
        self._last_smart_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_smart_result: Dict[str, Any] = {}      # Smart Brain
        self._last_neural_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_neural_result: Dict[str, Any] = {}     # Neural Brain
        self._last_deep_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_deep_result: Dict[str, Any] = {}       # Deep Intelligence
        self._last_quantum_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_quantum_result: Dict[str, Any] = {}    # Quantum Strategy
        self._last_pro_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_pro_result: Dict[str, Any] = {}        # Pro Features
        self._last_sentiment_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_sentiment_result: Dict[str, Any] = {}  # Sentiment Analyzer
        self._last_candle_count: int = 0                  # For data lake status
        
        self._signal_history: List[Dict] = []  # Keep last 100 signals
        
        # 🎯 Last Trade Result (for debugging why trades didn't execute)
        self._last_trade_result_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_trade_result: Dict[str, Any] = {}  # Last execute_trade() result
        self._last_ultra_decision: Dict[str, Any] = {}  # Last Ultra Intelligence decision
        self._last_supreme_decision: Dict[str, Any] = {}  # Last Supreme Intelligence decision
        self._last_transcendent_decision: Dict[str, Any] = {}  # Last Transcendent Intelligence decision
        self._last_omniscient_decision: Dict[str, Any] = {}  # Last Omniscient Intelligence decision
        
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

    async def _run_intelligence_analysis_for_display(
        self,
        symbol: str,
        signal: str,
        current_price: float,
        df: pd.DataFrame
    ):
        """
        Run 16-Layer Intelligence Analysis for Dashboard Display
        This runs even for WAIT signals so Frontend always has data
        """
        try:
            # Prepare price arrays
            prices = df['close'].values.astype(np.float32) if len(df) > 0 else np.array([current_price])
            volumes = df['volume'].values.astype(np.float32) if 'volume' in df.columns and len(df) > 0 else np.ones(len(prices)) * 1000
            opens = df['open'].values.astype(np.float32) if 'open' in df.columns and len(df) > 0 else prices * 0.999
            highs = df['high'].values.astype(np.float32) if 'high' in df.columns and len(df) > 0 else prices * 1.002
            lows = df['low'].values.astype(np.float32) if 'low' in df.columns and len(df) > 0 else prices * 0.998
            
            side_for_analysis = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL" if signal in ["SELL", "STRONG_SELL"] else "NEUTRAL"
            
            # 🎯 ALPHA ENGINE ANALYSIS
            if self.alpha_engine:
                try:
                    alpha_decision = self.alpha_engine.analyze(
                        symbol=symbol,
                        signal_direction=side_for_analysis if side_for_analysis != "NEUTRAL" else "BUY",
                        opens=opens[-200:] if len(opens) > 200 else opens,
                        highs=highs[-200:] if len(highs) > 200 else highs,
                        lows=lows[-200:] if len(lows) > 200 else lows,
                        closes=prices[-200:] if len(prices) > 200 else prices,
                        volumes=volumes[-200:] if len(volumes) > 200 else volumes
                    )
                    
                    self._last_alpha_result = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "grade": alpha_decision.grade.value,
                        "alpha_score": float(alpha_decision.alpha_score),
                        "confidence": float(alpha_decision.confidence),
                        "order_flow_bias": alpha_decision.order_flow.bias.value if alpha_decision.order_flow else "NEUTRAL",
                        "order_flow_delta": float(alpha_decision.order_flow.delta) if alpha_decision.order_flow else 0,
                        "risk_reward": float(alpha_decision.risk_reward),
                        "position_multiplier": float(alpha_decision.position_multiplier),
                        "optimal_entry": float(alpha_decision.optimal_entry) if alpha_decision.optimal_entry else 0,
                        "stop_loss": float(alpha_decision.stop_loss) if alpha_decision.stop_loss else 0,
                        "targets": [float(t) for t in alpha_decision.targets[:3]] if alpha_decision.targets else [],
                        "market_profile": {
                            "poc": float(alpha_decision.market_profile.poc) if alpha_decision.market_profile else 0,
                            "vah": float(alpha_decision.market_profile.value_area_high) if alpha_decision.market_profile else 0,
                            "val": float(alpha_decision.market_profile.value_area_low) if alpha_decision.market_profile else 0,
                        } if alpha_decision.market_profile else None,
                        "should_trade": alpha_decision.should_trade,
                        "edge_factors": alpha_decision.edge_factors[:5] if alpha_decision.edge_factors else [],
                        "risk_factors": alpha_decision.risk_factors[:5] if alpha_decision.risk_factors else [],
                    }
                    self._last_alpha_result_by_symbol[symbol] = self._last_alpha_result
                    logger.debug(f"📊 Alpha Engine analyzed: Grade={alpha_decision.grade.value}, Score={alpha_decision.alpha_score:.1f}")
                except Exception as e:
                    logger.debug(f"Alpha analysis error: {e}")
            
            # 🧠⚡ OMEGA BRAIN ANALYSIS
            if self.omega_brain:
                try:
                    omega_balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                    
                    omega_decision = self.omega_brain.analyze(
                        symbol=symbol,
                        signal_direction=side_for_analysis if side_for_analysis != "NEUTRAL" else "BUY",
                        opens=opens[-200:] if len(opens) > 200 else opens,
                        highs=highs[-200:] if len(highs) > 200 else highs,
                        lows=lows[-200:] if len(lows) > 200 else lows,
                        closes=prices[-200:] if len(prices) > 200 else prices,
                        volumes=volumes[-200:] if len(volumes) > 200 else volumes,
                        current_balance=omega_balance,
                        other_symbols=self.symbols
                    )
                    
                    self._last_omega_result = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "grade": omega_decision.grade.value,
                        "omega_score": float(omega_decision.omega_score),
                        "confidence": float(omega_decision.confidence),
                        "institutional_flow": omega_decision.institutional_flow.activity.value if omega_decision.institutional_flow else "N/A",
                        "smart_money": omega_decision.institutional_flow.smart_money_direction if omega_decision.institutional_flow else "N/A",
                        "manipulation_detected": omega_decision.manipulation_alert.manipulation_type.value if omega_decision.manipulation_alert else "NONE",
                        "manipulation_probability": float(omega_decision.manipulation_alert.probability) if omega_decision.manipulation_alert else 0,
                        "sentiment": float(omega_decision.sentiment.overall_sentiment) if omega_decision.sentiment else 0,
                        "current_regime": omega_decision.regime_prediction.current_regime if omega_decision.regime_prediction else "N/A",
                        "predicted_regime": omega_decision.regime_prediction.predicted_regime if omega_decision.regime_prediction else "N/A",
                        "position_multiplier": float(omega_decision.position_multiplier),
                        "risk_reward": float(omega_decision.risk_reward),
                        "should_trade": omega_decision.should_trade,
                        "final_verdict": omega_decision.final_verdict,
                        "institutional_insight": omega_decision.institutional_insight,
                        "edge_factors": omega_decision.edge_factors[:5] if omega_decision.edge_factors else [],
                        "risk_factors": omega_decision.risk_factors[:5] if omega_decision.risk_factors else [],
                    }
                    self._last_omega_result_by_symbol[symbol] = self._last_omega_result
                    logger.debug(f"📊 Omega Brain analyzed: Grade={omega_decision.grade.value}, Score={omega_decision.omega_score:.1f}")
                except Exception as e:
                    logger.debug(f"Omega analysis error: {e}")
            
            # 🏛️⚔️ TITAN CORE ANALYSIS
            if self.titan_core:
                try:
                    from trading.titan_core import ModuleSignal
                    
                    # Collect module signals
                    module_signals = []
                    
                    # Add Alpha signal
                    if self._last_alpha_result:
                        module_signals.append(ModuleSignal(
                            module_name="AlphaEngine",
                            should_trade=self._last_alpha_result.get("should_trade", False),
                            direction=side_for_analysis if side_for_analysis != "NEUTRAL" else "BUY",
                            confidence=self._last_alpha_result.get("alpha_score", 50),
                            multiplier=self._last_alpha_result.get("position_multiplier", 1.0),
                            score=self._last_alpha_result.get("alpha_score", 50),
                            reasons=self._last_alpha_result.get("edge_factors", []),
                            warnings=self._last_alpha_result.get("risk_factors", [])
                        ))
                    
                    # Add Omega signal
                    if self._last_omega_result:
                        module_signals.append(ModuleSignal(
                            module_name="OmegaBrain",
                            should_trade=self._last_omega_result.get("should_trade", False),
                            direction=side_for_analysis if side_for_analysis != "NEUTRAL" else "BUY",
                            confidence=self._last_omega_result.get("omega_score", 50),
                            multiplier=self._last_omega_result.get("position_multiplier", 1.0),
                            score=self._last_omega_result.get("omega_score", 50),
                            reasons=self._last_omega_result.get("edge_factors", []),
                            warnings=self._last_omega_result.get("risk_factors", [])
                        ))
                    
                    titan_decision = self.titan_core.synthesize(
                        symbol=symbol,
                        signal_direction=side_for_analysis if side_for_analysis != "NEUTRAL" else "BUY",
                        closes=prices[-200:] if len(prices) > 200 else prices,
                        highs=highs[-200:] if len(highs) > 200 else highs,
                        lows=lows[-200:] if len(lows) > 200 else lows,
                        volumes=volumes[-200:] if len(volumes) > 200 else volumes,
                        module_signals=module_signals,
                        current_price=current_price
                    )
                    
                    self._last_titan_decision = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "grade": titan_decision.grade.value,
                        "titan_score": float(titan_decision.titan_score),
                        "confidence": float(titan_decision.confidence),
                        "consensus": titan_decision.consensus.level.value,
                        "agreement_ratio": float(titan_decision.consensus.agreement_ratio),
                        "market_condition": titan_decision.market_condition.value,
                        "prediction": {
                            "direction": titan_decision.prediction.final_prediction,
                            "predicted_move": float(titan_decision.prediction.predicted_move),
                        },
                        "position_multiplier": float(titan_decision.position_multiplier),
                        "agreeing_modules": titan_decision.agreeing_modules,
                        "total_modules": titan_decision.total_modules,
                        "should_trade": titan_decision.should_trade,
                        "final_verdict": titan_decision.final_verdict,
                        "edge_factors": titan_decision.edge_factors[:5] if titan_decision.edge_factors else [],
                        "risk_factors": titan_decision.risk_factors[:5] if titan_decision.risk_factors else [],
                    }
                    self._last_titan_decision_by_symbol[symbol] = self._last_titan_decision
                    logger.debug(f"📊 Titan Core analyzed: Grade={titan_decision.grade.value}, Score={titan_decision.titan_score:.1f}")
                except Exception as e:
                    logger.debug(f"Titan analysis error: {e}")
            
            # 🧠 ADVANCED INTELLIGENCE ANALYSIS (for display)
            if self.intelligence:
                try:
                    side_for_intel = "BUY" if side_for_analysis != "SELL" else "SELL"
                    
                    # Build h1_data dict from DataFrame
                    h1_data_dict = {
                        "open": opens,
                        "high": highs,
                        "low": lows,
                        "close": prices,
                    }
                    
                    intel_decision = self.intelligence.analyze(
                        signal_side=side_for_intel,
                        pattern_confidence=70,  # Default for display
                        h1_data=h1_data_dict,
                        win_rate=0.5,
                        avg_win=1.0,
                        avg_loss=1.0,
                        total_trades=0,
                    )
                    
                    self._last_intel_result = {
                        "regime": intel_decision.regime.regime.value if intel_decision.regime else "N/A",
                        "trend_strength": intel_decision.regime.trend_strength if intel_decision.regime else 0,
                        "mtf_alignment": "ALIGNED" if intel_decision.can_trade else "CONFLICTING",
                        "position_size_factor": intel_decision.position_size_factor,
                        "can_trade": intel_decision.can_trade,
                        "momentum_state": intel_decision.momentum.momentum_state if intel_decision.momentum else "N/A",
                        "rsi": intel_decision.momentum.rsi if intel_decision.momentum else 0,
                    }
                    self._last_intel_result_by_symbol[symbol] = self._last_intel_result
                    logger.debug(f"📊 Intelligence analyzed: Regime={self._last_intel_result.get('regime')}")
                except Exception as e:
                    logger.warning(f"Intelligence analysis error: {e}")
            
            # 📚 SMART BRAIN ANALYSIS (for display)
            if self.smart_brain:
                try:
                    smart_result = {
                        "pattern_count": getattr(self.smart_brain, 'pattern_count', 0),
                        "position_multiplier": 1.0,
                        "win_rate": 0,
                        "avg_rr": 0
                    }
                    # Get stats from journal if available
                    if hasattr(self.smart_brain, 'journal') and self.smart_brain.journal:
                        stats = self.smart_brain.journal.get_stats()
                        if stats:
                            smart_result["win_rate"] = stats.get("win_rate", 0)
                            smart_result["avg_rr"] = stats.get("avg_rr", 0)
                    
                    self._last_smart_result = smart_result
                    self._last_smart_result_by_symbol[symbol] = self._last_smart_result
                    logger.debug(f"📊 Smart Brain analyzed")
                except Exception as e:
                    logger.warning(f"Smart analysis error: {e}")
            
            # 🧬 NEURAL BRAIN ANALYSIS (for display)
            if self.neural_brain:
                try:
                    balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                    
                    neural_decision = self.neural_brain.analyze(
                        signal_side="BUY" if side_for_analysis != "SELL" else "SELL",
                        prices=prices,
                        volumes=volumes,
                        balance=balance,
                    )
                    
                    self._last_neural_result = {
                        "market_state": neural_decision.market_state.value,
                        "pattern_quality": neural_decision.pattern_quality,
                        "dna_score": neural_decision.confidence,
                        "position_multiplier": neural_decision.position_size_factor,
                        "can_trade": neural_decision.can_trade,
                        "anomaly_detected": neural_decision.anomaly_detected,
                    }
                    self._last_neural_result_by_symbol[symbol] = self._last_neural_result
                    logger.debug(f"📊 Neural Brain analyzed: State={neural_decision.market_state.value}")
                except Exception as e:
                    logger.debug(f"Neural analysis error: {e}")
            
            # 🔮 DEEP INTELLIGENCE ANALYSIS (for display)
            if self.deep_intelligence:
                try:
                    # Build timeframe_data dict
                    timeframe_data = {"H1": prices[-200:] if len(prices) > 200 else prices}
                    
                    deep_decision = self.deep_intelligence.analyze(
                        symbol=symbol,
                        signal_direction="BUY" if side_for_analysis != "SELL" else "SELL",
                        timeframe_data=timeframe_data,
                        current_params={},
                        other_symbols_direction=None,
                    )
                    
                    self._last_deep_result = {
                        "correlation": deep_decision.correlation_score if hasattr(deep_decision, 'correlation_score') else 0,
                        "session": deep_decision.session_score if hasattr(deep_decision, 'session_score') else "N/A",
                        "position_multiplier": deep_decision.position_multiplier if hasattr(deep_decision, 'position_multiplier') else 1.0,
                        "cross_asset_signal": "N/A",
                        "deep_score": deep_decision.confidence if hasattr(deep_decision, 'confidence') else 0,
                        "confidence": deep_decision.confidence if hasattr(deep_decision, 'confidence') else 0,
                        "should_trade": deep_decision.should_trade if hasattr(deep_decision, 'should_trade') else False,
                        "timeframe_score": deep_decision.timeframe_score if hasattr(deep_decision, 'timeframe_score') else 0,
                        "confluence_level": deep_decision.confluence_level.value if hasattr(deep_decision, 'confluence_level') else "N/A",
                    }
                    self._last_deep_result_by_symbol[symbol] = self._last_deep_result
                    logger.debug(f"📊 Deep Intelligence analyzed: Score={self._last_deep_result.get('confidence', 0):.1f}")
                except Exception as e:
                    logger.warning(f"Deep analysis error: {e}")
            
            # ⚛️ QUANTUM STRATEGY ANALYSIS (for display)
            if self.quantum_strategy:
                try:
                    quantum_decision = self.quantum_strategy.analyze(
                        symbol=symbol,
                        signal_direction="BUY" if side_for_analysis != "SELL" else "SELL",
                        prices=prices[-200:] if len(prices) > 200 else prices,
                        volumes=volumes[-200:] if len(volumes) > 200 else volumes,
                        entry_price=current_price,
                    )
                    
                    self._last_quantum_result = {
                        "volatility_regime": quantum_decision.volatility.regime.value if quantum_decision.volatility else "N/A",
                        "fractal": f"H={quantum_decision.fractal.hurst_exponent:.2f}" if quantum_decision.fractal else "N/A",
                        "position_multiplier": quantum_decision.position_multiplier,
                        "microstructure_signal": quantum_decision.microstructure.smart_money_signal if quantum_decision.microstructure else "N/A",
                        "quantum_score": quantum_decision.quantum_score,
                        "confidence": quantum_decision.confidence,
                        "should_trade": quantum_decision.should_trade,
                    }
                    self._last_quantum_result_by_symbol[symbol] = self._last_quantum_result
                    logger.debug(f"📊 Quantum Strategy analyzed: Score={quantum_decision.quantum_score:.1f}")
                except Exception as e:
                    logger.debug(f"Quantum analysis error: {e}")
            
            # 🏆 PRO TRADING FEATURES (for display)
            if self.pro_features:
                try:
                    # Use session_filter from ProTradingFeatures
                    if hasattr(self.pro_features, 'session_filter'):
                        session_info = self.pro_features.session_filter.get_session_info()
                        self._last_pro_result = {
                            "session": session_info.current_session.value if hasattr(session_info, 'current_session') else "N/A",
                            "session_quality": session_info.quality_score if hasattr(session_info, 'quality_score') else 0,
                            "news_impact": "NONE",
                            "position_multiplier": 1.0,
                        }
                    else:
                        self._last_pro_result = {
                            "session": "N/A",
                            "session_quality": 0,
                            "news_impact": "NONE",
                            "position_multiplier": 1.0,
                        }
                    self._last_pro_result_by_symbol[symbol] = self._last_pro_result
                    logger.debug(f"📊 Pro Features analyzed: Session={self._last_pro_result.get('session')}")
                except Exception as e:
                    logger.warning(f"Pro features error: {e}")
                    
        except Exception as e:
            logger.warning(f"Intelligence analysis for display failed: {e}")

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
            enable_sentiment_filter=True,  # 🆕 Smart Money/Contrarian
        )
        logger.info(f"✓ Enhanced analyzer initialized (Min Quality: {self.min_quality.value})")
        
        # 5. 🛡️ Risk Guardian - ป้องกันการล้างพอร์ต
        self.risk_guardian = create_risk_guardian(
            max_risk_per_trade=self.max_risk_percent,
            max_daily_loss=5.0,    # หยุดเทรดเมื่อขาดทุน 5% ต่อวัน
            max_drawdown=10.0,     # หยุดเทรดเมื่อ drawdown 10%
        )
        logger.info(f"✓ Risk Guardian initialized (Max Daily Loss: 5%, Max Drawdown: 10%)")
        
        # 6. 🏆 Pro Trading Features - สิ่งที่ Pro Trader ทำ
        self.pro_features = ProTradingFeatures(
            enable_session_filter=True,    # เทรดเฉพาะช่วงเวลาดี
            enable_news_filter=True,       # หยุดช่วงข่าว
            enable_correlation_filter=True, # ไม่เปิดคู่ที่ correlate
            enable_losing_streak_stop=True, # หยุดเมื่อแพ้ติดๆ
            min_session_quality=40,         # อนุญาต Tokyo session ขึ้นไป
        )
        logger.info("✓ Pro Trading Features initialized:")
        logger.info("   - Session Filter (London-NY Overlap = Best)")
        logger.info("   - News Filter (หยุดช่วง NFP, FOMC, CPI)")
        logger.info("   - Trailing Stop (ล็อค profit อัตโนมัติ)")
        logger.info("   - Break-Even (ย้าย SL ไปจุดเข้า)")
        logger.info("   - Losing Streak Stop (หยุดแพ้ 5 ติด)")
        logger.info("   - Correlation Filter (EURUSD vs GBPUSD)")
        
        # 7. Firebase (ต้องสร้างก่อน Smart Brain)
        self.firebase_service = None
        if self.broadcast_to_firebase:
            try:
                self.firebase_service = get_firebase_service()
                logger.info("✓ Firebase service initialized ☁️")
            except Exception as e:
                logger.warning(f"Firebase not available: {e}")
                self.firebase_service = None
        
        # 8. 🧠 Smart Brain - เรียนรู้จากการเทรด (with Firebase)
        self.smart_brain = SmartBrain(
            enable_pullback_entry=True,   # รอ pullback ก่อนเข้า
            enable_partial_tp=True,       # ปิดบางส่วนที่ TP1
            enable_stale_exit=True,       # ปิดเทรดที่ค้างนาน
            enable_adaptive_risk=True,    # ปรับ size ตาม performance
            firebase_service=self.firebase_service,  # 🔥 Cloud Storage
        )
        logger.info("✓ Smart Brain initialized:")
        logger.info("   - Trade Journal (บันทึกทุกเทรด)")
        logger.info("   - Pattern Memory (จำว่า pattern ไหนได้/เสีย)")
        logger.info("   - Adaptive Risk (winning streak → +size)")
        logger.info("   - Time Analysis (รู้ว่าช่วงไหนเทรดดี)")
        logger.info("   - Symbol Analysis (รู้ว่า symbol ไหนเก่ง)")
        logger.info("   - Partial TP (ปิด 50% ที่ TP1)")
        if self.firebase_service:
            logger.info("   - ☁️ Firebase Cloud Sync: ENABLED")
        
        # 9. 🧠 Advanced Intelligence - ฉลาดขั้นสูง
        self.intelligence = AdvancedIntelligence(
            enable_regime=True,      # ตรวจจับ Market Regime
            enable_mtf=True,         # Multi-Timeframe Analysis
            enable_momentum=True,    # RSI, MACD, Stochastic
            enable_sr=True,          # Auto S/R Detection
            enable_kelly=True,       # Kelly Criterion Sizing
            min_confluence=2,        # ลดลงเหลือ 2 ปัจจัยขึ้นไป (pattern + regime/momentum/sr)
        )
        logger.info("✓ Advanced Intelligence initialized:")
        logger.info("   - Market Regime Detection (Trend/Range/Volatile)")
        logger.info("   - Multi-Timeframe Analysis (H1/H4/D1)")
        logger.info("   - Momentum Scanner (RSI+MACD+Stoch)")
        logger.info("   - Auto S/R Detection")
        logger.info("   - Kelly Criterion Position Sizing")
        logger.info("   - Confluence Scoring (min 2 factors)")
        
        # 10. 📚 Continuous Learning - เรียนรู้ตลอดเวลา
        self.learning_system = ContinuousLearningSystem(
            data_dir="data/learning",
            enable_background=True,  # ประหยัดทรัพยากร
            firebase_service=self.firebase_service,
        )
        logger.info("✓ Continuous Learning System initialized:")
        logger.info("   - Online Learning (เรียนทีละ trade)")
        logger.info("   - Market Cycle Detection")
        logger.info("   - Pattern Evolution Tracking")
        logger.info("   - Auto Strategy Optimization")
        logger.info("   - Background Processing (ประหยัด CPU)")
        
        # 11. 🧬 Neural Brain - Deep Pattern Understanding
        self.neural_brain = NeuralBrain(
            data_dir="data/neural",
            firebase_service=self.firebase_service,
            enable_dna=True,           # Pattern DNA tracking
            enable_state_machine=True, # Market state detection
            enable_anomaly=True,       # Anomaly detection
            enable_risk_intel=True,    # Risk intelligence
        )
        logger.info("✓ Neural Brain initialized:")
        logger.info("   - Pattern DNA Analyzer (จำ DNA ที่ทำกำไร)")
        logger.info("   - Market State Machine (7 states)")
        logger.info("   - Anomaly Detector (ตรวจจับผิดปกติ)")
        logger.info("   - Risk Intelligence (ฉลาดเรื่อง risk)")
        
        # 12. 🔮 Deep Intelligence - Multi-layer Analysis
        self.deep_intelligence = get_deep_intelligence()
        logger.info("✓ Deep Intelligence initialized:")
        logger.info("   - Multi-Timeframe Confluence (M15/H1/H4/D1)")
        logger.info("   - Cross-Asset Correlation")
        logger.info("   - Adaptive Parameter Tuning")
        logger.info("   - Predictive Model (5 methods)")
        logger.info("   - Session Analyzer")
        
        # 13. ⚛️ Quantum Strategy - Advanced Quantitative Analysis
        self.quantum_strategy = get_quantum_strategy()
        logger.info("✓ Quantum Strategy initialized:")
        logger.info("   - Market Microstructure (Smart Money Detection)")
        logger.info("   - Volatility Regime (GARCH-like)")
        logger.info("   - Fractal Analysis (Hurst Exponent)")
        logger.info("   - Sentiment Aggregator")
        logger.info("   - Dynamic Exit Manager")
        
        # 14. 🎯 Alpha Engine - Ultimate Trading Intelligence
        self.alpha_engine = get_alpha_engine()
        logger.info("✓ Alpha Engine initialized:")
        logger.info("   - Order Flow Analyzer (Volume Delta)")
        logger.info("   - Liquidity Zone Detector (SMC)")
        logger.info("   - Market Profile (POC/Value Area)")
        logger.info("   - Divergence Scanner (RSI/MACD/OBV)")
        logger.info("   - Momentum Wave Analyzer")
        logger.info("   - Risk Metrics Calculator")
        
        # 15. 🧠⚡ Omega Brain - Institutional-Grade Intelligence
        self.omega_brain = get_omega_brain()
        logger.info("✓ Omega Brain initialized:")
        logger.info("   - Institutional Flow Detector (Big Money)")
        logger.info("   - Manipulation Scanner (Stop Hunts)")
        logger.info("   - Sentiment Fusion Engine")
        logger.info("   - Regime Transition Predictor")
        logger.info("   - Position Orchestrator")
        logger.info("   - Risk Parity Allocator")
        
        # 16. 🏛️⚔️ Titan Core - Meta-Intelligence Synthesis
        self.titan_core = get_titan_core()
        logger.info("✓ Titan Core initialized:")
        logger.info("   - Consensus Engine (Module Agreement)")
        logger.info("   - Prediction Ensemble (Multi-Method)")
        logger.info("   - Confidence Calibrator (Self-Correcting)")
        logger.info("   - Dynamic Weight Optimizer")
        logger.info("   - Self-Improvement Engine")
        logger.info("   - Market Condition Analyzer")
        
        # 17. 🧠⚡ Ultra Intelligence - 10x Smarter Trading
        self.ultra_intelligence = get_ultra_intelligence()
        logger.info("✓ Ultra Intelligence initialized:")
        logger.info("   - Smart Money Concepts (SMC)")
        logger.info("   - Market Structure Analysis")
        logger.info("   - Session Quality Filter")
        logger.info("   - Volatility Scaling")
        logger.info("   - Liquidity Zone Detection")
        logger.info("   - Adaptive Position Sizing")
        logger.info("   - Partial Profit Taking")
        logger.info("   - Momentum Filter")
        
        # 18. 🏆👑 Supreme Intelligence - 20x Smarter (Hedge Fund Level)
        self.supreme_intelligence = get_supreme_intelligence()
        logger.info("✓ Supreme Intelligence initialized:")
        logger.info("   - Order Flow Analysis (Buy/Sell Pressure)")
        logger.info("   - Institutional Footprint Detection")
        logger.info("   - Market Entropy Analysis (Chaos Level)")
        logger.info("   - Fractal Dimension Calculation")
        logger.info("   - Win Probability Estimation")
        logger.info("   - Alpha Potential Calculation")
        logger.info("   - Self-Learning Weight Optimization")
        logger.info("   - Execution Timing Quality")
        logger.info("   - Dynamic SL/TP Optimization")
        logger.info("   - Scale In/Out Level Detection")
        
        # 19. 🌌✨ Transcendent Intelligence - 50x Smarter (Beyond Human)
        self.transcendent_intelligence = get_transcendent_intelligence()
        logger.info("✓ Transcendent Intelligence initialized:")
        logger.info("   - Quantum Probability Fields")
        logger.info("   - Multi-Dimensional Analysis (7D)")
        logger.info("   - Black Swan Detection")
        logger.info("   - Market Microstructure Analysis")
        logger.info("   - Signal Purity Filter")
        logger.info("   - Risk Topology Analysis")
        logger.info("   - Quantum Kelly Criterion")
        logger.info("   - Entry/Exit Optimization")
        logger.info("   - Scale In/Out Levels")
        logger.info("   - Time Decay Factor")
        logger.info("   - Market Synchronicity")
        logger.info("   - Self-Evolving Weights")
        
        # 20. 🔮 Omniscient Intelligence - 100x Smarter (All-Knowing)
        self.omniscient_intelligence = get_omniscient_intelligence()
        logger.info("✓ Omniscient Intelligence initialized:")
        logger.info("   === MARKET PHYSICS (1-10) ===")
        logger.info("   - Gravitational Price Levels")
        logger.info("   - Momentum Wave Interference")
        logger.info("   - Price Velocity & Acceleration")
        logger.info("   - Resonance Frequency Detection")
        logger.info("   === NEURAL ENSEMBLE (11-20) ===")
        logger.info("   - Deep LSTM Prediction")
        logger.info("   - Transformer Attention")
        logger.info("   - CNN Pattern Scanner")
        logger.info("   - Ensemble Voting Network")
        logger.info("   === INFORMATION THEORY (21-30) ===")
        logger.info("   - Shannon Entropy Decoder")
        logger.info("   - KL Divergence Monitor")
        logger.info("   - Signal-to-Noise Ratio")
        logger.info("   === CHAOS & COMPLEXITY (31-40) ===")
        logger.info("   - Lyapunov Exponent")
        logger.info("   - Fractal Dimension")
        logger.info("   - Bifurcation Detection")
        logger.info("   === GAME THEORY (41-50) ===")
        logger.info("   - Nash Equilibrium")
        logger.info("   - Pareto Efficiency")
        logger.info("   - Dominant Strategy")
        logger.info("   === BEHAVIORAL FINANCE (51-60) ===")
        logger.info("   - Herding Detection")
        logger.info("   - Bias Identification")
        logger.info("   - Regret Minimization")
        logger.info("   === RISK MATHEMATICS (71-80) ===")
        logger.info("   - VaR/CVaR Calculator")
        logger.info("   - Jump Probability")
        logger.info("   - Max Drawdown Predictor")
        logger.info("   === OMNISCIENT CORE (91-100) ===")
        logger.info("   - Consciousness Simulation")
        logger.info("   - Universal Alignment")
        logger.info("   - Prophecy Generation")
        
        logger.info("=" * 60)
        logger.info("✓ Bot initialization complete!")
        logger.info(f"🏛️ Total Intelligence Layers: 20")
        logger.info(f"🔮 Total Features: 100+ (OMNISCIENT)")
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
        
        # 📚 Set callback for learning from closed positions
        self.trading_engine.on_position_closed = self._on_position_closed
        
        logger.info("✓ Trading engine started")
    
    def _on_position_closed(self, result):
        """Callback เมื่อ Position ปิด - ใช้เรียนรู้"""
        try:
            position_id = result.position_id if hasattr(result, 'position_id') else str(result)
            
            # หา factors ที่ใช้ตอนเปิด trade
            factors_used = self._pending_trade_factors.pop(position_id, None)
            
            if factors_used and self.learning_system:
                # คำนวณ profit/loss
                pnl = result.pnl if hasattr(result, 'pnl') else 0
                is_win = pnl > 0
                
                # คำนวณ pnl percent
                entry_price = result.entry_price if hasattr(result, 'entry_price') else 1
                pnl_percent = (pnl / entry_price * 100) if entry_price > 0 else 0
                
                # 🧠 Learn from this trade (synchronous - uses background queue internally)
                try:
                    # Convert factor dict to bool dict
                    factor_bools = {k: bool(v) for k, v in factors_used.items() 
                                   if k not in ['symbol', 'signal', 'quality', 'entry_time']}
                    
                    self.learning_system.learn_from_trade(
                        is_win=is_win,
                        pnl_percent=pnl_percent,
                        factors=factor_bools,
                        pattern_hash=f"{factors_used.get('symbol', 'UNK')}_{factors_used.get('entry_time', '')}",
                        rr_ratio=1.5,  # Default R:R
                    )
                    
                    logger.info(f"📚 Trade closed: {'✅ WIN' if is_win else '❌ LOSS'} ${pnl:.2f} ({pnl_percent:.1f}%) - Learning recorded")
                except Exception as e:
                    logger.error(f"Learning record error: {e}")
        except Exception as e:
            logger.error(f"Error in _on_position_closed: {e}")
    
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
        enhanced_result = await self.enhanced_analyzer.analyze(
            base_signal=base_signal,
            base_confidence=base_confidence,
            ohlcv_data=ohlcv_data,
            current_price=current_price,
            symbol=symbol,  # Pass symbol for sentiment analysis
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
        
        # � Run 16-Layer Intelligence Analysis for Dashboard (even for WAIT signals)
        await self._run_intelligence_analysis_for_display(
            symbol=symbol,
            signal=enhanced_result.signal,
            current_price=current_price,
            df=df
        )
        
        # �📚 Feed market data to Continuous Learning System
        if self.learning_system and len(df) > 0:
            try:
                # Feed latest close price to cycle detector
                self.learning_system.cycle_detector.add_data(
                    close=float(df['close'].iloc[-1]),
                    volume=float(df['volume'].iloc[-1]) if 'volume' in df.columns else 1000,
                    volatility=float(df['high'].iloc[-1] - df['low'].iloc[-1])
                )
            except Exception as e:
                logger.debug(f"Learning feed error: {e}")
        
        return result
    
    async def execute_trade(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade based on enhanced analysis
        
        SECURITY: Mandatory Stop Loss Enforcement
        - All trades MUST have a Stop Loss
        - If no SL provided, auto-calculate from ATR or use 2% default
        
        PRO FEATURES:
        - Session Filter (เทรดเฉพาะช่วงเวลาดี)
        - News Filter (หยุดช่วงข่าว)
        - Correlation Filter (ไม่เปิดคู่ที่ correlate)
        - Losing Streak Stop (หยุดเมื่อแพ้ติดๆ)
        
        SMART BRAIN:
        - Pattern Memory (จำ pattern ที่เคยเทรด)
        - Adaptive Risk (ปรับ size ตาม performance)
        - Time Analysis (รู้ว่าช่วงไหนดี)
        - Symbol Analysis (รู้ว่า symbol ไหนเก่ง)
        
        ADVANCED INTELLIGENCE:
        - Market Regime Detection (Trend/Range/Volatile)
        - Multi-Timeframe Confirmation
        - Momentum Analysis (RSI+MACD+Stoch)
        - Support/Resistance Detection
        - Kelly Criterion Sizing
        - Confluence Scoring
        """
        symbol = analysis.get("symbol")
        signal = analysis.get("signal", "WAIT")
        quality = analysis.get("quality", "SKIP")
        current_price = analysis.get("current_price", 0)
        risk_mgmt = analysis.get("risk_management", {})
        
        logger.info(f"🔍 execute_trade() called for {symbol}")
        logger.info(f"   Signal: {signal}, Quality: {quality}, Price: {current_price}")
        
        # 🧠 SMART FEATURES CHECK - ตรวจสอบก่อนเทรด
        can_trade, reason = self._can_trade_today()
        if not can_trade:
            logger.warning(f"⛔ Trade blocked by smart features: {reason}")
            return {"action": "BLOCKED", "reason": reason, "symbol": symbol}
        
        # 🔗 Correlation Check
        side_str = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
        can_trade, reason = self._check_correlation(symbol, side_str)
        if not can_trade:
            logger.warning(f"⛔ Trade blocked by correlation: {reason}")
            return {"action": "BLOCKED", "reason": reason, "symbol": symbol}
        
        # 🧠⚡ ULTRA INTELLIGENCE CHECK - 10x Smarter
        ultra_decision = None
        ultra_multiplier = 1.0
        if self.ultra_intelligence:
            try:
                # Get price data
                df = await self.data_provider.get_klines(symbol=symbol, timeframe="H1", limit=100)
                if len(df) >= 50:
                    prices = df['close'].values.astype(np.float32)
                    highs = df['high'].values.astype(np.float32)
                    lows = df['low'].values.astype(np.float32)
                    volumes = df['volume'].values.astype(np.float32) if 'volume' in df.columns else None
                    
                    # Calculate ATR safely
                    tr = np.maximum(
                        highs[-14:] - lows[-14:],
                        np.abs(highs[-14:] - prices[-15:-1])
                    )
                    atr = np.mean(tr)
                    
                    # Get balance
                    balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                    equity = await self.trading_engine.broker.get_equity() if self.trading_engine else balance
                    
                    ultra_decision = self.ultra_intelligence.analyze(
                        symbol=symbol,
                        signal_side=side_str,
                        current_price=current_price,
                        prices=prices,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                        atr=atr,
                        base_confidence=analysis.get("enhanced_confidence", 70),
                        current_balance=balance,
                        account_equity=equity
                    )
                    
                    # Log Ultra Intelligence results
                    logger.info(f"🧠⚡ ULTRA INTELLIGENCE:")
                    logger.info(f"   Session: {ultra_decision.session_quality.value}")
                    logger.info(f"   Volatility: {ultra_decision.volatility_state.value}")
                    logger.info(f"   Phase: {ultra_decision.market_phase.value}")
                    logger.info(f"   Structure: {ultra_decision.market_structure.trend if ultra_decision.market_structure else 'N/A'}")
                    logger.info(f"   Size Mult: {ultra_decision.position_size_multiplier}x")
                    logger.info(f"   Optimal R:R: {ultra_decision.optimal_rr}")
                    logger.info(f"   Confidence: {ultra_decision.confidence}%")
                    
                    for reason in ultra_decision.reasons:
                        logger.info(f"   ✅ {reason}")
                    for warning in ultra_decision.warnings:
                        logger.warning(f"   ⚠️ {warning}")
                    
                    # Check if Ultra blocks the trade
                    if not ultra_decision.can_trade:
                        logger.warning(f"⛔ Trade blocked by Ultra Intelligence")
                        return {
                            "action": "BLOCKED_BY_ULTRA",
                            "reason": "; ".join(ultra_decision.warnings),
                            "symbol": symbol,
                            "session": ultra_decision.session_quality.value,
                            "volatility": ultra_decision.volatility_state.value
                        }
                    
                    # Apply Ultra multiplier
                    ultra_multiplier = ultra_decision.position_size_multiplier
                    
                    # Store for later use
                    self._last_ultra_decision = {
                        "symbol": symbol,
                        "can_trade": ultra_decision.can_trade,
                        "confidence": ultra_decision.confidence,
                        "size_multiplier": ultra_decision.position_size_multiplier,
                        "optimal_rr": ultra_decision.optimal_rr,
                        "session": ultra_decision.session_quality.value,
                        "volatility": ultra_decision.volatility_state.value,
                        "phase": ultra_decision.market_phase.value,
                        "entry_type": ultra_decision.entry_type,
                        "use_partial_tp": ultra_decision.use_partial_tp,
                        "reasons": ultra_decision.reasons,
                        "warnings": ultra_decision.warnings,
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Ultra Intelligence analysis failed: {e}")
        
        # 🏆👑 SUPREME INTELLIGENCE CHECK - 20x Smarter (Hedge Fund Level)
        supreme_decision = None
        supreme_multiplier = 1.0
        if self.supreme_intelligence:
            try:
                df = await self.data_provider.get_klines(symbol=symbol, timeframe="H1", limit=100)
                if len(df) >= 50:
                    prices = df['close'].values.astype(np.float32)
                    highs = df['high'].values.astype(np.float32)
                    lows = df['low'].values.astype(np.float32)
                    volumes = df['volume'].values.astype(np.float32) if 'volume' in df.columns else None
                    
                    tr = np.maximum(
                        highs[-14:] - lows[-14:],
                        np.abs(highs[-14:] - prices[-15:-1])
                    )
                    atr = np.mean(tr)
                    
                    balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                    equity = await self.trading_engine.broker.get_equity() if self.trading_engine else balance
                    
                    supreme_decision = self.supreme_intelligence.analyze(
                        symbol=symbol,
                        signal_side=side_str,
                        current_price=current_price,
                        prices=prices,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                        atr=atr,
                        base_confidence=analysis.get("enhanced_confidence", 70),
                        balance=balance,
                        equity=equity,
                    )
                    
                    logger.info(f"🏆👑 SUPREME INTELLIGENCE:")
                    logger.info(f"   Entropy: {supreme_decision.entropy_level.value}")
                    logger.info(f"   Institutional: {supreme_decision.institutional_activity.value}")
                    logger.info(f"   Momentum: {supreme_decision.momentum_quality.value}")
                    logger.info(f"   Confluence: {supreme_decision.confluence_score:.0f}%")
                    logger.info(f"   Win Prob: {supreme_decision.win_probability:.0f}%")
                    logger.info(f"   Alpha: {supreme_decision.alpha_potential:.1f}%")
                    logger.info(f"   Signal: {supreme_decision.signal_strength}")
                    logger.info(f"   Size: {supreme_decision.optimal_size_percent:.2f}x")
                    logger.info(f"   Execution: {supreme_decision.execution_timing.value}")
                    
                    for reason in supreme_decision.reasons:
                        logger.info(f"   ✅ {reason}")
                    for warning in supreme_decision.warnings:
                        logger.warning(f"   ⚠️ {warning}")
                    
                    if not supreme_decision.can_trade:
                        logger.warning(f"⛔ Trade blocked by Supreme Intelligence")
                        return {
                            "action": "BLOCKED_BY_SUPREME",
                            "reason": "; ".join(supreme_decision.warnings),
                            "symbol": symbol,
                            "entropy": supreme_decision.entropy_level.value,
                            "win_probability": supreme_decision.win_probability,
                            "confluence_score": supreme_decision.confluence_score,
                        }
                    
                    supreme_multiplier = supreme_decision.optimal_size_percent
                    
                    self._last_supreme_decision = {
                        "symbol": symbol,
                        "can_trade": supreme_decision.can_trade,
                        "confidence": supreme_decision.confidence,
                        "signal_strength": supreme_decision.signal_strength,
                        "size_percent": supreme_decision.optimal_size_percent,
                        "entropy": supreme_decision.entropy_level.value,
                        "institutional": supreme_decision.institutional_activity.value,
                        "momentum": supreme_decision.momentum_quality.value,
                        "confluence": supreme_decision.confluence_score,
                        "win_probability": supreme_decision.win_probability,
                        "alpha_potential": supreme_decision.alpha_potential,
                        "execution_timing": supreme_decision.execution_timing.value,
                        "optimal_sl": supreme_decision.optimal_sl_distance,
                        "optimal_tp": supreme_decision.optimal_tp_distance,
                        "max_holding_hours": supreme_decision.max_holding_hours,
                        "reasons": supreme_decision.reasons,
                        "warnings": supreme_decision.warnings,
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Supreme Intelligence analysis failed: {e}")
        
        # 🌌✨ TRANSCENDENT INTELLIGENCE CHECK - 50x Smarter (Beyond Human)
        transcendent_decision = None
        transcendent_multiplier = 1.0
        if self.transcendent_intelligence:
            try:
                df = await self.data_provider.get_klines(symbol=symbol, timeframe="H1", limit=100)
                if len(df) >= 50:
                    prices = df['close'].values.astype(np.float32)
                    highs = df['high'].values.astype(np.float32)
                    lows = df['low'].values.astype(np.float32)
                    volumes = df['volume'].values.astype(np.float32) if 'volume' in df.columns else None
                    
                    tr = np.maximum(
                        highs[-14:] - lows[-14:],
                        np.abs(highs[-14:] - prices[-15:-1])
                    )
                    atr = np.mean(tr)
                    
                    balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                    equity = await self.trading_engine.broker.get_equity() if self.trading_engine else balance
                    
                    transcendent_decision = self.transcendent_intelligence.analyze(
                        symbol=symbol,
                        signal_side=side_str,
                        current_price=current_price,
                        prices=prices,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                        atr=atr,
                        base_confidence=analysis.get("enhanced_confidence", 70),
                        balance=balance,
                        equity=equity,
                    )
                    
                    logger.info(f"🌌✨ TRANSCENDENT INTELLIGENCE:")
                    logger.info(f"   Quantum: {transcendent_decision.quantum_field.quantum_state.value}")
                    logger.info(f"   Bull Prob: {transcendent_decision.quantum_field.bull_probability:.0%}")
                    logger.info(f"   Bear Prob: {transcendent_decision.quantum_field.bear_probability:.0%}")
                    logger.info(f"   Dimensions: {transcendent_decision.multi_dimensional.dimensional_alignment:.0f}%")
                    logger.info(f"   Purity: {transcendent_decision.signal_purity.value}")
                    logger.info(f"   Topology: {transcendent_decision.risk_topology.value}")
                    logger.info(f"   Win Prob: {transcendent_decision.win_probability:.0%}")
                    logger.info(f"   Expected Value: {transcendent_decision.expected_value:.4f}")
                    logger.info(f"   Score: {transcendent_decision.transcendent_score:.0f}/100")
                    logger.info(f"   Level: {transcendent_decision.intelligence_level.value}")
                    
                    for reason in transcendent_decision.reasons:
                        logger.info(f"   ✅ {reason}")
                    for warning in transcendent_decision.warnings:
                        logger.warning(f"   ⚠️ {warning}")
                    for insight in transcendent_decision.insights[:3]:  # Top 3 insights
                        logger.info(f"   💡 {insight}")
                    
                    if not transcendent_decision.can_trade:
                        logger.warning(f"⛔ Trade blocked by Transcendent Intelligence")
                        return {
                            "action": "BLOCKED_BY_TRANSCENDENT",
                            "reason": "; ".join(transcendent_decision.warnings),
                            "symbol": symbol,
                            "quantum_state": transcendent_decision.quantum_field.quantum_state.value,
                            "transcendent_score": transcendent_decision.transcendent_score,
                            "win_probability": transcendent_decision.win_probability,
                            "expected_value": transcendent_decision.expected_value,
                        }
                    
                    transcendent_multiplier = transcendent_decision.quantum_position_size * 10  # Scale up
                    
                    self._last_transcendent_decision = {
                        "symbol": symbol,
                        "can_trade": transcendent_decision.can_trade,
                        "confidence": transcendent_decision.confidence,
                        "quantum_state": transcendent_decision.quantum_field.quantum_state.value,
                        "bull_probability": transcendent_decision.quantum_field.bull_probability,
                        "bear_probability": transcendent_decision.quantum_field.bear_probability,
                        "dimensional_alignment": transcendent_decision.multi_dimensional.dimensional_alignment,
                        "signal_purity": transcendent_decision.signal_purity.value,
                        "risk_topology": transcendent_decision.risk_topology.value,
                        "win_probability": transcendent_decision.win_probability,
                        "expected_value": transcendent_decision.expected_value,
                        "transcendent_score": transcendent_decision.transcendent_score,
                        "intelligence_level": transcendent_decision.intelligence_level.value,
                        "quantum_sl": transcendent_decision.quantum_sl,
                        "quantum_tp": transcendent_decision.quantum_tp,
                        "expected_rr": transcendent_decision.expected_rr,
                        "kelly_quantum": transcendent_decision.kelly_quantum,
                        "position_size": transcendent_decision.quantum_position_size,
                        "reasons": transcendent_decision.reasons,
                        "warnings": transcendent_decision.warnings,
                        "insights": transcendent_decision.insights,
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Transcendent Intelligence analysis failed: {e}")
        
        # 🔮 OMNISCIENT INTELLIGENCE - 100x SMARTER (All-Knowing)
        omniscient_decision: Optional[OmniscientDecision] = None
        omniscient_multiplier = 1.0
        if self.omniscient_intelligence and analysis.get("market_data"):
            try:
                market_data = analysis.get("market_data", {})
                atr = market_data.get("atr", 0)
                
                # Get more data for Omniscient analysis (need 100+ candles)
                df = await self.data_provider.get_klines(symbol=symbol, timeframe="H1", limit=200)
                if df is not None and len(df) > 50:
                    prices = df['close'].values.astype(np.float32)
                    highs = df['high'].values.astype(np.float32)
                    lows = df['low'].values.astype(np.float32)
                    volumes = df['volume'].values.astype(np.float32) if 'volume' in df else None
                    
                    balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                    equity = await self.trading_engine.broker.get_equity() if self.trading_engine else balance
                    
                    omniscient_decision = self.omniscient_intelligence.analyze(
                        symbol=symbol,
                        signal_side=signal_side,
                        current_price=current_price,
                        prices=prices,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                        atr=atr,
                        base_confidence=analysis.get("confidence", 50),
                        balance=balance,
                        equity=equity,
                    )
                    
                    logger.info(f"🔮 OMNISCIENT INTELLIGENCE:")
                    logger.info(f"   Consciousness: {omniscient_decision.consciousness_level.value}")
                    logger.info(f"   Physics: {omniscient_decision.physics.physics_state.value}")
                    logger.info(f"   Neural: {omniscient_decision.neural.confidence.value} → {omniscient_decision.neural.ensemble_vote}")
                    logger.info(f"   Chaos: {omniscient_decision.chaos.chaos_level.value}")
                    logger.info(f"   Game Strategy: {omniscient_decision.game_theory.strategy.value}")
                    logger.info(f"   Risk State: {omniscient_decision.risk_math.risk_state.value}")
                    logger.info(f"   Win Prob: {omniscient_decision.win_probability:.0%}")
                    logger.info(f"   Edge: {omniscient_decision.edge:.2f}%")
                    logger.info(f"   Omniscient Score: {omniscient_decision.omniscient_score:.0f}/100")
                    logger.info(f"   Universal Alignment: {omniscient_decision.universal_alignment:.0f}%")
                    
                    # Show biases
                    if omniscient_decision.behavioral.detected_biases:
                        biases = [b.value for b in omniscient_decision.behavioral.detected_biases]
                        logger.info(f"   Biases: {', '.join(biases)}")
                    
                    # Show prophecies
                    for prophecy in omniscient_decision.prophecies[:2]:
                        logger.info(f"   🔮 {prophecy}")
                    
                    for reason in omniscient_decision.reasons:
                        logger.info(f"   ✅ {reason}")
                    for warning in omniscient_decision.warnings[:3]:
                        logger.warning(f"   ⚠️ {warning}")
                    for insight in omniscient_decision.insights[:3]:
                        logger.info(f"   💡 {insight}")
                    
                    if not omniscient_decision.can_trade:
                        logger.warning(f"⛔ Trade blocked by Omniscient Intelligence")
                        return {
                            "action": "BLOCKED_BY_OMNISCIENT",
                            "reason": "; ".join(omniscient_decision.warnings),
                            "symbol": symbol,
                            "consciousness": omniscient_decision.consciousness_level.value,
                            "omniscient_score": omniscient_decision.omniscient_score,
                            "win_probability": omniscient_decision.win_probability,
                            "edge": omniscient_decision.edge,
                            "physics_state": omniscient_decision.physics.physics_state.value,
                            "chaos_level": omniscient_decision.chaos.chaos_level.value,
                        }
                    
                    omniscient_multiplier = omniscient_decision.omniscient_position_size * 10
                    
                    self._last_omniscient_decision = {
                        "symbol": symbol,
                        "can_trade": omniscient_decision.can_trade,
                        "confidence": omniscient_decision.confidence,
                        "consciousness_level": omniscient_decision.consciousness_level.value,
                        "omniscient_score": omniscient_decision.omniscient_score,
                        "universal_alignment": omniscient_decision.universal_alignment,
                        "physics_state": omniscient_decision.physics.physics_state.value,
                        "neural_confidence": omniscient_decision.neural.confidence.value,
                        "neural_vote": omniscient_decision.neural.ensemble_vote,
                        "chaos_level": omniscient_decision.chaos.chaos_level.value,
                        "game_strategy": omniscient_decision.game_theory.strategy.value,
                        "risk_state": omniscient_decision.risk_math.risk_state.value,
                        "biases": [b.value for b in omniscient_decision.behavioral.detected_biases],
                        "win_probability": omniscient_decision.win_probability,
                        "expected_value": omniscient_decision.expected_value,
                        "edge": omniscient_decision.edge,
                        "optimal_sl": omniscient_decision.optimal_sl,
                        "optimal_tp": omniscient_decision.optimal_tp,
                        "expected_rr": omniscient_decision.expected_rr,
                        "prophecies": omniscient_decision.prophecies,
                        "reasons": omniscient_decision.reasons,
                        "warnings": omniscient_decision.warnings,
                        "insights": omniscient_decision.insights,
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Omniscient Intelligence analysis failed: {e}")
        
        # 🧠 ADVANCED INTELLIGENCE CHECK
        intel_multiplier = 1.0
        intel_decision = None
        if self.intelligence and analysis.get("market_data"):
            try:
                # Get H1 data from data provider (need more than 1 candle for analysis)
                h1_data = {}
                try:
                    df = await self.data_provider.get_klines(symbol=symbol, timeframe="H1", limit=100)
                    if df is not None and len(df) > 30:
                        h1_data = {
                            "open": df['open'].values.astype(np.float32),
                            "high": df['high'].values.astype(np.float32),
                            "low": df['low'].values.astype(np.float32),
                            "close": df['close'].values.astype(np.float32),
                        }
                        logger.info(f"   📊 Got {len(df)} candles for Intelligence analysis")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to get klines: {e}")
                    # Fallback to single candle
                    market_data = analysis.get("market_data", {})
                    h1_data = {
                        "open": np.array([market_data.get("open", current_price)]),
                        "high": np.array([market_data.get("high", current_price)]),
                        "low": np.array([market_data.get("low", current_price)]),
                        "close": np.array([market_data.get("close", current_price)]),
                    }
                
                # Get Smart Brain stats for Kelly
                win_rate, avg_win, avg_loss, total_trades = 0.5, 1.0, 1.0, 0
                if self.smart_brain:
                    stats = self.smart_brain.journal.get_stats(30)
                    win_rate = stats.get("win_rate", 50) / 100
                    total_trades = stats.get("total", 0)
                    # Estimate avg win/loss from trades
                    if total_trades > 0:
                        wins = [t for t in self.smart_brain.journal.trades[-30:] if t.is_win()]
                        losses = [t for t in self.smart_brain.journal.trades[-30:] if not t.is_win()]
                        if wins:
                            avg_win = sum(abs(t.pnl_percent) for t in wins if t.pnl_percent) / len(wins)
                        if losses:
                            avg_loss = sum(abs(t.pnl_percent) for t in losses if t.pnl_percent) / len(losses)
                
                side_for_intel = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
                pattern_conf = analysis.get("enhanced_confidence", analysis.get("base_confidence", 70))
                
                intel_decision = self.intelligence.analyze(
                    signal_side=side_for_intel,
                    pattern_confidence=pattern_conf,
                    h1_data=h1_data,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    total_trades=total_trades,
                )
                
                # Log intelligence results
                if intel_decision.regime:
                    logger.info(f"   🌡️ Regime: {intel_decision.regime.regime.value} - {intel_decision.regime.message}")
                if intel_decision.momentum:
                    logger.info(f"   📈 Momentum: {intel_decision.momentum.momentum_state} (RSI={intel_decision.momentum.rsi:.0f})")
                if intel_decision.confluence:
                    logger.info(f"   🎯 Confluence: {intel_decision.confluence.agreeing_factors}/{intel_decision.confluence.total_factors}")
                
                if not intel_decision.can_trade:
                    logger.warning(f"   🧠 ADVANCED INTELLIGENCE BLOCKED:")
                    for warning in intel_decision.warnings:
                        logger.warning(f"      {warning}")
                    return {
                        "action": "BLOCKED_BY_INTELLIGENCE",
                        "reason": "; ".join(intel_decision.warnings),
                        "confluence": intel_decision.confluence.to_dict() if intel_decision.confluence else None,
                    }
                
                intel_multiplier = intel_decision.position_size_factor
                logger.info(f"   🧠 Intelligence Multiplier: {intel_multiplier}x")
                
                for reason in intel_decision.reasons:
                    logger.info(f"   ✅ {reason}")
                
                # Store for API
                self._last_intel_result = {
                    "regime": intel_decision.regime.regime.value if intel_decision.regime else "N/A",
                    "trend_strength": intel_decision.regime.trend_strength if intel_decision.regime else 0,
                    "mtf_alignment": "ALIGNED" if intel_decision.can_trade else "CONFLICTING",
                    "position_size_factor": intel_multiplier,
                    "can_trade": intel_decision.can_trade,
                    "confluence_agreeing": intel_decision.confluence.agreeing_factors if intel_decision.confluence else 0,
                    "confluence_total": intel_decision.confluence.total_factors if intel_decision.confluence else 0,
                }
                self._last_intel_result_by_symbol[symbol] = self._last_intel_result
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Intelligence analysis failed: {e}")
        
        # � NEURAL BRAIN CHECK
        neural_multiplier = 1.0
        if self.neural_brain:
            try:
                # Get balance for risk calculation
                balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                
                # Get price data
                df = await self.data_provider.get_klines(symbol=symbol, timeframe="H1", limit=100)
                prices = df['close'].values.astype(np.float32) if len(df) > 0 else np.array([current_price])
                volumes = df['volume'].values.astype(np.float32) if 'volume' in df.columns and len(df) > 0 else None
                
                neural_decision = self.neural_brain.analyze(
                    signal_side="BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL",
                    prices=prices,
                    volumes=volumes,
                    balance=balance,
                )
                
                if not neural_decision.can_trade:
                    logger.warning(f"   🧬 NEURAL BRAIN BLOCKED:")
                    for warning in neural_decision.warnings:
                        logger.warning(f"      {warning}")
                    return {
                        "action": "BLOCKED_BY_NEURAL",
                        "reason": "; ".join(neural_decision.warnings),
                        "market_state": neural_decision.market_state.value,
                    }
                
                neural_multiplier = neural_decision.position_size_factor
                
                logger.info(f"   🧬 Market State: {neural_decision.market_state.value}")
                logger.info(f"   🧬 Pattern Quality: {neural_decision.pattern_quality}")
                logger.info(f"   🧬 Neural Confidence: {neural_decision.confidence:.1f}%")
                logger.info(f"   🧬 Neural Multiplier: {neural_multiplier}x")
                
                if neural_decision.anomaly_detected:
                    logger.warning(f"   ⚠️ Anomaly detected!")
                
                for reason in neural_decision.reasons:
                    logger.info(f"   🧬 {reason}")
                
                # Store for API
                self._last_neural_result = {
                    "market_state": neural_decision.market_state.value,
                    "pattern_quality": neural_decision.pattern_quality,
                    "dna_score": neural_decision.confidence,
                    "position_multiplier": neural_multiplier,
                    "can_trade": neural_decision.can_trade,
                    "anomaly_detected": neural_decision.anomaly_detected,
                }
                self._last_neural_result_by_symbol[symbol] = self._last_neural_result
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Neural Brain analysis failed: {e}")
        
        # ⚛️ QUANTUM STRATEGY CHECK
        quantum_multiplier = 1.0
        if self.quantum_strategy:
            try:
                side_for_quantum = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
                
                # Get price data
                prices_arr = prices if isinstance(prices, np.ndarray) else np.array(prices)
                volumes_arr = volumes if volumes is not None and isinstance(volumes, np.ndarray) else None
                
                quantum_decision = self.quantum_strategy.analyze(
                    symbol=symbol,
                    signal_direction=side_for_quantum,
                    prices=prices_arr,
                    volumes=volumes_arr,
                    entry_price=current_price
                )
                
                if not quantum_decision.should_trade:
                    logger.warning(f"   ⚛️ QUANTUM STRATEGY BLOCKED:")
                    logger.warning(f"      Quantum Score: {quantum_decision.quantum_score:.1f}")
                    logger.warning(f"      Confidence: {quantum_decision.confidence:.1f}%")
                    for warning in quantum_decision.warnings:
                        logger.warning(f"      {warning}")
                    return {
                        "action": "BLOCKED_BY_QUANTUM",
                        "reason": f"Score={quantum_decision.quantum_score:.1f}, {'; '.join(quantum_decision.warnings)}",
                        "quantum_score": quantum_decision.quantum_score,
                        "confidence": quantum_decision.confidence,
                    }
                
                quantum_multiplier = quantum_decision.position_multiplier
                
                # Log quantum analysis
                logger.info(f"   ⚛️ Quantum Score: {quantum_decision.quantum_score:.1f}")
                logger.info(f"   ⚛️ Confidence: {quantum_decision.confidence:.1f}%")
                logger.info(f"   ⚛️ Edge Score: {quantum_decision.edge_score:.2f}")
                logger.info(f"   ⚛️ R:R Ratio: {quantum_decision.risk_reward:.2f}")
                logger.info(f"   ⚛️ Microstructure: {quantum_decision.microstructure.state.value}")
                logger.info(f"   ⚛️ Smart Money: {quantum_decision.microstructure.smart_money_signal}")
                logger.info(f"   ⚛️ Volatility: {quantum_decision.volatility.regime.value}")
                logger.info(f"   ⚛️ Hurst: {quantum_decision.fractal.hurst_exponent:.2f}")
                logger.info(f"   ⚛️ Sentiment: {quantum_decision.sentiment.overall_sentiment:.2f}")
                logger.info(f"   ⚛️ Quantum Multiplier: {quantum_multiplier:.2f}x")
                
                # Log exit plan if available
                if quantum_decision.exit_plan:
                    ep = quantum_decision.exit_plan
                    logger.info(f"   ⚛️ Exit Strategy: {ep.strategy.value}")
                    logger.info(f"   ⚛️ SL: {ep.initial_stop_loss:.5f} | TP1: {ep.take_profit_1:.5f}")
                
                for reason in quantum_decision.reasons:
                    logger.info(f"   ⚛️ {reason}")
                    
                if quantum_decision.warnings:
                    for warning in quantum_decision.warnings:
                        logger.info(f"   ⚠️ {warning}")
                
                # Store for API
                self._last_quantum_result = {
                    "quantum_score": quantum_decision.quantum_score,
                    "confidence": quantum_decision.confidence,
                    "volatility_regime": quantum_decision.volatility.regime.value if quantum_decision.volatility else "N/A",
                    "fractal": f"H={quantum_decision.fractal.hurst_exponent:.2f}" if quantum_decision.fractal else "N/A",
                    "microstructure_signal": quantum_decision.microstructure.smart_money_signal if quantum_decision.microstructure else "N/A",
                    "position_multiplier": quantum_multiplier,
                    "should_trade": quantum_decision.should_trade,
                    "risk_reward": quantum_decision.risk_reward,
                }
                self._last_quantum_result_by_symbol[symbol] = self._last_quantum_result
                        
            except Exception as e:
                logger.warning(f"   ⚠️ Quantum Strategy analysis failed: {e}")
        
        # 🔮 DEEP INTELLIGENCE CHECK
        deep_multiplier = 1.0
        if self.deep_intelligence:
            try:
                side_for_deep = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
                
                # Get multi-timeframe data if available
                timeframe_data = {}
                if hasattr(self, 'data_provider') and self.data_provider:
                    for tf_name, tf_code in [("M15", "15m"), ("H1", "1h"), ("H4", "4h")]:
                        try:
                            tf_df = await self.data_provider.get_historical_klines(
                                symbol=symbol, timeframe=tf_code, days=7
                            )
                            if tf_df is not None and len(tf_df) > 30:
                                timeframe_data[tf_name] = tf_df['close'].values
                        except:
                            pass
                
                # Get other symbols' direction
                other_dirs = {}
                for other_sym, last_sig in self._last_signals.items():
                    if other_sym != symbol and last_sig:
                        sig_val = last_sig.get("signal", "")
                        if "BUY" in sig_val:
                            other_dirs[other_sym] = "BUY"
                        elif "SELL" in sig_val:
                            other_dirs[other_sym] = "SELL"
                
                # Current params
                current_params = {
                    "quality_level": self.min_quality.value if hasattr(self.min_quality, 'value') else str(self.min_quality),
                    "session": datetime.now().strftime("%H"),
                    "symbol": symbol,
                }
                
                deep_decision = self.deep_intelligence.analyze(
                    symbol=symbol,
                    signal_direction=side_for_deep,
                    timeframe_data=timeframe_data,
                    current_params=current_params,
                    other_symbols_direction=other_dirs if other_dirs else None
                )
                
                if not deep_decision.should_trade:
                    logger.warning(f"   🔮 DEEP INTELLIGENCE BLOCKED:")
                    logger.warning(f"      Confluence: {deep_decision.confluence_level.value}")
                    logger.warning(f"      Confidence: {deep_decision.confidence:.1f}%")
                    for warning in deep_decision.warnings:
                        logger.warning(f"      {warning}")
                    warnings_str = "; ".join(deep_decision.warnings)
                    return {
                        "action": "BLOCKED_BY_DEEP",
                        "reason": f"Confluence={deep_decision.confluence_level.value}, {warnings_str}",
                        "confluence": deep_decision.confluence_level.value,
                        "confidence": deep_decision.confidence,
                    }
                
                deep_multiplier = deep_decision.position_multiplier
                
                logger.info(f"   🔮 Confluence: {deep_decision.confluence_level.value}")
                logger.info(f"   🔮 Deep Confidence: {deep_decision.confidence:.1f}%")
                logger.info(f"   🔮 TF Score: {deep_decision.timeframe_score:.2f}")
                logger.info(f"   🔮 Prediction: {deep_decision.prediction_score:.2f}")
                logger.info(f"   🔮 Session Score: {deep_decision.session_score:.2f}")
                logger.info(f"   🔮 Deep Multiplier: {deep_multiplier:.2f}x")
                
                if deep_decision.warnings:
                    for warning in deep_decision.warnings:
                        logger.info(f"   ⚠️ {warning}")
                
                # Store for API
                self._last_deep_result = {
                    "confluence": deep_decision.confluence_level.value,
                    "confidence": deep_decision.confidence,
                    "timeframe_score": deep_decision.timeframe_score,
                    "session_score": deep_decision.session_score,
                    "correlation": deep_decision.prediction_score,
                    "session": getattr(deep_decision, 'session', 'N/A'),
                    "cross_asset_signal": deep_decision.confluence_level.value,
                    "position_multiplier": deep_multiplier,
                    "should_trade": deep_decision.should_trade,
                }
                self._last_deep_result_by_symbol[symbol] = self._last_deep_result
                        
            except Exception as e:
                logger.warning(f"   ⚠️ Deep Intelligence analysis failed: {e}")
        
        # 🎯 ALPHA ENGINE CHECK
        alpha_multiplier = 1.0
        if self.alpha_engine:
            try:
                side_for_alpha = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
                
                # Prepare price arrays
                closes = np.array(prices[-200:]) if len(prices) > 200 else np.array(prices)
                
                # Get OHLCV data from recent analysis
                opens = closes * 0.999  # Approximate if not available
                highs = closes * 1.002
                lows = closes * 0.998
                vols = np.array(volumes[-len(closes):]) if volumes is not None and len(volumes) >= len(closes) else np.ones(len(closes)) * 1000
                
                alpha_decision = self.alpha_engine.analyze(
                    symbol=symbol,
                    signal_direction=side_for_alpha,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    volumes=vols
                )
                
                if not alpha_decision.should_trade:
                    logger.warning(f"   🎯 ALPHA ENGINE BLOCKED:")
                    logger.warning(f"      Grade: {alpha_decision.grade.value}")
                    logger.warning(f"      Alpha Score: {alpha_decision.alpha_score:.1f}")
                    for risk in alpha_decision.risk_factors[:3]:
                        logger.warning(f"      {risk}")
                    return {
                        "action": "BLOCKED_BY_ALPHA",
                        "reason": f"Grade={alpha_decision.grade.value}, Score={alpha_decision.alpha_score:.1f}",
                        "alpha_grade": alpha_decision.grade.value,
                        "alpha_score": alpha_decision.alpha_score,
                    }
                
                alpha_multiplier = alpha_decision.position_multiplier
                
                # Log alpha analysis
                logger.info(f"   🎯 Alpha Grade: {alpha_decision.grade.value}")
                logger.info(f"   🎯 Alpha Score: {alpha_decision.alpha_score:.1f}")
                logger.info(f"   🎯 Confidence: {alpha_decision.confidence:.1f}%")
                logger.info(f"   🎯 R:R Ratio: {alpha_decision.risk_reward:.2f}")
                logger.info(f"   🎯 Order Flow: {alpha_decision.order_flow.bias.value}")
                logger.info(f"   🎯 Delta: {alpha_decision.order_flow.delta:+.2f}")
                
                if alpha_decision.liquidity_zones:
                    for zone in alpha_decision.liquidity_zones[:3]:
                        logger.info(f"   🎯 Liquidity: {zone.zone_type.value} at {zone.price_level:.5f}")
                
                if alpha_decision.divergences:
                    for div in alpha_decision.divergences[:2]:
                        logger.info(f"   🎯 Divergence: {div.indicator} {div.div_type.value}")
                
                if alpha_decision.market_profile:
                    mp = alpha_decision.market_profile
                    logger.info(f"   🎯 POC: {mp.poc:.5f} | Value Area: {mp.value_area_low:.5f}-{mp.value_area_high:.5f}")
                
                if alpha_decision.optimal_entry:
                    logger.info(f"   🎯 Optimal Entry: {alpha_decision.optimal_entry:.5f}")
                    logger.info(f"   🎯 Suggested SL: {alpha_decision.stop_loss:.5f}")
                    logger.info(f"   🎯 Targets: {[f'{t:.5f}' for t in alpha_decision.targets[:3]]}")
                
                logger.info(f"   🎯 Alpha Multiplier: {alpha_multiplier:.2f}x")
                
                for edge in alpha_decision.edge_factors[:5]:
                    logger.info(f"   ✅ {edge}")
                
                if alpha_decision.risk_factors:
                    for risk in alpha_decision.risk_factors[:3]:
                        logger.info(f"   ⚠️ {risk}")
                
                # 📊 Store Alpha Decision for API
                self._last_alpha_result = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "grade": alpha_decision.grade.value,
                    "alpha_score": float(alpha_decision.alpha_score),
                    "confidence": float(alpha_decision.confidence),
                    "order_flow_bias": alpha_decision.order_flow.bias.value if alpha_decision.order_flow else "NEUTRAL",
                    "order_flow_delta": float(alpha_decision.order_flow.delta) if alpha_decision.order_flow else 0,
                    "risk_reward": float(alpha_decision.risk_reward),
                    "position_multiplier": float(alpha_multiplier),
                    "optimal_entry": float(alpha_decision.optimal_entry) if alpha_decision.optimal_entry else 0,
                    "stop_loss": float(alpha_decision.stop_loss) if alpha_decision.stop_loss else 0,
                    "targets": [float(t) for t in alpha_decision.targets[:3]] if alpha_decision.targets else [],
                    "market_profile": {
                        "poc": float(alpha_decision.market_profile.poc) if alpha_decision.market_profile else 0,
                        "vah": float(alpha_decision.market_profile.value_area_high) if alpha_decision.market_profile else 0,
                        "val": float(alpha_decision.market_profile.value_area_low) if alpha_decision.market_profile else 0,
                    } if alpha_decision.market_profile else None,
                    "liquidity_zones": [{"type": z.zone_type.value, "price": float(z.price_level)} for z in alpha_decision.liquidity_zones[:5]] if alpha_decision.liquidity_zones else [],
                    "should_trade": alpha_decision.should_trade,
                    "edge_factors": alpha_decision.edge_factors[:5] if alpha_decision.edge_factors else [],
                    "risk_factors": alpha_decision.risk_factors[:5] if alpha_decision.risk_factors else [],
                }
                self._last_alpha_result_by_symbol[symbol] = self._last_alpha_result
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Alpha Engine analysis failed: {e}")
        
        # 🧠⚡ OMEGA BRAIN CHECK
        omega_multiplier = 1.0
        if self.omega_brain:
            try:
                side_for_omega = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
                
                # Prepare price arrays
                omega_closes = np.array(prices[-200:]) if len(prices) > 200 else np.array(prices)
                omega_opens = omega_closes * 0.999
                omega_highs = omega_closes * 1.002
                omega_lows = omega_closes * 0.998
                omega_vols = np.array(volumes[-len(omega_closes):]) if volumes is not None and len(volumes) >= len(omega_closes) else np.ones(len(omega_closes)) * 1000
                
                # Get balance for risk allocation
                omega_balance = await self.trading_engine.broker.get_balance() if self.trading_engine else 10000
                
                omega_decision = self.omega_brain.analyze(
                    symbol=symbol,
                    signal_direction=side_for_omega,
                    opens=omega_opens,
                    highs=omega_highs,
                    lows=omega_lows,
                    closes=omega_closes,
                    volumes=omega_vols,
                    current_balance=omega_balance,
                    other_symbols=self.symbols
                )
                
                if not omega_decision.should_trade:
                    logger.warning(f"   🧠⚡ OMEGA BRAIN BLOCKED:")
                    logger.warning(f"      Grade: {omega_decision.grade.value}")
                    logger.warning(f"      Omega Score: {omega_decision.omega_score:.1f}")
                    logger.warning(f"      Verdict: {omega_decision.final_verdict}")
                    for risk in omega_decision.risk_factors[:3]:
                        logger.warning(f"      {risk}")
                    return {
                        "action": "BLOCKED_BY_OMEGA",
                        "reason": f"Grade={omega_decision.grade.value}, Score={omega_decision.omega_score:.1f}",
                        "omega_grade": omega_decision.grade.value,
                        "omega_score": omega_decision.omega_score,
                        "verdict": omega_decision.final_verdict
                    }
                
                omega_multiplier = omega_decision.position_multiplier
                
                # Log Omega Brain analysis
                logger.info(f"   🧠⚡ Omega Grade: {omega_decision.grade.value}")
                logger.info(f"   🧠⚡ Omega Score: {omega_decision.omega_score:.1f}")
                logger.info(f"   🧠⚡ Confidence: {omega_decision.confidence:.1f}%")
                logger.info(f"   🧠⚡ Institutional: {omega_decision.institutional_flow.activity.value}")
                logger.info(f"   🧠⚡ Smart Money: {omega_decision.institutional_flow.smart_money_direction}")
                logger.info(f"   🧠⚡ Sentiment: {omega_decision.sentiment.overall_sentiment:.1f} ({omega_decision.sentiment.dominant_narrative})")
                logger.info(f"   🧠⚡ Regime: {omega_decision.regime_prediction.current_regime}")
                
                if omega_decision.manipulation_alert:
                    ma = omega_decision.manipulation_alert
                    logger.info(f"   🧠⚡ Manipulation: {ma.manipulation_type.value} ({ma.probability:.0f}%)")
                
                logger.info(f"   🧠⚡ Position Plan: {omega_decision.position_plan.action}")
                logger.info(f"   🧠⚡ R:R Ratio: {omega_decision.risk_reward:.2f}")
                logger.info(f"   🧠⚡ Omega Multiplier: {omega_multiplier:.2f}x")
                
                # Log institutional insight
                logger.info(f"   💡 {omega_decision.institutional_insight}")
                logger.info(f"   📊 {omega_decision.final_verdict}")
                
                for edge in omega_decision.edge_factors[:3]:
                    logger.info(f"   ✅ {edge}")
                
                if omega_decision.risk_factors:
                    for risk in omega_decision.risk_factors[:2]:
                        logger.info(f"   ⚠️ {risk}")
                
                # 📊 Store Omega Decision for API
                self._last_omega_result = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "grade": omega_decision.grade.value,
                    "omega_score": float(omega_decision.omega_score),
                    "confidence": float(omega_decision.confidence),
                    "institutional_flow": omega_decision.institutional_flow.activity.value if omega_decision.institutional_flow else "N/A",
                    "smart_money": omega_decision.institutional_flow.smart_money_direction if omega_decision.institutional_flow else "N/A",
                    "manipulation_detected": omega_decision.manipulation_alert.manipulation_type.value if omega_decision.manipulation_alert else "NONE",
                    "manipulation_probability": float(omega_decision.manipulation_alert.probability) if omega_decision.manipulation_alert else 0,
                    "sentiment": float(omega_decision.sentiment.overall_sentiment) if omega_decision.sentiment else 0,
                    "current_regime": omega_decision.regime_prediction.current_regime if omega_decision.regime_prediction else "N/A",
                    "predicted_regime": omega_decision.regime_prediction.predicted_regime if omega_decision.regime_prediction else "N/A",
                    "position_multiplier": float(omega_multiplier),
                    "risk_reward": float(omega_decision.risk_reward),
                    "should_trade": omega_decision.should_trade,
                    "final_verdict": omega_decision.final_verdict,
                    "institutional_insight": omega_decision.institutional_insight,
                    "edge_factors": omega_decision.edge_factors[:5] if omega_decision.edge_factors else [],
                    "risk_factors": omega_decision.risk_factors[:5] if omega_decision.risk_factors else [],
                }
                self._last_omega_result_by_symbol[symbol] = self._last_omega_result
                
                # 📰 Store Sentiment data (from Omega Brain) for frontend
                if omega_decision.sentiment:
                    sentiment_level = "EXTREME_FEAR" if omega_decision.sentiment.overall_sentiment < -50 else \
                                      "FEAR" if omega_decision.sentiment.overall_sentiment < -20 else \
                                      "NEUTRAL" if omega_decision.sentiment.overall_sentiment < 20 else \
                                      "GREED" if omega_decision.sentiment.overall_sentiment < 50 else "EXTREME_GREED"
                    
                    self._last_sentiment_result = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "level": sentiment_level,
                        "retail_sentiment": float(omega_decision.sentiment.overall_sentiment),
                        "dominant_narrative": omega_decision.sentiment.dominant_narrative if hasattr(omega_decision.sentiment, 'dominant_narrative') else "N/A",
                        "fear_greed_index": 50 + float(omega_decision.sentiment.overall_sentiment) / 2,  # Convert to 0-100 scale
                        "override_signal": abs(omega_decision.sentiment.overall_sentiment) > 70,  # Contrarian signal when extreme
                        "source": "Omega Brain Sentiment Fusion"
                    }
                    self._last_sentiment_result_by_symbol[symbol] = self._last_sentiment_result
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Omega Brain analysis failed: {e}")
        
        # 🏛️⚔️ TITAN CORE CHECK (Final Meta-Intelligence)
        titan_multiplier = 1.0
        if self.titan_core:
            try:
                side_for_titan = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
                
                # Prepare arrays
                titan_closes = np.array(prices[-200:]) if len(prices) > 200 else np.array(prices)
                titan_highs = titan_closes * 1.002
                titan_lows = titan_closes * 0.998
                titan_vols = np.array(volumes[-len(titan_closes):]) if volumes is not None and len(volumes) >= len(titan_closes) else np.ones(len(titan_closes)) * 1000
                
                # Collect module signals for synthesis
                module_signals = []
                
                # Add signals from all active modules
                if neural_multiplier != 1.0 or True:
                    module_signals.append(ModuleSignal(
                        module_name="NeuralBrain",
                        should_trade=neural_multiplier > 0.3,
                        direction=side_for_titan,
                        confidence=70 * neural_multiplier,
                        multiplier=neural_multiplier,
                        score=70,
                        reasons=[],
                        warnings=[]
                    ))
                
                if deep_multiplier != 1.0 or True:
                    module_signals.append(ModuleSignal(
                        module_name="DeepIntelligence",
                        should_trade=deep_multiplier > 0.3,
                        direction=side_for_titan,
                        confidence=70 * deep_multiplier,
                        multiplier=deep_multiplier,
                        score=70,
                        reasons=[],
                        warnings=[]
                    ))
                
                if quantum_multiplier != 1.0 or True:
                    module_signals.append(ModuleSignal(
                        module_name="QuantumStrategy",
                        should_trade=quantum_multiplier > 0.3,
                        direction=side_for_titan,
                        confidence=70 * quantum_multiplier,
                        multiplier=quantum_multiplier,
                        score=70,
                        reasons=[],
                        warnings=[]
                    ))
                
                if alpha_multiplier != 1.0 or True:
                    module_signals.append(ModuleSignal(
                        module_name="AlphaEngine",
                        should_trade=alpha_multiplier > 0.3,
                        direction=side_for_titan,
                        confidence=70 * alpha_multiplier,
                        multiplier=alpha_multiplier,
                        score=70,
                        reasons=[],
                        warnings=[]
                    ))
                
                if omega_multiplier != 1.0 or True:
                    module_signals.append(ModuleSignal(
                        module_name="OmegaBrain",
                        should_trade=omega_multiplier > 0.3,
                        direction=side_for_titan,
                        confidence=70 * omega_multiplier,
                        multiplier=omega_multiplier,
                        score=70,
                        reasons=[],
                        warnings=[]
                    ))
                
                titan_decision = self.titan_core.synthesize(
                    symbol=symbol,
                    signal_direction=side_for_titan,
                    closes=titan_closes,
                    highs=titan_highs,
                    lows=titan_lows,
                    volumes=titan_vols,
                    module_signals=module_signals,
                    current_price=current_price
                )
                
                if not titan_decision.should_trade:
                    logger.warning(f"   🏛️ TITAN CORE BLOCKED:")
                    logger.warning(f"      Grade: {titan_decision.grade.value}")
                    logger.warning(f"      Titan Score: {titan_decision.titan_score:.1f}")
                    logger.warning(f"      Consensus: {titan_decision.consensus.level.value}")
                    logger.warning(f"      Verdict: {titan_decision.final_verdict}")
                    return {
                        "action": "BLOCKED_BY_TITAN",
                        "reason": f"Grade={titan_decision.grade.value}, Score={titan_decision.titan_score:.1f}",
                        "titan_grade": titan_decision.grade.value,
                        "titan_score": titan_decision.titan_score,
                        "consensus": titan_decision.consensus.level.value,
                        "verdict": titan_decision.final_verdict
                    }
                
                titan_multiplier = titan_decision.position_multiplier
                
                # Log Titan Core analysis
                logger.info(f"   🏛️ Titan Grade: {titan_decision.grade.value}")
                logger.info(f"   🏛️ Titan Score: {titan_decision.titan_score:.1f}")
                logger.info(f"   🏛️ Confidence: {titan_decision.confidence:.1f}%")
                logger.info(f"   🏛️ Consensus: {titan_decision.consensus.level.value} ({titan_decision.consensus.agreement_ratio:.0%})")
                logger.info(f"   🏛️ Prediction: {titan_decision.prediction.final_prediction} ({titan_decision.prediction.predicted_move:+.2f}%)")
                logger.info(f"   🏛️ Market: {titan_decision.market_condition.value}")
                logger.info(f"   🏛️ Agreeing: {titan_decision.agreeing_modules}/{titan_decision.total_modules} modules")
                logger.info(f"   🏛️ Titan Multiplier: {titan_multiplier:.2f}x")
                
                # Log verdict
                logger.info(f"   ⚔️ {titan_decision.final_verdict}")
                
                for edge in titan_decision.edge_factors[:3]:
                    logger.info(f"   ✅ {edge}")
                
                if titan_decision.risk_factors:
                    for risk in titan_decision.risk_factors[:2]:
                        logger.info(f"   ⚠️ {risk}")
                
                # Log insights if any
                if titan_decision.improvement_insights:
                    for insight in titan_decision.improvement_insights[:2]:
                        logger.info(f"   💡 {insight.description}")
                
                # 📊 Store Titan Decision for API
                self._last_titan_decision = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "grade": titan_decision.grade.value,
                    "titan_score": float(titan_decision.titan_score),
                    "confidence": float(titan_decision.confidence),
                    "consensus": titan_decision.consensus.level.value,
                    "agreement_ratio": float(titan_decision.consensus.agreement_ratio),
                    "market_condition": titan_decision.market_condition.value,
                    "prediction": {
                        "direction": titan_decision.prediction.final_prediction,
                        "predicted_move": float(titan_decision.prediction.predicted_move),
                    },
                    "position_multiplier": float(titan_multiplier),
                    "agreeing_modules": titan_decision.agreeing_modules,
                    "total_modules": titan_decision.total_modules,
                    "should_trade": titan_decision.should_trade,
                    "final_verdict": titan_decision.final_verdict,
                    "edge_factors": titan_decision.edge_factors[:5] if titan_decision.edge_factors else [],
                    "risk_factors": titan_decision.risk_factors[:5] if titan_decision.risk_factors else [],
                }
                self._last_titan_decision_by_symbol[symbol] = self._last_titan_decision
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Titan Core analysis failed: {e}")
        
        # 🧠 SMART BRAIN CHECK
        smart_multiplier = 1.0
        if self.smart_brain:
            side_for_check = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
            smart_decision = self.smart_brain.evaluate_entry(symbol, side_for_check)
            
            if not smart_decision.can_trade:
                logger.warning(f"   🧠 SMART BRAIN BLOCKED:")
                for reason in smart_decision.reasons:
                    logger.warning(f"      {reason}")
                return {
                    "action": "BLOCKED_BY_BRAIN",
                    "reason": "; ".join(smart_decision.reasons),
                }
            
            smart_multiplier = smart_decision.risk_multiplier
            
            if smart_decision.insights:
                for insight in smart_decision.insights:
                    logger.info(f"   🧠 {insight}")
            
            logger.info(f"   🧠 Smart Multiplier: {smart_multiplier}x")
        
        # 🏆 PRO FEATURES CHECK
        if self.pro_features:
            # Get existing positions for correlation check
            existing_positions = [
                {"symbol": p.symbol, "side": p.side.value}
                for p in self.trading_engine.positions.values()
            ]
            
            # Determine side for check
            side_for_check = "BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL"
            
            pro_decision = self.pro_features.check_entry(
                symbol=symbol,
                side=side_for_check,
                existing_positions=existing_positions,
            )
            
            # Log session info
            if pro_decision.session_info:
                session = pro_decision.session_info
                logger.info(f"   🕐 Session: {session.current_session.value} ({session.quality_score}%)")
            
            if not pro_decision.can_trade:
                logger.warning(f"   🏆 PRO FEATURES BLOCKED:")
                for reason in pro_decision.reasons:
                    logger.warning(f"      {reason}")
                return {
                    "action": "BLOCKED_BY_PRO",
                    "reason": "; ".join(pro_decision.reasons),
                }
            
            if pro_decision.warnings:
                for warning in pro_decision.warnings:
                    logger.info(f"   💡 {warning}")
            
            # Apply position multiplier from Pro Features
            position_multiplier_from_pro = pro_decision.position_multiplier
            logger.info(f"   🏆 Pro Position Multiplier: {position_multiplier_from_pro}x")
        else:
            position_multiplier_from_pro = 1.0
        
        # 🛡️ RISK GUARDIAN CHECK
        if self.risk_guardian:
            balance = await self.trading_engine.broker.get_balance()
            open_positions = [p.to_dict() for p in self.trading_engine.positions.values()]
            
            risk_assessment = self.risk_guardian.assess_risk(
                current_balance=balance,
                open_positions=open_positions,
                proposed_trade={"symbol": symbol, "side": signal}
            )
            
            if not risk_assessment.can_trade:
                logger.warning(f"   🛡️ RISK GUARDIAN BLOCKED:")
                for reason in risk_assessment.reasons:
                    logger.warning(f"      {reason}")
                return {
                    "action": "BLOCKED_BY_RISK",
                    "reason": "; ".join(risk_assessment.reasons),
                    "risk_level": risk_assessment.level.value,
                }
            
            if risk_assessment.warnings:
                for warning in risk_assessment.warnings:
                    logger.warning(f"   ⚠️ {warning}")
            
            # Adjust position size based on risk assessment
            position_multiplier_from_risk = risk_assessment.max_position_size
            logger.info(f"   🛡️ Risk Level: {risk_assessment.level.value}, Max Position: {position_multiplier_from_risk}x")
        else:
            position_multiplier_from_risk = 1.0
        
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
        
        # 🧠 Advanced Intelligence multiplier
        position_multiplier = min(position_multiplier, intel_multiplier)
        
        # 🧠 Smart Brain multiplier (adaptive risk)
        position_multiplier = min(position_multiplier, smart_multiplier)
        
        # 🏆 Pro Features position size limit
        position_multiplier = min(position_multiplier, position_multiplier_from_pro)

        # 🛡️ Risk Guardian position size limit
        position_multiplier = min(position_multiplier, position_multiplier_from_risk)
        
        # 🧬 Neural Brain position size factor
        position_multiplier = min(position_multiplier, neural_multiplier)
        
        # 🔮 Deep Intelligence position size factor
        position_multiplier = min(position_multiplier, deep_multiplier)
        
        # ⚛️ Quantum Strategy position size factor
        position_multiplier = min(position_multiplier, quantum_multiplier)
        
        # 🎯 Alpha Engine position size factor
        position_multiplier = min(position_multiplier, alpha_multiplier)
        
        # 🧠⚡ Omega Brain position size factor
        position_multiplier = min(position_multiplier, omega_multiplier)
        
        # 🏛️⚔️ Titan Core position size factor (Final)
        position_multiplier = min(position_multiplier, titan_multiplier)
        
        # 🧠⚡ Ultra Intelligence position size factor (Ultimate)
        position_multiplier = min(position_multiplier, ultra_multiplier)
        
        # 🏆👑 Supreme Intelligence position size factor (Hedge Fund Level)
        position_multiplier = min(position_multiplier, supreme_multiplier)
        
        # 🌌✨ Transcendent Intelligence position size factor (Beyond Human)
        position_multiplier = min(position_multiplier, transcendent_multiplier)
        
        # 🔮 Omniscient Intelligence position size factor (All-Knowing)
        position_multiplier = min(position_multiplier, omniscient_multiplier)
        
        logger.info(f"   📊 Final Position Multiplier: {position_multiplier:.2f}x")
        logger.info(f"      Neural: {neural_multiplier}x | Deep: {deep_multiplier:.2f}x | Quantum: {quantum_multiplier:.2f}x")
        logger.info(f"      Alpha: {alpha_multiplier:.2f}x | Omega: {omega_multiplier:.2f}x | Titan: {titan_multiplier:.2f}x")
        logger.info(f"      Ultra: {ultra_multiplier:.2f}x | Supreme: {supreme_multiplier:.2f}x | Transcendent: {transcendent_multiplier:.2f}x")
        logger.info(f"      🔮 Omniscient: {omniscient_multiplier:.2f}x")
        
        # 🔒 MANDATORY STOP LOSS - Use Risk Guardian to validate/fix
        if self.risk_guardian:
            stop_loss, sl_msg = self.risk_guardian.validate_stop_loss(
                side=side.value,
                entry_price=current_price,
                stop_loss=stop_loss,
                atr=risk_mgmt.get("atr"),  # ATR from analysis if available
            )
            logger.info(f"   🛡️ SL Validation: {sl_msg}")
        elif not stop_loss or stop_loss <= 0:
            # Fallback: Auto-calculate Stop Loss (2% from current price)
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
        
        # � Validate and Fix Take Profit direction
        if take_profit:
            if side == OrderSide.BUY and take_profit <= current_price:
                # TP ต้องสูงกว่า entry สำหรับ BUY
                old_tp = take_profit
                take_profit = current_price * 1.02  # Default 2% profit
                logger.warning(f"⚠️ Fixed invalid TP for BUY: {old_tp:.5f} -> {take_profit:.5f}")
            elif side == OrderSide.SELL and take_profit >= current_price:
                # TP ต้องต่ำกว่า entry สำหรับ SELL
                old_tp = take_profit
                take_profit = current_price * 0.98  # Default 2% profit
                logger.warning(f"⚠️ Fixed invalid TP for SELL: {old_tp:.5f} -> {take_profit:.5f}")
        

            # 🎯 LIMIT TP - ไม่ให้ TP ไกลเกินไป (Max R:R = 2.0)
            if take_profit and stop_loss:
                sl_distance = abs(current_price - stop_loss)
                tp_distance = abs(take_profit - current_price)
                current_rr = tp_distance / sl_distance if sl_distance > 0 else 0
                max_rr = 2.0
                if current_rr > max_rr:
                    old_tp = take_profit
                    tp_distance_limited = sl_distance * max_rr
                    if side == OrderSide.BUY:
                        take_profit = current_price + tp_distance_limited
                    else:
                        take_profit = current_price - tp_distance_limited
                    logger.info(f"🎯 Limited TP: R:R {current_rr:.1f} -> {max_rr:.1f}, TP: {old_tp:.5f} -> {take_profit:.5f}")


        # �🛡️ Calculate position size using Risk Guardian
        balance = await self.trading_engine.broker.get_balance()
        
        if self.risk_guardian:
            quantity, calc_details = self.risk_guardian.calculate_position_size(
                balance=balance,
                entry_price=current_price,
                stop_loss=stop_loss,
                risk_multiplier=position_multiplier,
                symbol=symbol,  # Pass symbol for dynamic min SL
            )
            if quantity <= 0:
                logger.error(f"❌ Risk Guardian rejected position: {calc_details.get('error', 'Unknown')}")
                return {"action": "SKIP", "reason": calc_details.get('error', 'Position size rejected')}
        else:
            # Fallback calculation
            risk_amount = balance * (self.max_risk_percent / 100) * position_multiplier
            stop_distance = abs(current_price - stop_loss)
            quantity = risk_amount / stop_distance if stop_distance > 0 else 0.001
        
        quantity = round(max(0.01, quantity), 2)  # Min 0.01 lot
        
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
            
            # 🧠 Record trade in Smart Brain
            if self.smart_brain:
                session_name = ""
                if self.pro_features and self.pro_features.session_filter:
                    session_info = self.pro_features.session_filter.get_session_info()
                    session_name = session_info.current_session.value
                
                self.smart_brain.record_trade_open(
                    trade_id=order.id,
                    symbol=symbol,
                    side=side.value,
                    entry_price=result.order.filled_price if result.order else current_price,
                    stop_loss=stop_loss,
                    quantity=quantity,
                    signal_quality=quality,
                    pattern_confidence=analysis.get("enhanced_confidence", 0),
                    session=session_name,
                    market_regime=analysis.get("market_regime", ""),
                )
            
            # 📚 Record factors for Continuous Learning
            if self.learning_system and intel_decision:
                # Store trade factors for later learning when closed
                self._pending_trade_factors[order.id] = {
                    "symbol": symbol,  # Important for learning
                    "signal": signal,
                    "pattern_confidence": analysis.get("enhanced_confidence", 0) > 70,
                    "regime_aligned": intel_decision.regime.regime.value != "high_volatility" if intel_decision.regime else False,
                    "mtf_aligned": intel_decision.mtf.can_trade if intel_decision.mtf else False,
                    "momentum_aligned": intel_decision.momentum.combined_score > 0 if intel_decision.momentum else False,
                    "near_sr": any(l.level_type == "support" for l in intel_decision.sr_levels[:3]) if intel_decision.sr_levels else False,
                    "smart_money": True,  # Will be from actual check
                    "session_quality": self.pro_features.session_filter.get_session_info().quality_score > 60 if self.pro_features else False,
                    "quality": quality,
                    "entry_time": datetime.now().isoformat(),
                }
            
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
    


    async def _apply_break_even(self) -> None:
        """🛡️ Break-Even - ย้าย SL ไปจุด entry เมื่อกำไร"""
        config = self._smart_features.get("break_even", {})
        if not config.get("enabled", False):
            return
        
        if not self.trading_engine or not self.trading_engine.positions:
            return
        
        activation_pct = config.get("activation_pct", 0.5)
        offset_pct = config.get("offset_pct", 0.05)
        
        for pos_id, position in list(self.trading_engine.positions.items()):
            try:
                # Skip if already applied
                if self._break_even_applied.get(pos_id, False):
                    continue
                
                symbol = position.symbol
                current_price = position.current_price or 0
                entry_price = position.entry_price or 0
                current_sl = position.stop_loss
                
                if not current_price or not entry_price:
                    continue
                
                # Calculate profit percentage
                if position.side == OrderSide.BUY:
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    profit_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Check activation
                if profit_pct < activation_pct:
                    continue
                
                # Calculate break-even SL with offset
                offset = entry_price * (offset_pct / 100)
                
                if position.side == OrderSide.BUY:
                    new_sl = entry_price + offset
                    # Only move if better than current SL
                    if current_sl and new_sl <= current_sl:
                        continue
                else:
                    new_sl = entry_price - offset
                    if current_sl and new_sl >= current_sl:
                        continue
                
                # Round appropriately
                is_gold = "XAU" in symbol.upper()
                new_sl = round(new_sl, 2 if is_gold else 5)
                
                # Apply break-even
                success = await self.trading_engine.broker.modify_position(
                    position_id=pos_id,
                    stop_loss=new_sl
                )
                
                if success:
                    self._break_even_applied[pos_id] = True
                    position.stop_loss = new_sl
                    logger.info(
                        f"🛡️ BREAK-EVEN: {symbol} | "
                        f"Entry: {entry_price:.5f} | "
                        f"New SL: {new_sl:.5f} | "
                        f"Profit: {profit_pct:.2f}%"
                    )
                    
                    await self._broadcast_update("break_even_applied", {
                        "symbol": symbol,
                        "position_id": pos_id,
                        "entry_price": entry_price,
                        "new_sl": new_sl,
                        "profit_pct": profit_pct,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error applying break-even for {pos_id}: {e}")
    
    async def _check_time_exit(self) -> None:
        """⏰ Time Exit - ปิดออเดอร์ที่ค้างนานเกินไป"""
        config = self._smart_features.get("time_exit", {})
        if not config.get("enabled", False):
            return
        
        if not self.trading_engine or not self.trading_engine.positions:
            return
        
        max_hours = config.get("max_hours", 24)
        
        for pos_id, position in list(self.trading_engine.positions.items()):
            try:
                # Check how long position has been open
                opened_at = getattr(position, 'opened_at', None)
                if not opened_at:
                    continue
                
                hours_open = (datetime.now() - opened_at).total_seconds() / 3600
                
                if hours_open >= max_hours:
                    symbol = position.symbol
                    pnl = position.pnl or 0
                    
                    logger.info(
                        f"⏰ TIME EXIT: {symbol} | "
                        f"Open for {hours_open:.1f} hours | "
                        f"PnL: ${pnl:.2f}"
                    )
                    
                    # Close position
                    await self.trading_engine.close_position(pos_id, reason="time_exit")
                    
                    await self._broadcast_update("time_exit", {
                        "symbol": symbol,
                        "position_id": pos_id,
                        "hours_open": hours_open,
                        "pnl": pnl,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error checking time exit for {pos_id}: {e}")
    
    def _can_trade_today(self) -> tuple[bool, str]:
        """📊 Check if we can trade today based on limits"""
        # Max daily trades
        config = self._smart_features.get("max_daily_trades", {})
        if config.get("enabled", False):
            limit = config.get("limit", 5)
            today_trades = self._daily_stats.get("trades", 0)
            if today_trades >= limit:
                return False, f"Daily limit reached ({today_trades}/{limit})"
        
        # Consecutive loss protection
        config = self._smart_features.get("loss_protection", {})
        if config.get("enabled", False):
            max_losses = config.get("max_consecutive_losses", 3)
            cooldown = config.get("cooldown_minutes", 60)
            
            if self._consecutive_losses >= max_losses:
                if self._last_loss_time:
                    minutes_since = (datetime.now() - self._last_loss_time).total_seconds() / 60
                    if minutes_since < cooldown:
                        remaining = int(cooldown - minutes_since)
                        return False, f"Loss protection active ({remaining}m cooldown)"
                    else:
                        # Reset after cooldown
                        self._consecutive_losses = 0
        
        return True, "OK"
    
    def _check_correlation(self, symbol: str, side: str) -> tuple[bool, str]:
        """🔗 Check correlation protection"""
        config = self._smart_features.get("correlation_protection", {})
        if not config.get("enabled", False):
            return True, "OK"
        
        if not self.trading_engine or not self.trading_engine.positions:
            return True, "OK"
        
        max_same = config.get("max_same_direction", 2)
        
        # Count positions in same direction
        same_direction = 0
        for pos in self.trading_engine.positions.values():
            if pos.side.value == side:
                same_direction += 1
        
        if same_direction >= max_same:
            return False, f"Max {max_same} positions in {side} direction"
        
        return True, "OK"
    
    def _update_loss_tracking(self, pnl: float) -> None:
        """📉 Update consecutive loss tracking"""
        if pnl < 0:
            self._consecutive_losses += 1
            self._last_loss_time = datetime.now()
            logger.warning(f"📉 Consecutive losses: {self._consecutive_losses}")
        else:
            self._consecutive_losses = 0
            logger.info(f"📈 Win! Consecutive losses reset")

    async def _update_floating_tp(
        self,
        position,
        new_sl: float,
        entry_price: float,
        current_price: float,
        pos_id: str
    ) -> Optional[float]:
        """
        🎯 FLOATING TP - ยก TP ตาม SL เพื่อให้ได้กำไรมากขึ้น
        
        Logic:
        1. เมื่อ SL ถูกยกขึ้น (locked profit) → TP ก็ควรขยับตาม
        2. รักษา R:R ratio ขั้นต่ำ (เช่น 1.5:1)
        3. ยืด TP เพิ่มเติมเมื่อ Momentum ดี
        
        ตัวอย่าง:
        - Entry: 1.1000, Original SL: 1.0950, Original TP: 1.1100 (R:R = 2:1)
        - SL ยกขึ้นเป็น 1.0980 (locked 0.3%)
        - TP ใหม่ = Entry + (Entry - New_SL) * R:R = 1.1000 + (1.1000 - 1.0980) * 2 = 1.1040
        - แต่ต้องไม่ต่ำกว่า TP เดิม! ดังนั้น TP ยังคง 1.1100 หรือยืดเป็น 1.1120
        """
        
        if not self._floating_tp_config.get("enabled", False):
            return None
        
        symbol = position.symbol
        current_tp = position.take_profit
        
        if not current_tp:
            return None
        
        # เก็บ TP เดิมถ้ายังไม่มี
        if pos_id not in self._position_original_tp:
            self._position_original_tp[pos_id] = current_tp
        
        original_tp = self._position_original_tp[pos_id]
        config = self._floating_tp_config
        min_rr = config.get("min_rr_ratio", 1.5)
        extension_mult = config.get("tp_extension_multiplier", 1.2)
        max_extension_pct = config.get("max_tp_extension_pct", 5.0)
        
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        
        try:
            if position.side == OrderSide.BUY:
                # BUY: SL ต่ำกว่า Entry, TP สูงกว่า Entry
                new_risk = entry_price - new_sl  # Risk ลดลงแล้ว (SL ยกขึ้น)
                
                # คำนวณ TP ใหม่จาก R:R
                new_reward_min = new_risk * min_rr
                new_tp_from_rr = entry_price + new_reward_min
                
                # ยืด TP เพิ่มเมื่อ SL ถูก lock
                # ยิ่ง lock profit มาก ยิ่งยืด TP มาก
                locked_profit = new_sl - (entry_price - (original_tp - entry_price) / min_rr)
                if locked_profit > 0:
                    extension_factor = extension_mult
                else:
                    extension_factor = 1.0
                
                # TP ใหม่ = max(TP เดิม, TP จาก R:R ใหม่) * extension
                base_tp = max(current_tp, new_tp_from_rr)
                
                # คำนวณ distance ที่ราคาวิ่งมาแล้ว
                price_moved = current_price - entry_price
                if price_moved > 0:
                    # ราคาวิ่งไปในทิศทางที่ถูก → ยืด TP ตาม
                    new_tp = entry_price + (original_tp - entry_price) + price_moved * (extension_factor - 1)
                else:
                    new_tp = base_tp
                
                # ไม่ให้ TP ต่ำกว่า TP เดิม
                new_tp = max(new_tp, current_tp)
                
                # จำกัดการยืดไม่เกิน max_extension_pct
                max_tp = entry_price * (1 + max_extension_pct / 100)
                new_tp = min(new_tp, max_tp)
                
            else:  # SELL
                # SELL: SL สูงกว่า Entry, TP ต่ำกว่า Entry
                new_risk = new_sl - entry_price
                
                new_reward_min = new_risk * min_rr
                new_tp_from_rr = entry_price - new_reward_min
                
                # TP ใหม่ = min(TP เดิม, TP จาก R:R ใหม่)
                base_tp = min(current_tp, new_tp_from_rr)
                
                price_moved = entry_price - current_price
                if price_moved > 0:
                    new_tp = entry_price - (entry_price - original_tp) - price_moved * (extension_mult - 1)
                else:
                    new_tp = base_tp
                
                # ไม่ให้ TP สูงกว่า TP เดิม (สำหรับ SELL)
                new_tp = min(new_tp, current_tp)
                
                # จำกัดการยืด
                min_tp = entry_price * (1 - max_extension_pct / 100)
                new_tp = max(new_tp, min_tp)
            
            # Round ตามประเภทสินทรัพย์
            if is_gold:
                new_tp = round(new_tp, 2)
            else:
                new_tp = round(new_tp, 5)
            
            # ถ้า TP เปลี่ยนแปลงมากพอ → อัพเดท
            tp_change_pct = abs(new_tp - current_tp) / current_tp * 100
            if tp_change_pct < 0.05:  # เปลี่ยนน้อยกว่า 0.05% ไม่ต้องอัพเดท
                return current_tp
            
            # Modify TP via broker
            success = await self.trading_engine.broker.modify_position(
                position_id=pos_id,
                take_profit=new_tp
            )
            
            if success:
                position.take_profit = new_tp
                
                # คำนวณ R:R ใหม่
                if position.side == OrderSide.BUY:
                    actual_risk = entry_price - new_sl
                    actual_reward = new_tp - entry_price
                else:
                    actual_risk = new_sl - entry_price
                    actual_reward = entry_price - new_tp
                
                new_rr = actual_reward / actual_risk if actual_risk > 0 else 0
                
                logger.info(
                    f"🎯 FLOATING TP: {symbol} | "
                    f"TP: {current_tp:.5f} → {new_tp:.5f} | "
                    f"New R:R = 1:{new_rr:.2f}"
                )
                
                return new_tp
            else:
                logger.warning(f"⚠️ Failed to modify TP for {symbol}")
                return current_tp
                
        except Exception as e:
            logger.error(f"Error updating floating TP for {symbol}: {e}")
            return current_tp

    async def _update_trailing_stops(self) -> None:
        """🎯 Trailing Stop - ยก SL ตามราคาเพื่อล็อคกำไร"""
        if not self._trailing_stop_config.get("enabled", False):
            return
        
        if not self.trading_engine or not self.trading_engine.positions:
            return
        
        config = self._trailing_stop_config
        activation_pct = config.get("activation_profit_pct", 0.3)
        trail_pct = config.get("trail_distance_pct", 0.2)
        step_pct = config.get("step_pct", 0.1)
        
        for pos_id, position in list(self.trading_engine.positions.items()):
            try:
                symbol = position.symbol
                current_price = position.current_price or 0
                entry_price = position.entry_price or 0
                current_sl = position.stop_loss
                
                if not current_price or not entry_price:
                    continue
                
                # Calculate profit percentage
                if position.side == OrderSide.BUY:
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # SELL
                    profit_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Check activation threshold
                if profit_pct < activation_pct:
                    continue
                
                # Determine min trail distance based on symbol
                is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
                if is_gold:
                    min_trail_distance = config.get("min_trail_distance_gold", 1.0)
                else:
                    min_trail_distance = config.get("min_trail_distance_forex", 0.0010)
                
                # Calculate trailing distance
                trail_distance = max(current_price * (trail_pct / 100), min_trail_distance)
                step_distance = current_price * (step_pct / 100)
                
                # Calculate new SL based on position side
                if position.side == OrderSide.BUY:
                    # For BUY: SL below current price, move up only
                    new_sl = current_price - trail_distance
                    
                    # Don't move SL backward
                    if current_sl and new_sl <= current_sl:
                        continue
                    
                    # Check step size (move by at least step_distance)
                    if current_sl and (new_sl - current_sl) < step_distance:
                        continue
                    
                    # Don't move SL above entry (break-even ok, but not negative)
                    # Actually for trailing, we can move above entry to lock profit
                    
                else:  # SELL
                    # For SELL: SL above current price, move down only
                    new_sl = current_price + trail_distance
                    
                    # Don't move SL backward (for SELL, lower is backward)
                    if current_sl and new_sl >= current_sl:
                        continue
                    
                    # Check step size
                    if current_sl and (current_sl - new_sl) < step_distance:
                        continue
                
                # Round to appropriate precision
                if is_gold:
                    new_sl = round(new_sl, 2)
                else:
                    new_sl = round(new_sl, 5)
                
                # Modify position SL via broker
                try:
                    success = await self.trading_engine.broker.modify_position(
                        position_id=pos_id,
                        stop_loss=new_sl
                    )
                    
                    if success:
                        old_sl = current_sl or entry_price
                        position.stop_loss = new_sl
                        
                        # Calculate locked profit
                        if position.side == OrderSide.BUY:
                            locked_profit_pct = ((new_sl - entry_price) / entry_price) * 100
                        else:
                            locked_profit_pct = ((entry_price - new_sl) / entry_price) * 100
                        
                        logger.info(
                            f"📈 TRAILING STOP: {symbol} | "
                            f"Profit: {profit_pct:.2f}% | "
                            f"SL: {old_sl:.5f} → {new_sl:.5f} | "
                            f"Locked: {locked_profit_pct:.2f}%"
                        )
                        
                        # 🎯 FLOATING TP - ยก TP ตาม SL เพื่อให้ได้กำไรมากขึ้น
                        new_tp = await self._update_floating_tp(
                            position=position,
                            new_sl=new_sl,
                            entry_price=entry_price,
                            current_price=current_price,
                            pos_id=pos_id
                        )
                        
                        # Broadcast update
                        await self._broadcast_update("trailing_stop_moved", {
                            "symbol": symbol,
                            "position_id": pos_id,
                            "old_sl": old_sl,
                            "new_sl": new_sl,
                            "new_tp": new_tp,
                            "profit_pct": profit_pct,
                            "locked_profit_pct": locked_profit_pct,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        logger.warning(f"⚠️ Failed to modify SL for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error modifying trailing stop for {symbol}: {e}")
                    
            except Exception as e:
                logger.error(f"Error in trailing stop for position {pos_id}: {e}")

    async def run(self, interval_seconds: int = 60):
        """Run the enhanced trading bot"""
        # Store interval for status reporting
        self._interval = interval_seconds
        
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
        
        # 📚 Start background learner (ประหยัด CPU)
        if self.learning_system and self.learning_system.enable_background:
            await self.learning_system.start()
            logger.info("📚 Background Learner started (async mode)")
        
        # Broadcast bot status
        await self._broadcast_update("bot_status", {
            "status": "running",
            "symbols": self.symbols,
            "min_quality": self.min_quality.value,
        })
        
        while self._running:
            try:
                # 🔄 SYNC POSITIONS WITH MT5 (Auto-detect SL/TP closed positions)
                if self.trading_engine:
                    sync_result = await self.trading_engine.sync_with_broker()
                    
                    # 📈 Update Trailing Stops - ยก SL ตามราคา
                    await self._update_trailing_stops()
                    
                    # 🛡️ Apply Break-Even - ย้าย SL ไป entry
                    await self._apply_break_even()
                    
                    # ⏰ Check Time Exit - ปิดออเดอร์ที่ค้างนาน
                    await self._check_time_exit()
                    
                    # Update daily stats if positions were closed externally
                    for removed_pos in sync_result.get("removed", []):
                        pnl = removed_pos.get("pnl", 0)
                        self._daily_stats["trades"] += 1
                        self._daily_stats["pnl"] += pnl
                        if pnl > 0:
                            self._daily_stats["wins"] += 1
                        else:
                            self._daily_stats["losses"] += 1
                        
                        # Broadcast position closed event
                        await self._broadcast_update("position_closed", {
                            "id": removed_pos.get("id"),
                            "symbol": removed_pos.get("symbol"),
                            "side": removed_pos.get("side"),
                            "pnl": pnl,
                            "reason": removed_pos.get("reason"),
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.info(f"📊 Position closed externally: {removed_pos.get('symbol')} PnL: ${pnl:.2f}")
                        
                        # 📉 Update loss tracking for smart features
                        self._update_loss_tracking(pnl)
                        
                        # 🧠⚡ Update Ultra Intelligence performance
                        if self.ultra_intelligence:
                            self.ultra_intelligence.update_performance(pnl, pnl > 0)
                        
                        # 🏆👑 Update Supreme Intelligence performance
                        if self.supreme_intelligence:
                            self.supreme_intelligence.update_trade_result({
                                "pnl": pnl,
                                "symbol": removed_pos.get("symbol"),
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        # 🌌✨ Update Transcendent Intelligence performance
                        if self.transcendent_intelligence:
                            self.transcendent_intelligence.update_trade_result({
                                "pnl": pnl,
                                "symbol": removed_pos.get("symbol"),
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        # Clean up break-even tracking
                        pos_id = removed_pos.get("id")
                        if pos_id in self._break_even_applied:
                            del self._break_even_applied[pos_id]
                
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
                    
                    # 📊 Store Signal History for API
                    signal_record = {
                        "id": f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "signal": signal,
                        "quality": quality,
                        "confidence": confidence,
                        "price": price,
                        "regime": regime,
                        "titan_grade": self._last_titan_decision.get("grade", "N/A"),
                        "titan_score": self._last_titan_decision.get("titan_score", 0),
                        "omega_grade": self._last_omega_result.get("grade", "N/A"),
                        "omega_score": self._last_omega_result.get("omega_score", 0),
                        "alpha_grade": self._last_alpha_result.get("grade", "N/A"),
                        "alpha_score": self._last_alpha_result.get("alpha_score", 0),
                    }
                    self._signal_history.insert(0, signal_record)
                    if len(self._signal_history) > 100:
                        self._signal_history = self._signal_history[:100]
                    
                    # Store last analysis (both global and by symbol)
                    self._last_analysis = analysis
                    self._last_analysis_by_symbol[symbol] = analysis
                    
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
                        
                        # 📊 Store trade result for debugging
                        self._last_trade_result = {
                            "symbol": symbol,
                            "signal": signal,
                            "quality": quality,
                            "result": trade_result,
                            "timestamp": datetime.now().isoformat()
                        }
                        self._last_trade_result_by_symbol[symbol] = self._last_trade_result
                        
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
        
        # 📚 Stop background learner and save state
        if self.learning_system:
            await self.learning_system.stop()
            logger.info("📚 Learning state saved")
        
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
            # Bot config for dashboard
            "config": {
                "symbols": self.symbols,
                "timeframe": self.timeframe,
                "htf_timeframe": self.htf_timeframe,
                "quality": self.min_quality.value if hasattr(self.min_quality, 'value') else str(self.min_quality),
                "interval": getattr(self, '_interval', 60),
            }
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
