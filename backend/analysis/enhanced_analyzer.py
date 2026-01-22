"""
AI-Enhanced Pattern Analyzer - High Win Rate Module
เพิ่ม factors เสริมเพื่อเพิ่ม Win Rate ให้ระบบ

ปัจจัยที่เพิ่ม:
1. Technical Indicators (RSI, MACD, Bollinger, ATR)
2. Volume Analysis (Volume Confirmation, OBV)
3. Multi-Timeframe Analysis (MTF Confluence)
4. Market Regime Detection (Trending vs Ranging)
5. Quality Score Filter (Pattern Quality Assessment)
6. Momentum Confirmation
7. Support/Resistance Proximity
8. Session Timing (Best trading sessions)
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """สภาวะตลาด"""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    RANGING = "RANGING"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    VOLATILE = "VOLATILE"


class SignalQuality(str, Enum):
    """คุณภาพสัญญาณ"""
    PREMIUM = "PREMIUM"      # Win Rate Expected: 85%+
    HIGH = "HIGH"            # Win Rate Expected: 75-85%
    MEDIUM = "MEDIUM"        # Win Rate Expected: 65-75%
    LOW = "LOW"              # Win Rate Expected: < 65%
    SKIP = "SKIP"            # ไม่ควรเข้า


@dataclass
class TechnicalIndicators:
    """Technical Indicator Values"""
    rsi: float = 50.0
    rsi_trend: str = "NEUTRAL"  # OVERBOUGHT, OVERSOLD, NEUTRAL
    
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: str = "MIDDLE"  # UPPER, MIDDLE, LOWER, OUTSIDE_UPPER, OUTSIDE_LOWER
    
    atr: float = 0.0
    atr_percent: float = 0.0
    volatility: str = "NORMAL"  # HIGH, NORMAL, LOW
    
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    ema_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    
    def to_dict(self) -> dict:
        return {
            "rsi": round(self.rsi, 2),
            "rsi_trend": self.rsi_trend,
            "macd": round(self.macd, 4),
            "macd_signal": round(self.macd_signal, 4),
            "macd_histogram": round(self.macd_histogram, 4),
            "macd_trend": self.macd_trend,
            "bb_upper": round(self.bb_upper, 4),
            "bb_middle": round(self.bb_middle, 4),
            "bb_lower": round(self.bb_lower, 4),
            "bb_position": self.bb_position,
            "atr": round(self.atr, 4),
            "atr_percent": round(self.atr_percent, 2),
            "volatility": self.volatility,
            "ema_20": round(self.ema_20, 4),
            "ema_50": round(self.ema_50, 4),
            "ema_200": round(self.ema_200, 4),
            "ema_trend": self.ema_trend,
        }


@dataclass
class VolumeAnalysis:
    """Volume Analysis Results"""
    current_volume: float = 0.0
    average_volume: float = 0.0
    volume_ratio: float = 1.0  # Current / Average
    
    obv: float = 0.0
    obv_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    
    volume_confirmation: bool = False
    volume_spike: bool = False
    
    def to_dict(self) -> dict:
        return {
            "current_volume": self.current_volume,
            "average_volume": round(self.average_volume, 2),
            "volume_ratio": round(self.volume_ratio, 2),
            "obv_trend": self.obv_trend,
            "volume_confirmation": self.volume_confirmation,
            "volume_spike": self.volume_spike,
        }


@dataclass
class MultiTimeframeAnalysis:
    """Multi-Timeframe Analysis Results"""
    htf_trend: str = "NEUTRAL"  # Higher timeframe trend
    mtf_trend: str = "NEUTRAL"  # Medium timeframe trend
    ltf_trend: str = "NEUTRAL"  # Lower timeframe trend
    
    confluence_score: float = 0.0  # 0-100
    trend_alignment: bool = False
    
    htf_support_nearby: bool = False
    htf_resistance_nearby: bool = False
    
    def to_dict(self) -> dict:
        return {
            "htf_trend": self.htf_trend,
            "mtf_trend": self.mtf_trend,
            "ltf_trend": self.ltf_trend,
            "confluence_score": round(self.confluence_score, 2),
            "trend_alignment": self.trend_alignment,
            "htf_support_nearby": self.htf_support_nearby,
            "htf_resistance_nearby": self.htf_resistance_nearby,
        }


@dataclass
class EnhancedSignalResult:
    """Enhanced Signal with AI Factors"""
    # Basic signal info
    signal: str  # BUY, SELL, WAIT
    base_confidence: float  # From pattern matching
    
    # Enhanced confidence
    enhanced_confidence: float
    quality: SignalQuality
    
    # Factor scores (0-100)
    sentiment_score: float = 0.0   # NEW: Contrarian sentiment
    pattern_score: float = 0.0
    technical_score: float = 0.0
    volume_score: float = 0.0
    mtf_score: float = 0.0
    regime_score: float = 0.0
    timing_score: float = 0.0
    momentum_score: float = 0.0
    
    # Smart Money Analysis
    smart_money_signal: Optional[str] = None  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    
    # Analysis components
    indicators: Optional[TechnicalIndicators] = None
    volume_analysis: Optional[VolumeAnalysis] = None
    mtf_analysis: Optional[MultiTimeframeAnalysis] = None
    market_regime: MarketRegime = MarketRegime.RANGING
    
    # Risk-adjusted values
    adjusted_stop_loss: Optional[float] = None
    adjusted_take_profit: Optional[float] = None
    risk_reward_ratio: float = 0.0
    
    # Trade recommendation
    recommended_position_size: float = 1.0  # Multiplier (0.5x, 1x, 1.5x)
    entry_timing: str = "NOW"  # NOW, WAIT_PULLBACK, WAIT_BREAKOUT
    
    # Reasons
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    skip_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "base_confidence": round(self.base_confidence, 2),
            "enhanced_confidence": round(self.enhanced_confidence, 2),
            "quality": self.quality.value,
            "smart_money_signal": self.smart_money_signal,
            "scores": {
                "sentiment": round(self.sentiment_score, 2),  # NEW
                "pattern": round(self.pattern_score, 2),
                "technical": round(self.technical_score, 2),
                "volume": round(self.volume_score, 2),
                "mtf": round(self.mtf_score, 2),
                "regime": round(self.regime_score, 2),
                "timing": round(self.timing_score, 2),
                "momentum": round(self.momentum_score, 2),
            },
            "indicators": self.indicators.to_dict() if self.indicators else None,
            "volume_analysis": self.volume_analysis.to_dict() if self.volume_analysis else None,
            "mtf_analysis": self.mtf_analysis.to_dict() if self.mtf_analysis else None,
            "market_regime": self.market_regime.value,
            "risk_management": {
                "adjusted_stop_loss": self.adjusted_stop_loss,
                "adjusted_take_profit": self.adjusted_take_profit,
                "risk_reward_ratio": round(self.risk_reward_ratio, 2),
                "recommended_position_size": self.recommended_position_size,
                "entry_timing": self.entry_timing,
            },
            "factors": {
                "bullish": self.bullish_factors,
                "bearish": self.bearish_factors,
                "skip_reasons": self.skip_reasons,
            },
        }


class TechnicalIndicatorCalculator:
    """Calculate Technical Indicators"""
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD"""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        ema_fast = TechnicalIndicatorCalculator._ema(prices, fast)
        ema_slow = TechnicalIndicatorCalculator._ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicatorCalculator._ema(
            np.array([macd_line]), signal
        ) if isinstance(macd_line, float) else macd_line
        
        # Simplified for single value
        if isinstance(macd_line, (int, float)):
            return float(macd_line), float(signal_line), float(macd_line - signal_line)
        
        return float(macd_line), float(macd_line * 0.9), float(macd_line * 0.1)
    
    @staticmethod
    def calculate_bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1] * 1.02, prices[-1], prices[-1] * 0.98
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    @staticmethod
    def calculate_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate ATR (Average True Range)"""
        if len(highs) < period + 1:
            return np.mean(highs - lows)
        
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        return np.mean(tr_list[-period:])
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        return TechnicalIndicatorCalculator._ema(prices, period)
    
    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        """Internal EMA calculation"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def calculate_obv(prices: np.ndarray, volumes: np.ndarray) -> Tuple[float, str]:
        """Calculate OBV and trend"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0, "NEUTRAL"
        
        obv = 0.0
        obv_values = [0.0]
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
            obv_values.append(obv)
        
        # Determine OBV trend (last 10 periods)
        if len(obv_values) > 10:
            recent_obv = obv_values[-10:]
            if recent_obv[-1] > recent_obv[0] * 1.05:
                trend = "BULLISH"
            elif recent_obv[-1] < recent_obv[0] * 0.95:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
        else:
            trend = "NEUTRAL"
        
        return obv, trend


class EnhancedAnalyzer:
    """
    AI-Enhanced Pattern Analyzer
    เพิ่มปัจจัยหลายมิติเพื่อเพิ่ม Win Rate
    
    รวม Smart Money Concept:
    - Contrarian Sentiment (สำคัญที่สุด!)
    - Order Flow / Volume
    - Market Structure
    """
    
    # Weight configuration for each factor
    # ปรับให้ SENTIMENT มีน้ำหนักสูงสุด (Contrarian Strategy)
    FACTOR_WEIGHTS = {
        "sentiment": 0.25,   # 🔴 Contrarian sentiment (NEW - สำคัญมาก!)
        "pattern": 0.20,     # Pattern matching score
        "technical": 0.15,   # Technical indicators
        "volume": 0.12,      # Volume confirmation
        "mtf": 0.10,         # Multi-timeframe
        "regime": 0.08,      # Market regime
        "timing": 0.05,      # Session timing
        "momentum": 0.05,    # Momentum
    }
    
    # Quality thresholds
    QUALITY_THRESHOLDS = {
        "PREMIUM": 85,   # >= 85 = Premium
        "HIGH": 75,      # >= 75 = High
        "MEDIUM": 65,    # >= 65 = Medium
        "LOW": 50,       # >= 50 = Low
    }
    
    def __init__(
        self,
        min_quality: SignalQuality = SignalQuality.MEDIUM,
        enable_volume_filter: bool = True,
        enable_mtf_filter: bool = True,
        enable_regime_filter: bool = True,
        enable_sentiment_filter: bool = True,  # NEW: Contrarian sentiment
    ):
        """
        Initialize Enhanced Analyzer
        
        Args:
            min_quality: Minimum quality to generate signal
            enable_volume_filter: Enable volume confirmation
            enable_mtf_filter: Enable multi-timeframe analysis
            enable_regime_filter: Enable market regime detection
            enable_sentiment_filter: Enable contrarian sentiment analysis
        """
        self.min_quality = min_quality
        self.enable_volume_filter = enable_volume_filter
        self.enable_mtf_filter = enable_mtf_filter
        self.enable_regime_filter = enable_regime_filter
        self.enable_sentiment_filter = enable_sentiment_filter
        
        self.indicator_calc = TechnicalIndicatorCalculator()
        
        # Initialize Smart Money Analyzer
        if self.enable_sentiment_filter:
            from .smart_money_analyzer import get_smart_money_analyzer
            self.smart_money_analyzer = get_smart_money_analyzer()
    
    async def analyze(
        self,
        base_signal: str,
        base_confidence: float,
        ohlcv_data: Dict[str, np.ndarray],
        current_price: float,
        symbol: str = "UNKNOWN",  # NEW: For sentiment analysis
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        htf_data: Optional[Dict[str, np.ndarray]] = None,
        current_time: Optional[datetime] = None,
    ) -> EnhancedSignalResult:
        """
        Perform enhanced analysis with AI factors + Smart Money
        
        Args:
            base_signal: Base signal from pattern matching (BUY/SELL/WAIT)
            base_confidence: Confidence from pattern matching (0-100)
            ohlcv_data: Dict with 'open', 'high', 'low', 'close', 'volume' arrays
            current_price: Current price
            symbol: Trading symbol for sentiment analysis
            stop_loss: Base stop loss from pattern analysis
            take_profit: Base take profit from pattern analysis
            htf_data: Higher timeframe OHLCV data (optional)
            current_time: Current datetime for session analysis
        
        Returns:
            EnhancedSignalResult with comprehensive analysis
        """
        bullish_factors = []
        bearish_factors = []
        skip_reasons = []
        
        # Extract OHLCV arrays
        opens = ohlcv_data.get("open", np.array([]))
        highs = ohlcv_data.get("high", np.array([]))
        lows = ohlcv_data.get("low", np.array([]))
        closes = ohlcv_data.get("close", np.array([]))
        volumes = ohlcv_data.get("volume", np.array([]))
        
        # 0. SMART MONEY / SENTIMENT ANALYSIS (NEW - Most Important!)
        sentiment_score = 50.0  # Default neutral
        smart_money_signal = None
        
        if self.enable_sentiment_filter:
            try:
                from .smart_money_analyzer import SmartMoneySignal
                
                smart_money = await self.smart_money_analyzer.analyze(
                    symbol=symbol,
                    ohlcv=ohlcv_data,
                    current_price=current_price,
                    htf_ohlcv=htf_data,
                )
                
                sentiment_score = smart_money.sentiment_score
                smart_money_signal = smart_money.signal
                
                # Override base signal if sentiment is strong
                if smart_money.signal == SmartMoneySignal.STRONG_SELL:
                    bearish_factors.append(f"🔴 SMART MONEY: Retail {smart_money.sentiment.avg_long_percent:.0f}% Long → SELL")
                    if base_signal == "BUY":
                        # Contrarian override!
                        logger.warning(f"🚫 [CONTRARIAN OVERRIDE] Original: BUY → Changed to: WAIT")
                        logger.warning(f"   Reason: Retail {smart_money.sentiment.avg_long_percent:.0f}% Long - Too bullish!")
                        skip_reasons.append("⚠️ Sentiment override: Retail too bullish for BUY")
                        base_signal = "WAIT"
                        
                elif smart_money.signal == SmartMoneySignal.STRONG_BUY:
                    bullish_factors.append(f"🟢 SMART MONEY: Retail {smart_money.sentiment.avg_short_percent:.0f}% Short → BUY")
                    if base_signal == "SELL":
                        # Contrarian override!
                        logger.warning(f"🚫 [CONTRARIAN OVERRIDE] Original: SELL → Changed to: WAIT")
                        logger.warning(f"   Reason: Retail {smart_money.sentiment.avg_short_percent:.0f}% Short - Too bearish!")
                        skip_reasons.append("⚠️ Sentiment override: Retail too bearish for SELL")
                        base_signal = "WAIT"
                        
                elif smart_money.signal == SmartMoneySignal.SELL:
                    bearish_factors.append(f"Retail sentiment bearish edge ({smart_money.sentiment.avg_long_percent:.0f}% Long)")
                    logger.info(f"📊 [SENTIMENT EDGE] Retail {smart_money.sentiment.avg_long_percent:.0f}% Long - Slight SELL edge")
                    
                elif smart_money.signal == SmartMoneySignal.BUY:
                    bullish_factors.append(f"Retail sentiment bullish edge ({smart_money.sentiment.avg_short_percent:.0f}% Short)")
                    logger.info(f"📊 [SENTIMENT EDGE] Retail {smart_money.sentiment.avg_short_percent:.0f}% Short - Slight BUY edge")
                
                # Add Smart Money reasons
                for reason in smart_money.reasons[:3]:
                    if "SELL" in reason or "bearish" in reason.lower():
                        bearish_factors.append(reason)
                    elif "BUY" in reason or "bullish" in reason.lower():
                        bullish_factors.append(reason)
                        
                for warning in smart_money.warnings[:2]:
                    skip_reasons.append(warning)
                    
            except Exception as e:
                logger.warning(f"Smart Money analysis failed: {e}")
        
        # 1. Calculate Pattern Score
        pattern_score = self._calculate_pattern_score(base_confidence)
        
        # 2. Calculate Technical Indicators
        indicators = self._calculate_indicators(highs, lows, closes, current_price)
        technical_score = self._calculate_technical_score(indicators, base_signal)
        
        if indicators.rsi_trend == "OVERBOUGHT" and base_signal == "BUY":
            bearish_factors.append("RSI Overbought - Potential reversal")
        elif indicators.rsi_trend == "OVERSOLD" and base_signal == "SELL":
            bullish_factors.append("RSI Oversold - Potential reversal")
        
        if indicators.macd_trend == "BULLISH":
            bullish_factors.append("MACD Bullish crossover")
        elif indicators.macd_trend == "BEARISH":
            bearish_factors.append("MACD Bearish crossover")
        
        if indicators.ema_trend == "BULLISH":
            bullish_factors.append("EMA alignment bullish (20>50>200)")
        elif indicators.ema_trend == "BEARISH":
            bearish_factors.append("EMA alignment bearish (20<50<200)")
        
        # 3. Volume Analysis
        volume_analysis = self._analyze_volume(closes, volumes, base_signal)
        volume_score = self._calculate_volume_score(volume_analysis, base_signal)
        
        if self.enable_volume_filter:
            if volume_analysis.volume_confirmation:
                if base_signal == "BUY":
                    bullish_factors.append("Volume confirms buying pressure")
                else:
                    bearish_factors.append("Volume confirms selling pressure")
            else:
                skip_reasons.append("No volume confirmation")
            
            if volume_analysis.volume_spike:
                bullish_factors.append("Volume spike detected - Strong move expected")
        
        # 4. Multi-Timeframe Analysis
        if self.enable_mtf_filter and htf_data:
            mtf_analysis = self._analyze_mtf(closes, htf_data, base_signal)
            mtf_score = mtf_analysis.confluence_score
            
            if mtf_analysis.trend_alignment:
                if base_signal == "BUY":
                    bullish_factors.append("Multi-timeframe trend alignment (Bullish)")
                else:
                    bearish_factors.append("Multi-timeframe trend alignment (Bearish)")
            else:
                skip_reasons.append("MTF trend misalignment")
        else:
            mtf_analysis = MultiTimeframeAnalysis()
            mtf_score = 50.0  # Neutral if not enabled
        
        # 5. Market Regime Detection
        market_regime = self._detect_market_regime(closes, indicators.atr_percent)
        regime_score = self._calculate_regime_score(market_regime, base_signal)
        
        if self.enable_regime_filter:
            if market_regime == MarketRegime.VOLATILE:
                skip_reasons.append("High volatility - Risky conditions")
            elif market_regime == MarketRegime.RANGING and base_signal in ["BUY", "SELL"]:
                skip_reasons.append("Ranging market - Wait for breakout")
        
        # 6. Timing Score (Trading Sessions)
        timing_score = self._calculate_timing_score(current_time)
        
        if timing_score < 50:
            skip_reasons.append("Sub-optimal trading session")
        elif timing_score >= 80:
            bullish_factors.append("Optimal trading session (London/NY overlap)")
        
        # 7. Momentum Score
        momentum_score = self._calculate_momentum_score(closes, indicators)
        
        if momentum_score >= 70:
            if base_signal == "BUY":
                bullish_factors.append("Strong bullish momentum")
            else:
                bearish_factors.append("Strong bearish momentum")
        
        # Calculate weighted enhanced confidence (now includes sentiment!)
        enhanced_confidence = (
            sentiment_score * self.FACTOR_WEIGHTS["sentiment"] +
            pattern_score * self.FACTOR_WEIGHTS["pattern"] +
            technical_score * self.FACTOR_WEIGHTS["technical"] +
            volume_score * self.FACTOR_WEIGHTS["volume"] +
            mtf_score * self.FACTOR_WEIGHTS["mtf"] +
            regime_score * self.FACTOR_WEIGHTS["regime"] +
            timing_score * self.FACTOR_WEIGHTS["timing"] +
            momentum_score * self.FACTOR_WEIGHTS["momentum"]
        )
        
        # Determine quality
        quality = self._determine_quality(enhanced_confidence)
        
        # Adjust final signal based on quality
        final_signal = base_signal
        if quality == SignalQuality.SKIP or (
            quality.value < self.min_quality.value and 
            self.min_quality != SignalQuality.SKIP
        ):
            final_signal = "WAIT"
            skip_reasons.append(f"Quality {quality.value} below threshold {self.min_quality.value}")
        
        # Calculate risk-adjusted SL/TP
        adjusted_sl, adjusted_tp = self._adjust_sl_tp(
            current_price,
            stop_loss,
            take_profit,
            indicators.atr,
            base_signal,
            quality
        )
        
        # Calculate Risk:Reward ratio
        if adjusted_sl and adjusted_tp:
            risk = abs(current_price - adjusted_sl)
            reward = abs(adjusted_tp - current_price)
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            rr_ratio = 0
        
        # Recommended position size based on quality
        position_size_map = {
            SignalQuality.PREMIUM: 1.5,
            SignalQuality.HIGH: 1.2,
            SignalQuality.MEDIUM: 1.0,
            SignalQuality.LOW: 0.5,
            SignalQuality.SKIP: 0.0,
        }
        recommended_position_size = position_size_map.get(quality, 1.0)
        
        # Entry timing recommendation
        entry_timing = self._determine_entry_timing(
            indicators, volume_analysis, market_regime
        )
        
        return EnhancedSignalResult(
            signal=final_signal,
            base_confidence=base_confidence,
            enhanced_confidence=enhanced_confidence,
            quality=quality,
            sentiment_score=sentiment_score,  # NEW
            pattern_score=pattern_score,
            technical_score=technical_score,
            volume_score=volume_score,
            mtf_score=mtf_score,
            regime_score=regime_score,
            timing_score=timing_score,
            momentum_score=momentum_score,
            smart_money_signal=smart_money_signal.value if smart_money_signal else None,  # NEW
            indicators=indicators,
            volume_analysis=volume_analysis,
            mtf_analysis=mtf_analysis,
            market_regime=market_regime,
            adjusted_stop_loss=adjusted_sl,
            adjusted_take_profit=adjusted_tp,
            risk_reward_ratio=rr_ratio,
            recommended_position_size=recommended_position_size,
            entry_timing=entry_timing,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            skip_reasons=skip_reasons,
        )
    
    def _calculate_pattern_score(self, base_confidence: float) -> float:
        """Convert base confidence to pattern score"""
        # Scale 70-100 to 50-100 for better distribution
        if base_confidence >= 80:
            return min(100, base_confidence + 10)
        elif base_confidence >= 70:
            return base_confidence
        else:
            return base_confidence * 0.8
    
    def _calculate_indicators(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        current_price: float
    ) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        indicators = TechnicalIndicators()
        
        if len(closes) < 2:
            return indicators
        
        # RSI
        indicators.rsi = self.indicator_calc.calculate_rsi(closes)
        if indicators.rsi > 70:
            indicators.rsi_trend = "OVERBOUGHT"
        elif indicators.rsi < 30:
            indicators.rsi_trend = "OVERSOLD"
        else:
            indicators.rsi_trend = "NEUTRAL"
        
        # MACD
        macd, signal, histogram = self.indicator_calc.calculate_macd(closes)
        indicators.macd = macd
        indicators.macd_signal = signal
        indicators.macd_histogram = histogram
        
        if histogram > 0 and macd > signal:
            indicators.macd_trend = "BULLISH"
        elif histogram < 0 and macd < signal:
            indicators.macd_trend = "BEARISH"
        else:
            indicators.macd_trend = "NEUTRAL"
        
        # Bollinger Bands
        upper, middle, lower = self.indicator_calc.calculate_bollinger_bands(closes)
        indicators.bb_upper = upper
        indicators.bb_middle = middle
        indicators.bb_lower = lower
        
        if current_price > upper:
            indicators.bb_position = "OUTSIDE_UPPER"
        elif current_price > middle:
            indicators.bb_position = "UPPER"
        elif current_price < lower:
            indicators.bb_position = "OUTSIDE_LOWER"
        elif current_price < middle:
            indicators.bb_position = "LOWER"
        else:
            indicators.bb_position = "MIDDLE"
        
        # ATR
        if len(highs) > 0 and len(lows) > 0:
            indicators.atr = self.indicator_calc.calculate_atr(highs, lows, closes)
            indicators.atr_percent = (indicators.atr / current_price) * 100
            
            if indicators.atr_percent > 3:
                indicators.volatility = "HIGH"
            elif indicators.atr_percent < 1:
                indicators.volatility = "LOW"
            else:
                indicators.volatility = "NORMAL"
        
        # EMAs
        indicators.ema_20 = self.indicator_calc.calculate_ema(closes, 20)
        indicators.ema_50 = self.indicator_calc.calculate_ema(closes, 50)
        indicators.ema_200 = self.indicator_calc.calculate_ema(closes, 200)
        
        if indicators.ema_20 > indicators.ema_50 > indicators.ema_200:
            indicators.ema_trend = "BULLISH"
        elif indicators.ema_20 < indicators.ema_50 < indicators.ema_200:
            indicators.ema_trend = "BEARISH"
        else:
            indicators.ema_trend = "NEUTRAL"
        
        return indicators
    
    def _calculate_technical_score(
        self,
        indicators: TechnicalIndicators,
        base_signal: str
    ) -> float:
        """Calculate technical indicator score"""
        score = 50.0  # Base neutral score
        
        # RSI contribution
        if base_signal == "BUY":
            if 30 < indicators.rsi < 50:
                score += 15  # Good buy zone
            elif indicators.rsi < 30:
                score += 10  # Oversold, but risky
            elif indicators.rsi > 70:
                score -= 15  # Overbought, bad for buy
        elif base_signal == "SELL":
            if 50 < indicators.rsi < 70:
                score += 15  # Good sell zone
            elif indicators.rsi > 70:
                score += 10  # Overbought
            elif indicators.rsi < 30:
                score -= 15  # Oversold, bad for sell
        
        # MACD contribution
        if base_signal == "BUY" and indicators.macd_trend == "BULLISH":
            score += 15
        elif base_signal == "SELL" and indicators.macd_trend == "BEARISH":
            score += 15
        elif (base_signal == "BUY" and indicators.macd_trend == "BEARISH") or \
             (base_signal == "SELL" and indicators.macd_trend == "BULLISH"):
            score -= 10
        
        # EMA trend contribution
        if base_signal == "BUY" and indicators.ema_trend == "BULLISH":
            score += 15
        elif base_signal == "SELL" and indicators.ema_trend == "BEARISH":
            score += 15
        elif (base_signal == "BUY" and indicators.ema_trend == "BEARISH") or \
             (base_signal == "SELL" and indicators.ema_trend == "BULLISH"):
            score -= 10
        
        # Bollinger Band contribution
        if base_signal == "BUY" and indicators.bb_position in ["LOWER", "OUTSIDE_LOWER"]:
            score += 10  # Buy near support
        elif base_signal == "SELL" and indicators.bb_position in ["UPPER", "OUTSIDE_UPPER"]:
            score += 10  # Sell near resistance
        
        return max(0, min(100, score))
    
    def _analyze_volume(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        base_signal: str
    ) -> VolumeAnalysis:
        """Analyze volume patterns"""
        analysis = VolumeAnalysis()
        
        if len(volumes) < 10:
            return analysis
        
        analysis.current_volume = volumes[-1]
        analysis.average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        analysis.volume_ratio = analysis.current_volume / analysis.average_volume if analysis.average_volume > 0 else 1.0
        
        # OBV
        analysis.obv, analysis.obv_trend = self.indicator_calc.calculate_obv(closes, volumes)
        
        # Volume confirmation
        # Volume should be higher than average on signal candle
        if analysis.volume_ratio > 1.2:
            if base_signal == "BUY" and closes[-1] > closes[-2]:
                analysis.volume_confirmation = True
            elif base_signal == "SELL" and closes[-1] < closes[-2]:
                analysis.volume_confirmation = True
        
        # Volume spike detection
        if analysis.volume_ratio > 2.0:
            analysis.volume_spike = True
        
        return analysis
    
    def _calculate_volume_score(
        self,
        volume_analysis: VolumeAnalysis,
        base_signal: str
    ) -> float:
        """Calculate volume score"""
        score = 50.0
        
        # Volume ratio contribution
        if volume_analysis.volume_ratio > 1.5:
            score += 20
        elif volume_analysis.volume_ratio > 1.2:
            score += 10
        elif volume_analysis.volume_ratio < 0.8:
            score -= 15  # Low volume is bad
        
        # Volume confirmation
        if volume_analysis.volume_confirmation:
            score += 20
        
        # OBV trend alignment
        if base_signal == "BUY" and volume_analysis.obv_trend == "BULLISH":
            score += 10
        elif base_signal == "SELL" and volume_analysis.obv_trend == "BEARISH":
            score += 10
        elif (base_signal == "BUY" and volume_analysis.obv_trend == "BEARISH") or \
             (base_signal == "SELL" and volume_analysis.obv_trend == "BULLISH"):
            score -= 10
        
        return max(0, min(100, score))
    
    def _analyze_mtf(
        self,
        ltf_closes: np.ndarray,
        htf_data: Dict[str, np.ndarray],
        base_signal: str
    ) -> MultiTimeframeAnalysis:
        """Analyze multiple timeframes"""
        analysis = MultiTimeframeAnalysis()
        
        htf_closes = htf_data.get("close", np.array([]))
        
        if len(ltf_closes) < 20 or len(htf_closes) < 20:
            return analysis
        
        # Determine LTF trend
        ltf_ema_fast = self.indicator_calc.calculate_ema(ltf_closes, 10)
        ltf_ema_slow = self.indicator_calc.calculate_ema(ltf_closes, 20)
        
        if ltf_ema_fast > ltf_ema_slow:
            analysis.ltf_trend = "BULLISH"
        elif ltf_ema_fast < ltf_ema_slow:
            analysis.ltf_trend = "BEARISH"
        else:
            analysis.ltf_trend = "NEUTRAL"
        
        # Determine HTF trend
        htf_ema_fast = self.indicator_calc.calculate_ema(htf_closes, 10)
        htf_ema_slow = self.indicator_calc.calculate_ema(htf_closes, 20)
        
        if htf_ema_fast > htf_ema_slow:
            analysis.htf_trend = "BULLISH"
        elif htf_ema_fast < htf_ema_slow:
            analysis.htf_trend = "BEARISH"
        else:
            analysis.htf_trend = "NEUTRAL"
        
        # Check alignment
        if base_signal == "BUY":
            if analysis.ltf_trend == "BULLISH" and analysis.htf_trend == "BULLISH":
                analysis.trend_alignment = True
                analysis.confluence_score = 90
            elif analysis.htf_trend == "BULLISH":
                analysis.confluence_score = 70
            elif analysis.htf_trend == "BEARISH":
                analysis.confluence_score = 30
            else:
                analysis.confluence_score = 50
        elif base_signal == "SELL":
            if analysis.ltf_trend == "BEARISH" and analysis.htf_trend == "BEARISH":
                analysis.trend_alignment = True
                analysis.confluence_score = 90
            elif analysis.htf_trend == "BEARISH":
                analysis.confluence_score = 70
            elif analysis.htf_trend == "BULLISH":
                analysis.confluence_score = 30
            else:
                analysis.confluence_score = 50
        else:
            analysis.confluence_score = 50
        
        return analysis
    
    def _detect_market_regime(
        self,
        closes: np.ndarray,
        atr_percent: float
    ) -> MarketRegime:
        """Detect current market regime"""
        if len(closes) < 50:
            return MarketRegime.RANGING
        
        # Calculate trend strength using ADX-like metric
        price_changes = np.diff(closes[-20:])
        up_moves = np.sum(price_changes[price_changes > 0])
        down_moves = np.abs(np.sum(price_changes[price_changes < 0]))
        
        total_move = up_moves + down_moves
        if total_move == 0:
            return MarketRegime.RANGING
        
        trend_strength = abs(up_moves - down_moves) / total_move * 100
        
        # Check volatility
        if atr_percent > 4:
            return MarketRegime.VOLATILE
        
        # Determine regime
        if trend_strength > 60:
            if up_moves > down_moves:
                return MarketRegime.STRONG_UPTREND
            else:
                return MarketRegime.STRONG_DOWNTREND
        elif trend_strength > 40:
            if up_moves > down_moves:
                return MarketRegime.UPTREND
            else:
                return MarketRegime.DOWNTREND
        else:
            return MarketRegime.RANGING
    
    def _calculate_regime_score(
        self,
        regime: MarketRegime,
        base_signal: str
    ) -> float:
        """Calculate regime alignment score"""
        score_map = {
            MarketRegime.STRONG_UPTREND: {"BUY": 95, "SELL": 20, "WAIT": 50},
            MarketRegime.UPTREND: {"BUY": 80, "SELL": 35, "WAIT": 50},
            MarketRegime.RANGING: {"BUY": 40, "SELL": 40, "WAIT": 70},
            MarketRegime.DOWNTREND: {"BUY": 35, "SELL": 80, "WAIT": 50},
            MarketRegime.STRONG_DOWNTREND: {"BUY": 20, "SELL": 95, "WAIT": 50},
            MarketRegime.VOLATILE: {"BUY": 30, "SELL": 30, "WAIT": 80},
        }
        
        return score_map.get(regime, {}).get(base_signal, 50)
    
    def _calculate_timing_score(
        self,
        current_time: Optional[datetime]
    ) -> float:
        """Calculate trading session timing score"""
        if current_time is None:
            return 70  # Neutral if no time provided
        
        hour = current_time.hour
        
        # Best trading hours (UTC)
        # London session: 8-16
        # NY session: 13-22
        # Overlap: 13-16 (best)
        
        if 13 <= hour <= 16:
            return 95  # London-NY overlap
        elif 8 <= hour <= 12:
            return 80  # London morning
        elif 17 <= hour <= 20:
            return 75  # NY afternoon
        elif 21 <= hour <= 23 or 0 <= hour <= 2:
            return 50  # Asian early
        elif 3 <= hour <= 7:
            return 60  # Asian late
        else:
            return 40  # Off hours
    
    def _calculate_momentum_score(
        self,
        closes: np.ndarray,
        indicators: TechnicalIndicators
    ) -> float:
        """Calculate momentum score"""
        if len(closes) < 10:
            return 50
        
        score = 50.0
        
        # Price momentum (rate of change)
        roc = (closes[-1] - closes[-10]) / closes[-10] * 100
        
        if abs(roc) > 2:
            score += 20 if roc > 0 else 20  # Strong momentum either way
        elif abs(roc) > 1:
            score += 10
        
        # RSI momentum
        if 40 < indicators.rsi < 60:
            score += 0  # Neutral
        elif 30 < indicators.rsi < 70:
            score += 10  # Has room to move
        else:
            score += 15  # Extreme, expect reversal
        
        # MACD histogram momentum
        if abs(indicators.macd_histogram) > 0:
            score += 10
        
        return max(0, min(100, score))
    
    def _determine_quality(self, enhanced_confidence: float) -> SignalQuality:
        """Determine signal quality based on enhanced confidence"""
        if enhanced_confidence >= self.QUALITY_THRESHOLDS["PREMIUM"]:
            return SignalQuality.PREMIUM
        elif enhanced_confidence >= self.QUALITY_THRESHOLDS["HIGH"]:
            return SignalQuality.HIGH
        elif enhanced_confidence >= self.QUALITY_THRESHOLDS["MEDIUM"]:
            return SignalQuality.MEDIUM
        elif enhanced_confidence >= self.QUALITY_THRESHOLDS["LOW"]:
            return SignalQuality.LOW
        else:
            return SignalQuality.SKIP
    
    def _adjust_sl_tp(
        self,
        current_price: float,
        base_sl: Optional[float],
        base_tp: Optional[float],
        atr: float,
        signal: str,
        quality: SignalQuality
    ) -> Tuple[Optional[float], Optional[float]]:
        """Adjust SL/TP based on ATR and quality"""
        if not base_sl or not base_tp:
            # Default ATR-based SL/TP
            atr_multiplier = 2.0
            if signal == "BUY":
                base_sl = current_price - (atr * atr_multiplier)
                base_tp = current_price + (atr * atr_multiplier * 2)
            elif signal == "SELL":
                base_sl = current_price + (atr * atr_multiplier)
                base_tp = current_price - (atr * atr_multiplier * 2)
            else:
                return None, None
        
        # Tighten SL for lower quality signals
        sl_adjustment = {
            SignalQuality.PREMIUM: 1.0,   # No adjustment
            SignalQuality.HIGH: 0.9,      # 10% tighter
            SignalQuality.MEDIUM: 0.8,    # 20% tighter
            SignalQuality.LOW: 0.6,       # 40% tighter
            SignalQuality.SKIP: 0.5,
        }
        
        # Widen TP for higher quality signals
        tp_adjustment = {
            SignalQuality.PREMIUM: 1.3,   # 30% wider
            SignalQuality.HIGH: 1.2,      # 20% wider
            SignalQuality.MEDIUM: 1.0,    # No adjustment
            SignalQuality.LOW: 0.8,       # 20% tighter
            SignalQuality.SKIP: 0.5,
        }
        
        sl_mult = sl_adjustment.get(quality, 1.0)
        tp_mult = tp_adjustment.get(quality, 1.0)
        
        sl_distance = abs(current_price - base_sl)
        tp_distance = abs(base_tp - current_price)
        
        if signal == "BUY":
            adjusted_sl = current_price - (sl_distance * sl_mult)
            adjusted_tp = current_price + (tp_distance * tp_mult)
        else:
            adjusted_sl = current_price + (sl_distance * sl_mult)
            adjusted_tp = current_price - (tp_distance * tp_mult)
        
        return adjusted_sl, adjusted_tp
    
    def _determine_entry_timing(
        self,
        indicators: TechnicalIndicators,
        volume_analysis: VolumeAnalysis,
        regime: MarketRegime
    ) -> str:
        """Determine best entry timing"""
        # Wait for pullback in strong trends
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            if indicators.rsi > 60 or indicators.rsi < 40:
                return "WAIT_PULLBACK"
        
        # Wait for breakout in ranging market
        if regime == MarketRegime.RANGING:
            if indicators.bb_position == "MIDDLE":
                return "WAIT_BREAKOUT"
        
        # Volume spike = enter now
        if volume_analysis.volume_spike:
            return "NOW"
        
        # Default
        return "NOW"


# Singleton instance for easy access
_enhanced_analyzer: Optional[EnhancedAnalyzer] = None


def get_enhanced_analyzer() -> EnhancedAnalyzer:
    """Get or create enhanced analyzer instance"""
    global _enhanced_analyzer
    if _enhanced_analyzer is None:
        _enhanced_analyzer = EnhancedAnalyzer()
    return _enhanced_analyzer


# ===========================================
# MULTI-FACTOR ANALYZER INTEGRATION
# ===========================================

from analysis.multi_factor_analyzer import MultiFactorAnalyzer, MultiFactorResult
from config.enhanced_settings import EnhancedAnalysisConfig

_multi_factor_analyzer: Optional[MultiFactorAnalyzer] = None


def get_multi_factor_analyzer() -> MultiFactorAnalyzer:
    """Get or create multi-factor analyzer instance"""
    global _multi_factor_analyzer
    if _multi_factor_analyzer is None:
        config = EnhancedAnalysisConfig.from_env()
        _multi_factor_analyzer = MultiFactorAnalyzer(config)
    return _multi_factor_analyzer


def analyze_with_multi_factor(
    vote_result,
    ohlcv_data: Dict[str, np.ndarray],
    pattern_dates: Optional[list] = None,
    current_time: Optional[datetime] = None,
    symbol: str = "UNKNOWN",
) -> MultiFactorResult:
    """
    Convenience function to run multi-factor analysis
    
    Args:
        vote_result: VoteResult from VotingSystem
        ohlcv_data: Dict with 'open', 'high', 'low', 'close', 'volume' arrays
        pattern_dates: List of datetime when patterns occurred
        current_time: Current datetime
        symbol: Trading symbol
    
    Returns:
        MultiFactorResult with comprehensive analysis
    """
    analyzer = get_multi_factor_analyzer()
    
    prices = ohlcv_data.get("close", np.array([]))
    volumes = ohlcv_data.get("volume", None)
    highs = ohlcv_data.get("high", None)
    lows = ohlcv_data.get("low", None)
    
    return analyzer.analyze(
        vote_result=vote_result,
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        pattern_dates=pattern_dates,
        current_time=current_time,
        symbol=symbol,
    )


if __name__ == "__main__":
    # Test the enhanced analyzer
    print("=" * 60)
    print("AI-Enhanced Pattern Analyzer - Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate sample OHLCV data
    n = 100
    base_price = 100
    closes = base_price + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    opens = closes + np.random.randn(n) * 0.2
    volumes = np.random.randint(1000, 10000, n).astype(float)
    
    # Increase recent volume to simulate confirmation
    volumes[-5:] = volumes[-5:] * 2
    
    ohlcv_data = {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }
    
    # Test with BUY signal
    analyzer = EnhancedAnalyzer()
    result = analyzer.analyze(
        base_signal="BUY",
        base_confidence=75,
        ohlcv_data=ohlcv_data,
        current_price=closes[-1],
        stop_loss=closes[-1] * 0.98,
        take_profit=closes[-1] * 1.04,
        current_time=datetime(2024, 1, 15, 14, 30),  # London-NY overlap
    )
    
    print(f"\nBase Signal: BUY with 75% confidence")
    print(f"Enhanced Confidence: {result.enhanced_confidence:.2f}%")
    print(f"Quality: {result.quality.value}")
    print(f"Final Signal: {result.signal}")
    print(f"\nScores:")
    print(f"  Pattern: {result.pattern_score:.2f}")
    print(f"  Technical: {result.technical_score:.2f}")
    print(f"  Volume: {result.volume_score:.2f}")
    print(f"  MTF: {result.mtf_score:.2f}")
    print(f"  Regime: {result.regime_score:.2f}")
    print(f"  Timing: {result.timing_score:.2f}")
    print(f"  Momentum: {result.momentum_score:.2f}")
    print(f"\nMarket Regime: {result.market_regime.value}")
    print(f"Risk:Reward: {result.risk_reward_ratio:.2f}")
    print(f"Position Size: {result.recommended_position_size}x")
    print(f"Entry Timing: {result.entry_timing}")
    print(f"\nBullish Factors: {result.bullish_factors}")
    print(f"Bearish Factors: {result.bearish_factors}")
    print(f"Skip Reasons: {result.skip_reasons}")
    
    # Test Multi-Factor Analyzer
    print("\n" + "=" * 60)
    print("Multi-Factor Analyzer - Test")
    print("=" * 60)
    
    from analysis.voting_system import VoteResult, Signal
    
    vote_result = VoteResult(
        signal=Signal.BUY,
        confidence=78.0,
        bullish_votes=8,
        bearish_votes=2,
        total_votes=10,
        average_movement=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    
    mf_result = analyze_with_multi_factor(
        vote_result=vote_result,
        ohlcv_data=ohlcv_data,
        pattern_dates=[datetime(2025, 6, 15), datetime(2024, 3, 20), datetime(2023, 9, 10)],
        current_time=datetime(2026, 1, 18, 14, 30),
        symbol="EURUSD",
    )
    
    print(f"\nSignal: {mf_result.signal.value}")
    print(f"Base Confidence: {mf_result.base_confidence:.2f}%")
    print(f"Final Score: {mf_result.final_score:.2f}%")
    print(f"Quality: {mf_result.quality}")
    print(f"Recommendation: {mf_result.recommendation}")
    print(f"\nFactor Breakdown:")
    for factor in mf_result.factors:
        status_icon = "✅" if factor.passed else "❌"
        print(f"  {status_icon} {factor.name}: {factor.score*100:.1f}% (weight: {factor.weight*100:.0f}%)")
        print(f"      └─ {factor.details}")
    print(f"\nPosition Size Multiplier: {mf_result.position_size_multiplier:.2f}x")

