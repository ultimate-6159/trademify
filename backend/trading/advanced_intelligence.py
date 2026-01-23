"""
Advanced Intelligence Module - สมองขั้นสูง
ทำให้ Bot ฉลาดเหมือน Hedge Fund

Features:
1. Market Regime Detection - รู้ว่าตลาด Trending/Ranging/Volatile
2. Multi-Timeframe Confirmation - ยืนยันกับ TF ใหญ่
3. Momentum Scanner - RSI + MACD + Stochastic รวมกัน
4. Support/Resistance Finder - หา S/R อัตโนมัติ
5. Breakout Detector - จับ breakout
6. Kelly Criterion - Position size ที่ optimal
7. Volatility Adjuster - ปรับ size ตาม volatility
8. Confluence Score - นับว่ากี่ปัจจัยเห็นด้วย
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# =====================
# Market Regime Detection
# =====================

class MarketRegime(Enum):
    """สภาพตลาด"""
    STRONG_UPTREND = "strong_uptrend"      # แนวโน้มขึ้นแรง
    WEAK_UPTREND = "weak_uptrend"          # แนวโน้มขึ้นอ่อน
    RANGING = "ranging"                     # Sideway
    WEAK_DOWNTREND = "weak_downtrend"      # แนวโน้มลงอ่อน
    STRONG_DOWNTREND = "strong_downtrend"  # แนวโน้มลงแรง
    HIGH_VOLATILITY = "high_volatility"    # ผันผวนมาก
    LOW_VOLATILITY = "low_volatility"      # นิ่งมาก


@dataclass
class RegimeInfo:
    """ข้อมูล Market Regime"""
    regime: MarketRegime
    confidence: float  # 0-100%
    trend_strength: float  # -100 to +100 (negative = downtrend)
    volatility_percentile: float  # 0-100 (current vs historical)
    atr_value: float
    message: str
    
    # Trading guidance
    recommended_strategy: str  # "trend_follow", "mean_revert", "breakout", "wait"
    position_size_factor: float  # 0.5-1.5
    
    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 1),
            "trend_strength": round(self.trend_strength, 1),
            "volatility_percentile": round(self.volatility_percentile, 1),
            "atr_value": round(self.atr_value, 5),
            "message": self.message,
            "recommended_strategy": self.recommended_strategy,
            "position_size_factor": round(self.position_size_factor, 2),
        }


class MarketRegimeDetector:
    """
    ตรวจจับสภาพตลาด
    
    ใช้:
    - ADX (Average Directional Index) - วัดความแรงของ trend
    - ATR (Average True Range) - วัด volatility
    - Moving Average slope - ทิศทาง trend
    - Bollinger Band width - วัด volatility
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        ma_period: int = 50,
        volatility_lookback: int = 100,
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.ma_period = ma_period
        self.volatility_lookback = volatility_lookback
    
    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> RegimeInfo:
        """ตรวจจับ market regime จาก OHLC data"""
        if len(closes) < self.volatility_lookback:
            return RegimeInfo(
                regime=MarketRegime.RANGING,
                confidence=50,
                trend_strength=0,
                volatility_percentile=50,
                atr_value=0,
                message="Not enough data",
                recommended_strategy="wait",
                position_size_factor=0.5,
            )
        
        # Calculate indicators
        adx = self._calculate_adx(highs, lows, closes)
        atr = self._calculate_atr(highs, lows, closes)
        ma_slope = self._calculate_ma_slope(closes)
        volatility_pct = self._calculate_volatility_percentile(atr)
        
        # Determine regime
        regime, confidence, message, strategy = self._classify_regime(
            adx, ma_slope, volatility_pct
        )
        
        # Position size factor
        size_factor = self._get_size_factor(regime, volatility_pct)
        
        return RegimeInfo(
            regime=regime,
            confidence=confidence,
            trend_strength=ma_slope * 100,  # Convert to -100 to +100
            volatility_percentile=volatility_pct,
            atr_value=atr[-1] if len(atr) > 0 else 0,
            message=message,
            recommended_strategy=strategy,
            position_size_factor=size_factor,
        )
    
    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> float:
        """Calculate ADX (Average Directional Index)"""
        n = self.adx_period
        
        if len(closes) < n + 1:
            return 25  # Default neutral
        
        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        
        # +DM and -DM
        plus_dm = np.where(
            (highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
            np.maximum(highs[1:] - highs[:-1], 0),
            0
        )
        minus_dm = np.where(
            (lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
            np.maximum(lows[:-1] - lows[1:], 0),
            0
        )
        
        # Smoothed averages
        atr = self._ema(tr, n)
        plus_di = 100 * self._ema(plus_dm, n) / (atr + 1e-10)
        minus_di = 100 * self._ema(minus_dm, n) / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, n)
        
        return float(adx[-1]) if len(adx) > 0 else 25
    
    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """Calculate ATR (Average True Range)"""
        if len(closes) < 2:
            return np.array([0])
        
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        
        atr = self._ema(tr, self.atr_period)
        return atr
    
    def _calculate_ma_slope(self, closes: np.ndarray) -> float:
        """Calculate MA slope (-1 to +1)"""
        if len(closes) < self.ma_period + 10:
            return 0
        
        # Simple MA
        ma = np.convolve(closes, np.ones(self.ma_period) / self.ma_period, mode='valid')
        
        if len(ma) < 10:
            return 0
        
        # Slope of last 10 MA values (normalized)
        recent_ma = ma[-10:]
        slope = (recent_ma[-1] - recent_ma[0]) / (recent_ma[0] + 1e-10)
        
        # Clamp to -1 to +1
        return max(-1, min(1, slope * 10))
    
    def _calculate_volatility_percentile(self, atr: np.ndarray) -> float:
        """Calculate current volatility percentile (0-100)"""
        if len(atr) < 20:
            return 50
        
        current_atr = atr[-1]
        historical_atr = atr[-self.volatility_lookback:] if len(atr) >= self.volatility_lookback else atr
        
        # Percentile rank
        percentile = (np.sum(historical_atr < current_atr) / len(historical_atr)) * 100
        return float(percentile)
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        if len(data) < period:
            return data
        
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _classify_regime(
        self,
        adx: float,
        ma_slope: float,
        volatility_pct: float,
    ) -> Tuple[MarketRegime, float, str, str]:
        """Classify market regime"""
        
        # High volatility override
        if volatility_pct > 85:
            return (
                MarketRegime.HIGH_VOLATILITY,
                90,
                "⚠️ Volatility สูงมาก! ระวัง",
                "wait",
            )
        
        # Low volatility
        if volatility_pct < 15:
            return (
                MarketRegime.LOW_VOLATILITY,
                80,
                "😴 ตลาดนิ่ง รอ breakout",
                "breakout",
            )
        
        # Strong trend (ADX > 25)
        if adx > 30:
            if ma_slope > 0.3:
                return (
                    MarketRegime.STRONG_UPTREND,
                    min(95, 70 + adx),
                    "🚀 Uptrend แรง! BUY only",
                    "trend_follow",
                )
            elif ma_slope < -0.3:
                return (
                    MarketRegime.STRONG_DOWNTREND,
                    min(95, 70 + adx),
                    "📉 Downtrend แรง! SELL only",
                    "trend_follow",
                )
        
        # Weak trend (ADX 20-30)
        if adx > 20:
            if ma_slope > 0.1:
                return (
                    MarketRegime.WEAK_UPTREND,
                    65,
                    "📈 Uptrend อ่อน, ระวัง pullback",
                    "trend_follow",
                )
            elif ma_slope < -0.1:
                return (
                    MarketRegime.WEAK_DOWNTREND,
                    65,
                    "📉 Downtrend อ่อน, ระวัง pullback",
                    "trend_follow",
                )
        
        # Ranging market (ADX < 20)
        return (
            MarketRegime.RANGING,
            70,
            "↔️ Sideway - เทรด S/R หรือรอ breakout",
            "mean_revert",
        )
    
    def _get_size_factor(self, regime: MarketRegime, volatility_pct: float) -> float:
        """Get position size factor based on regime"""
        base_factors = {
            MarketRegime.STRONG_UPTREND: 1.2,
            MarketRegime.STRONG_DOWNTREND: 1.2,
            MarketRegime.WEAK_UPTREND: 1.0,
            MarketRegime.WEAK_DOWNTREND: 1.0,
            MarketRegime.RANGING: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.LOW_VOLATILITY: 0.7,
        }
        
        factor = base_factors.get(regime, 1.0)
        
        # Reduce size in high volatility
        if volatility_pct > 70:
            factor *= 0.8
        
        return factor


# =====================
# Multi-Timeframe Analyzer
# =====================

@dataclass
class MTFAnalysis:
    """Multi-Timeframe Analysis Result"""
    h1_trend: str  # "up", "down", "neutral"
    h4_trend: str
    d1_trend: str
    alignment: str  # "aligned_up", "aligned_down", "mixed"
    alignment_score: float  # 0-100
    message: str
    can_trade: bool
    
    def to_dict(self) -> dict:
        return {
            "h1_trend": self.h1_trend,
            "h4_trend": self.h4_trend,
            "d1_trend": self.d1_trend,
            "alignment": self.alignment,
            "alignment_score": round(self.alignment_score, 1),
            "message": self.message,
            "can_trade": self.can_trade,
        }


class MultiTimeframeAnalyzer:
    """
    วิเคราะห์หลาย Timeframe
    
    กฎ: Trade only when higher TF agrees
    - H1 signal + H4 same direction + D1 not against = Strong
    - H1 signal + H4 against = Weak/Skip
    """
    
    def __init__(self, ma_fast: int = 20, ma_slow: int = 50):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
    
    def analyze(
        self,
        h1_closes: np.ndarray,
        h4_closes: np.ndarray,
        d1_closes: np.ndarray,
    ) -> MTFAnalysis:
        """วิเคราะห์ทุก TF"""
        h1_trend = self._get_trend(h1_closes)
        h4_trend = self._get_trend(h4_closes)
        d1_trend = self._get_trend(d1_closes)
        
        # Check alignment
        trends = [h1_trend, h4_trend, d1_trend]
        up_count = trends.count("up")
        down_count = trends.count("down")
        
        if up_count >= 2 and down_count == 0:
            alignment = "aligned_up"
            score = 80 + (up_count - 2) * 10
            message = "✅ ทุก TF เห็นด้วย BUY"
            can_trade = True
        elif down_count >= 2 and up_count == 0:
            alignment = "aligned_down"
            score = 80 + (down_count - 2) * 10
            message = "✅ ทุก TF เห็นด้วย SELL"
            can_trade = True
        elif h1_trend != "neutral" and h4_trend == h1_trend:
            alignment = f"partial_{h1_trend}"
            score = 60
            message = f"⚠️ H1+H4 {h1_trend}, D1 ไม่เห็นด้วย"
            can_trade = True
        elif h1_trend != "neutral":
            # ผ่อนปรน: อนุญาตถ้า H1 มีทิศทางชัดเจน
            alignment = f"h1_{h1_trend}"
            score = 45
            message = f"⚠️ H1 {h1_trend} เท่านั้น - ระวัง"
            can_trade = True  # Trial mode: อนุญาตให้เทรด
        else:
            alignment = "mixed"
            score = 30
            message = "⚠️ TF ไม่ชัดเจน - ใช้ความระวัง"
            can_trade = True  # Trial mode: อนุญาตให้เทรด
        
        return MTFAnalysis(
            h1_trend=h1_trend,
            h4_trend=h4_trend,
            d1_trend=d1_trend,
            alignment=alignment,
            alignment_score=score,
            message=message,
            can_trade=can_trade,
        )
    
    def _get_trend(self, closes: np.ndarray) -> str:
        """Get trend from closes"""
        if len(closes) < self.ma_slow + 5:
            return "neutral"
        
        # Calculate MAs
        ma_fast = np.mean(closes[-self.ma_fast:])
        ma_slow = np.mean(closes[-self.ma_slow:])
        current_price = closes[-1]
        
        # Price above both MAs = uptrend
        if current_price > ma_fast > ma_slow:
            return "up"
        elif current_price < ma_fast < ma_slow:
            return "down"
        else:
            return "neutral"


# =====================
# Momentum Scanner
# =====================

@dataclass
class MomentumScore:
    """Momentum Analysis Result"""
    rsi: float
    macd_signal: str  # "bullish", "bearish", "neutral"
    stoch_signal: str
    combined_score: float  # -100 to +100
    momentum_state: str  # "oversold", "neutral", "overbought"
    message: str
    
    def to_dict(self) -> dict:
        return {
            "rsi": round(self.rsi, 1),
            "macd_signal": self.macd_signal,
            "stoch_signal": self.stoch_signal,
            "combined_score": round(self.combined_score, 1),
            "momentum_state": self.momentum_state,
            "message": self.message,
        }


class MomentumScanner:
    """
    สแกน Momentum จากหลาย indicators
    
    รวม:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Stochastic
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def scan(self, closes: np.ndarray, highs: np.ndarray = None, lows: np.ndarray = None) -> MomentumScore:
        """Scan momentum from price data"""
        if len(closes) < 50:
            return MomentumScore(
                rsi=50,
                macd_signal="neutral",
                stoch_signal="neutral",
                combined_score=0,
                momentum_state="neutral",
                message="Not enough data",
            )
        
        # Calculate indicators
        rsi = self._calculate_rsi(closes)
        macd_signal, macd_score = self._calculate_macd(closes)
        
        if highs is not None and lows is not None:
            stoch_signal, stoch_score = self._calculate_stochastic(highs, lows, closes)
        else:
            stoch_signal, stoch_score = "neutral", 0
        
        # Combined score (-100 to +100)
        rsi_score = (rsi - 50) * 2  # -100 to +100
        combined = (rsi_score * 0.4 + macd_score * 0.4 + stoch_score * 0.2)
        
        # Determine state
        if rsi < self.rsi_oversold:
            state = "oversold"
            message = f"📉 Oversold (RSI={rsi:.0f}) - หา BUY"
        elif rsi > self.rsi_overbought:
            state = "overbought"
            message = f"📈 Overbought (RSI={rsi:.0f}) - หา SELL"
        else:
            state = "neutral"
            message = f"↔️ Neutral zone (RSI={rsi:.0f})"
        
        return MomentumScore(
            rsi=rsi,
            macd_signal=macd_signal,
            stoch_signal=stoch_signal,
            combined_score=combined,
            momentum_state=state,
            message=message,
        )
    
    def _calculate_rsi(self, closes: np.ndarray) -> float:
        """Calculate RSI"""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, closes: np.ndarray) -> Tuple[str, float]:
        """Calculate MACD signal"""
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(macd_line, 9)
        
        macd_current = macd_line[-1]
        signal_current = signal_line[-1]
        macd_prev = macd_line[-2] if len(macd_line) > 1 else macd_current
        signal_prev = signal_line[-2] if len(signal_line) > 1 else signal_current
        
        # Crossover detection
        if macd_prev <= signal_prev and macd_current > signal_current:
            return "bullish", 50  # Bullish crossover
        elif macd_prev >= signal_prev and macd_current < signal_current:
            return "bearish", -50  # Bearish crossover
        elif macd_current > signal_current:
            return "bullish", 25
        elif macd_current < signal_current:
            return "bearish", -25
        else:
            return "neutral", 0
    
    def _calculate_stochastic(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[str, float]:
        """Calculate Stochastic"""
        if len(closes) < k_period:
            return "neutral", 0
        
        lowest_low = np.min(lows[-k_period:])
        highest_high = np.max(highs[-k_period:])
        
        if highest_high == lowest_low:
            k = 50
        else:
            k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        if k < 20:
            return "oversold", 40
        elif k > 80:
            return "overbought", -40
        else:
            return "neutral", 0
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        return ema


# =====================
# Support/Resistance Finder
# =====================

@dataclass
class SRLevel:
    """Support/Resistance Level"""
    price: float
    level_type: str  # "support", "resistance"
    strength: int  # Number of touches
    last_touch: int  # Candles ago
    
    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 5),
            "type": self.level_type,
            "strength": self.strength,
            "last_touch": self.last_touch,
        }


class SupportResistanceFinder:
    """
    หา Support/Resistance อัตโนมัติ
    
    วิธี: หา Swing High/Low แล้ว cluster ใกล้กัน
    """
    
    def __init__(
        self,
        lookback: int = 100,
        swing_strength: int = 5,
        cluster_threshold: float = 0.001,  # 0.1% clustering
    ):
        self.lookback = lookback
        self.swing_strength = swing_strength
        self.cluster_threshold = cluster_threshold
    
    def find_levels(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> List[SRLevel]:
        """Find S/R levels"""
        if len(closes) < self.lookback:
            return []
        
        # Use recent data
        highs = highs[-self.lookback:]
        lows = lows[-self.lookback:]
        closes = closes[-self.lookback:]
        current_price = closes[-1]
        
        # Find swing points
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)
        
        # Cluster swing points
        resistance_levels = self._cluster_levels(swing_highs, current_price, "resistance")
        support_levels = self._cluster_levels(swing_lows, current_price, "support")
        
        # Combine and sort by distance from current price
        all_levels = resistance_levels + support_levels
        all_levels.sort(key=lambda x: abs(x.price - current_price))
        
        return all_levels[:10]  # Top 10 nearest levels
    
    def _find_swing_highs(self, highs: np.ndarray) -> List[Tuple[int, float]]:
        """Find swing high points"""
        swings = []
        n = self.swing_strength
        
        for i in range(n, len(highs) - n):
            if highs[i] == max(highs[i-n:i+n+1]):
                swings.append((i, highs[i]))
        
        return swings
    
    def _find_swing_lows(self, lows: np.ndarray) -> List[Tuple[int, float]]:
        """Find swing low points"""
        swings = []
        n = self.swing_strength
        
        for i in range(n, len(lows) - n):
            if lows[i] == min(lows[i-n:i+n+1]):
                swings.append((i, lows[i]))
        
        return swings
    
    def _cluster_levels(
        self,
        swings: List[Tuple[int, float]],
        current_price: float,
        level_type: str,
    ) -> List[SRLevel]:
        """Cluster nearby swing points into S/R levels"""
        if not swings:
            return []
        
        # Sort by price
        swings.sort(key=lambda x: x[1])
        
        clusters = []
        current_cluster = [swings[0]]
        
        for i in range(1, len(swings)):
            price_diff = abs(swings[i][1] - current_cluster[-1][1]) / current_cluster[-1][1]
            
            if price_diff < self.cluster_threshold:
                current_cluster.append(swings[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [swings[i]]
        
        clusters.append(current_cluster)
        
        # Convert clusters to levels
        levels = []
        for cluster in clusters:
            avg_price = np.mean([s[1] for s in cluster])
            strength = len(cluster)
            last_touch = len(swings) - max(s[0] for s in cluster)
            
            levels.append(SRLevel(
                price=avg_price,
                level_type=level_type,
                strength=strength,
                last_touch=last_touch,
            ))
        
        return levels
    
    def get_nearest_sr(
        self,
        current_price: float,
        levels: List[SRLevel],
    ) -> Tuple[Optional[SRLevel], Optional[SRLevel]]:
        """Get nearest support and resistance"""
        supports = [l for l in levels if l.level_type == "support" and l.price < current_price]
        resistances = [l for l in levels if l.level_type == "resistance" and l.price > current_price]
        
        nearest_support = max(supports, key=lambda x: x.price) if supports else None
        nearest_resistance = min(resistances, key=lambda x: x.price) if resistances else None
        
        return nearest_support, nearest_resistance


# =====================
# Kelly Criterion Position Sizing
# =====================

class KellyCalculator:
    """
    Kelly Criterion - Position Size ที่ Optimal
    
    Formula: f* = (bp - q) / b
    Where:
    - f* = fraction of bankroll to bet
    - b = odds received (reward/risk ratio)
    - p = probability of winning
    - q = probability of losing (1-p)
    
    ใช้ Half Kelly หรือ Quarter Kelly เพื่อความปลอดภัย
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Quarter Kelly (conservative)
        min_trades_required: int = 20,
        max_position_size: float = 5.0,  # Max 5% of account
    ):
        self.kelly_fraction = kelly_fraction
        self.min_trades_required = min_trades_required
        self.max_position_size = max_position_size
    
    def calculate(
        self,
        win_rate: float,  # 0-1
        avg_win: float,   # Average winning trade %
        avg_loss: float,  # Average losing trade % (positive number)
        total_trades: int,
    ) -> Tuple[float, str]:
        """
        Calculate optimal position size
        
        Returns: (position_size_percent, explanation)
        """
        # Not enough data
        if total_trades < self.min_trades_required:
            return (
                1.0,
                f"ยังไม่พอ (ต้องการ {self.min_trades_required} trades, มี {total_trades})"
            )
        
        # Invalid data
        if avg_loss <= 0 or avg_win <= 0:
            return (1.0, "Invalid avg win/loss data")
        
        # Kelly calculation
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss  # Reward/Risk ratio
        
        # Kelly formula
        kelly = (b * p - q) / b
        
        # Negative Kelly means edge is negative (don't trade!)
        if kelly <= 0:
            return (
                0,
                f"❌ Negative edge! Win={win_rate*100:.0f}%, RR={b:.2f}"
            )
        
        # Apply fraction (Quarter/Half Kelly)
        position_size = kelly * self.kelly_fraction * 100
        
        # Cap at maximum
        position_size = min(position_size, self.max_position_size)
        
        return (
            position_size,
            f"Kelly={kelly*100:.1f}% × {self.kelly_fraction} = {position_size:.1f}%"
        )


# =====================
# Confluence Score
# =====================

@dataclass
class ConfluenceResult:
    """Confluence Analysis Result"""
    total_factors: int
    agreeing_factors: int
    score: float  # 0-100
    factors: Dict[str, bool]  # Which factors agree
    recommendation: str  # "strong_buy", "buy", "neutral", "sell", "strong_sell"
    can_trade: bool
    message: str
    
    def to_dict(self) -> dict:
        return {
            "total_factors": self.total_factors,
            "agreeing_factors": self.agreeing_factors,
            "score": round(self.score, 1),
            "factors": self.factors,
            "recommendation": self.recommendation,
            "can_trade": self.can_trade,
            "message": self.message,
        }


class ConfluenceAnalyzer:
    """
    นับว่ากี่ปัจจัยเห็นด้วย
    
    ปัจจัย:
    - Pattern Signal
    - Market Regime
    - MTF Alignment
    - Momentum
    - Near S/R
    - Smart Money
    - Sentiment (Contrarian)
    """
    
    def __init__(self, min_confluence: int = 4):
        self.min_confluence = min_confluence
    
    def analyze(
        self,
        signal_side: str,  # "BUY" or "SELL"
        pattern_confidence: float,
        regime: RegimeInfo = None,
        mtf: MTFAnalysis = None,
        momentum: MomentumScore = None,
        near_sr: bool = False,
        smart_money_agrees: bool = None,
        sentiment_agrees: bool = None,
    ) -> ConfluenceResult:
        """Analyze confluence of all factors"""
        factors = {}
        
        # 1. Pattern Signal
        factors["pattern"] = pattern_confidence >= 70
        
        # 2. Market Regime
        if regime:
            if signal_side == "BUY":
                factors["regime"] = regime.regime in [
                    MarketRegime.STRONG_UPTREND,
                    MarketRegime.WEAK_UPTREND,
                    MarketRegime.RANGING,  # Can buy support in range
                ]
            else:
                factors["regime"] = regime.regime in [
                    MarketRegime.STRONG_DOWNTREND,
                    MarketRegime.WEAK_DOWNTREND,
                    MarketRegime.RANGING,  # Can sell resistance in range
                ]
        
        # 3. MTF Alignment
        if mtf:
            if signal_side == "BUY":
                factors["mtf"] = mtf.alignment in ["aligned_up", "partial_up"]
            else:
                factors["mtf"] = mtf.alignment in ["aligned_down", "partial_down"]
        
        # 4. Momentum
        if momentum:
            if signal_side == "BUY":
                factors["momentum"] = momentum.combined_score > 0 or momentum.momentum_state == "oversold"
            else:
                factors["momentum"] = momentum.combined_score < 0 or momentum.momentum_state == "overbought"
        
        # 5. Near S/R
        factors["sr_level"] = near_sr
        
        # 6. Smart Money
        if smart_money_agrees is not None:
            factors["smart_money"] = smart_money_agrees
        
        # 7. Sentiment
        if sentiment_agrees is not None:
            factors["sentiment"] = sentiment_agrees
        
        # Calculate score
        total = len(factors)
        agreeing = sum(1 for v in factors.values() if v)
        score = (agreeing / total) * 100 if total > 0 else 0
        
        # Recommendation - ผ่อนปรนสำหรับ trial mode
        if agreeing >= 5:
            rec = f"strong_{signal_side.lower()}"
            can_trade = True
            message = f"🎯 Confluence สูง! {agreeing}/{total} ปัจจัยเห็นด้วย"
        elif agreeing >= self.min_confluence:
            rec = signal_side.lower()
            can_trade = True
            message = f"✅ Confluence พอใช้ได้ {agreeing}/{total}"
        elif agreeing >= 2:
            rec = "weak_" + signal_side.lower()
            can_trade = True  # Trial mode: อนุญาตให้เทรด
            message = f"⚠️ Confluence ต่ำ {agreeing}/{total} - ระวัง"
        else:
            rec = "very_weak_" + signal_side.lower()
            can_trade = True  # Trial mode: อนุญาตให้เทรด
            message = f"⚠️ Confluence ต่ำมาก {agreeing}/{total} - ใช้ position เล็ก"
        
        return ConfluenceResult(
            total_factors=total,
            agreeing_factors=agreeing,
            score=score,
            factors=factors,
            recommendation=rec,
            can_trade=can_trade,
            message=message,
        )


# =====================
# Master Intelligence Class
# =====================

@dataclass
class IntelligenceDecision:
    """ผลการตัดสินใจจาก Advanced Intelligence"""
    can_trade: bool
    confidence: float
    position_size_factor: float
    
    # Analysis results
    regime: Optional[RegimeInfo] = None
    mtf: Optional[MTFAnalysis] = None
    momentum: Optional[MomentumScore] = None
    sr_levels: List[SRLevel] = field(default_factory=list)
    confluence: Optional[ConfluenceResult] = None
    kelly_size: float = 1.0
    
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "can_trade": self.can_trade,
            "confidence": round(self.confidence, 1),
            "position_size_factor": round(self.position_size_factor, 2),
            "regime": self.regime.to_dict() if self.regime else None,
            "mtf": self.mtf.to_dict() if self.mtf else None,
            "momentum": self.momentum.to_dict() if self.momentum else None,
            "sr_levels": [l.to_dict() for l in self.sr_levels[:5]],
            "confluence": self.confluence.to_dict() if self.confluence else None,
            "kelly_size": round(self.kelly_size, 2),
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


class AdvancedIntelligence:
    """
    🧠 Advanced Intelligence - สมองขั้นสูง
    
    รวมทุก feature:
    - Market Regime Detection
    - Multi-Timeframe Analysis
    - Momentum Scanning
    - S/R Detection
    - Kelly Position Sizing
    - Confluence Scoring
    """
    
    def __init__(
        self,
        enable_regime: bool = True,
        enable_mtf: bool = True,
        enable_momentum: bool = True,
        enable_sr: bool = True,
        enable_kelly: bool = True,
        min_confluence: int = 3,
    ):
        self.enable_regime = enable_regime
        self.enable_mtf = enable_mtf
        self.enable_momentum = enable_momentum
        self.enable_sr = enable_sr
        self.enable_kelly = enable_kelly
        
        # Initialize analyzers
        self.regime_detector = MarketRegimeDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.momentum_scanner = MomentumScanner()
        self.sr_finder = SupportResistanceFinder()
        self.kelly = KellyCalculator()
        self.confluence = ConfluenceAnalyzer(min_confluence=min_confluence)
        
        logger.info("🧠 Advanced Intelligence initialized")
        logger.info(f"   - Regime Detection: {enable_regime}")
        logger.info(f"   - Multi-Timeframe: {enable_mtf}")
        logger.info(f"   - Momentum Scanner: {enable_momentum}")
        logger.info(f"   - S/R Detection: {enable_sr}")
        logger.info(f"   - Kelly Sizing: {enable_kelly}")
    
    def analyze(
        self,
        signal_side: str,
        pattern_confidence: float,
        h1_data: Dict[str, np.ndarray],  # {"open", "high", "low", "close"}
        h4_data: Dict[str, np.ndarray] = None,
        d1_data: Dict[str, np.ndarray] = None,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        total_trades: int = 0,
        smart_money_agrees: bool = None,
        sentiment_agrees: bool = None,
    ) -> IntelligenceDecision:
        """
        วิเคราะห์ครบทุกมิติ
        
        Args:
            signal_side: "BUY" or "SELL"
            pattern_confidence: Confidence from pattern matching
            h1_data: H1 OHLC data
            h4_data: H4 OHLC data (optional)
            d1_data: D1 OHLC data (optional)
            win_rate: Historical win rate
            avg_win: Average winning trade %
            avg_loss: Average losing trade %
            total_trades: Total number of trades
            smart_money_agrees: Smart Money signal agrees
            sentiment_agrees: Sentiment (contrarian) agrees
        """
        reasons = []
        warnings = []
        position_factor = 1.0
        
        # Extract H1 data
        h1_open = h1_data.get("open", np.array([]))
        h1_high = h1_data.get("high", np.array([]))
        h1_low = h1_data.get("low", np.array([]))
        h1_close = h1_data.get("close", np.array([]))
        
        # 1. Market Regime
        regime = None
        if self.enable_regime and len(h1_close) > 50:
            regime = self.regime_detector.detect(h1_high, h1_low, h1_close)
            
            if regime.regime == MarketRegime.HIGH_VOLATILITY:
                warnings.append("⚠️ Volatility สูง!")
                position_factor *= regime.position_size_factor
            elif regime.regime == MarketRegime.STRONG_UPTREND and signal_side == "BUY":
                reasons.append("✅ Strong Uptrend - BUY aligned")
            elif regime.regime == MarketRegime.STRONG_DOWNTREND and signal_side == "SELL":
                reasons.append("✅ Strong Downtrend - SELL aligned")
            
            position_factor *= regime.position_size_factor
        
        # 2. Multi-Timeframe
        mtf = None
        if self.enable_mtf and h4_data and d1_data:
            h4_close = h4_data.get("close", np.array([]))
            d1_close = d1_data.get("close", np.array([]))
            
            if len(h4_close) > 50 and len(d1_close) > 50:
                mtf = self.mtf_analyzer.analyze(h1_close, h4_close, d1_close)
                
                if mtf.can_trade:
                    reasons.append(f"✅ MTF Aligned: {mtf.alignment}")
                else:
                    warnings.append(f"⚠️ MTF Mixed: {mtf.message}")
        
        # 3. Momentum
        momentum = None
        if self.enable_momentum and len(h1_close) > 50:
            momentum = self.momentum_scanner.scan(h1_close, h1_high, h1_low)
            
            if signal_side == "BUY" and momentum.momentum_state == "oversold":
                reasons.append(f"✅ Oversold - Good for BUY")
            elif signal_side == "SELL" and momentum.momentum_state == "overbought":
                reasons.append(f"✅ Overbought - Good for SELL")
        
        # 4. S/R Levels
        sr_levels = []
        near_sr = False
        if self.enable_sr and len(h1_close) > 50:
            sr_levels = self.sr_finder.find_levels(h1_high, h1_low, h1_close)
            
            current_price = h1_close[-1]
            support, resistance = self.sr_finder.get_nearest_sr(current_price, sr_levels)
            
            if signal_side == "BUY" and support:
                dist = (current_price - support.price) / current_price
                if dist < 0.005:  # Within 0.5%
                    reasons.append(f"✅ Near Support ({support.price:.5f})")
                    near_sr = True
            
            if signal_side == "SELL" and resistance:
                dist = (resistance.price - current_price) / current_price
                if dist < 0.005:
                    reasons.append(f"✅ Near Resistance ({resistance.price:.5f})")
                    near_sr = True
        
        # 5. Kelly Sizing
        kelly_size = 1.0
        if self.enable_kelly and total_trades >= 10:
            kelly_size, kelly_msg = self.kelly.calculate(
                win_rate, avg_win, avg_loss, total_trades
            )
            if kelly_size <= 0:
                warnings.append(f"⚠️ Kelly says no edge!")
            else:
                reasons.append(f"📊 Kelly: {kelly_msg}")
        
        # 6. Confluence
        confluence_result = self.confluence.analyze(
            signal_side=signal_side,
            pattern_confidence=pattern_confidence,
            regime=regime,
            mtf=mtf,
            momentum=momentum,
            near_sr=near_sr,
            smart_money_agrees=smart_money_agrees,
            sentiment_agrees=sentiment_agrees,
        )
        
        # Final decision
        can_trade = confluence_result.can_trade
        
        # Override: Kelly warning only (ไม่บล็อก ใน trial mode)
        if kelly_size <= 0:
            # can_trade = False  # Disabled for trial
            warnings.append("⚠️ Kelly says no edge - use small size")
        
        # Override: High volatility warning only (ไม่บล็อก ใน trial mode)
        if regime and regime.regime == MarketRegime.HIGH_VOLATILITY:
            if confluence_result.agreeing_factors < 5:
                # can_trade = False  # Disabled for trial
                warnings.append("⚠️ High volatility + low confluence - reduce size")
        
        # Calculate final position factor
        final_factor = position_factor
        if kelly_size > 0:
            final_factor *= min(kelly_size / 2, 1.5)  # Cap Kelly influence
        
        return IntelligenceDecision(
            can_trade=can_trade,
            confidence=confluence_result.score,
            position_size_factor=final_factor,
            regime=regime,
            mtf=mtf,
            momentum=momentum,
            sr_levels=sr_levels,
            confluence=confluence_result,
            kelly_size=kelly_size,
            reasons=reasons,
            warnings=warnings,
        )


# Singleton
_intelligence: Optional[AdvancedIntelligence] = None


def get_intelligence() -> AdvancedIntelligence:
    """Get Advanced Intelligence singleton"""
    global _intelligence
    if _intelligence is None:
        _intelligence = AdvancedIntelligence()
    return _intelligence


# Test
if __name__ == "__main__":
    import random
    
    print("=" * 60)
    print("  ADVANCED INTELLIGENCE TEST")
    print("=" * 60)
    
    # Generate fake OHLC data
    np.random.seed(42)
    n = 200
    
    # Simulated uptrend
    base = 1.1000
    trend = np.linspace(0, 0.01, n)
    noise = np.random.randn(n) * 0.001
    closes = base + trend + noise
    highs = closes + np.abs(np.random.randn(n) * 0.0005)
    lows = closes - np.abs(np.random.randn(n) * 0.0005)
    opens = closes - np.random.randn(n) * 0.0003
    
    h1_data = {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    }
    
    # Test
    intel = AdvancedIntelligence()
    
    result = intel.analyze(
        signal_side="BUY",
        pattern_confidence=80,
        h1_data=h1_data,
        win_rate=0.55,
        avg_win=1.5,
        avg_loss=1.0,
        total_trades=30,
        smart_money_agrees=True,
    )
    
    print("\n📊 Analysis Result:")
    print(f"   Can Trade: {result.can_trade}")
    print(f"   Confidence: {result.confidence:.1f}%")
    print(f"   Position Factor: {result.position_size_factor:.2f}x")
    
    if result.regime:
        print(f"\n🌡️ Regime: {result.regime.regime.value}")
        print(f"   {result.regime.message}")
    
    if result.momentum:
        print(f"\n📈 Momentum: {result.momentum.momentum_state}")
        print(f"   RSI: {result.momentum.rsi:.1f}")
    
    if result.confluence:
        print(f"\n🎯 Confluence: {result.confluence.agreeing_factors}/{result.confluence.total_factors}")
        print(f"   {result.confluence.message}")
    
    print("\n✅ Reasons:")
    for r in result.reasons:
        print(f"   {r}")
    
    print("\n⚠️ Warnings:")
    for w in result.warnings:
        print(f"   {w}")
    
    print("\n" + "=" * 60)
    print("  Advanced Intelligence Ready!")
    print("=" * 60)
