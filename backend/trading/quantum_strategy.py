"""
⚛️ Quantum Strategy Module
===========================
Advanced quantitative trading strategies and market analysis.

Components:
1. MicrostructureAnalyzer - Order flow & spread analysis
2. VolatilityRegime - GARCH-like volatility forecasting
3. FractalAnalyzer - Market fractal pattern detection
4. SentimentAggregator - Multi-source sentiment scoring
5. DynamicExitManager - Adaptive SL/TP management
6. QuantumDecisionEngine - Combine all into final decision

Author: Trademify AI
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import math

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATA CLASSES
# ============================================================

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    EXTREMELY_LOW = "extremely_low"   # < 25th percentile
    LOW = "low"                       # 25-40th percentile
    NORMAL = "normal"                 # 40-60th percentile
    HIGH = "high"                     # 60-75th percentile
    EXTREMELY_HIGH = "extremely_high" # > 75th percentile
    EXPLOSIVE = "explosive"           # Sudden spike


class MarketMicrostructure(Enum):
    """Market microstructure state"""
    ACCUMULATION = "accumulation"     # Smart money buying
    DISTRIBUTION = "distribution"     # Smart money selling
    MARKUP = "markup"                 # Price rising phase
    MARKDOWN = "markdown"             # Price falling phase
    RANGING = "ranging"               # No clear direction
    SQUEEZE = "squeeze"               # Volatility squeeze


class FractalType(Enum):
    """Fractal pattern types"""
    BULLISH_FRACTAL = "bullish_fractal"
    BEARISH_FRACTAL = "bearish_fractal"
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    NONE = "none"


class ExitStrategy(Enum):
    """Exit strategy types"""
    FIXED = "fixed"                   # Fixed SL/TP
    TRAILING = "trailing"             # Trailing stop
    CHANDELIER = "chandelier"         # ATR-based trailing
    PARABOLIC = "parabolic"           # SAR-style
    BREAKEVEN_PLUS = "breakeven_plus" # Move to BE + buffer
    SCALE_OUT = "scale_out"           # Partial exits


@dataclass
class MicrostructureData:
    """Market microstructure analysis result"""
    state: MarketMicrostructure
    buy_pressure: float      # 0-1
    sell_pressure: float     # 0-1
    spread_percentile: float # Current spread vs history
    volume_imbalance: float  # Buy vs sell volume
    smart_money_signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float        # 0-1


@dataclass
class VolatilityData:
    """Volatility analysis result"""
    regime: VolatilityRegime
    current_vol: float       # Current volatility
    forecast_vol: float      # Predicted next volatility
    vol_percentile: float    # 0-100
    is_expanding: bool       # Volatility increasing
    is_contracting: bool     # Volatility decreasing
    breakout_probability: float  # 0-1


@dataclass
class FractalData:
    """Fractal analysis result"""
    pattern: FractalType
    fractal_dimension: float # 1-2 (1=trending, 2=random)
    hurst_exponent: float    # <0.5=mean revert, 0.5=random, >0.5=trending
    support_levels: List[float]
    resistance_levels: List[float]
    next_target: Optional[float]
    pattern_confidence: float


@dataclass
class SentimentData:
    """Sentiment analysis result"""
    overall_sentiment: float  # -1 to 1
    trend_sentiment: float    # Based on price trend
    momentum_sentiment: float # Based on momentum
    volume_sentiment: float   # Based on volume
    contrarian_signal: bool   # Extreme sentiment = contrarian
    sentiment_divergence: bool # Price vs sentiment diverge


@dataclass
class ExitPlan:
    """Dynamic exit plan"""
    strategy: ExitStrategy
    initial_stop_loss: float
    current_stop_loss: float
    take_profit_1: float     # First TP (partial)
    take_profit_2: float     # Second TP (partial)
    take_profit_3: float     # Final TP
    trailing_distance: float
    breakeven_trigger: float # Price level to move SL to BE
    scale_out_levels: List[Tuple[float, float]]  # (price, %)


@dataclass
class QuantumDecision:
    """Final quantum strategy decision"""
    should_trade: bool
    direction: str           # "BUY", "SELL", "WAIT"
    confidence: float        # 0-100
    position_multiplier: float  # 0-1.5
    
    # Component data
    microstructure: MicrostructureData
    volatility: VolatilityData
    fractal: FractalData
    sentiment: SentimentData
    exit_plan: Optional[ExitPlan]
    
    # Scores
    quantum_score: float     # -100 to 100
    edge_score: float        # Statistical edge estimate
    risk_reward: float       # Expected R:R ratio
    
    # Warnings & reasons
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


# ============================================================
# MICROSTRUCTURE ANALYZER
# ============================================================

class MicrostructureAnalyzer:
    """
    Analyze market microstructure for smart money detection.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.spread_history: deque = deque(maxlen=lookback)
        self.volume_history: deque = deque(maxlen=lookback)
        self.price_history: deque = deque(maxlen=lookback)
    
    def analyze(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None
    ) -> MicrostructureData:
        """Analyze market microstructure"""
        
        if len(prices) < 20:
            return MicrostructureData(
                state=MarketMicrostructure.RANGING,
                buy_pressure=0.5,
                sell_pressure=0.5,
                spread_percentile=50.0,
                volume_imbalance=0.0,
                smart_money_signal="NEUTRAL",
                confidence=0.0
            )
        
        # Calculate buy/sell pressure using price-volume relationship
        buy_pressure, sell_pressure = self._calculate_pressure(prices, volumes)
        
        # Volume imbalance
        volume_imbalance = buy_pressure - sell_pressure
        
        # Determine microstructure state
        state = self._determine_state(prices, volumes, buy_pressure, sell_pressure)
        
        # Smart money signal
        smart_money_signal = self._detect_smart_money(prices, volumes, state)
        
        # Spread analysis (using high-low as proxy if no bid-ask)
        spread_percentile = 50.0
        if highs is not None and lows is not None:
            spreads = (highs - lows) / prices
            current_spread = spreads[-1]
            spread_percentile = self._percentile_rank(spreads, current_spread)
        
        # Confidence based on volume and consistency
        confidence = min(1.0, np.mean(volumes[-20:]) / (np.mean(volumes) + 1e-10))
        
        return MicrostructureData(
            state=state,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            spread_percentile=spread_percentile,
            volume_imbalance=volume_imbalance,
            smart_money_signal=smart_money_signal,
            confidence=confidence
        )
    
    def _calculate_pressure(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate buying and selling pressure"""
        
        price_changes = np.diff(prices)
        
        # Volume on up moves vs down moves
        up_volume = 0.0
        down_volume = 0.0
        
        for i, change in enumerate(price_changes):
            if i + 1 < len(volumes):
                if change > 0:
                    up_volume += volumes[i + 1]
                elif change < 0:
                    down_volume += volumes[i + 1]
        
        total = up_volume + down_volume + 1e-10
        buy_pressure = up_volume / total
        sell_pressure = down_volume / total
        
        return buy_pressure, sell_pressure
    
    def _determine_state(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        buy_pressure: float,
        sell_pressure: float
    ) -> MarketMicrostructure:
        """Determine market microstructure state"""
        
        # Price trend
        sma = np.mean(prices[-20:])
        price_trend = (prices[-1] - sma) / sma
        
        # Volume trend
        vol_recent = np.mean(volumes[-10:])
        vol_older = np.mean(volumes[-30:-10]) if len(volumes) >= 30 else np.mean(volumes)
        vol_expanding = vol_recent > vol_older * 1.2
        
        # Volatility squeeze
        recent_range = np.max(prices[-20:]) - np.min(prices[-20:])
        older_range = np.max(prices[-50:-20]) - np.min(prices[-50:-20]) if len(prices) >= 50 else recent_range
        is_squeeze = recent_range < older_range * 0.5
        
        if is_squeeze:
            return MarketMicrostructure.SQUEEZE
        elif buy_pressure > 0.6 and price_trend < 0:
            return MarketMicrostructure.ACCUMULATION
        elif sell_pressure > 0.6 and price_trend > 0:
            return MarketMicrostructure.DISTRIBUTION
        elif price_trend > 0.01 and vol_expanding:
            return MarketMicrostructure.MARKUP
        elif price_trend < -0.01 and vol_expanding:
            return MarketMicrostructure.MARKDOWN
        else:
            return MarketMicrostructure.RANGING
    
    def _detect_smart_money(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        state: MarketMicrostructure
    ) -> str:
        """Detect smart money activity"""
        
        if state == MarketMicrostructure.ACCUMULATION:
            return "BULLISH"
        elif state == MarketMicrostructure.DISTRIBUTION:
            return "BEARISH"
        elif state == MarketMicrostructure.MARKUP:
            return "BULLISH"
        elif state == MarketMicrostructure.MARKDOWN:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _percentile_rank(self, data: np.ndarray, value: float) -> float:
        """Calculate percentile rank of value in data"""
        return (np.sum(data < value) / len(data)) * 100


# ============================================================
# VOLATILITY REGIME ANALYZER
# ============================================================

class VolatilityRegimeAnalyzer:
    """
    GARCH-like volatility forecasting and regime detection.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.vol_history: deque = deque(maxlen=lookback)
    
    def analyze(self, prices: np.ndarray) -> VolatilityData:
        """Analyze volatility regime"""
        
        if len(prices) < 30:
            return VolatilityData(
                regime=VolatilityRegime.NORMAL,
                current_vol=0.0,
                forecast_vol=0.0,
                vol_percentile=50.0,
                is_expanding=False,
                is_contracting=False,
                breakout_probability=0.5
            )
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Current volatility (20-period)
        current_vol = np.std(returns[-20:]) * np.sqrt(252)  # Annualized
        
        # Historical volatility for percentile
        all_vol = []
        for i in range(20, len(returns)):
            all_vol.append(np.std(returns[i-20:i]) * np.sqrt(252))
        
        vol_percentile = self._percentile_rank(np.array(all_vol), current_vol) if all_vol else 50.0
        
        # GARCH-like forecast (simplified)
        omega = 0.000001
        alpha = 0.1
        beta = 0.85
        
        # Exponential smoothing of squared returns
        var_forecast = omega + alpha * returns[-1]**2 + beta * (current_vol/np.sqrt(252))**2
        forecast_vol = np.sqrt(var_forecast) * np.sqrt(252)
        
        # Determine regime
        regime = self._determine_regime(vol_percentile, current_vol, forecast_vol)
        
        # Volatility trend
        recent_vol = np.std(returns[-10:]) * np.sqrt(252)
        older_vol = np.std(returns[-30:-10]) * np.sqrt(252) if len(returns) >= 30 else current_vol
        
        is_expanding = recent_vol > older_vol * 1.1
        is_contracting = recent_vol < older_vol * 0.9
        
        # Breakout probability (higher after squeeze)
        breakout_probability = self._calculate_breakout_prob(
            vol_percentile, is_contracting, prices
        )
        
        # Store for history
        self.vol_history.append(current_vol)
        
        return VolatilityData(
            regime=regime,
            current_vol=current_vol,
            forecast_vol=forecast_vol,
            vol_percentile=vol_percentile,
            is_expanding=is_expanding,
            is_contracting=is_contracting,
            breakout_probability=breakout_probability
        )
    
    def _determine_regime(
        self,
        percentile: float,
        current: float,
        forecast: float
    ) -> VolatilityRegime:
        """Determine volatility regime"""
        
        # Check for explosive volatility
        if forecast > current * 1.5:
            return VolatilityRegime.EXPLOSIVE
        
        if percentile < 25:
            return VolatilityRegime.EXTREMELY_LOW
        elif percentile < 40:
            return VolatilityRegime.LOW
        elif percentile < 60:
            return VolatilityRegime.NORMAL
        elif percentile < 75:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREMELY_HIGH
    
    def _calculate_breakout_prob(
        self,
        vol_percentile: float,
        is_contracting: bool,
        prices: np.ndarray
    ) -> float:
        """Calculate probability of volatility breakout"""
        
        prob = 0.5
        
        # Low volatility increases breakout probability
        if vol_percentile < 25:
            prob += 0.2
        elif vol_percentile < 40:
            prob += 0.1
        
        # Contracting volatility increases probability
        if is_contracting:
            prob += 0.15
        
        # Bollinger Band squeeze
        sma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        upper = sma + 2 * std
        lower = sma - 2 * std
        band_width = (upper - lower) / sma
        
        # Narrow bands = higher breakout probability
        if band_width < 0.02:  # Very narrow
            prob += 0.2
        elif band_width < 0.04:
            prob += 0.1
        
        return min(0.95, prob)
    
    def _percentile_rank(self, data: np.ndarray, value: float) -> float:
        return (np.sum(data < value) / len(data)) * 100 if len(data) > 0 else 50.0


# ============================================================
# FRACTAL ANALYZER
# ============================================================

class FractalAnalyzer:
    """
    Analyze market fractals and calculate Hurst exponent.
    """
    
    def __init__(self):
        self.fractal_history: List[FractalData] = []
    
    def analyze(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None
    ) -> FractalData:
        """Analyze fractal patterns"""
        
        if len(prices) < 50:
            return FractalData(
                pattern=FractalType.NONE,
                fractal_dimension=1.5,
                hurst_exponent=0.5,
                support_levels=[],
                resistance_levels=[],
                next_target=None,
                pattern_confidence=0.0
            )
        
        # Use prices for highs/lows if not provided
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        
        # Calculate Hurst exponent
        hurst = self._calculate_hurst(prices)
        
        # Calculate fractal dimension
        fractal_dim = 2 - hurst
        
        # Detect Williams fractals
        bullish_fractals, bearish_fractals = self._detect_williams_fractals(highs, lows)
        
        # Find support/resistance from fractals
        support_levels = sorted(bullish_fractals)[-3:] if bullish_fractals else []
        resistance_levels = sorted(bearish_fractals)[:3] if bearish_fractals else []
        
        # Detect pattern
        pattern, confidence = self._detect_pattern(prices, highs, lows)
        
        # Calculate next target
        next_target = self._calculate_target(prices, support_levels, resistance_levels)
        
        return FractalData(
            pattern=pattern,
            fractal_dimension=fractal_dim,
            hurst_exponent=hurst,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            next_target=next_target,
            pattern_confidence=confidence
        )
    
    def _calculate_hurst(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        
        if len(prices) < max_lag * 2:
            return 0.5
        
        returns = np.diff(np.log(prices))
        
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Calculate R/S for this lag
            chunks = [returns[i:i+lag] for i in range(0, len(returns)-lag, lag)]
            
            rs_values = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(chunk)
                if s > 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        if len(tau) < 3:
            return 0.5
        
        # Fit log-log line
        log_lags = np.log(list(lags)[:len(tau)])
        log_tau = np.log(tau)
        
        try:
            hurst = np.polyfit(log_lags, log_tau, 1)[0]
            return np.clip(hurst, 0, 1)
        except:
            return 0.5
    
    def _detect_williams_fractals(
        self,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Detect Williams fractals"""
        
        bullish = []  # Low point fractals (support)
        bearish = []  # High point fractals (resistance)
        
        for i in range(2, len(highs) - 2):
            # Bearish fractal (high point)
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                bearish.append(highs[i])
            
            # Bullish fractal (low point)
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                bullish.append(lows[i])
        
        return bullish, bearish
    
    def _detect_pattern(
        self,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Tuple[FractalType, float]:
        """Detect chart patterns"""
        
        # Simple pattern detection
        recent_prices = prices[-30:]
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                peaks.append((i, recent_prices[i]))
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                troughs.append((i, recent_prices[i]))
        
        if len(troughs) >= 2:
            # Double bottom
            last_two = troughs[-2:]
            if abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] < 0.02:
                return FractalType.DOUBLE_BOTTOM, 0.7
        
        if len(peaks) >= 2:
            # Double top
            last_two = peaks[-2:]
            if abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] < 0.02:
                return FractalType.DOUBLE_TOP, 0.7
        
        # Head and shoulders (simplified)
        if len(peaks) >= 3:
            p1, p2, p3 = [p[1] for p in peaks[-3:]]
            if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.02:
                return FractalType.HEAD_SHOULDERS, 0.6
        
        if len(troughs) >= 3:
            t1, t2, t3 = [t[1] for t in troughs[-3:]]
            if t2 < t1 and t2 < t3 and abs(t1 - t3) / t1 < 0.02:
                return FractalType.INVERSE_HEAD_SHOULDERS, 0.6
        
        return FractalType.NONE, 0.0
    
    def _calculate_target(
        self,
        prices: np.ndarray,
        supports: List[float],
        resistances: List[float]
    ) -> Optional[float]:
        """Calculate next price target"""
        
        current = prices[-1]
        
        # Find nearest resistance above
        above = [r for r in resistances if r > current]
        if above:
            return min(above)
        
        # Find nearest support below
        below = [s for s in supports if s < current]
        if below:
            return max(below)
        
        return None


# ============================================================
# SENTIMENT AGGREGATOR
# ============================================================

class SentimentAggregator:
    """
    Aggregate sentiment from multiple technical sources.
    """
    
    def __init__(self):
        self.sentiment_history: deque = deque(maxlen=100)
    
    def analyze(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> SentimentData:
        """Aggregate sentiment from technical indicators"""
        
        if len(prices) < 30:
            return SentimentData(
                overall_sentiment=0.0,
                trend_sentiment=0.0,
                momentum_sentiment=0.0,
                volume_sentiment=0.0,
                contrarian_signal=False,
                sentiment_divergence=False
            )
        
        # Trend sentiment
        trend_sentiment = self._calculate_trend_sentiment(prices)
        
        # Momentum sentiment
        momentum_sentiment = self._calculate_momentum_sentiment(prices)
        
        # Volume sentiment
        volume_sentiment = 0.0
        if volumes is not None and len(volumes) > 0:
            volume_sentiment = self._calculate_volume_sentiment(prices, volumes)
        
        # Overall sentiment (weighted)
        overall = (
            trend_sentiment * 0.4 +
            momentum_sentiment * 0.4 +
            volume_sentiment * 0.2
        )
        
        # Contrarian signal (extreme sentiment)
        contrarian_signal = abs(overall) > 0.8
        
        # Divergence detection
        sentiment_divergence = self._detect_divergence(
            prices, trend_sentiment, momentum_sentiment
        )
        
        # Store history
        self.sentiment_history.append(overall)
        
        return SentimentData(
            overall_sentiment=overall,
            trend_sentiment=trend_sentiment,
            momentum_sentiment=momentum_sentiment,
            volume_sentiment=volume_sentiment,
            contrarian_signal=contrarian_signal,
            sentiment_divergence=sentiment_divergence
        )
    
    def _calculate_trend_sentiment(self, prices: np.ndarray) -> float:
        """Calculate sentiment from trend"""
        
        # Multiple MAs
        ma5 = np.mean(prices[-5:])
        ma10 = np.mean(prices[-10:])
        ma20 = np.mean(prices[-20:])
        
        current = prices[-1]
        
        score = 0.0
        
        # Price vs MAs
        if current > ma5:
            score += 0.3
        else:
            score -= 0.3
        
        if current > ma10:
            score += 0.3
        else:
            score -= 0.3
        
        if current > ma20:
            score += 0.4
        else:
            score -= 0.4
        
        # MA alignment
        if ma5 > ma10 > ma20:
            score += 0.3
        elif ma5 < ma10 < ma20:
            score -= 0.3
        
        return np.clip(score, -1, 1)
    
    def _calculate_momentum_sentiment(self, prices: np.ndarray) -> float:
        """Calculate sentiment from momentum"""
        
        score = 0.0
        
        # RSI
        rsi = self._calculate_rsi(prices)
        if rsi > 70:
            score += 0.3  # Bullish but overbought
        elif rsi > 50:
            score += 0.5
        elif rsi > 30:
            score -= 0.3
        else:
            score -= 0.5  # Bearish but oversold
        
        # Rate of Change
        roc = (prices[-1] - prices[-10]) / prices[-10]
        score += np.clip(roc * 10, -0.5, 0.5)
        
        return np.clip(score, -1, 1)
    
    def _calculate_volume_sentiment(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """Calculate sentiment from volume"""
        
        if len(volumes) < 10:
            return 0.0
        
        # On-Balance Volume trend
        obv = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
        
        # Volume on recent up vs down days
        recent_up_vol = 0
        recent_down_vol = 0
        
        for i in range(-10, 0):
            if i + 1 < len(prices) and prices[i] > prices[i-1]:
                recent_up_vol += volumes[i]
            else:
                recent_down_vol += volumes[i]
        
        total = recent_up_vol + recent_down_vol + 1e-10
        ratio = (recent_up_vol - recent_down_vol) / total
        
        return np.clip(ratio, -1, 1)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        
        if len(prices) < period + 1:
            return 50.0
        
        changes = np.diff(prices[-(period+1):])
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses) + 1e-10
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _detect_divergence(
        self,
        prices: np.ndarray,
        trend: float,
        momentum: float
    ) -> bool:
        """Detect price vs sentiment divergence"""
        
        # Price direction
        price_up = prices[-1] > prices[-10]
        
        # Sentiment direction
        sentiment_up = (trend + momentum) / 2 > 0
        
        return price_up != sentiment_up


# ============================================================
# DYNAMIC EXIT MANAGER
# ============================================================

class DynamicExitManager:
    """
    Manage dynamic stop loss and take profit levels.
    """
    
    def __init__(self):
        self.active_exits: Dict[str, ExitPlan] = {}
    
    def create_exit_plan(
        self,
        side: str,
        entry_price: float,
        prices: np.ndarray,
        volatility_data: VolatilityData,
        fractal_data: FractalData
    ) -> ExitPlan:
        """Create dynamic exit plan based on market conditions"""
        
        # Calculate ATR
        atr = self._calculate_atr(prices)
        
        # Determine strategy based on volatility
        if volatility_data.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH]:
            strategy = ExitStrategy.CHANDELIER
            sl_multiplier = 3.0
            tp_multiplier = 2.0
        elif volatility_data.regime == VolatilityRegime.EXTREMELY_LOW:
            strategy = ExitStrategy.BREAKEVEN_PLUS
            sl_multiplier = 1.5
            tp_multiplier = 3.0
        else:
            strategy = ExitStrategy.TRAILING
            sl_multiplier = 2.0
            tp_multiplier = 2.5
        
        # Calculate levels
        if side == "BUY":
            initial_sl = entry_price - (atr * sl_multiplier)
            tp1 = entry_price + (atr * tp_multiplier)
            tp2 = entry_price + (atr * tp_multiplier * 1.5)
            tp3 = entry_price + (atr * tp_multiplier * 2.0)
            
            # Use fractal resistance if available
            if fractal_data.resistance_levels:
                nearest_resistance = min([r for r in fractal_data.resistance_levels 
                                         if r > entry_price], default=tp1)
                tp1 = min(tp1, nearest_resistance * 0.995)  # Just below resistance
        else:
            initial_sl = entry_price + (atr * sl_multiplier)
            tp1 = entry_price - (atr * tp_multiplier)
            tp2 = entry_price - (atr * tp_multiplier * 1.5)
            tp3 = entry_price - (atr * tp_multiplier * 2.0)
            
            # Use fractal support if available
            if fractal_data.support_levels:
                nearest_support = max([s for s in fractal_data.support_levels 
                                      if s < entry_price], default=tp1)
                tp1 = max(tp1, nearest_support * 1.005)  # Just above support
        
        # Trailing distance
        trailing_distance = atr * 1.5
        
        # Breakeven trigger (after 1R profit)
        if side == "BUY":
            breakeven_trigger = entry_price + atr * sl_multiplier
        else:
            breakeven_trigger = entry_price - atr * sl_multiplier
        
        # Scale out levels
        scale_out = [
            (tp1, 0.4),   # 40% at TP1
            (tp2, 0.35),  # 35% at TP2
            (tp3, 0.25),  # 25% at TP3
        ]
        
        return ExitPlan(
            strategy=strategy,
            initial_stop_loss=initial_sl,
            current_stop_loss=initial_sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            trailing_distance=trailing_distance,
            breakeven_trigger=breakeven_trigger,
            scale_out_levels=scale_out
        )
    
    def update_exit(
        self,
        trade_id: str,
        current_price: float,
        side: str
    ) -> Optional[ExitPlan]:
        """Update exit plan based on current price"""
        
        if trade_id not in self.active_exits:
            return None
        
        plan = self.active_exits[trade_id]
        
        if plan.strategy == ExitStrategy.TRAILING:
            # Update trailing stop
            if side == "BUY":
                new_sl = current_price - plan.trailing_distance
                if new_sl > plan.current_stop_loss:
                    plan.current_stop_loss = new_sl
            else:
                new_sl = current_price + plan.trailing_distance
                if new_sl < plan.current_stop_loss:
                    plan.current_stop_loss = new_sl
        
        elif plan.strategy == ExitStrategy.BREAKEVEN_PLUS:
            # Move to breakeven + buffer after trigger
            if side == "BUY" and current_price >= plan.breakeven_trigger:
                plan.current_stop_loss = max(
                    plan.current_stop_loss,
                    plan.initial_stop_loss + (plan.trailing_distance * 0.5)
                )
            elif side == "SELL" and current_price <= plan.breakeven_trigger:
                plan.current_stop_loss = min(
                    plan.current_stop_loss,
                    plan.initial_stop_loss - (plan.trailing_distance * 0.5)
                )
        
        return plan
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate ATR from prices"""
        
        if len(prices) < period + 1:
            return np.std(prices) * 2
        
        # True Range approximation
        tr = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])  # Simplified
            tr.append(high_low)
        
        return np.mean(tr[-period:])


# ============================================================
# QUANTUM DECISION ENGINE
# ============================================================

class QuantumStrategy:
    """
    Main class combining all quantum strategy components.
    """
    
    def __init__(self):
        self.microstructure = MicrostructureAnalyzer()
        self.volatility = VolatilityRegimeAnalyzer()
        self.fractal = FractalAnalyzer()
        self.sentiment = SentimentAggregator()
        self.exit_manager = DynamicExitManager()
        
        self.decision_history: deque = deque(maxlen=100)
        self.last_decisions: Dict[str, QuantumDecision] = {}
        
        logger.info("⚛️ Quantum Strategy initialized")
        logger.info("   - Microstructure Analyzer: ✓")
        logger.info("   - Volatility Regime: ✓")
        logger.info("   - Fractal Analyzer: ✓")
        logger.info("   - Sentiment Aggregator: ✓")
        logger.info("   - Dynamic Exit Manager: ✓")
    
    def analyze(
        self,
        symbol: str,
        signal_direction: str,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        entry_price: Optional[float] = None
    ) -> QuantumDecision:
        """
        Perform quantum analysis and return decision.
        """
        
        warnings = []
        reasons = []
        
        if volumes is None:
            volumes = np.ones_like(prices) * np.mean(prices) * 1000
        
        # 1. Microstructure Analysis
        micro_data = self.microstructure.analyze(prices, volumes, highs, lows)
        
        # Check alignment with signal
        if signal_direction == "BUY" and micro_data.smart_money_signal == "BEARISH":
            warnings.append("⚠️ Smart money against BUY signal")
        elif signal_direction == "SELL" and micro_data.smart_money_signal == "BULLISH":
            warnings.append("⚠️ Smart money against SELL signal")
        
        if micro_data.state == MarketMicrostructure.SQUEEZE:
            reasons.append("🔥 Volatility squeeze detected - breakout imminent")
        
        # 2. Volatility Analysis
        vol_data = self.volatility.analyze(prices)
        
        if vol_data.regime == VolatilityRegime.EXPLOSIVE:
            warnings.append("⚠️ Explosive volatility - high risk")
        elif vol_data.regime == VolatilityRegime.EXTREMELY_LOW:
            reasons.append("📊 Low volatility - potential breakout setup")
        
        if vol_data.breakout_probability > 0.7:
            reasons.append(f"🎯 High breakout probability: {vol_data.breakout_probability:.0%}")
        
        # 3. Fractal Analysis
        fractal_data = self.fractal.analyze(prices, highs, lows)
        
        # Hurst exponent insight
        if fractal_data.hurst_exponent > 0.6:
            reasons.append(f"📈 Trending market (H={fractal_data.hurst_exponent:.2f})")
        elif fractal_data.hurst_exponent < 0.4:
            reasons.append(f"📉 Mean-reverting market (H={fractal_data.hurst_exponent:.2f})")
        
        # Pattern detection
        if fractal_data.pattern != FractalType.NONE:
            if signal_direction == "BUY" and fractal_data.pattern in [
                FractalType.DOUBLE_BOTTOM, FractalType.INVERSE_HEAD_SHOULDERS, FractalType.BULLISH_FRACTAL
            ]:
                reasons.append(f"✅ Bullish pattern: {fractal_data.pattern.value}")
            elif signal_direction == "SELL" and fractal_data.pattern in [
                FractalType.DOUBLE_TOP, FractalType.HEAD_SHOULDERS, FractalType.BEARISH_FRACTAL
            ]:
                reasons.append(f"✅ Bearish pattern: {fractal_data.pattern.value}")
            else:
                warnings.append(f"⚠️ Pattern conflicts: {fractal_data.pattern.value}")
        
        # 4. Sentiment Analysis
        sent_data = self.sentiment.analyze(prices, volumes)
        
        if sent_data.contrarian_signal:
            if (signal_direction == "BUY" and sent_data.overall_sentiment > 0.8) or \
               (signal_direction == "SELL" and sent_data.overall_sentiment < -0.8):
                warnings.append("⚠️ Extreme sentiment - contrarian risk")
        
        if sent_data.sentiment_divergence:
            warnings.append("⚠️ Price/sentiment divergence detected")
        
        # 5. Create Exit Plan
        exit_plan = None
        if entry_price:
            exit_plan = self.exit_manager.create_exit_plan(
                side=signal_direction,
                entry_price=entry_price,
                prices=prices,
                volatility_data=vol_data,
                fractal_data=fractal_data
            )
        
        # ============================================
        # QUANTUM SCORE CALCULATION
        # ============================================
        
        quantum_score = 0.0
        
        # Microstructure contribution (±25)
        if micro_data.smart_money_signal == signal_direction.replace("BUY", "BULLISH").replace("SELL", "BEARISH"):
            quantum_score += 25 * micro_data.confidence
        elif micro_data.smart_money_signal != "NEUTRAL":
            quantum_score -= 15 * micro_data.confidence
        
        # Volatility contribution (±20)
        if vol_data.regime in [VolatilityRegime.NORMAL, VolatilityRegime.LOW]:
            quantum_score += 15
        elif vol_data.regime == VolatilityRegime.EXTREMELY_LOW and vol_data.breakout_probability > 0.6:
            quantum_score += 20
        elif vol_data.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH]:
            quantum_score -= 10
        elif vol_data.regime == VolatilityRegime.EXPLOSIVE:
            quantum_score -= 25
        
        # Fractal contribution (±25)
        if fractal_data.hurst_exponent > 0.55:  # Trending
            quantum_score += 15
        elif fractal_data.hurst_exponent < 0.45:  # Mean reverting
            quantum_score += 5 if signal_direction == "SELL" else -5
        
        if fractal_data.pattern_confidence > 0.5:
            quantum_score += 10 * fractal_data.pattern_confidence
        
        # Sentiment contribution (±20)
        if signal_direction == "BUY":
            quantum_score += sent_data.overall_sentiment * 20
        else:
            quantum_score -= sent_data.overall_sentiment * 20
        
        # Warning penalty
        quantum_score -= len(warnings) * 5
        
        # Calculate confidence
        confidence = 50 + quantum_score / 2
        confidence = np.clip(confidence, 0, 100)
        
        # Edge score (simplified statistical edge)
        win_factors = sum([
            1 if micro_data.smart_money_signal != "NEUTRAL" else 0,
            1 if vol_data.breakout_probability > 0.6 else 0,
            1 if fractal_data.hurst_exponent > 0.55 else 0,
            1 if abs(sent_data.overall_sentiment) < 0.7 else 0,
        ])
        edge_score = win_factors / 4
        
        # Risk/Reward estimate
        if exit_plan:
            if signal_direction == "BUY":
                reward = exit_plan.take_profit_1 - (entry_price or prices[-1])
                risk = (entry_price or prices[-1]) - exit_plan.initial_stop_loss
            else:
                reward = (entry_price or prices[-1]) - exit_plan.take_profit_1
                risk = exit_plan.initial_stop_loss - (entry_price or prices[-1])
            
            risk_reward = reward / risk if risk > 0 else 1.0
        else:
            risk_reward = 1.5  # Default assumption
        
        # Should trade decision
        should_trade = (
            confidence >= 55 and
            quantum_score >= 0 and
            len(warnings) <= 2 and
            vol_data.regime != VolatilityRegime.EXPLOSIVE and
            edge_score >= 0.5
        )
        
        # Position multiplier
        if confidence >= 75 and quantum_score >= 30:
            position_multiplier = 1.2
        elif confidence >= 65 and quantum_score >= 15:
            position_multiplier = 1.0
        elif confidence >= 55:
            position_multiplier = 0.7
        else:
            position_multiplier = 0.5
        
        # Adjust for volatility
        if vol_data.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH]:
            position_multiplier *= 0.7
        
        direction = signal_direction if should_trade else "WAIT"
        
        # Create decision
        decision = QuantumDecision(
            should_trade=should_trade,
            direction=direction,
            confidence=confidence,
            position_multiplier=position_multiplier,
            microstructure=micro_data,
            volatility=vol_data,
            fractal=fractal_data,
            sentiment=sent_data,
            exit_plan=exit_plan,
            quantum_score=quantum_score,
            edge_score=edge_score,
            risk_reward=risk_reward,
            warnings=warnings,
            reasons=reasons
        )
        
        # Store
        self.last_decisions[symbol] = decision
        self.decision_history.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "decision": decision
        })
        
        return decision
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        return {
            "active_symbols": list(self.last_decisions.keys()),
            "total_decisions": len(self.decision_history),
            "avg_quantum_score": np.mean([
                d["decision"].quantum_score 
                for d in self.decision_history
            ]) if self.decision_history else 0
        }


# ============================================================
# SINGLETON & FACTORY
# ============================================================

_quantum_strategy: Optional[QuantumStrategy] = None


def get_quantum_strategy() -> QuantumStrategy:
    """Get singleton Quantum Strategy instance"""
    global _quantum_strategy
    if _quantum_strategy is None:
        _quantum_strategy = QuantumStrategy()
    return _quantum_strategy


def init_quantum_strategy() -> QuantumStrategy:
    """Initialize and return Quantum Strategy"""
    global _quantum_strategy
    _quantum_strategy = QuantumStrategy()
    return _quantum_strategy
