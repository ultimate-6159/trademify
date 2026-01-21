"""
🏆 Alpha Engine Module
======================
Ultimate trading intelligence combining institutional-grade analytics.

Components:
1. OrderFlowAnalyzer - Simulated order flow analysis
2. LiquidityZoneDetector - Smart money liquidity zones
3. MarketProfileAnalyzer - Value area & POC detection
4. DivergenceScanner - Multi-indicator divergence
5. MomentumWaveAnalyzer - Wave pattern analysis
6. RiskMetricsCalculator - Sharpe, Sortino, Calmar
7. AlphaScorer - Final trade quality scoring

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

class OrderFlowBias(Enum):
    """Order flow directional bias"""
    STRONG_BUYING = "strong_buying"
    BUYING = "buying"
    NEUTRAL = "neutral"
    SELLING = "selling"
    STRONG_SELLING = "strong_selling"


class LiquidityType(Enum):
    """Type of liquidity zone"""
    BUY_SIDE = "buy_side"       # Stop losses above highs
    SELL_SIDE = "sell_side"     # Stop losses below lows
    FAIR_VALUE_GAP = "fvg"      # Imbalance zones
    ORDER_BLOCK = "order_block" # Institutional entry zones


class WaveType(Enum):
    """Momentum wave type"""
    IMPULSE_UP = "impulse_up"     # Strong move up (wave 1, 3, 5)
    IMPULSE_DOWN = "impulse_down" # Strong move down
    CORRECTION_UP = "correction_up"   # Correction up (wave 2, 4)
    CORRECTION_DOWN = "correction_down"
    EXTENSION = "extension"       # Extended wave
    EXHAUSTION = "exhaustion"     # Trend exhaustion


class DivergenceType(Enum):
    """Divergence type"""
    REGULAR_BULLISH = "regular_bullish"   # Price lower low, indicator higher low
    REGULAR_BEARISH = "regular_bearish"   # Price higher high, indicator lower high
    HIDDEN_BULLISH = "hidden_bullish"     # Price higher low, indicator lower low
    HIDDEN_BEARISH = "hidden_bearish"     # Price lower high, indicator higher high
    NONE = "none"


class TradeGrade(Enum):
    """Trade quality grade"""
    A_PLUS = "A+"   # 90-100 - Perfect setup
    A = "A"         # 80-89 - Excellent
    B_PLUS = "B+"   # 70-79 - Very good
    B = "B"         # 60-69 - Good
    C = "C"         # 50-59 - Average
    D = "D"         # 40-49 - Below average
    F = "F"         # <40 - Avoid


@dataclass
class OrderFlowData:
    """Order flow analysis result"""
    bias: OrderFlowBias
    delta: float              # Net buying - selling
    cumulative_delta: float   # Running total
    absorption_detected: bool # Large orders absorbing flow
    imbalance_ratio: float    # Buy/sell imbalance
    aggressive_side: str      # "BUYERS" or "SELLERS"
    confidence: float


@dataclass
class LiquidityZone:
    """Liquidity zone data"""
    zone_type: LiquidityType
    price_level: float
    strength: float        # 0-1
    times_tested: int
    is_fresh: bool         # Not yet touched
    expected_reaction: str # "BOUNCE" or "BREAK"


@dataclass
class MarketProfileData:
    """Market profile analysis"""
    poc: float              # Point of Control (highest volume price)
    value_area_high: float  # Upper 70% volume bound
    value_area_low: float   # Lower 70% volume bound
    profile_shape: str      # "P", "b", "D", "B" shapes
    developing_poc: float   # Current session POC
    single_prints: List[float]  # Low volume nodes


@dataclass
class DivergenceData:
    """Divergence analysis result"""
    divergence_type: DivergenceType
    indicator: str          # Which indicator shows divergence
    strength: float         # 0-1
    price_swing_1: float
    price_swing_2: float
    indicator_swing_1: float
    indicator_swing_2: float
    bars_between: int


@dataclass
class WaveData:
    """Wave analysis result"""
    current_wave: WaveType
    wave_count: int         # Estimated wave number (1-5 or A-C)
    wave_progress: float    # 0-1 (how far into current wave)
    trend_strength: float   # Overall trend strength
    next_expected: str      # Expected next wave
    invalidation_level: float


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics"""
    sharpe_ratio: float     # Risk-adjusted return
    sortino_ratio: float    # Downside risk-adjusted
    calmar_ratio: float     # Return / Max Drawdown
    win_rate: float
    profit_factor: float    # Gross profit / Gross loss
    expectancy: float       # Average expected return per trade
    max_consecutive_losses: int
    recovery_factor: float  # Net profit / Max Drawdown


@dataclass
class AlphaDecision:
    """Final Alpha Engine decision"""
    should_trade: bool
    direction: str
    grade: TradeGrade
    alpha_score: float      # 0-100
    confidence: float       # 0-100
    position_multiplier: float
    
    # Component data
    order_flow: OrderFlowData
    liquidity_zones: List[LiquidityZone]
    market_profile: MarketProfileData
    divergences: List[DivergenceData]
    wave_data: WaveData
    risk_metrics: RiskMetrics
    
    # Entry/Exit optimization
    optimal_entry: float
    stop_loss: float
    targets: List[float]
    risk_reward: float
    
    # Analysis
    edge_factors: List[str]
    risk_factors: List[str]
    recommendation: str


# ============================================================
# ORDER FLOW ANALYZER
# ============================================================

class OrderFlowAnalyzer:
    """
    Simulate order flow analysis from OHLCV data.
    Real order flow requires tick data, this approximates from candles.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.delta_history: deque = deque(maxlen=lookback)
        self.cumulative_delta = 0.0
    
    def analyze(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> OrderFlowData:
        """Analyze order flow from OHLCV"""
        
        if len(closes) < 10:
            return OrderFlowData(
                bias=OrderFlowBias.NEUTRAL,
                delta=0.0,
                cumulative_delta=0.0,
                absorption_detected=False,
                imbalance_ratio=1.0,
                aggressive_side="NEUTRAL",
                confidence=0.0
            )
        
        # Calculate delta (buy - sell volume approximation)
        deltas = self._calculate_deltas(opens, highs, lows, closes, volumes)
        
        # Current delta
        current_delta = deltas[-1]
        
        # Cumulative delta
        self.cumulative_delta += current_delta
        self.delta_history.append(current_delta)
        
        # Recent delta sum
        recent_delta = np.sum(deltas[-10:])
        
        # Imbalance ratio
        buy_vol = np.sum(np.maximum(deltas[-20:], 0))
        sell_vol = np.abs(np.sum(np.minimum(deltas[-20:], 0)))
        imbalance_ratio = buy_vol / (sell_vol + 1e-10)
        
        # Determine bias
        if recent_delta > 0:
            if imbalance_ratio > 2.0:
                bias = OrderFlowBias.STRONG_BUYING
            else:
                bias = OrderFlowBias.BUYING
        elif recent_delta < 0:
            if imbalance_ratio < 0.5:
                bias = OrderFlowBias.STRONG_SELLING
            else:
                bias = OrderFlowBias.SELLING
        else:
            bias = OrderFlowBias.NEUTRAL
        
        # Absorption detection (large volume with small price move)
        absorption = self._detect_absorption(
            opens[-10:], highs[-10:], lows[-10:], closes[-10:], volumes[-10:]
        )
        
        # Aggressive side
        aggressive = "BUYERS" if recent_delta > 0 else "SELLERS" if recent_delta < 0 else "NEUTRAL"
        
        # Confidence
        confidence = min(1.0, abs(recent_delta) / (np.mean(volumes[-20:]) * 0.5 + 1e-10))
        
        return OrderFlowData(
            bias=bias,
            delta=current_delta,
            cumulative_delta=self.cumulative_delta,
            absorption_detected=absorption,
            imbalance_ratio=imbalance_ratio,
            aggressive_side=aggressive,
            confidence=confidence
        )
    
    def _calculate_deltas(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """Calculate volume delta for each candle"""
        
        deltas = []
        for i in range(len(closes)):
            # Candle range
            range_size = highs[i] - lows[i]
            if range_size == 0:
                deltas.append(0)
                continue
            
            # Buy volume = volume * (close - low) / range
            # Sell volume = volume * (high - close) / range
            buy_vol = volumes[i] * (closes[i] - lows[i]) / range_size
            sell_vol = volumes[i] * (highs[i] - closes[i]) / range_size
            
            deltas.append(buy_vol - sell_vol)
        
        return np.array(deltas)
    
    def _detect_absorption(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> bool:
        """Detect volume absorption (large volume, small move)"""
        
        avg_range = np.mean(highs - lows)
        avg_volume = np.mean(volumes)
        
        # Look for candles with high volume but small range
        for i in range(len(closes)):
            candle_range = highs[i] - lows[i]
            if volumes[i] > avg_volume * 1.5 and candle_range < avg_range * 0.5:
                return True
        
        return False


# ============================================================
# LIQUIDITY ZONE DETECTOR
# ============================================================

class LiquidityZoneDetector:
    """
    Detect institutional liquidity zones (Smart Money Concepts).
    """
    
    def __init__(self):
        self.zones: List[LiquidityZone] = []
    
    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        current_price: float
    ) -> List[LiquidityZone]:
        """Detect all liquidity zones"""
        
        zones = []
        
        # Buy-side liquidity (above swing highs)
        buy_side = self._find_buy_side_liquidity(highs, closes)
        zones.extend(buy_side)
        
        # Sell-side liquidity (below swing lows)
        sell_side = self._find_sell_side_liquidity(lows, closes)
        zones.extend(sell_side)
        
        # Fair Value Gaps
        fvgs = self._find_fair_value_gaps(highs, lows, closes)
        zones.extend(fvgs)
        
        # Order Blocks
        obs = self._find_order_blocks(highs, lows, closes)
        zones.extend(obs)
        
        # Sort by proximity to current price
        zones.sort(key=lambda z: abs(z.price_level - current_price))
        
        self.zones = zones
        return zones[:10]  # Return top 10 nearest
    
    def _find_buy_side_liquidity(
        self,
        highs: np.ndarray,
        closes: np.ndarray
    ) -> List[LiquidityZone]:
        """Find buy-side liquidity pools (above swing highs)"""
        
        zones = []
        
        for i in range(2, len(highs) - 2):
            # Swing high detection
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                
                # Count how many times tested
                times_tested = np.sum(np.abs(highs[i+2:] - highs[i]) / highs[i] < 0.002)
                
                zones.append(LiquidityZone(
                    zone_type=LiquidityType.BUY_SIDE,
                    price_level=highs[i],
                    strength=min(1.0, times_tested / 3),
                    times_tested=int(times_tested),
                    is_fresh=times_tested == 0,
                    expected_reaction="BREAK" if times_tested >= 3 else "BOUNCE"
                ))
        
        return zones
    
    def _find_sell_side_liquidity(
        self,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> List[LiquidityZone]:
        """Find sell-side liquidity pools (below swing lows)"""
        
        zones = []
        
        for i in range(2, len(lows) - 2):
            # Swing low detection
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                
                times_tested = np.sum(np.abs(lows[i+2:] - lows[i]) / lows[i] < 0.002)
                
                zones.append(LiquidityZone(
                    zone_type=LiquidityType.SELL_SIDE,
                    price_level=lows[i],
                    strength=min(1.0, times_tested / 3),
                    times_tested=int(times_tested),
                    is_fresh=times_tested == 0,
                    expected_reaction="BREAK" if times_tested >= 3 else "BOUNCE"
                ))
        
        return zones
    
    def _find_fair_value_gaps(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> List[LiquidityZone]:
        """Find Fair Value Gaps (imbalance zones)"""
        
        zones = []
        
        for i in range(1, len(highs) - 1):
            # Bullish FVG: gap between high[i-1] and low[i+1]
            if lows[i+1] > highs[i-1]:
                gap_mid = (lows[i+1] + highs[i-1]) / 2
                zones.append(LiquidityZone(
                    zone_type=LiquidityType.FAIR_VALUE_GAP,
                    price_level=gap_mid,
                    strength=0.7,
                    times_tested=0,
                    is_fresh=True,
                    expected_reaction="BOUNCE"
                ))
            
            # Bearish FVG: gap between low[i-1] and high[i+1]
            if highs[i+1] < lows[i-1]:
                gap_mid = (highs[i+1] + lows[i-1]) / 2
                zones.append(LiquidityZone(
                    zone_type=LiquidityType.FAIR_VALUE_GAP,
                    price_level=gap_mid,
                    strength=0.7,
                    times_tested=0,
                    is_fresh=True,
                    expected_reaction="BOUNCE"
                ))
        
        return zones[-5:]  # Recent FVGs only
    
    def _find_order_blocks(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> List[LiquidityZone]:
        """Find order blocks (institutional entry zones)"""
        
        zones = []
        
        for i in range(3, len(closes) - 1):
            # Bullish OB: last down candle before strong up move
            if (closes[i-1] < closes[i-2] and  # Down candle
                closes[i] > closes[i-1] and    # Up candle
                closes[i] > highs[i-1]):       # Break above
                
                zones.append(LiquidityZone(
                    zone_type=LiquidityType.ORDER_BLOCK,
                    price_level=(highs[i-1] + lows[i-1]) / 2,
                    strength=0.8,
                    times_tested=0,
                    is_fresh=True,
                    expected_reaction="BOUNCE"
                ))
            
            # Bearish OB: last up candle before strong down move
            if (closes[i-1] > closes[i-2] and  # Up candle
                closes[i] < closes[i-1] and    # Down candle
                closes[i] < lows[i-1]):        # Break below
                
                zones.append(LiquidityZone(
                    zone_type=LiquidityType.ORDER_BLOCK,
                    price_level=(highs[i-1] + lows[i-1]) / 2,
                    strength=0.8,
                    times_tested=0,
                    is_fresh=True,
                    expected_reaction="BOUNCE"
                ))
        
        return zones[-5:]


# ============================================================
# MARKET PROFILE ANALYZER
# ============================================================

class MarketProfileAnalyzer:
    """
    Market Profile analysis (Volume at Price).
    """
    
    def __init__(self, num_bins: int = 30):
        self.num_bins = num_bins
    
    def analyze(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> MarketProfileData:
        """Create market profile analysis"""
        
        if len(closes) < 20:
            poc = closes[-1]
            return MarketProfileData(
                poc=poc,
                value_area_high=poc * 1.01,
                value_area_low=poc * 0.99,
                profile_shape="D",
                developing_poc=poc,
                single_prints=[]
            )
        
        # Create price bins
        price_min = np.min(lows)
        price_max = np.max(highs)
        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Distribute volume to price bins
        volume_profile = np.zeros(self.num_bins)
        
        for i in range(len(closes)):
            # Each candle contributes volume to bins it touches
            for j in range(self.num_bins):
                if bins[j] <= highs[i] and bins[j+1] >= lows[i]:
                    # Proportional volume based on overlap
                    overlap = min(highs[i], bins[j+1]) - max(lows[i], bins[j])
                    candle_range = highs[i] - lows[i] + 1e-10
                    volume_profile[j] += volumes[i] * (overlap / candle_range)
        
        # POC (Point of Control) - highest volume price
        poc_idx = np.argmax(volume_profile)
        poc = bin_centers[poc_idx]
        
        # Value Area (70% of volume)
        total_vol = np.sum(volume_profile)
        target_vol = total_vol * 0.7
        
        # Expand from POC until 70%
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        current_vol = volume_profile[poc_idx]
        
        while current_vol < target_vol and (va_low_idx > 0 or va_high_idx < self.num_bins - 1):
            low_vol = volume_profile[va_low_idx - 1] if va_low_idx > 0 else 0
            high_vol = volume_profile[va_high_idx + 1] if va_high_idx < self.num_bins - 1 else 0
            
            if low_vol >= high_vol and va_low_idx > 0:
                va_low_idx -= 1
                current_vol += low_vol
            elif va_high_idx < self.num_bins - 1:
                va_high_idx += 1
                current_vol += high_vol
            else:
                break
        
        va_high = bin_centers[va_high_idx]
        va_low = bin_centers[va_low_idx]
        
        # Profile shape
        shape = self._determine_shape(volume_profile, poc_idx)
        
        # Developing POC (recent session)
        recent_profile = np.zeros(self.num_bins)
        for i in range(-min(20, len(closes)), 0):
            for j in range(self.num_bins):
                if bins[j] <= highs[i] and bins[j+1] >= lows[i]:
                    recent_profile[j] += volumes[i]
        
        dev_poc_idx = np.argmax(recent_profile)
        developing_poc = bin_centers[dev_poc_idx]
        
        # Single prints (low volume nodes)
        avg_vol = np.mean(volume_profile)
        single_prints = [bin_centers[i] for i in range(self.num_bins) 
                        if volume_profile[i] < avg_vol * 0.3]
        
        return MarketProfileData(
            poc=poc,
            value_area_high=va_high,
            value_area_low=va_low,
            profile_shape=shape,
            developing_poc=developing_poc,
            single_prints=single_prints[:5]
        )
    
    def _determine_shape(self, profile: np.ndarray, poc_idx: int) -> str:
        """Determine profile shape (P, b, D, B)"""
        
        n = len(profile)
        upper_vol = np.sum(profile[poc_idx:])
        lower_vol = np.sum(profile[:poc_idx])
        
        # P-shape: high volume at top (buying)
        if upper_vol > lower_vol * 1.5 and poc_idx > n * 0.6:
            return "P"
        
        # b-shape: high volume at bottom (selling)
        if lower_vol > upper_vol * 1.5 and poc_idx < n * 0.4:
            return "b"
        
        # B-shape: volume at both ends (balanced/ranging)
        upper_third = np.sum(profile[int(n*0.67):])
        lower_third = np.sum(profile[:int(n*0.33)])
        middle_third = np.sum(profile[int(n*0.33):int(n*0.67)])
        
        if upper_third > middle_third and lower_third > middle_third:
            return "B"
        
        # D-shape: normal distribution
        return "D"


# ============================================================
# DIVERGENCE SCANNER
# ============================================================

class DivergenceScanner:
    """
    Scan for price/indicator divergences.
    """
    
    def __init__(self):
        pass
    
    def scan(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None
    ) -> List[DivergenceData]:
        """Scan for all divergences"""
        
        if len(prices) < 30:
            return []
        
        divergences = []
        
        # RSI divergence
        rsi = self._calculate_rsi(prices)
        rsi_div = self._find_divergence(prices, rsi, "RSI")
        if rsi_div:
            divergences.append(rsi_div)
        
        # MACD divergence
        macd = self._calculate_macd(prices)
        macd_div = self._find_divergence(prices, macd, "MACD")
        if macd_div:
            divergences.append(macd_div)
        
        # Stochastic divergence
        if highs is not None and lows is not None:
            stoch = self._calculate_stochastic(highs, lows, prices)
            stoch_div = self._find_divergence(prices, stoch, "Stochastic")
            if stoch_div:
                divergences.append(stoch_div)
        
        # OBV divergence
        obv = self._calculate_obv_like(prices)
        obv_div = self._find_divergence(prices, obv, "OBV")
        if obv_div:
            divergences.append(obv_div)
        
        return divergences
    
    def _find_divergence(
        self,
        prices: np.ndarray,
        indicator: np.ndarray,
        indicator_name: str
    ) -> Optional[DivergenceData]:
        """Find divergence between price and indicator"""
        
        # Find recent swing points
        price_swings = self._find_swings(prices)
        ind_swings = self._find_swings(indicator)
        
        if len(price_swings['highs']) < 2 or len(price_swings['lows']) < 2:
            return None
        
        # Check for regular bearish divergence (higher high price, lower high indicator)
        if len(price_swings['highs']) >= 2:
            ph1, pi1 = price_swings['highs'][-2]
            ph2, pi2 = price_swings['highs'][-1]
            
            ih1 = indicator[pi1] if pi1 < len(indicator) else indicator[-1]
            ih2 = indicator[pi2] if pi2 < len(indicator) else indicator[-1]
            
            if ph2 > ph1 and ih2 < ih1:
                return DivergenceData(
                    divergence_type=DivergenceType.REGULAR_BEARISH,
                    indicator=indicator_name,
                    strength=abs(ih1 - ih2) / (abs(ih1) + 1e-10),
                    price_swing_1=ph1,
                    price_swing_2=ph2,
                    indicator_swing_1=ih1,
                    indicator_swing_2=ih2,
                    bars_between=pi2 - pi1
                )
        
        # Check for regular bullish divergence (lower low price, higher low indicator)
        if len(price_swings['lows']) >= 2:
            pl1, pi1 = price_swings['lows'][-2]
            pl2, pi2 = price_swings['lows'][-1]
            
            il1 = indicator[pi1] if pi1 < len(indicator) else indicator[-1]
            il2 = indicator[pi2] if pi2 < len(indicator) else indicator[-1]
            
            if pl2 < pl1 and il2 > il1:
                return DivergenceData(
                    divergence_type=DivergenceType.REGULAR_BULLISH,
                    indicator=indicator_name,
                    strength=abs(il2 - il1) / (abs(il1) + 1e-10),
                    price_swing_1=pl1,
                    price_swing_2=pl2,
                    indicator_swing_1=il1,
                    indicator_swing_2=il2,
                    bars_between=pi2 - pi1
                )
        
        return None
    
    def _find_swings(self, data: np.ndarray) -> Dict[str, List[Tuple[float, int]]]:
        """Find swing highs and lows"""
        
        highs = []
        lows = []
        
        for i in range(2, len(data) - 2):
            if data[i] > data[i-1] and data[i] > data[i-2] and \
               data[i] > data[i+1] and data[i] > data[i+2]:
                highs.append((data[i], i))
            
            if data[i] < data[i-1] and data[i] < data[i-2] and \
               data[i] < data[i+1] and data[i] < data[i+2]:
                lows.append((data[i], i))
        
        return {'highs': highs, 'lows': lows}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        
        changes = np.diff(prices)
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))
        
        rsi = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            avg_gain = np.mean(gains[i-period:i])
            avg_loss = np.mean(losses[i-period:i]) + 1e-10
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> np.ndarray:
        """Calculate MACD histogram"""
        
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd_line = ema12 - ema26
        signal = self._ema(macd_line, 9)
        
        return macd_line - signal  # Histogram
    
    def _calculate_stochastic(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate Stochastic %K"""
        
        stoch = np.zeros(len(closes))
        
        for i in range(period, len(closes)):
            highest = np.max(highs[i-period:i])
            lowest = np.min(lows[i-period:i])
            stoch[i] = (closes[i] - lowest) / (highest - lowest + 1e-10) * 100
        
        return stoch
    
    def _calculate_obv_like(self, prices: np.ndarray) -> np.ndarray:
        """Calculate OBV-like indicator from price"""
        
        obv = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv[i] = obv[i-1] + abs(prices[i] - prices[i-1])
            elif prices[i] < prices[i-1]:
                obv[i] = obv[i-1] - abs(prices[i] - prices[i-1])
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        
        ema = np.zeros(len(data))
        ema[0] = data[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema


# ============================================================
# MOMENTUM WAVE ANALYZER
# ============================================================

class MomentumWaveAnalyzer:
    """
    Analyze momentum waves (Elliott Wave inspired).
    """
    
    def __init__(self):
        self.wave_history: List[WaveData] = []
    
    def analyze(self, prices: np.ndarray) -> WaveData:
        """Analyze current wave structure"""
        
        if len(prices) < 30:
            return WaveData(
                current_wave=WaveType.IMPULSE_UP,
                wave_count=1,
                wave_progress=0.5,
                trend_strength=0.0,
                next_expected="Unknown",
                invalidation_level=prices[-1] * 0.95
            )
        
        # Find major swings
        swings = self._find_major_swings(prices)
        
        # Determine trend
        trend = self._determine_trend(prices)
        trend_strength = abs(trend)
        
        # Count waves
        wave_count, current_wave = self._count_waves(swings, trend)
        
        # Wave progress
        if len(swings) >= 2:
            last_swing = swings[-1][0]
            prev_swing = swings[-2][0]
            current = prices[-1]
            
            wave_range = abs(last_swing - prev_swing)
            if wave_range > 0:
                progress = abs(current - prev_swing) / wave_range
                wave_progress = min(1.0, progress)
            else:
                wave_progress = 0.5
        else:
            wave_progress = 0.5
        
        # Predict next wave
        next_expected = self._predict_next_wave(current_wave, wave_count)
        
        # Invalidation level
        if current_wave in [WaveType.IMPULSE_UP, WaveType.CORRECTION_UP]:
            invalidation = min(prices[-20:]) * 0.99
        else:
            invalidation = max(prices[-20:]) * 1.01
        
        wave_data = WaveData(
            current_wave=current_wave,
            wave_count=wave_count,
            wave_progress=wave_progress,
            trend_strength=trend_strength,
            next_expected=next_expected,
            invalidation_level=invalidation
        )
        
        self.wave_history.append(wave_data)
        return wave_data
    
    def _find_major_swings(self, prices: np.ndarray) -> List[Tuple[float, int]]:
        """Find major swing points"""
        
        swings = []
        threshold = np.std(prices) * 0.5
        
        last_swing = prices[0]
        last_idx = 0
        
        for i in range(1, len(prices)):
            if abs(prices[i] - last_swing) > threshold:
                swings.append((last_swing, last_idx))
                last_swing = prices[i]
                last_idx = i
        
        swings.append((prices[-1], len(prices) - 1))
        return swings
    
    def _determine_trend(self, prices: np.ndarray) -> float:
        """Determine trend direction and strength"""
        
        sma20 = np.mean(prices[-20:])
        sma50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma20
        
        trend = (sma20 - sma50) / sma50
        return np.clip(trend * 100, -1, 1)
    
    def _count_waves(
        self,
        swings: List[Tuple[float, int]],
        trend: float
    ) -> Tuple[int, WaveType]:
        """Count waves in current structure"""
        
        if len(swings) < 3:
            return 1, WaveType.IMPULSE_UP if trend > 0 else WaveType.IMPULSE_DOWN
        
        # Count alternating moves
        up_moves = 0
        down_moves = 0
        
        for i in range(1, len(swings)):
            if swings[i][0] > swings[i-1][0]:
                up_moves += 1
            else:
                down_moves += 1
        
        total_waves = up_moves + down_moves
        wave_count = min(5, (total_waves % 5) + 1)
        
        # Determine current wave type
        last_move_up = swings[-1][0] > swings[-2][0] if len(swings) >= 2 else True
        
        if trend > 0:
            if last_move_up:
                if wave_count in [1, 3, 5]:
                    current = WaveType.IMPULSE_UP
                else:
                    current = WaveType.CORRECTION_UP
            else:
                current = WaveType.CORRECTION_DOWN
        else:
            if not last_move_up:
                if wave_count in [1, 3, 5]:
                    current = WaveType.IMPULSE_DOWN
                else:
                    current = WaveType.CORRECTION_DOWN
            else:
                current = WaveType.CORRECTION_UP
        
        return wave_count, current
    
    def _predict_next_wave(self, current: WaveType, count: int) -> str:
        """Predict next expected wave"""
        
        predictions = {
            (WaveType.IMPULSE_UP, 1): "Wave 2 correction down",
            (WaveType.CORRECTION_DOWN, 2): "Wave 3 impulse up (strongest)",
            (WaveType.IMPULSE_UP, 3): "Wave 4 correction down",
            (WaveType.CORRECTION_DOWN, 4): "Wave 5 final impulse",
            (WaveType.IMPULSE_UP, 5): "ABC correction or trend reversal",
            (WaveType.IMPULSE_DOWN, 1): "Wave 2 correction up",
            (WaveType.CORRECTION_UP, 2): "Wave 3 impulse down",
            (WaveType.IMPULSE_DOWN, 3): "Wave 4 correction up",
            (WaveType.CORRECTION_UP, 4): "Wave 5 final down",
            (WaveType.IMPULSE_DOWN, 5): "ABC correction or reversal",
        }
        
        return predictions.get((current, count), "Continuation likely")


# ============================================================
# RISK METRICS CALCULATOR
# ============================================================

class RiskMetricsCalculator:
    """
    Calculate advanced risk-adjusted performance metrics.
    """
    
    def __init__(self):
        self.returns_history: deque = deque(maxlen=500)
    
    def calculate(self, trade_results: List[float]) -> RiskMetrics:
        """Calculate all risk metrics"""
        
        if len(trade_results) < 5:
            return RiskMetrics(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                win_rate=0.5,
                profit_factor=1.0,
                expectancy=0.0,
                max_consecutive_losses=0,
                recovery_factor=0.0
            )
        
        returns = np.array(trade_results)
        
        # Basic stats
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / (gross_loss + 1e-10)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # Sortino ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino = np.mean(returns) / (downside_std + 1e-10) * np.sqrt(252)
        
        # Max drawdown for Calmar
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns)
        
        # Calmar ratio
        total_return = np.sum(returns)
        calmar = total_return / (max_dd + 1e-10)
        
        # Expectancy
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1e-10
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Max consecutive losses
        max_consec = self._max_consecutive_losses(returns)
        
        # Recovery factor
        recovery = total_return / (max_dd + 1e-10)
        
        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_consecutive_losses=max_consec,
            recovery_factor=recovery
        )
    
    def _max_consecutive_losses(self, returns: np.ndarray) -> int:
        """Find maximum consecutive losses"""
        
        max_streak = 0
        current_streak = 0
        
        for r in returns:
            if r < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak


# ============================================================
# ALPHA SCORER (Main Decision Engine)
# ============================================================

class AlphaEngine:
    """
    Ultimate Alpha Engine combining all components.
    """
    
    def __init__(self):
        self.order_flow = OrderFlowAnalyzer()
        self.liquidity = LiquidityZoneDetector()
        self.market_profile = MarketProfileAnalyzer()
        self.divergence = DivergenceScanner()
        self.wave_analyzer = MomentumWaveAnalyzer()
        self.risk_calc = RiskMetricsCalculator()
        
        self.trade_history: List[float] = []
        self.decision_history: deque = deque(maxlen=100)
        
        logger.info("🏆 Alpha Engine initialized")
        logger.info("   - Order Flow Analyzer: ✓")
        logger.info("   - Liquidity Zone Detector: ✓")
        logger.info("   - Market Profile Analyzer: ✓")
        logger.info("   - Divergence Scanner: ✓")
        logger.info("   - Momentum Wave Analyzer: ✓")
        logger.info("   - Risk Metrics Calculator: ✓")
    
    def analyze(
        self,
        symbol: str,
        signal_direction: str,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> AlphaDecision:
        """Perform comprehensive alpha analysis"""
        
        edge_factors = []
        risk_factors = []
        current_price = closes[-1]
        
        # 1. Order Flow Analysis
        of_data = self.order_flow.analyze(opens, highs, lows, closes, volumes)
        
        if of_data.bias == OrderFlowBias.STRONG_BUYING and signal_direction == "BUY":
            edge_factors.append("✓ Strong buying pressure aligns with signal")
        elif of_data.bias == OrderFlowBias.STRONG_SELLING and signal_direction == "SELL":
            edge_factors.append("✓ Strong selling pressure aligns with signal")
        elif of_data.aggressive_side != "NEUTRAL" and \
             of_data.aggressive_side != signal_direction.replace("BUY", "BUYERS").replace("SELL", "SELLERS"):
            risk_factors.append("✗ Order flow opposes signal")
        
        if of_data.absorption_detected:
            risk_factors.append("⚠ Volume absorption detected - potential reversal")
        
        # 2. Liquidity Zones
        liq_zones = self.liquidity.detect(highs, lows, closes, current_price)
        
        nearby_zones = [z for z in liq_zones if abs(z.price_level - current_price) / current_price < 0.01]
        
        for zone in nearby_zones[:3]:
            if zone.is_fresh:
                edge_factors.append(f"✓ Fresh {zone.zone_type.value} near entry")
            if zone.times_tested >= 3:
                risk_factors.append(f"⚠ {zone.zone_type.value} tested {zone.times_tested}x - may break")
        
        # 3. Market Profile
        mp_data = self.market_profile.analyze(highs, lows, closes, volumes)
        
        if mp_data.value_area_low < current_price < mp_data.value_area_high:
            edge_factors.append("✓ Price within value area")
        else:
            risk_factors.append("⚠ Price outside value area - may revert")
        
        if abs(current_price - mp_data.poc) / current_price < 0.005:
            edge_factors.append("✓ Near POC - high probability zone")
        
        # 4. Divergence
        div_data = self.divergence.scan(closes, highs, lows)
        
        for div in div_data:
            if div.divergence_type == DivergenceType.REGULAR_BULLISH and signal_direction == "BUY":
                edge_factors.append(f"✓ Bullish {div.indicator} divergence supports BUY")
            elif div.divergence_type == DivergenceType.REGULAR_BEARISH and signal_direction == "SELL":
                edge_factors.append(f"✓ Bearish {div.indicator} divergence supports SELL")
            elif div.divergence_type in [DivergenceType.REGULAR_BULLISH, DivergenceType.REGULAR_BEARISH]:
                risk_factors.append(f"⚠ {div.indicator} divergence conflicts")
        
        # 5. Wave Analysis
        wave_data = self.wave_analyzer.analyze(closes)
        
        if wave_data.current_wave == WaveType.IMPULSE_UP and signal_direction == "BUY":
            if wave_data.wave_count in [1, 3]:
                edge_factors.append(f"✓ Wave {wave_data.wave_count} impulse - trend continuation")
            elif wave_data.wave_count == 5:
                risk_factors.append("⚠ Wave 5 - potential exhaustion")
        elif wave_data.current_wave == WaveType.IMPULSE_DOWN and signal_direction == "SELL":
            if wave_data.wave_count in [1, 3]:
                edge_factors.append(f"✓ Wave {wave_data.wave_count} impulse down")
            elif wave_data.wave_count == 5:
                risk_factors.append("⚠ Wave 5 down - potential reversal")
        
        # 6. Risk Metrics
        risk_metrics = self.risk_calc.calculate(self.trade_history if self.trade_history else [0.01, -0.005, 0.008])
        
        if risk_metrics.win_rate < 0.4:
            risk_factors.append(f"⚠ Low historical win rate: {risk_metrics.win_rate:.1%}")
        elif risk_metrics.win_rate > 0.55:
            edge_factors.append(f"✓ Strong win rate: {risk_metrics.win_rate:.1%}")
        
        if risk_metrics.max_consecutive_losses >= 5:
            risk_factors.append(f"⚠ Recent losing streak: {risk_metrics.max_consecutive_losses}")
        
        # ============================================
        # ALPHA SCORE CALCULATION
        # ============================================
        
        alpha_score = 50.0
        
        # Order flow contribution (+/- 15)
        if of_data.bias in [OrderFlowBias.STRONG_BUYING, OrderFlowBias.STRONG_SELLING]:
            if (of_data.bias == OrderFlowBias.STRONG_BUYING and signal_direction == "BUY") or \
               (of_data.bias == OrderFlowBias.STRONG_SELLING and signal_direction == "SELL"):
                alpha_score += 15
            else:
                alpha_score -= 10
        
        # Liquidity zones (+/- 10)
        fresh_zones_aligned = sum(1 for z in nearby_zones if z.is_fresh)
        alpha_score += fresh_zones_aligned * 5
        
        # Market profile (+/- 10)
        if mp_data.value_area_low < current_price < mp_data.value_area_high:
            alpha_score += 8
        
        # Divergence (+/- 15)
        bullish_div = any(d.divergence_type == DivergenceType.REGULAR_BULLISH for d in div_data)
        bearish_div = any(d.divergence_type == DivergenceType.REGULAR_BEARISH for d in div_data)
        
        if (bullish_div and signal_direction == "BUY") or (bearish_div and signal_direction == "SELL"):
            alpha_score += 15
        elif bullish_div or bearish_div:
            alpha_score -= 10
        
        # Wave analysis (+/- 10)
        if wave_data.wave_count in [1, 3]:
            alpha_score += 10
        elif wave_data.wave_count == 5:
            alpha_score -= 5
        
        # Risk metrics adjustment
        alpha_score += (risk_metrics.expectancy * 100)
        
        # Apply penalties
        alpha_score -= len(risk_factors) * 3
        alpha_score += len(edge_factors) * 2
        
        alpha_score = np.clip(alpha_score, 0, 100)
        
        # Determine grade
        grade = self._calculate_grade(alpha_score)
        
        # Should trade
        should_trade = (
            alpha_score >= 55 and
            grade.value not in ["D", "F"] and
            len(risk_factors) <= 3
        )
        
        # Position multiplier
        if alpha_score >= 80:
            position_multiplier = 1.3
        elif alpha_score >= 70:
            position_multiplier = 1.1
        elif alpha_score >= 60:
            position_multiplier = 0.9
        else:
            position_multiplier = 0.6
        
        # Calculate optimal entry and exits
        optimal_entry, stop_loss, targets = self._calculate_levels(
            signal_direction, current_price, highs, lows, mp_data, liq_zones
        )
        
        # Risk/Reward
        if signal_direction == "BUY":
            risk = current_price - stop_loss
            reward = targets[0] - current_price if targets else risk * 2
        else:
            risk = stop_loss - current_price
            reward = current_price - targets[0] if targets else risk * 2
        
        risk_reward = reward / (risk + 1e-10)
        
        # Confidence
        confidence = alpha_score
        
        # Recommendation
        recommendation = self._generate_recommendation(
            should_trade, signal_direction, grade, edge_factors, risk_factors
        )
        
        decision = AlphaDecision(
            should_trade=should_trade,
            direction=signal_direction if should_trade else "WAIT",
            grade=grade,
            alpha_score=alpha_score,
            confidence=confidence,
            position_multiplier=position_multiplier,
            order_flow=of_data,
            liquidity_zones=liq_zones,
            market_profile=mp_data,
            divergences=div_data,
            wave_data=wave_data,
            risk_metrics=risk_metrics,
            optimal_entry=optimal_entry,
            stop_loss=stop_loss,
            targets=targets,
            risk_reward=risk_reward,
            edge_factors=edge_factors,
            risk_factors=risk_factors,
            recommendation=recommendation
        )
        
        self.decision_history.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "decision": decision
        })
        
        return decision
    
    def _calculate_grade(self, score: float) -> TradeGrade:
        """Calculate trade grade from score"""
        
        if score >= 90:
            return TradeGrade.A_PLUS
        elif score >= 80:
            return TradeGrade.A
        elif score >= 70:
            return TradeGrade.B_PLUS
        elif score >= 60:
            return TradeGrade.B
        elif score >= 50:
            return TradeGrade.C
        elif score >= 40:
            return TradeGrade.D
        else:
            return TradeGrade.F
    
    def _calculate_levels(
        self,
        direction: str,
        current: float,
        highs: np.ndarray,
        lows: np.ndarray,
        mp: MarketProfileData,
        zones: List[LiquidityZone]
    ) -> Tuple[float, float, List[float]]:
        """Calculate optimal entry, SL, and targets"""
        
        atr = np.mean(highs - lows)
        
        if direction == "BUY":
            # Entry at slight pullback
            optimal_entry = current - atr * 0.3
            
            # Stop below recent low or sell-side liquidity
            sell_side_zones = [z.price_level for z in zones if z.zone_type == LiquidityType.SELL_SIDE]
            recent_low = np.min(lows[-20:])
            stop_loss = min(sell_side_zones + [recent_low]) - atr * 0.5 if sell_side_zones else recent_low - atr
            
            # Targets at resistance levels
            targets = [
                mp.value_area_high,
                np.max(highs[-30:]),
                np.max(highs[-50:]) if len(highs) >= 50 else np.max(highs) * 1.02
            ]
            targets = sorted([t for t in targets if t > current])[:3]
            
        else:  # SELL
            optimal_entry = current + atr * 0.3
            
            buy_side_zones = [z.price_level for z in zones if z.zone_type == LiquidityType.BUY_SIDE]
            recent_high = np.max(highs[-20:])
            stop_loss = max(buy_side_zones + [recent_high]) + atr * 0.5 if buy_side_zones else recent_high + atr
            
            targets = [
                mp.value_area_low,
                np.min(lows[-30:]),
                np.min(lows[-50:]) if len(lows) >= 50 else np.min(lows) * 0.98
            ]
            targets = sorted([t for t in targets if t < current], reverse=True)[:3]
        
        if not targets:
            targets = [current * (1.02 if direction == "BUY" else 0.98)]
        
        return optimal_entry, stop_loss, targets
    
    def _generate_recommendation(
        self,
        should_trade: bool,
        direction: str,
        grade: TradeGrade,
        edges: List[str],
        risks: List[str]
    ) -> str:
        """Generate trading recommendation"""
        
        if not should_trade:
            if len(risks) > len(edges):
                return f"AVOID: Too many risk factors ({len(risks)}). Wait for better setup."
            else:
                return f"WAIT: Grade {grade.value} insufficient. Need more confluence."
        
        if grade == TradeGrade.A_PLUS:
            return f"EXCELLENT {direction}: A+ setup with {len(edges)} edge factors. Full position recommended."
        elif grade == TradeGrade.A:
            return f"STRONG {direction}: A grade setup. Standard position with tight management."
        elif grade == TradeGrade.B_PLUS:
            return f"GOOD {direction}: B+ setup. Reduced position recommended."
        elif grade == TradeGrade.B:
            return f"ACCEPTABLE {direction}: B grade. Conservative position only."
        else:
            return f"MARGINAL {direction}: Proceed with caution. Minimal position."
    
    def record_trade_result(self, result: float):
        """Record trade result for metrics"""
        self.trade_history.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get engine summary"""
        
        return {
            "total_decisions": len(self.decision_history),
            "trades_recorded": len(self.trade_history),
            "current_metrics": self.risk_calc.calculate(self.trade_history).__dict__ if self.trade_history else {}
        }


# ============================================================
# SINGLETON & FACTORY
# ============================================================

_alpha_engine: Optional[AlphaEngine] = None


def get_alpha_engine() -> AlphaEngine:
    """Get singleton Alpha Engine instance"""
    global _alpha_engine
    if _alpha_engine is None:
        _alpha_engine = AlphaEngine()
    return _alpha_engine


def init_alpha_engine() -> AlphaEngine:
    """Initialize and return Alpha Engine"""
    global _alpha_engine
    _alpha_engine = AlphaEngine()
    return _alpha_engine
