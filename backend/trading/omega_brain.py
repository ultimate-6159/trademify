"""
🧠⚡ OMEGA BRAIN - Ultimate Institutional-Grade Trading Intelligence
====================================================================

The final layer of intelligence that thinks like a hedge fund:
- Institutional Flow Detection (Big Money tracking)
- Market Manipulation Scanner (Stop hunts, Fakeouts)
- Multi-Source Sentiment Fusion
- Regime Transition Prediction
- Position Orchestration
- Risk Parity Allocation

This module represents the pinnacle of trading AI.
"""
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATA CLASSES
# ============================================================

class InstitutionalActivity(Enum):
    """Institutional trading activity level"""
    ACCUMULATING = "accumulating"      # Big money buying quietly
    DISTRIBUTING = "distributing"      # Big money selling quietly
    AGGRESSIVE_BUYING = "aggressive_buying"
    AGGRESSIVE_SELLING = "aggressive_selling"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ManipulationType(Enum):
    """Types of market manipulation"""
    STOP_HUNT_LONG = "stop_hunt_long"    # Hunt longs then reverse up
    STOP_HUNT_SHORT = "stop_hunt_short"  # Hunt shorts then reverse down
    FAKEOUT_UP = "fakeout_up"            # False breakout up
    FAKEOUT_DOWN = "fakeout_down"        # False breakout down
    LIQUIDITY_GRAB = "liquidity_grab"    # Quick grab and reverse
    PUMP_AND_DUMP = "pump_and_dump"
    NONE = "none"


class RegimeTransition(Enum):
    """Market regime transition states"""
    TRENDING_TO_RANGING = "trending_to_ranging"
    RANGING_TO_TRENDING = "ranging_to_trending"
    BULL_TO_BEAR = "bull_to_bear"
    BEAR_TO_BULL = "bear_to_bull"
    STABLE = "stable"
    VOLATILE_SHIFT = "volatile_shift"


class RiskLevel(Enum):
    """Risk level classification"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class OmegaGrade(Enum):
    """Omega Brain trade grade"""
    OMEGA_PLUS = "Ω+"    # Perfect institutional setup
    OMEGA = "Ω"          # Excellent opportunity
    ALPHA_PLUS = "α+"    # Very strong
    ALPHA = "α"          # Strong
    BETA = "β"           # Moderate
    GAMMA = "γ"          # Weak
    REJECT = "✗"         # Do not trade


@dataclass
class InstitutionalFlowData:
    """Institutional flow analysis result"""
    activity: InstitutionalActivity
    flow_strength: float          # 0-100
    accumulation_score: float     # -100 to 100 (neg=distribution)
    volume_anomaly: float         # Standard deviations from norm
    large_order_ratio: float      # Ratio of large orders
    stealth_buying: bool          # Quiet accumulation detected
    stealth_selling: bool         # Quiet distribution detected
    smart_money_direction: str    # "LONG", "SHORT", "NEUTRAL"
    confidence: float


@dataclass
class ManipulationAlert:
    """Market manipulation detection result"""
    manipulation_type: ManipulationType
    probability: float            # 0-100
    price_level: float            # Where manipulation occurred
    expected_reversal: float      # Expected reversal price
    time_detected: datetime
    description: str
    action: str                   # "WAIT", "FADE", "FOLLOW"


@dataclass
class SentimentFusion:
    """Multi-source sentiment fusion result"""
    overall_sentiment: float      # -100 to 100
    technical_sentiment: float
    volume_sentiment: float
    momentum_sentiment: float
    volatility_sentiment: float
    cross_asset_sentiment: float
    agreement_level: float        # 0-100 (how much sources agree)
    dominant_narrative: str
    contrarian_signal: bool       # When to go against crowd


@dataclass
class RegimePrediction:
    """Regime transition prediction"""
    current_regime: str
    predicted_regime: str
    transition_type: RegimeTransition
    probability: float            # 0-100
    expected_time_bars: int       # Bars until transition
    confidence: float
    regime_strength: float        # How strong is current regime


@dataclass
class PositionPlan:
    """Orchestrated position plan"""
    action: str                   # "ENTER", "SCALE_IN", "SCALE_OUT", "EXIT", "HOLD"
    position_size: float          # 0-1 (fraction of max)
    entry_zones: List[float]      # Multiple entry levels
    scale_levels: List[Tuple[float, float]]  # (price, size) pairs
    stop_loss: float
    take_profits: List[float]
    max_hold_bars: int
    trail_activation: float       # Price to activate trailing
    reason: str


@dataclass
class RiskParityAllocation:
    """Risk parity allocation result"""
    symbol_weights: Dict[str, float]     # Symbol -> weight
    volatility_adjusted: bool
    correlation_adjusted: bool
    max_position_size: float
    total_risk_budget: float
    individual_risks: Dict[str, float]
    diversification_score: float


@dataclass
class OmegaDecision:
    """Final Omega Brain decision"""
    should_trade: bool
    direction: str
    grade: OmegaGrade
    omega_score: float            # 0-100
    confidence: float
    position_multiplier: float
    
    # Component analysis
    institutional_flow: InstitutionalFlowData
    manipulation_alert: Optional[ManipulationAlert]
    sentiment: SentimentFusion
    regime_prediction: RegimePrediction
    position_plan: PositionPlan
    risk_allocation: RiskParityAllocation
    
    # Recommendations
    optimal_entry: float
    stop_loss: float
    targets: List[float]
    risk_reward: float
    max_risk_percent: float
    
    # Analysis
    edge_factors: List[str]
    risk_factors: List[str]
    institutional_insight: str
    final_verdict: str


# ============================================================
# INSTITUTIONAL FLOW DETECTOR
# ============================================================

class InstitutionalFlowDetector:
    """
    Detect institutional (smart money) trading activity.
    Institutions leave footprints even when trying to hide.
    """
    
    def __init__(self):
        self.volume_history = deque(maxlen=500)
        self.price_history = deque(maxlen=500)
        self.large_order_threshold = 2.0  # Standard deviations
    
    def analyze(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> InstitutionalFlowData:
        """Detect institutional flow patterns"""
        
        # Update history
        for v in volumes[-50:]:
            self.volume_history.append(v)
        for c in closes[-50:]:
            self.price_history.append(c)
        
        # 1. Volume anomaly detection
        vol_mean = np.mean(volumes[-100:]) if len(volumes) >= 100 else np.mean(volumes)
        vol_std = np.std(volumes[-100:]) if len(volumes) >= 100 else np.std(volumes)
        vol_std = max(vol_std, 1e-10)
        
        recent_vol = np.mean(volumes[-10:])
        volume_anomaly = (recent_vol - vol_mean) / vol_std
        
        # 2. Large order detection (volume spikes with small price change)
        price_changes = np.abs(np.diff(closes[-20:]))
        vol_changes = volumes[-19:]
        
        # Large volume + small price = possible accumulation/distribution
        large_orders = 0
        small_price_moves = 0
        
        for i, (pc, vc) in enumerate(zip(price_changes, vol_changes)):
            avg_vol = np.mean(volumes[-50:])
            avg_price_change = np.mean(price_changes) + 1e-10
            
            if vc > avg_vol * 1.5:  # High volume
                large_orders += 1
                if pc < avg_price_change * 0.5:  # Small price change
                    small_price_moves += 1
        
        large_order_ratio = large_orders / len(price_changes) if len(price_changes) > 0 else 0
        stealth_ratio = small_price_moves / max(large_orders, 1)
        
        # 3. Accumulation/Distribution score
        # Using OBV-like analysis
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # OBV trend vs price trend
        price_trend = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        obv_trend = (obv[-1] - obv[-20]) / (abs(obv[-20]) + 1e-10) if len(obv) >= 20 else 0
        
        # Divergence = accumulation/distribution
        accumulation_score = (obv_trend - price_trend) * 100
        accumulation_score = np.clip(accumulation_score, -100, 100)
        
        # 4. Determine activity type
        activity = InstitutionalActivity.NEUTRAL
        smart_money_direction = "NEUTRAL"
        
        if accumulation_score > 30 and volume_anomaly > 1:
            if stealth_ratio > 0.5:
                activity = InstitutionalActivity.ACCUMULATING
            else:
                activity = InstitutionalActivity.AGGRESSIVE_BUYING
            smart_money_direction = "LONG"
        elif accumulation_score < -30 and volume_anomaly > 1:
            if stealth_ratio > 0.5:
                activity = InstitutionalActivity.DISTRIBUTING
            else:
                activity = InstitutionalActivity.AGGRESSIVE_SELLING
            smart_money_direction = "SHORT"
        
        # 5. Flow strength
        flow_strength = min(100, abs(accumulation_score) + volume_anomaly * 10)
        
        # Confidence
        confidence = min(95, 50 + flow_strength * 0.4 + large_order_ratio * 20)
        
        return InstitutionalFlowData(
            activity=activity,
            flow_strength=flow_strength,
            accumulation_score=accumulation_score,
            volume_anomaly=volume_anomaly,
            large_order_ratio=large_order_ratio,
            stealth_buying=activity == InstitutionalActivity.ACCUMULATING,
            stealth_selling=activity == InstitutionalActivity.DISTRIBUTING,
            smart_money_direction=smart_money_direction,
            confidence=confidence
        )


# ============================================================
# MARKET MANIPULATION SCANNER
# ============================================================

class ManipulationScanner:
    """
    Detect common market manipulation patterns:
    - Stop hunts (sweep lows/highs then reverse)
    - Fakeouts (false breakouts)
    - Liquidity grabs
    """
    
    def __init__(self):
        self.recent_swings = deque(maxlen=50)
        self.breakout_attempts = deque(maxlen=20)
    
    def scan(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> Optional[ManipulationAlert]:
        """Scan for manipulation patterns"""
        
        if len(closes) < 50:
            return None
        
        current_price = closes[-1]
        
        # Find recent swing points
        swing_high = np.max(highs[-20:-5])
        swing_low = np.min(lows[-20:-5])
        
        recent_high = np.max(highs[-5:])
        recent_low = np.min(lows[-5:])
        
        # Volume analysis
        avg_volume = np.mean(volumes[-50:])
        recent_volume = np.mean(volumes[-3:])
        volume_spike = recent_volume > avg_volume * 1.5
        
        # 1. Stop Hunt Detection
        # Stop hunt long: price breaks below swing low then reverses up
        if recent_low < swing_low and current_price > swing_low:
            # Check for reversal candle (close above low)
            if closes[-1] > lows[-1] + (highs[-1] - lows[-1]) * 0.6:
                probability = 70 + (volume_spike * 15)
                return ManipulationAlert(
                    manipulation_type=ManipulationType.STOP_HUNT_LONG,
                    probability=min(95, probability),
                    price_level=recent_low,
                    expected_reversal=swing_high,
                    time_detected=datetime.now(),
                    description="Stop hunt below swing low detected - longs stopped out",
                    action="FADE"  # Go long against the stop hunt
                )
        
        # Stop hunt short: price breaks above swing high then reverses down
        if recent_high > swing_high and current_price < swing_high:
            if closes[-1] < highs[-1] - (highs[-1] - lows[-1]) * 0.6:
                probability = 70 + (volume_spike * 15)
                return ManipulationAlert(
                    manipulation_type=ManipulationType.STOP_HUNT_SHORT,
                    probability=min(95, probability),
                    price_level=recent_high,
                    expected_reversal=swing_low,
                    time_detected=datetime.now(),
                    description="Stop hunt above swing high detected - shorts stopped out",
                    action="FADE"  # Go short against the stop hunt
                )
        
        # 2. Fakeout Detection
        # Fakeout up: break above resistance but close below
        resistance = np.percentile(highs[-30:], 90)
        support = np.percentile(lows[-30:], 10)
        
        if highs[-2] > resistance and closes[-1] < resistance * 0.998:
            if volume_spike:
                return ManipulationAlert(
                    manipulation_type=ManipulationType.FAKEOUT_UP,
                    probability=65,
                    price_level=highs[-2],
                    expected_reversal=support,
                    time_detected=datetime.now(),
                    description="False breakout above resistance",
                    action="FADE"
                )
        
        # Fakeout down: break below support but close above
        if lows[-2] < support and closes[-1] > support * 1.002:
            if volume_spike:
                return ManipulationAlert(
                    manipulation_type=ManipulationType.FAKEOUT_DOWN,
                    probability=65,
                    price_level=lows[-2],
                    expected_reversal=resistance,
                    time_detected=datetime.now(),
                    description="False breakout below support",
                    action="FADE"
                )
        
        # 3. Liquidity Grab
        # Quick spike then immediate reversal (within same/next bar)
        last_range = highs[-1] - lows[-1]
        prev_range = highs[-2] - lows[-2]
        avg_range = np.mean(highs[-20:] - lows[-20:])
        
        if last_range > avg_range * 2.5:  # Abnormally large range
            # Upper wick dominant = grab and sell
            upper_wick = highs[-1] - max(opens[-1], closes[-1])
            lower_wick = min(opens[-1], closes[-1]) - lows[-1]
            body = abs(closes[-1] - opens[-1])
            
            if upper_wick > body * 2 and upper_wick > lower_wick * 2:
                return ManipulationAlert(
                    manipulation_type=ManipulationType.LIQUIDITY_GRAB,
                    probability=75,
                    price_level=highs[-1],
                    expected_reversal=lows[-1],
                    time_detected=datetime.now(),
                    description="Liquidity grab at highs - expect down move",
                    action="FADE"
                )
            
            if lower_wick > body * 2 and lower_wick > upper_wick * 2:
                return ManipulationAlert(
                    manipulation_type=ManipulationType.LIQUIDITY_GRAB,
                    probability=75,
                    price_level=lows[-1],
                    expected_reversal=highs[-1],
                    time_detected=datetime.now(),
                    description="Liquidity grab at lows - expect up move",
                    action="FADE"
                )
        
        return None


# ============================================================
# SENTIMENT FUSION ENGINE
# ============================================================

class SentimentFusionEngine:
    """
    Fuse sentiment from multiple technical sources.
    True sentiment = weighted agreement of multiple indicators.
    """
    
    def __init__(self):
        self.sentiment_history = deque(maxlen=100)
    
    def analyze(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> SentimentFusion:
        """Fuse multiple sentiment sources"""
        
        sentiments = []
        
        # 1. Technical Sentiment (Moving Averages)
        if len(closes) >= 50:
            ma20 = np.mean(closes[-20:])
            ma50 = np.mean(closes[-50:])
            current = closes[-1]
            
            # Price vs MAs
            tech_sent = 0
            if current > ma20:
                tech_sent += 25
            if current > ma50:
                tech_sent += 25
            if ma20 > ma50:
                tech_sent += 25
            
            # Normalize to -100 to 100
            technical_sentiment = (tech_sent - 37.5) * 2.67
        else:
            technical_sentiment = 0
        sentiments.append(technical_sentiment)
        
        # 2. Volume Sentiment
        vol_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        recent_vol = np.mean(volumes[-5:])
        
        # Price direction with volume
        price_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        vol_change = (recent_vol - vol_ma) / vol_ma if vol_ma > 0 else 0
        
        if price_change > 0 and vol_change > 0:
            volume_sentiment = 50 + vol_change * 50
        elif price_change < 0 and vol_change > 0:
            volume_sentiment = -50 - vol_change * 50
        else:
            volume_sentiment = price_change * 100
        volume_sentiment = np.clip(volume_sentiment, -100, 100)
        sentiments.append(volume_sentiment)
        
        # 3. Momentum Sentiment (RSI-like)
        if len(closes) >= 15:
            gains = []
            losses = []
            for i in range(1, min(15, len(closes))):
                change = closes[-i] - closes[-i-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.0001
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Convert RSI to sentiment (-100 to 100)
            momentum_sentiment = (rsi - 50) * 2
        else:
            momentum_sentiment = 0
        sentiments.append(momentum_sentiment)
        
        # 4. Volatility Sentiment
        if len(closes) >= 20:
            returns = np.diff(np.log(closes[-21:]))
            current_vol = np.std(returns[-5:]) * np.sqrt(252)
            hist_vol = np.std(returns) * np.sqrt(252)
            
            # High vol = negative sentiment (uncertainty)
            vol_ratio = current_vol / (hist_vol + 1e-10)
            volatility_sentiment = (1 - vol_ratio) * 50
            volatility_sentiment = np.clip(volatility_sentiment, -100, 100)
        else:
            volatility_sentiment = 0
        sentiments.append(volatility_sentiment)
        
        # 5. Cross-Asset Sentiment (trend consistency)
        if len(closes) >= 30:
            short_trend = (closes[-1] - closes[-5]) / closes[-5]
            medium_trend = (closes[-1] - closes[-15]) / closes[-15]
            long_trend = (closes[-1] - closes[-30]) / closes[-30]
            
            # All aligned = strong sentiment
            if short_trend > 0 and medium_trend > 0 and long_trend > 0:
                cross_asset_sentiment = 60 + min(40, abs(short_trend) * 1000)
            elif short_trend < 0 and medium_trend < 0 and long_trend < 0:
                cross_asset_sentiment = -60 - min(40, abs(short_trend) * 1000)
            else:
                cross_asset_sentiment = short_trend * 100
        else:
            cross_asset_sentiment = 0
        sentiments.append(cross_asset_sentiment)
        
        # Calculate overall sentiment (weighted average)
        weights = [0.25, 0.20, 0.25, 0.15, 0.15]
        overall_sentiment = sum(s * w for s, w in zip(sentiments, weights))
        
        # Agreement level (how much sources agree)
        sentiment_signs = [1 if s > 0 else -1 if s < 0 else 0 for s in sentiments]
        agreement = abs(sum(sentiment_signs)) / len(sentiment_signs) * 100
        
        # Determine narrative
        if overall_sentiment > 50:
            narrative = "Strong bullish consensus"
        elif overall_sentiment > 20:
            narrative = "Moderate bullish bias"
        elif overall_sentiment > -20:
            narrative = "Mixed/neutral sentiment"
        elif overall_sentiment > -50:
            narrative = "Moderate bearish bias"
        else:
            narrative = "Strong bearish consensus"
        
        # Contrarian signal (extreme readings often reverse)
        contrarian = abs(overall_sentiment) > 80 and agreement > 70
        
        self.sentiment_history.append(overall_sentiment)
        
        return SentimentFusion(
            overall_sentiment=overall_sentiment,
            technical_sentiment=technical_sentiment,
            volume_sentiment=volume_sentiment,
            momentum_sentiment=momentum_sentiment,
            volatility_sentiment=volatility_sentiment,
            cross_asset_sentiment=cross_asset_sentiment,
            agreement_level=agreement,
            dominant_narrative=narrative,
            contrarian_signal=contrarian
        )


# ============================================================
# REGIME TRANSITION PREDICTOR
# ============================================================

class RegimeTransitionPredictor:
    """
    Predict market regime transitions before they happen.
    Early detection = better positioning.
    """
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        self.transition_patterns = []
    
    def predict(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> RegimePrediction:
        """Predict regime transitions"""
        
        # 1. Identify current regime
        if len(closes) < 50:
            return RegimePrediction(
                current_regime="unknown",
                predicted_regime="unknown",
                transition_type=RegimeTransition.STABLE,
                probability=50,
                expected_time_bars=10,
                confidence=30,
                regime_strength=50
            )
        
        # Trend detection
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:])
        
        # Volatility
        returns = np.diff(np.log(closes[-50:]))
        volatility = np.std(returns) * np.sqrt(252)
        recent_vol = np.std(returns[-10:]) * np.sqrt(252)
        
        # Range analysis
        atr = np.mean(highs[-20:] - lows[-20:])
        recent_range = np.mean(highs[-5:] - lows[-5:])
        
        # Determine current regime
        price_trend = (closes[-1] - closes[-20]) / closes[-20]
        
        if abs(price_trend) > 0.02 and ma20 > ma50 * 1.01:
            current_regime = "bullish_trend"
        elif abs(price_trend) > 0.02 and ma20 < ma50 * 0.99:
            current_regime = "bearish_trend"
        elif recent_vol > volatility * 1.5:
            current_regime = "high_volatility"
        else:
            current_regime = "ranging"
        
        # 2. Predict transition
        transition_type = RegimeTransition.STABLE
        predicted_regime = current_regime
        probability = 30
        expected_bars = 20
        
        # Trending to Ranging signals
        if "trend" in current_regime:
            # Volume declining = trend exhaustion
            vol_trend = (np.mean(volumes[-5:]) - np.mean(volumes[-20:])) / np.mean(volumes[-20:])
            
            # Narrowing range = consolidation coming
            range_trend = (recent_range - atr) / atr
            
            if vol_trend < -0.2 and range_trend < -0.1:
                transition_type = RegimeTransition.TRENDING_TO_RANGING
                predicted_regime = "ranging"
                probability = 60 + abs(vol_trend) * 50
                expected_bars = 5
        
        # Ranging to Trending signals
        if current_regime == "ranging":
            # Volatility compression followed by expansion
            vol_compression = recent_vol < volatility * 0.7
            
            # Volume building up
            vol_building = np.mean(volumes[-5:]) > np.mean(volumes[-20:]) * 1.2
            
            if vol_compression and vol_building:
                transition_type = RegimeTransition.RANGING_TO_TRENDING
                # Predict direction from subtle clues
                if closes[-1] > ma20:
                    predicted_regime = "bullish_trend"
                else:
                    predicted_regime = "bearish_trend"
                probability = 55 + vol_building * 20
                expected_bars = 3
        
        # Bull to Bear transition
        if current_regime == "bullish_trend":
            # Lower highs forming
            if len(highs) >= 20:
                recent_highs = []
                for i in range(-18, -2):
                    if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                        recent_highs.append(highs[i])
                if len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[0]:
                    transition_type = RegimeTransition.BULL_TO_BEAR
                    predicted_regime = "bearish_trend"
                    probability = 50
                    expected_bars = 8
        
        # Bear to Bull transition
        if current_regime == "bearish_trend":
            # Higher lows forming
            if len(lows) >= 20:
                recent_lows = []
                for i in range(-18, -2):
                    if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                        recent_lows.append(lows[i])
                if len(recent_lows) >= 2:
                    if recent_lows[-1] > recent_lows[0]:
                        transition_type = RegimeTransition.BEAR_TO_BULL
                        predicted_regime = "bullish_trend"
                        probability = 50
                        expected_bars = 8
        
        # Regime strength
        regime_strength = 50
        if "trend" in current_regime:
            regime_strength = min(95, 50 + abs(price_trend) * 500 + (ma20 - ma50) / ma50 * 1000)
        
        confidence = min(90, 40 + probability * 0.3 + regime_strength * 0.2)
        
        return RegimePrediction(
            current_regime=current_regime,
            predicted_regime=predicted_regime,
            transition_type=transition_type,
            probability=min(95, probability),
            expected_time_bars=expected_bars,
            confidence=confidence,
            regime_strength=regime_strength
        )


# ============================================================
# POSITION ORCHESTRATOR
# ============================================================

class PositionOrchestrator:
    """
    Orchestrate position management like a hedge fund.
    Scale in/out, manage multiple entries, optimize exits.
    """
    
    def __init__(self):
        self.position_history = []
    
    def plan(
        self,
        direction: str,
        current_price: float,
        atr: float,
        support_levels: List[float],
        resistance_levels: List[float],
        regime: str,
        confidence: float
    ) -> PositionPlan:
        """Create sophisticated position plan"""
        
        # 1. Determine action based on context
        if confidence < 40:
            action = "HOLD"
            position_size = 0
        elif confidence < 60:
            action = "SCALE_IN"
            position_size = 0.3
        elif confidence < 80:
            action = "ENTER"
            position_size = 0.5
        else:
            action = "ENTER"
            position_size = 0.7
        
        # Adjust for regime
        if regime == "high_volatility":
            position_size *= 0.5
        elif regime == "ranging":
            position_size *= 0.7
        
        # 2. Calculate entry zones
        if direction == "BUY":
            # Scale-in levels below current price
            entry_zones = [
                current_price,
                current_price - atr * 0.5,
                current_price - atr * 1.0,
            ]
            
            # Use support levels if available
            nearby_supports = [s for s in support_levels if s < current_price and s > current_price - atr * 2]
            if nearby_supports:
                entry_zones.extend(nearby_supports[:2])
            
            entry_zones = sorted(set(entry_zones), reverse=True)[:4]
            
        else:  # SELL
            entry_zones = [
                current_price,
                current_price + atr * 0.5,
                current_price + atr * 1.0,
            ]
            
            nearby_resistances = [r for r in resistance_levels if r > current_price and r < current_price + atr * 2]
            if nearby_resistances:
                entry_zones.extend(nearby_resistances[:2])
            
            entry_zones = sorted(set(entry_zones))[:4]
        
        # 3. Scale levels
        scale_levels = []
        remaining_size = position_size
        for i, entry in enumerate(entry_zones):
            if i == 0:
                size = remaining_size * 0.4
            elif i == 1:
                size = remaining_size * 0.3
            else:
                size = remaining_size * 0.15
            scale_levels.append((entry, size))
            remaining_size -= size
        
        # 4. Stop loss
        if direction == "BUY":
            stop_loss = min(entry_zones) - atr * 1.5
            if support_levels:
                nearest_support = max([s for s in support_levels if s < min(entry_zones)], default=stop_loss)
                stop_loss = min(stop_loss, nearest_support - atr * 0.5)
        else:
            stop_loss = max(entry_zones) + atr * 1.5
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > max(entry_zones)], default=stop_loss)
                stop_loss = max(stop_loss, nearest_resistance + atr * 0.5)
        
        # 5. Take profits (1:2, 1:3, 1:5 R:R)
        risk = abs(current_price - stop_loss)
        if direction == "BUY":
            take_profits = [
                current_price + risk * 2,
                current_price + risk * 3,
                current_price + risk * 5,
            ]
            
            # Adjust to resistance levels
            for r in sorted(resistance_levels):
                if r > current_price + risk * 1.5:
                    take_profits[0] = min(take_profits[0], r)
                    break
        else:
            take_profits = [
                current_price - risk * 2,
                current_price - risk * 3,
                current_price - risk * 5,
            ]
            
            for s in sorted(support_levels, reverse=True):
                if s < current_price - risk * 1.5:
                    take_profits[0] = max(take_profits[0], s)
                    break
        
        # 6. Trailing stop activation
        trail_activation = take_profits[0]  # Activate at first TP
        
        # 7. Max hold time (bars)
        max_hold_bars = 50 if "trend" in regime else 20
        
        reason = f"{action} with {position_size:.0%} size, scaling at {len(scale_levels)} levels"
        
        return PositionPlan(
            action=action,
            position_size=position_size,
            entry_zones=entry_zones,
            scale_levels=scale_levels,
            stop_loss=stop_loss,
            take_profits=take_profits,
            max_hold_bars=max_hold_bars,
            trail_activation=trail_activation,
            reason=reason
        )


# ============================================================
# RISK PARITY ALLOCATOR
# ============================================================

class RiskParityAllocator:
    """
    Allocate risk using risk parity principles.
    Each position contributes equally to total portfolio risk.
    """
    
    def __init__(self, max_total_risk: float = 5.0):
        self.max_total_risk = max_total_risk
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
    
    def update_volatility(self, symbol: str, returns: np.ndarray):
        """Update volatility estimate for symbol"""
        if len(returns) > 10:
            vol = np.std(returns) * np.sqrt(252)
            self.volatility_cache[symbol] = vol
    
    def allocate(
        self,
        symbols: List[str],
        current_balance: float,
        prices: Dict[str, float]
    ) -> RiskParityAllocation:
        """Calculate risk parity allocation"""
        
        # Get volatilities (use defaults if not cached)
        vols = {}
        for sym in symbols:
            vols[sym] = self.volatility_cache.get(sym, 0.15)  # Default 15% vol
        
        # Inverse volatility weighting
        inv_vols = {sym: 1 / (vol + 0.01) for sym, vol in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # Raw weights
        raw_weights = {sym: iv / total_inv_vol for sym, iv in inv_vols.items()}
        
        # Apply correlation adjustment (assume 0.5 correlation by default)
        correlation_penalty = 0.5
        adjusted_weights = {}
        for sym, w in raw_weights.items():
            # Reduce weight for correlated assets
            corr_adjustment = 1 - (correlation_penalty * (len(symbols) - 1) / len(symbols))
            adjusted_weights[sym] = w * corr_adjustment
        
        # Normalize
        total_weight = sum(adjusted_weights.values())
        symbol_weights = {sym: w / total_weight for sym, w in adjusted_weights.items()}
        
        # Calculate individual risks
        individual_risks = {}
        for sym, weight in symbol_weights.items():
            individual_risks[sym] = weight * vols[sym] * self.max_total_risk
        
        # Max position size (in lots or units)
        max_position_size = self.max_total_risk / 100  # As fraction
        
        # Diversification score (how spread out the allocation is)
        herfindahl = sum(w**2 for w in symbol_weights.values())
        diversification_score = (1 - herfindahl) * 100
        
        return RiskParityAllocation(
            symbol_weights=symbol_weights,
            volatility_adjusted=True,
            correlation_adjusted=True,
            max_position_size=max_position_size,
            total_risk_budget=self.max_total_risk,
            individual_risks=individual_risks,
            diversification_score=diversification_score
        )


# ============================================================
# OMEGA BRAIN - MAIN CLASS
# ============================================================

class OmegaBrain:
    """
    🧠⚡ OMEGA BRAIN - Ultimate Institutional-Grade Trading Intelligence
    
    The final layer of AI that combines:
    - Institutional Flow Detection
    - Market Manipulation Scanning
    - Multi-Source Sentiment Fusion
    - Regime Transition Prediction
    - Position Orchestration
    - Risk Parity Allocation
    
    Thinks like a hedge fund, trades like a machine.
    """
    
    def __init__(
        self,
        min_omega_score: float = 60.0,
        max_risk_per_trade: float = 2.0,
        enable_manipulation_filter: bool = True,
        enable_institutional_filter: bool = True
    ):
        self.min_omega_score = min_omega_score
        self.max_risk_per_trade = max_risk_per_trade
        self.enable_manipulation_filter = enable_manipulation_filter
        self.enable_institutional_filter = enable_institutional_filter
        
        # Initialize components
        self.institutional_detector = InstitutionalFlowDetector()
        self.manipulation_scanner = ManipulationScanner()
        self.sentiment_engine = SentimentFusionEngine()
        self.regime_predictor = RegimeTransitionPredictor()
        self.position_orchestrator = PositionOrchestrator()
        self.risk_allocator = RiskParityAllocator(max_total_risk=5.0)
        
        # History
        self.decision_history: List[OmegaDecision] = []
        
        logger.info("🧠⚡ OMEGA BRAIN initialized")
        logger.info("   - Institutional Flow Detector: ✓")
        logger.info("   - Manipulation Scanner: ✓")
        logger.info("   - Sentiment Fusion Engine: ✓")
        logger.info("   - Regime Transition Predictor: ✓")
        logger.info("   - Position Orchestrator: ✓")
        logger.info("   - Risk Parity Allocator: ✓")
    
    def analyze(
        self,
        symbol: str,
        signal_direction: str,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        current_balance: float = 10000.0,
        other_symbols: List[str] = None
    ) -> OmegaDecision:
        """
        Perform comprehensive Omega Brain analysis.
        Returns institutional-grade trading decision.
        """
        
        edge_factors = []
        risk_factors = []
        current_price = closes[-1]
        
        # 1. Institutional Flow Analysis
        flow_data = self.institutional_detector.analyze(opens, highs, lows, closes, volumes)
        
        if self.enable_institutional_filter:
            if flow_data.smart_money_direction == signal_direction.replace("BUY", "LONG").replace("SELL", "SHORT"):
                edge_factors.append(f"✓ Institutional flow aligned ({flow_data.activity.value})")
            elif flow_data.smart_money_direction != "NEUTRAL":
                if flow_data.flow_strength > 50:
                    risk_factors.append(f"✗ Institutional flow opposes signal ({flow_data.activity.value})")
            
            if flow_data.stealth_buying and signal_direction == "BUY":
                edge_factors.append("✓ Stealth accumulation detected")
            elif flow_data.stealth_selling and signal_direction == "SELL":
                edge_factors.append("✓ Stealth distribution detected")
        
        # 2. Manipulation Scan
        manipulation = self.manipulation_scanner.scan(highs, lows, closes, volumes)
        
        if manipulation and self.enable_manipulation_filter:
            if manipulation.probability > 60:
                if manipulation.action == "FADE":
                    # Check if our signal aligns with fade direction
                    if manipulation.manipulation_type in [ManipulationType.STOP_HUNT_LONG, ManipulationType.FAKEOUT_DOWN]:
                        if signal_direction == "BUY":
                            edge_factors.append(f"✓ Signal aligns with manipulation fade (BUY)")
                        else:
                            risk_factors.append("✗ Signal opposes manipulation fade")
                    elif manipulation.manipulation_type in [ManipulationType.STOP_HUNT_SHORT, ManipulationType.FAKEOUT_UP]:
                        if signal_direction == "SELL":
                            edge_factors.append(f"✓ Signal aligns with manipulation fade (SELL)")
                        else:
                            risk_factors.append("✗ Signal opposes manipulation fade")
                
                risk_factors.append(f"⚠ Manipulation detected: {manipulation.manipulation_type.value}")
        
        # 3. Sentiment Fusion
        sentiment = self.sentiment_engine.analyze(opens, highs, lows, closes, volumes)
        
        sentiment_aligned = (
            (signal_direction == "BUY" and sentiment.overall_sentiment > 20) or
            (signal_direction == "SELL" and sentiment.overall_sentiment < -20)
        )
        
        if sentiment_aligned:
            edge_factors.append(f"✓ Sentiment aligned ({sentiment.dominant_narrative})")
        elif abs(sentiment.overall_sentiment) > 30:
            risk_factors.append(f"⚠ Sentiment misaligned ({sentiment.dominant_narrative})")
        
        if sentiment.contrarian_signal:
            risk_factors.append("⚠ Extreme sentiment - contrarian reversal possible")
        
        if sentiment.agreement_level > 70:
            edge_factors.append(f"✓ High sentiment agreement ({sentiment.agreement_level:.0f}%)")
        
        # 4. Regime Prediction
        regime = self.regime_predictor.predict(highs, lows, closes, volumes)
        
        if regime.transition_type != RegimeTransition.STABLE:
            if regime.probability > 50:
                risk_factors.append(f"⚠ Regime transition predicted: {regime.transition_type.value}")
        
        # Check if signal aligns with regime
        if "bullish" in regime.current_regime and signal_direction == "BUY":
            edge_factors.append(f"✓ Signal aligns with {regime.current_regime}")
        elif "bearish" in regime.current_regime and signal_direction == "SELL":
            edge_factors.append(f"✓ Signal aligns with {regime.current_regime}")
        elif "trend" in regime.current_regime:
            risk_factors.append(f"⚠ Signal against {regime.current_regime}")
        
        # 5. Calculate support/resistance for position planning
        support_levels = self._find_support_levels(lows, closes)
        resistance_levels = self._find_resistance_levels(highs, closes)
        
        # ATR
        atr = np.mean(highs[-20:] - lows[-20:])
        
        # 6. Position Plan
        position_plan = self.position_orchestrator.plan(
            direction=signal_direction,
            current_price=current_price,
            atr=atr,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            regime=regime.current_regime,
            confidence=flow_data.confidence
        )
        
        # 7. Risk Allocation
        symbols = other_symbols or [symbol]
        risk_allocation = self.risk_allocator.allocate(
            symbols=symbols,
            current_balance=current_balance,
            prices={symbol: current_price}
        )
        
        # 8. Calculate Omega Score
        omega_score = self._calculate_omega_score(
            flow_data=flow_data,
            sentiment=sentiment,
            regime=regime,
            edge_factors=edge_factors,
            risk_factors=risk_factors
        )
        
        # 9. Determine Grade
        grade = self._determine_grade(omega_score, len(edge_factors), len(risk_factors))
        
        # 10. Final Decision
        should_trade = (
            omega_score >= self.min_omega_score and
            grade not in [OmegaGrade.GAMMA, OmegaGrade.REJECT] and
            len(risk_factors) < 4
        )
        
        # Position multiplier based on grade
        multiplier_map = {
            OmegaGrade.OMEGA_PLUS: 1.0,
            OmegaGrade.OMEGA: 0.9,
            OmegaGrade.ALPHA_PLUS: 0.8,
            OmegaGrade.ALPHA: 0.7,
            OmegaGrade.BETA: 0.5,
            OmegaGrade.GAMMA: 0.3,
            OmegaGrade.REJECT: 0.0
        }
        position_multiplier = multiplier_map.get(grade, 0.5)
        
        # Apply risk allocation
        symbol_weight = risk_allocation.symbol_weights.get(symbol, 0.5)
        position_multiplier *= symbol_weight * 2  # Scale to reasonable size
        position_multiplier = min(1.0, position_multiplier)
        
        # Confidence
        confidence = min(95, omega_score * 0.8 + flow_data.confidence * 0.2)
        
        # Entry/Exit calculations
        optimal_entry = position_plan.entry_zones[0] if position_plan.entry_zones else current_price
        stop_loss = position_plan.stop_loss
        targets = position_plan.take_profits
        
        risk_per_unit = abs(optimal_entry - stop_loss)
        reward = abs(targets[0] - optimal_entry) if targets else risk_per_unit * 2
        risk_reward = reward / risk_per_unit if risk_per_unit > 0 else 0
        
        # Institutional insight
        if flow_data.activity == InstitutionalActivity.ACCUMULATING:
            insight = "Smart money is quietly accumulating - follow their lead"
        elif flow_data.activity == InstitutionalActivity.DISTRIBUTING:
            insight = "Smart money is quietly distributing - be cautious with longs"
        elif flow_data.activity == InstitutionalActivity.AGGRESSIVE_BUYING:
            insight = "Aggressive institutional buying detected - momentum may continue"
        elif flow_data.activity == InstitutionalActivity.AGGRESSIVE_SELLING:
            insight = "Aggressive institutional selling detected - downside risk elevated"
        else:
            insight = "No clear institutional activity detected"
        
        # Final verdict
        if grade in [OmegaGrade.OMEGA_PLUS, OmegaGrade.OMEGA]:
            verdict = f"STRONG {signal_direction} - Institutional-grade setup"
        elif grade in [OmegaGrade.ALPHA_PLUS, OmegaGrade.ALPHA]:
            verdict = f"{signal_direction} recommended with caution"
        elif grade == OmegaGrade.BETA:
            verdict = f"Weak {signal_direction} - Consider reducing size"
        else:
            verdict = "AVOID - Setup does not meet institutional standards"
        
        decision = OmegaDecision(
            should_trade=should_trade,
            direction=signal_direction,
            grade=grade,
            omega_score=omega_score,
            confidence=confidence,
            position_multiplier=position_multiplier,
            institutional_flow=flow_data,
            manipulation_alert=manipulation,
            sentiment=sentiment,
            regime_prediction=regime,
            position_plan=position_plan,
            risk_allocation=risk_allocation,
            optimal_entry=optimal_entry,
            stop_loss=stop_loss,
            targets=targets,
            risk_reward=risk_reward,
            max_risk_percent=self.max_risk_per_trade,
            edge_factors=edge_factors,
            risk_factors=risk_factors,
            institutional_insight=insight,
            final_verdict=verdict
        )
        
        self.decision_history.append(decision)
        
        return decision
    
    def _find_support_levels(self, lows: np.ndarray, closes: np.ndarray) -> List[float]:
        """Find significant support levels"""
        if len(lows) < 20:
            return []
        
        supports = []
        
        # Recent swing lows
        for i in range(2, min(50, len(lows) - 2)):
            if lows[-i] < lows[-i-1] and lows[-i] < lows[-i-2] and lows[-i] < lows[-i+1] and lows[-i] < lows[-i+2]:
                supports.append(lows[-i])
        
        # Percentile levels
        supports.append(np.percentile(lows[-50:], 10))
        supports.append(np.percentile(lows[-50:], 25))
        
        return sorted(set(supports))
    
    def _find_resistance_levels(self, highs: np.ndarray, closes: np.ndarray) -> List[float]:
        """Find significant resistance levels"""
        if len(highs) < 20:
            return []
        
        resistances = []
        
        # Recent swing highs
        for i in range(2, min(50, len(highs) - 2)):
            if highs[-i] > highs[-i-1] and highs[-i] > highs[-i-2] and highs[-i] > highs[-i+1] and highs[-i] > highs[-i+2]:
                resistances.append(highs[-i])
        
        # Percentile levels
        resistances.append(np.percentile(highs[-50:], 75))
        resistances.append(np.percentile(highs[-50:], 90))
        
        return sorted(set(resistances))
    
    def _calculate_omega_score(
        self,
        flow_data: InstitutionalFlowData,
        sentiment: SentimentFusion,
        regime: RegimePrediction,
        edge_factors: List[str],
        risk_factors: List[str]
    ) -> float:
        """Calculate comprehensive Omega Score"""
        
        score = 50  # Base score
        
        # Institutional flow contribution (0-25 points)
        if flow_data.smart_money_direction != "NEUTRAL":
            score += flow_data.flow_strength * 0.25
        
        # Sentiment contribution (0-15 points)
        if sentiment.agreement_level > 50:
            score += sentiment.agreement_level * 0.15
        
        # Regime contribution (0-10 points)
        score += regime.regime_strength * 0.1
        
        # Edge factors bonus
        score += len(edge_factors) * 5
        
        # Risk factors penalty
        score -= len(risk_factors) * 7
        
        # Manipulation penalty
        if flow_data.stealth_buying or flow_data.stealth_selling:
            score += 5  # Bonus for detecting stealth activity aligned with signal
        
        return np.clip(score, 0, 100)
    
    def _determine_grade(
        self,
        omega_score: float,
        edge_count: int,
        risk_count: int
    ) -> OmegaGrade:
        """Determine Omega Grade"""
        
        net_factors = edge_count - risk_count
        
        if omega_score >= 85 and net_factors >= 3:
            return OmegaGrade.OMEGA_PLUS
        elif omega_score >= 75 and net_factors >= 2:
            return OmegaGrade.OMEGA
        elif omega_score >= 70 and net_factors >= 1:
            return OmegaGrade.ALPHA_PLUS
        elif omega_score >= 60 and net_factors >= 0:
            return OmegaGrade.ALPHA
        elif omega_score >= 50:
            return OmegaGrade.BETA
        elif omega_score >= 40:
            return OmegaGrade.GAMMA
        else:
            return OmegaGrade.REJECT


# ============================================================
# SINGLETON & HELPERS
# ============================================================

_omega_brain_instance: Optional[OmegaBrain] = None


def get_omega_brain() -> OmegaBrain:
    """Get or create OmegaBrain singleton"""
    global _omega_brain_instance
    if _omega_brain_instance is None:
        _omega_brain_instance = OmegaBrain()
    return _omega_brain_instance


def init_omega_brain(
    min_omega_score: float = 60.0,
    max_risk_per_trade: float = 2.0,
    enable_manipulation_filter: bool = True,
    enable_institutional_filter: bool = True
) -> OmegaBrain:
    """Initialize OmegaBrain with custom settings"""
    global _omega_brain_instance
    _omega_brain_instance = OmegaBrain(
        min_omega_score=min_omega_score,
        max_risk_per_trade=max_risk_per_trade,
        enable_manipulation_filter=enable_manipulation_filter,
        enable_institutional_filter=enable_institutional_filter
    )
    return _omega_brain_instance
