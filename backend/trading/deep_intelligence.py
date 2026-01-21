"""
🧠 Deep Intelligence Module
===========================
Advanced multi-layer intelligence system for trading decisions.

Components:
1. MultiTimeframeAnalyzer - Confluence across M15, H1, H4, D1
2. CrossAssetCorrelation - Track symbol correlations
3. AdaptiveParameterTuner - Self-optimize based on performance
4. PredictiveModel - Short-term price forecasting
5. SessionAnalyzer - Optimal trading sessions
6. ConfluenceEngine - Combine all signals

Author: Trademify AI
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATA CLASSES
# ============================================================

class TimeframeSignal(Enum):
    """Signal from each timeframe"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class TradingSession(Enum):
    """Major trading sessions"""
    ASIAN = "asian"          # 00:00-08:00 UTC
    LONDON = "london"        # 08:00-16:00 UTC
    NEW_YORK = "new_york"    # 13:00-21:00 UTC
    OVERLAP = "overlap"      # 13:00-16:00 UTC (London + NY)
    OFF_HOURS = "off_hours"  # Low liquidity periods


class ConfluenceLevel(Enum):
    """Level of signal confluence"""
    PERFECT = "perfect"      # All timeframes agree
    STRONG = "strong"        # Most timeframes agree
    MODERATE = "moderate"    # Mixed signals
    WEAK = "weak"           # Conflicting signals
    NONE = "none"           # No clear direction


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe"""
    timeframe: str
    signal: TimeframeSignal
    trend_strength: float  # 0-1
    momentum: float        # -1 to 1
    support_resistance: Dict[str, float]
    key_levels: List[float]


@dataclass
class CorrelationData:
    """Correlation between assets"""
    symbol1: str
    symbol2: str
    correlation: float
    rolling_corr: float  # Recent correlation
    is_diverging: bool   # Unusual divergence detected


@dataclass
class PredictionResult:
    """Price prediction result"""
    predicted_direction: str  # "UP", "DOWN", "SIDEWAYS"
    confidence: float         # 0-1
    predicted_move: float     # Expected % move
    support_level: float
    resistance_level: float
    time_horizon: str         # "short", "medium"


@dataclass
class DeepDecision:
    """Final decision from Deep Intelligence"""
    should_trade: bool
    direction: str           # "BUY", "SELL", "WAIT"
    confluence_level: ConfluenceLevel
    confidence: float        # 0-100
    position_multiplier: float  # 0-1.5
    
    # Component scores
    timeframe_score: float   # -2 to 2
    correlation_score: float # -1 to 1
    prediction_score: float  # -1 to 1
    session_score: float     # 0-1
    
    # Warnings/Notes
    warnings: List[str] = field(default_factory=list)
    reasoning: str = ""


# ============================================================
# MULTI-TIMEFRAME ANALYZER
# ============================================================

class MultiTimeframeAnalyzer:
    """
    Analyze multiple timeframes for confluence.
    Timeframes: M15, H1, H4, D1
    """
    
    TIMEFRAMES = ["M15", "H1", "H4", "D1"]
    WEIGHTS = {"M15": 0.15, "H1": 0.30, "H4": 0.35, "D1": 0.20}
    
    def __init__(self):
        self.analyses: Dict[str, Dict[str, TimeframeAnalysis]] = {}
        self.history: deque = deque(maxlen=100)
    
    def analyze_timeframe(
        self,
        timeframe: str,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> TimeframeAnalysis:
        """Analyze a single timeframe"""
        
        if len(prices) < 20:
            return TimeframeAnalysis(
                timeframe=timeframe,
                signal=TimeframeSignal.NEUTRAL,
                trend_strength=0.0,
                momentum=0.0,
                support_resistance={},
                key_levels=[]
            )
        
        # Calculate indicators
        sma_fast = np.mean(prices[-10:])
        sma_slow = np.mean(prices[-20:])
        current_price = prices[-1]
        
        # Trend
        trend = (sma_fast - sma_slow) / sma_slow if sma_slow != 0 else 0
        trend_strength = min(abs(trend) * 100, 1.0)
        
        # Momentum (Rate of Change)
        roc = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
        momentum = np.clip(roc * 50, -1, 1)
        
        # RSI-like momentum
        gains = np.maximum(np.diff(prices[-15:]), 0)
        losses = np.abs(np.minimum(np.diff(prices[-15:]), 0))
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Support/Resistance (simple pivot points)
        high = np.max(prices[-20:])
        low = np.min(prices[-20:])
        pivot = (high + low + current_price) / 3
        
        support_resistance = {
            "pivot": pivot,
            "support1": 2 * pivot - high,
            "support2": pivot - (high - low),
            "resistance1": 2 * pivot - low,
            "resistance2": pivot + (high - low)
        }
        
        # Key levels (recent highs/lows)
        key_levels = [high, low, pivot]
        
        # Determine signal
        signal = self._determine_signal(trend, momentum, rsi, current_price, sma_fast)
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            signal=signal,
            trend_strength=trend_strength,
            momentum=momentum,
            support_resistance=support_resistance,
            key_levels=key_levels
        )
    
    def _determine_signal(
        self,
        trend: float,
        momentum: float,
        rsi: float,
        price: float,
        sma: float
    ) -> TimeframeSignal:
        """Determine signal from indicators"""
        
        score = 0
        
        # Trend contribution
        if trend > 0.002:
            score += 1
        elif trend < -0.002:
            score -= 1
        
        # Momentum contribution  
        if momentum > 0.3:
            score += 1
        elif momentum < -0.3:
            score -= 1
        
        # RSI contribution
        if rsi > 70:
            score -= 0.5  # Overbought
        elif rsi < 30:
            score += 0.5  # Oversold
        elif rsi > 50:
            score += 0.3
        else:
            score -= 0.3
        
        # Price vs SMA
        if price > sma * 1.002:
            score += 0.5
        elif price < sma * 0.998:
            score -= 0.5
        
        # Convert to signal
        if score >= 2:
            return TimeframeSignal.STRONG_BUY
        elif score >= 1:
            return TimeframeSignal.BUY
        elif score <= -2:
            return TimeframeSignal.STRONG_SELL
        elif score <= -1:
            return TimeframeSignal.SELL
        else:
            return TimeframeSignal.NEUTRAL
    
    def get_confluence(
        self,
        symbol: str,
        timeframe_data: Dict[str, np.ndarray]
    ) -> Tuple[float, ConfluenceLevel, Dict[str, TimeframeAnalysis]]:
        """
        Get confluence score across all timeframes.
        
        Returns:
            (weighted_score, confluence_level, analyses)
        """
        
        analyses = {}
        weighted_score = 0.0
        signals_agree = 0
        
        for tf in self.TIMEFRAMES:
            if tf in timeframe_data and len(timeframe_data[tf]) > 0:
                analysis = self.analyze_timeframe(tf, timeframe_data[tf])
                analyses[tf] = analysis
                
                # Weighted score
                weight = self.WEIGHTS.get(tf, 0.25)
                weighted_score += analysis.signal.value * weight
        
        # Count agreement
        if analyses:
            primary_direction = "BUY" if weighted_score > 0 else "SELL" if weighted_score < 0 else "NEUTRAL"
            
            for tf, analysis in analyses.items():
                if primary_direction == "BUY" and analysis.signal.value > 0:
                    signals_agree += 1
                elif primary_direction == "SELL" and analysis.signal.value < 0:
                    signals_agree += 1
                elif primary_direction == "NEUTRAL" and analysis.signal.value == 0:
                    signals_agree += 1
        
        # Determine confluence level
        total_tfs = len(analyses)
        if total_tfs == 0:
            confluence = ConfluenceLevel.NONE
        elif signals_agree == total_tfs:
            confluence = ConfluenceLevel.PERFECT
        elif signals_agree >= total_tfs * 0.75:
            confluence = ConfluenceLevel.STRONG
        elif signals_agree >= total_tfs * 0.5:
            confluence = ConfluenceLevel.MODERATE
        else:
            confluence = ConfluenceLevel.WEAK
        
        # Store for history
        self.analyses[symbol] = analyses
        
        return weighted_score, confluence, analyses


# ============================================================
# CROSS-ASSET CORRELATION
# ============================================================

class CrossAssetCorrelation:
    """
    Track correlations between trading symbols.
    Detect unusual divergences for opportunities/risks.
    """
    
    # Known correlations (positive or negative)
    KNOWN_PAIRS = {
        ("EURUSDm", "GBPUSDm"): 0.85,    # High positive
        ("EURUSDm", "USDJPYm"): -0.60,   # Negative
        ("GBPUSDm", "USDJPYm"): -0.55,   # Negative
        ("XAUUSDm", "USDJPYm"): -0.40,   # Gold vs Yen
        ("XAUUSDm", "EURUSDm"): 0.30,    # Weak positive
    }
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.price_history: Dict[str, deque] = {}
        self.correlations: Dict[Tuple[str, str], CorrelationData] = {}
    
    def update_price(self, symbol: str, price: float):
        """Update price history for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback)
        self.price_history[symbol].append(price)
    
    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[CorrelationData]:
        """Calculate correlation between two symbols"""
        
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return None
        
        prices1 = np.array(self.price_history[symbol1])
        prices2 = np.array(self.price_history[symbol2])
        
        min_len = min(len(prices1), len(prices2))
        if min_len < 20:
            return None
        
        # Use returns instead of prices
        returns1 = np.diff(prices1[-min_len:]) / prices1[-min_len:-1]
        returns2 = np.diff(prices2[-min_len:]) / prices2[-min_len:-1]
        
        # Full correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        # Rolling correlation (recent 10 periods)
        rolling_corr = np.corrcoef(returns1[-10:], returns2[-10:])[0, 1]
        
        # Check for divergence
        expected_corr = self.KNOWN_PAIRS.get((symbol1, symbol2), 
                        self.KNOWN_PAIRS.get((symbol2, symbol1), 0.0))
        
        is_diverging = abs(rolling_corr - expected_corr) > 0.3
        
        corr_data = CorrelationData(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            rolling_corr=rolling_corr,
            is_diverging=is_diverging
        )
        
        self.correlations[(symbol1, symbol2)] = corr_data
        return corr_data
    
    def get_correlation_score(
        self,
        symbol: str,
        direction: str,
        other_symbols_direction: Dict[str, str]
    ) -> Tuple[float, List[str]]:
        """
        Check if trade direction aligns with correlated assets.
        
        Returns:
            (score, warnings)
        """
        
        score = 0.0
        warnings = []
        
        for other_symbol, other_dir in other_symbols_direction.items():
            pair = (symbol, other_symbol)
            reverse_pair = (other_symbol, symbol)
            
            expected = self.KNOWN_PAIRS.get(pair, self.KNOWN_PAIRS.get(reverse_pair, 0))
            
            if abs(expected) < 0.3:
                continue  # Weak correlation, ignore
            
            # Check alignment
            same_direction = (direction == other_dir)
            
            if expected > 0:
                # Positive correlation - should move same direction
                if same_direction:
                    score += 0.2
                else:
                    score -= 0.3
                    warnings.append(f"⚠️ {symbol} vs {other_symbol}: Divergence (expected +corr)")
            else:
                # Negative correlation - should move opposite
                if not same_direction:
                    score += 0.2
                else:
                    score -= 0.3
                    warnings.append(f"⚠️ {symbol} vs {other_symbol}: Divergence (expected -corr)")
        
        return np.clip(score, -1, 1), warnings


# ============================================================
# ADAPTIVE PARAMETER TUNER
# ============================================================

class AdaptiveParameterTuner:
    """
    Self-optimize trading parameters based on recent performance.
    Tracks what works and adjusts accordingly.
    """
    
    def __init__(self):
        self.trade_history: deque = deque(maxlen=100)
        self.parameter_performance: Dict[str, Dict[str, float]] = {
            "quality_level": {"PREMIUM": 0, "HIGH": 0, "MEDIUM": 0},
            "session": {"asian": 0, "london": 0, "new_york": 0, "overlap": 0},
            "confluence": {"PERFECT": 0, "STRONG": 0, "MODERATE": 0},
            "market_state": {},
        }
        self.current_params: Dict[str, Any] = {
            "min_quality": "HIGH",
            "best_session": "overlap",
            "min_confluence": "MODERATE",
            "position_scale": 1.0
        }
    
    def record_trade(
        self,
        result: float,  # Profit/Loss in %
        params: Dict[str, Any]
    ):
        """Record trade result with parameters used"""
        
        self.trade_history.append({
            "timestamp": datetime.now(),
            "result": result,
            "params": params
        })
        
        # Update parameter performance
        for key, value in params.items():
            if key in self.parameter_performance:
                if value in self.parameter_performance[key]:
                    # Exponential moving average
                    old = self.parameter_performance[key][value]
                    self.parameter_performance[key][value] = old * 0.9 + result * 0.1
                else:
                    self.parameter_performance[key][value] = result
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get currently optimal parameters based on history"""
        
        if len(self.trade_history) < 10:
            return self.current_params
        
        # Find best performing values for each parameter
        for param_name, values in self.parameter_performance.items():
            if values:
                best_value = max(values.keys(), key=lambda k: values[k])
                self.current_params[f"best_{param_name}"] = best_value
        
        # Adjust position scale based on recent performance
        recent_trades = list(self.trade_history)[-20:]
        recent_pnl = sum(t["result"] for t in recent_trades)
        win_rate = sum(1 for t in recent_trades if t["result"] > 0) / len(recent_trades)
        
        if win_rate > 0.6 and recent_pnl > 0:
            self.current_params["position_scale"] = min(1.3, self.current_params["position_scale"] + 0.1)
        elif win_rate < 0.4 or recent_pnl < 0:
            self.current_params["position_scale"] = max(0.5, self.current_params["position_scale"] - 0.1)
        
        return self.current_params
    
    def should_trade_with_params(self, current_params: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Check if we should trade with given parameters.
        
        Returns:
            (should_trade, confidence_adjustment, reason)
        """
        
        # Check if parameters have historically performed well
        score = 0
        reasons = []
        
        for param_name, value in current_params.items():
            if param_name in self.parameter_performance:
                perf = self.parameter_performance[param_name].get(value, 0)
                if perf > 0.5:
                    score += 1
                    reasons.append(f"✓ {param_name}={value} performs well")
                elif perf < -0.5:
                    score -= 1
                    reasons.append(f"✗ {param_name}={value} underperforms")
        
        # Recent overall performance
        if len(self.trade_history) >= 10:
            recent = list(self.trade_history)[-10:]
            recent_pnl = sum(t["result"] for t in recent)
            if recent_pnl < -5:
                score -= 2
                reasons.append("✗ Recent losing streak")
            elif recent_pnl > 5:
                score += 1
                reasons.append("✓ Recent winning streak")
        
        should_trade = score >= 0
        confidence_adj = np.clip(score * 5, -20, 20)
        
        return should_trade, confidence_adj, "; ".join(reasons)


# ============================================================
# PREDICTIVE MODEL
# ============================================================

class PredictiveModel:
    """
    Short-term price prediction using technical indicators.
    Combines multiple methods for ensemble prediction.
    """
    
    def __init__(self):
        self.prediction_history: deque = deque(maxlen=50)
        self.accuracy_tracker: Dict[str, List[bool]] = {}
    
    def predict(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        horizon: str = "short"  # "short" = 1-5 candles, "medium" = 5-20 candles
    ) -> PredictionResult:
        """
        Predict price direction and magnitude.
        """
        
        if len(prices) < 30:
            return PredictionResult(
                predicted_direction="SIDEWAYS",
                confidence=0.0,
                predicted_move=0.0,
                support_level=prices[-1] * 0.99,
                resistance_level=prices[-1] * 1.01,
                time_horizon=horizon
            )
        
        predictions = []
        
        # Method 1: Moving Average Crossover
        ma_pred = self._ma_prediction(prices)
        predictions.append(ma_pred)
        
        # Method 2: Momentum
        mom_pred = self._momentum_prediction(prices)
        predictions.append(mom_pred)
        
        # Method 3: Mean Reversion
        mr_pred = self._mean_reversion_prediction(prices)
        predictions.append(mr_pred)
        
        # Method 4: Trend Continuation
        trend_pred = self._trend_prediction(prices)
        predictions.append(trend_pred)
        
        # Method 5: Volatility Breakout
        vol_pred = self._volatility_prediction(prices)
        predictions.append(vol_pred)
        
        # Ensemble: weighted vote
        up_votes = sum(1 for p in predictions if p > 0)
        down_votes = sum(1 for p in predictions if p < 0)
        avg_magnitude = np.mean([abs(p) for p in predictions])
        
        if up_votes > down_votes + 1:
            direction = "UP"
            confidence = up_votes / len(predictions)
        elif down_votes > up_votes + 1:
            direction = "DOWN"
            confidence = down_votes / len(predictions)
        else:
            direction = "SIDEWAYS"
            confidence = 0.3
        
        # Calculate support/resistance
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        atr = self._calculate_atr(prices)
        
        support = recent_low - atr * 0.5
        resistance = recent_high + atr * 0.5
        
        # Predicted move
        predicted_move = avg_magnitude * 100  # Convert to %
        if direction == "DOWN":
            predicted_move = -predicted_move
        
        return PredictionResult(
            predicted_direction=direction,
            confidence=confidence,
            predicted_move=predicted_move,
            support_level=support,
            resistance_level=resistance,
            time_horizon=horizon
        )
    
    def _ma_prediction(self, prices: np.ndarray) -> float:
        """Moving average based prediction"""
        ma5 = np.mean(prices[-5:])
        ma20 = np.mean(prices[-20:])
        
        diff = (ma5 - ma20) / ma20
        return np.clip(diff * 100, -1, 1)
    
    def _momentum_prediction(self, prices: np.ndarray) -> float:
        """Momentum based prediction"""
        roc = (prices[-1] - prices[-10]) / prices[-10]
        return np.clip(roc * 20, -1, 1)
    
    def _mean_reversion_prediction(self, prices: np.ndarray) -> float:
        """Mean reversion prediction"""
        mean = np.mean(prices[-30:])
        std = np.std(prices[-30:])
        z_score = (prices[-1] - mean) / std if std > 0 else 0
        
        # High z-score = expect reversion
        return np.clip(-z_score * 0.3, -1, 1)
    
    def _trend_prediction(self, prices: np.ndarray) -> float:
        """Trend continuation prediction"""
        # Linear regression slope
        x = np.arange(len(prices[-20:]))
        slope = np.polyfit(x, prices[-20:], 1)[0]
        
        normalized_slope = slope / prices[-1] * 100
        return np.clip(normalized_slope, -1, 1)
    
    def _volatility_prediction(self, prices: np.ndarray) -> float:
        """Volatility breakout prediction"""
        recent_vol = np.std(prices[-10:])
        longer_vol = np.std(prices[-30:])
        
        # Contracting volatility often leads to breakout
        if recent_vol < longer_vol * 0.7:
            # Direction based on last move
            last_move = prices[-1] - prices[-5]
            return 0.5 if last_move > 0 else -0.5
        
        return 0.0
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return np.std(prices) * 2
        
        highs = np.maximum(prices[1:], prices[:-1])
        lows = np.minimum(prices[1:], prices[:-1])
        tr = highs - lows
        
        return np.mean(tr[-period:])
    
    def record_prediction(self, symbol: str, prediction: str, actual: str):
        """Record prediction accuracy"""
        if symbol not in self.accuracy_tracker:
            self.accuracy_tracker[symbol] = []
        
        correct = prediction == actual
        self.accuracy_tracker[symbol].append(correct)
        
        # Keep last 50
        if len(self.accuracy_tracker[symbol]) > 50:
            self.accuracy_tracker[symbol] = self.accuracy_tracker[symbol][-50:]
    
    def get_accuracy(self, symbol: str) -> float:
        """Get prediction accuracy for symbol"""
        if symbol not in self.accuracy_tracker or len(self.accuracy_tracker[symbol]) < 10:
            return 0.5  # Unknown
        
        return sum(self.accuracy_tracker[symbol]) / len(self.accuracy_tracker[symbol])


# ============================================================
# SESSION ANALYZER
# ============================================================

class SessionAnalyzer:
    """
    Analyze trading sessions for optimal trading times.
    """
    
    # Session times in UTC
    SESSIONS = {
        TradingSession.ASIAN: (0, 8),
        TradingSession.LONDON: (8, 16),
        TradingSession.NEW_YORK: (13, 21),
        TradingSession.OVERLAP: (13, 16),
    }
    
    # Symbol-Session performance (empirical data)
    OPTIMAL_SESSIONS = {
        "EURUSDm": [TradingSession.LONDON, TradingSession.OVERLAP],
        "GBPUSDm": [TradingSession.LONDON, TradingSession.OVERLAP],
        "USDJPYm": [TradingSession.ASIAN, TradingSession.NEW_YORK],
        "XAUUSDm": [TradingSession.NEW_YORK, TradingSession.OVERLAP],
    }
    
    def __init__(self):
        self.session_performance: Dict[str, Dict[TradingSession, List[float]]] = {}
    
    def get_current_session(self, utc_hour: Optional[int] = None) -> TradingSession:
        """Get current trading session"""
        if utc_hour is None:
            utc_hour = datetime.utcnow().hour
        
        # Check overlap first (most specific)
        if 13 <= utc_hour < 16:
            return TradingSession.OVERLAP
        elif 8 <= utc_hour < 16:
            return TradingSession.LONDON
        elif 13 <= utc_hour < 21:
            return TradingSession.NEW_YORK
        elif 0 <= utc_hour < 8:
            return TradingSession.ASIAN
        else:
            return TradingSession.OFF_HOURS
    
    def get_session_score(self, symbol: str, session: Optional[TradingSession] = None) -> float:
        """
        Get score for trading this symbol in current/given session.
        Returns 0-1 score.
        """
        if session is None:
            session = self.get_current_session()
        
        optimal = self.OPTIMAL_SESSIONS.get(symbol, [])
        
        if session in optimal:
            return 1.0
        elif session == TradingSession.OFF_HOURS:
            return 0.3
        else:
            return 0.6
    
    def record_session_result(self, symbol: str, session: TradingSession, result: float):
        """Record trade result for a session"""
        if symbol not in self.session_performance:
            self.session_performance[symbol] = {}
        
        if session not in self.session_performance[symbol]:
            self.session_performance[symbol][session] = []
        
        self.session_performance[symbol][session].append(result)
        
        # Keep last 50 per session
        if len(self.session_performance[symbol][session]) > 50:
            self.session_performance[symbol][session] = \
                self.session_performance[symbol][session][-50:]
    
    def get_best_session(self, symbol: str) -> TradingSession:
        """Get historically best session for a symbol"""
        if symbol not in self.session_performance:
            # Return default from empirical data
            optimal = self.OPTIMAL_SESSIONS.get(symbol, [TradingSession.LONDON])
            return optimal[0] if optimal else TradingSession.LONDON
        
        best_session = TradingSession.LONDON
        best_avg = float('-inf')
        
        for session, results in self.session_performance[symbol].items():
            if len(results) >= 5:
                avg = sum(results) / len(results)
                if avg > best_avg:
                    best_avg = avg
                    best_session = session
        
        return best_session


# ============================================================
# CONFLUENCE ENGINE - MAIN CLASS
# ============================================================

class DeepIntelligence:
    """
    Main class that combines all intelligence components.
    Provides final trading decision with high confidence.
    """
    
    def __init__(self):
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.correlation = CrossAssetCorrelation()
        self.param_tuner = AdaptiveParameterTuner()
        self.predictor = PredictiveModel()
        self.session_analyzer = SessionAnalyzer()
        
        # State
        self.last_decisions: Dict[str, DeepDecision] = {}
        self.decision_history: deque = deque(maxlen=100)
        
        logger.info("🧠 Deep Intelligence initialized")
        logger.info("   - Multi-Timeframe Analyzer: ✓")
        logger.info("   - Cross-Asset Correlation: ✓")
        logger.info("   - Adaptive Parameter Tuner: ✓")
        logger.info("   - Predictive Model: ✓")
        logger.info("   - Session Analyzer: ✓")
    
    def analyze(
        self,
        symbol: str,
        signal_direction: str,  # "BUY" or "SELL"
        timeframe_data: Dict[str, np.ndarray],  # {"M15": prices, "H1": prices, ...}
        current_params: Dict[str, Any],
        other_symbols_direction: Optional[Dict[str, str]] = None
    ) -> DeepDecision:
        """
        Perform deep analysis and return final decision.
        
        Args:
            symbol: Trading symbol
            signal_direction: Proposed direction from main signal
            timeframe_data: Price data for each timeframe
            current_params: Current trading parameters
            other_symbols_direction: Direction of other symbols being traded
        """
        
        warnings = []
        
        # 1. Multi-Timeframe Analysis
        if timeframe_data:
            tf_score, confluence, tf_analyses = self.mtf_analyzer.get_confluence(
                symbol, timeframe_data
            )
        else:
            tf_score = 0.0
            confluence = ConfluenceLevel.NONE
            tf_analyses = {}
        
        # Check if timeframes agree with signal
        if signal_direction == "BUY" and tf_score < -0.5:
            warnings.append("⚠️ Timeframes disagree with BUY signal")
        elif signal_direction == "SELL" and tf_score > 0.5:
            warnings.append("⚠️ Timeframes disagree with SELL signal")
        
        # 2. Cross-Asset Correlation
        corr_score = 0.0
        if other_symbols_direction:
            corr_score, corr_warnings = self.correlation.get_correlation_score(
                symbol, signal_direction, other_symbols_direction
            )
            warnings.extend(corr_warnings)
        
        # 3. Prediction
        # Use H1 data for prediction if available
        pred_data = timeframe_data.get("H1", timeframe_data.get("M15", np.array([])))
        if len(pred_data) > 30:
            prediction = self.predictor.predict(pred_data)
            pred_score = 0.0
            
            if prediction.predicted_direction == "UP" and signal_direction == "BUY":
                pred_score = prediction.confidence
            elif prediction.predicted_direction == "DOWN" and signal_direction == "SELL":
                pred_score = prediction.confidence
            elif prediction.predicted_direction == "SIDEWAYS":
                pred_score = 0.0
            else:
                pred_score = -prediction.confidence * 0.5
                warnings.append(f"⚠️ Prediction ({prediction.predicted_direction}) conflicts with signal")
        else:
            pred_score = 0.0
            prediction = None
        
        # 4. Session Analysis
        session_score = self.session_analyzer.get_session_score(symbol)
        current_session = self.session_analyzer.get_current_session()
        
        if session_score < 0.5:
            warnings.append(f"⚠️ Not optimal session ({current_session.value}) for {symbol}")
        
        # 5. Parameter Tuning Check
        should_trade_params, conf_adj, param_reason = self.param_tuner.should_trade_with_params(
            current_params
        )
        
        if not should_trade_params:
            warnings.append(f"⚠️ Parameters underperforming: {param_reason}")
        
        # ============================================
        # FINAL DECISION CALCULATION
        # ============================================
        
        # Weighted scores
        total_score = (
            tf_score * 0.35 +           # Timeframe confluence
            corr_score * 0.15 +         # Correlation
            pred_score * 0.25 +         # Prediction
            (session_score - 0.5) * 0.15 +  # Session (centered)
            (conf_adj / 20) * 0.10      # Parameter performance
        )
        
        # Confidence calculation
        base_confidence = 50.0
        
        # Add from confluence
        if confluence == ConfluenceLevel.PERFECT:
            base_confidence += 25
        elif confluence == ConfluenceLevel.STRONG:
            base_confidence += 15
        elif confluence == ConfluenceLevel.MODERATE:
            base_confidence += 5
        elif confluence == ConfluenceLevel.WEAK:
            base_confidence -= 10
        
        # Add from prediction
        if prediction and prediction.confidence > 0.6:
            base_confidence += prediction.confidence * 15
        
        # Add from session
        base_confidence += (session_score - 0.5) * 20
        
        # Subtract from warnings
        base_confidence -= len(warnings) * 5
        
        # Final confidence
        confidence = np.clip(base_confidence, 0, 100)
        
        # Should we trade?
        should_trade = (
            confidence >= 55 and
            confluence not in [ConfluenceLevel.WEAK, ConfluenceLevel.NONE] and
            len(warnings) <= 2 and
            should_trade_params
        )
        
        # Direction
        if not should_trade:
            direction = "WAIT"
        else:
            direction = signal_direction
        
        # Position multiplier
        if confidence >= 75 and confluence == ConfluenceLevel.PERFECT:
            position_multiplier = 1.2
        elif confidence >= 65 and confluence in [ConfluenceLevel.PERFECT, ConfluenceLevel.STRONG]:
            position_multiplier = 1.0
        elif confidence >= 55:
            position_multiplier = 0.8
        else:
            position_multiplier = 0.5
        
        # Apply session adjustment
        position_multiplier *= session_score
        
        # Build reasoning
        reasoning_parts = [
            f"TF Score: {tf_score:.2f}",
            f"Confluence: {confluence.value}",
            f"Prediction: {pred_score:.2f}",
            f"Session: {current_session.value} ({session_score:.1f})",
        ]
        reasoning = " | ".join(reasoning_parts)
        
        # Create decision
        decision = DeepDecision(
            should_trade=should_trade,
            direction=direction,
            confluence_level=confluence,
            confidence=confidence,
            position_multiplier=position_multiplier,
            timeframe_score=tf_score,
            correlation_score=corr_score,
            prediction_score=pred_score,
            session_score=session_score,
            warnings=warnings,
            reasoning=reasoning
        )
        
        # Store
        self.last_decisions[symbol] = decision
        self.decision_history.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "decision": decision
        })
        
        return decision
    
    def record_result(
        self,
        symbol: str,
        result: float,
        params: Dict[str, Any],
        session: TradingSession
    ):
        """Record trade result for learning"""
        self.param_tuner.record_trade(result, params)
        self.session_analyzer.record_session_result(symbol, session, result)
        
        # Record prediction accuracy if we made one
        if symbol in self.last_decisions:
            last = self.last_decisions[symbol]
            actual_dir = "UP" if result > 0 else "DOWN" if result < 0 else "SIDEWAYS"
            self.predictor.record_prediction(symbol, last.direction, actual_dir)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of intelligence state"""
        return {
            "active_symbols": list(self.last_decisions.keys()),
            "total_decisions": len(self.decision_history),
            "optimal_params": self.param_tuner.get_optimal_parameters(),
            "prediction_accuracy": {
                symbol: self.predictor.get_accuracy(symbol)
                for symbol in self.predictor.accuracy_tracker.keys()
            }
        }


# ============================================================
# SINGLETON & FACTORY
# ============================================================

_deep_intelligence: Optional[DeepIntelligence] = None


def get_deep_intelligence() -> DeepIntelligence:
    """Get singleton Deep Intelligence instance"""
    global _deep_intelligence
    if _deep_intelligence is None:
        _deep_intelligence = DeepIntelligence()
    return _deep_intelligence


def init_deep_intelligence() -> DeepIntelligence:
    """Initialize and return Deep Intelligence"""
    global _deep_intelligence
    _deep_intelligence = DeepIntelligence()
    return _deep_intelligence
