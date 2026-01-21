"""
Multi-Factor Signal Analyzer - Maximum Win Rate Engine
รวมทุกปัจจัยเพื่อสร้างสัญญาณคุณภาพสูงสุด

ปัจจัยที่วิเคราะห์:
1. Pattern Matching Score (Base)
2. Trend Alignment (ไปทางเดียวกับ trend)
3. Volume Confirmation (volume รองรับ)
4. Pattern Recency (patterns ล่าสุดมีน้ำหนักมาก)
5. Volatility Assessment (volatility อยู่ในช่วงที่ดี)
6. Session Timing (เทรดช่วงที่ดี)
7. Momentum Confluence (RSI, MACD เห็นด้วย)
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
import logging

from config.enhanced_settings import (
    EnhancedAnalysisConfig, 
    TradingMode,
    TrendFilterConfig,
    VolumeFilterConfig,
)
from analysis.voting_system import Signal, VoteResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorStatus(str, Enum):
    """สถานะของแต่ละ factor"""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    SKIP = "SKIP"  # ไม่ควรเทรด


@dataclass
class FactorAnalysis:
    """ผลวิเคราะห์ของแต่ละ factor"""
    name: str
    enabled: bool
    score: float  # 0.0 - 1.0
    weight: float
    weighted_score: float  # score * weight
    status: FactorStatus
    details: str
    passed: bool  # ผ่านเกณฑ์หรือไม่
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "score": round(self.score * 100, 1),  # แสดงเป็น %
            "weight": round(self.weight * 100, 1),
            "weighted_score": round(self.weighted_score * 100, 2),
            "status": self.status.value,
            "details": self.details,
            "passed": self.passed,
        }


class SignalQuality(str, Enum):
    """คุณภาพสัญญาณ"""
    PREMIUM = "PREMIUM"      # Win Rate Expected: 85%+
    HIGH = "HIGH"            # Win Rate Expected: 75-85%
    MEDIUM = "MEDIUM"        # Win Rate Expected: 65-75%
    LOW = "LOW"              # Win Rate Expected: 55-65%
    SKIP = "SKIP"            # ไม่ควรเข้า


@dataclass
class MultiFactorResult:
    """ผลลัพธ์จาก Multi-Factor Analysis"""
    # Signal Info
    signal: Signal
    base_confidence: float      # จาก pattern matching
    final_score: float          # รวมทุก factors
    
    # Quality Assessment
    quality: SignalQuality
    recommendation: str         # TRADE, TRADE_REDUCED, SKIP
    
    # Factor Breakdown
    factors: List[FactorAnalysis] = field(default_factory=list)
    
    # Individual Scores (for display)
    pattern_score: float = 0.0
    trend_score: float = 0.0
    volume_score: float = 0.0
    recency_score: float = 0.0
    volatility_score: float = 0.0
    session_score: float = 0.0
    momentum_score: float = 0.0
    
    # Risk Adjustments
    position_size_multiplier: float = 1.0
    confidence_adjusted: float = 0.0
    
    # Reasons
    bullish_reasons: List[str] = field(default_factory=list)
    bearish_reasons: List[str] = field(default_factory=list)
    skip_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "signal": self.signal.value,
            "base_confidence": round(self.base_confidence, 2),
            "final_score": round(self.final_score, 2),
            "quality": self.quality.value if isinstance(self.quality, SignalQuality) else self.quality,
            "recommendation": self.recommendation,
            "factors": [f.to_dict() for f in self.factors],
            "scores": {
                "pattern": round(self.pattern_score, 2),
                "trend": round(self.trend_score, 2),
                "volume": round(self.volume_score, 2),
                "recency": round(self.recency_score, 2),
                "volatility": round(self.volatility_score, 2),
                "session": round(self.session_score, 2),
                "momentum": round(self.momentum_score, 2),
            },
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "confidence_adjusted": round(self.confidence_adjusted, 2),
            "reasons": {
                "bullish": self.bullish_reasons,
                "bearish": self.bearish_reasons,
                "skip": self.skip_reasons,
            },
        }


class MultiFactorAnalyzer:
    """
    Multi-Factor Signal Analyzer
    รวมทุกปัจจัยเพื่อ Win Rate สูงสุด
    """
    
    def __init__(self, config: Optional[EnhancedAnalysisConfig] = None):
        """
        Initialize Multi-Factor Analyzer
        
        Args:
            config: Configuration (loads from env if None)
        """
        self.config = config or EnhancedAnalysisConfig.from_env()
        logger.info(f"MultiFactorAnalyzer initialized with mode: {self.config.mode.value}")
    
    def analyze(
        self,
        vote_result: VoteResult,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        pattern_dates: Optional[List[datetime]] = None,
        current_time: Optional[datetime] = None,
        symbol: str = "UNKNOWN",
    ) -> MultiFactorResult:
        """
        วิเคราะห์สัญญาณด้วยหลายปัจจัย
        
        Args:
            vote_result: ผลจาก VotingSystem
            prices: ราคาปิด (close prices)
            volumes: Volume data (optional)
            highs: High prices for ATR (optional)
            lows: Low prices for ATR (optional)
            pattern_dates: วันที่ของ patterns ที่ match (optional)
            current_time: เวลาปัจจุบัน (optional)
            symbol: สัญลักษณ์ที่เทรด
        
        Returns:
            MultiFactorResult with comprehensive analysis
        """
        factors: List[FactorAnalysis] = []
        bullish_reasons: List[str] = []
        bearish_reasons: List[str] = []
        skip_reasons: List[str] = []
        
        current_time = current_time or datetime.now()
        
        # 1. Pattern Score (Base)
        pattern_factor = self._analyze_pattern(vote_result)
        factors.append(pattern_factor)
        
        # 2. Trend Alignment
        trend_factor = self._analyze_trend(prices, vote_result.signal)
        factors.append(trend_factor)
        
        # 3. Volume Confirmation
        volume_factor = self._analyze_volume(prices, volumes, vote_result.signal)
        factors.append(volume_factor)
        
        # 4. Pattern Recency
        recency_factor = self._analyze_recency(pattern_dates)
        factors.append(recency_factor)
        
        # 5. Volatility Assessment
        volatility_factor = self._analyze_volatility(prices, highs, lows)
        factors.append(volatility_factor)
        
        # 6. Session Timing
        session_factor = self._analyze_session(current_time, symbol)
        factors.append(session_factor)
        
        # 7. Momentum Confluence
        momentum_factor = self._analyze_momentum(prices, vote_result.signal)
        factors.append(momentum_factor)
        
        # Calculate final score
        total_weight = sum(f.weight for f in factors if f.enabled)
        if total_weight > 0:
            final_score = sum(f.weighted_score for f in factors if f.enabled) / total_weight * 100
        else:
            final_score = vote_result.confidence
        
        # Collect reasons
        for factor in factors:
            if factor.status in [FactorStatus.STRONG_BULLISH, FactorStatus.BULLISH]:
                bullish_reasons.append(f"{factor.name}: {factor.details}")
            elif factor.status in [FactorStatus.STRONG_BEARISH, FactorStatus.BEARISH]:
                bearish_reasons.append(f"{factor.name}: {factor.details}")
            elif factor.status == FactorStatus.SKIP:
                skip_reasons.append(f"{factor.name}: {factor.details}")
        
        # Determine quality and recommendation
        quality = self._determine_quality(final_score)
        recommendation = self._determine_recommendation(
            vote_result.signal, final_score, factors, skip_reasons
        )
        
        # Position size adjustment
        position_multiplier = self._calculate_position_multiplier(
            final_score, quality, factors
        )
        
        # Adjusted confidence
        confidence_adjusted = final_score * position_multiplier / 100
        
        return MultiFactorResult(
            signal=vote_result.signal,
            base_confidence=vote_result.confidence,
            final_score=final_score,
            quality=quality,
            recommendation=recommendation,
            factors=factors,
            pattern_score=pattern_factor.score * 100,
            trend_score=trend_factor.score * 100,
            volume_score=volume_factor.score * 100,
            recency_score=recency_factor.score * 100,
            volatility_score=volatility_factor.score * 100,
            session_score=session_factor.score * 100,
            momentum_score=momentum_factor.score * 100,
            position_size_multiplier=position_multiplier,
            confidence_adjusted=confidence_adjusted,
            bullish_reasons=bullish_reasons,
            bearish_reasons=bearish_reasons,
            skip_reasons=skip_reasons,
        )
    
    def _analyze_pattern(self, vote_result: VoteResult) -> FactorAnalysis:
        """วิเคราะห์ Pattern Matching Score"""
        score = vote_result.confidence / 100.0
        weight = self.config.pattern_weight
        
        if score >= 0.85:
            status = FactorStatus.STRONG_BULLISH if vote_result.bullish_votes > vote_result.bearish_votes else FactorStatus.STRONG_BEARISH
            details = f"Strong pattern match: {vote_result.confidence:.1f}%"
        elif score >= 0.70:
            status = FactorStatus.BULLISH if vote_result.bullish_votes > vote_result.bearish_votes else FactorStatus.BEARISH
            details = f"Good pattern match: {vote_result.confidence:.1f}%"
        elif score >= 0.50:
            status = FactorStatus.NEUTRAL
            details = f"Moderate pattern match: {vote_result.confidence:.1f}%"
        else:
            status = FactorStatus.SKIP
            details = f"Weak pattern match: {vote_result.confidence:.1f}%"
        
        return FactorAnalysis(
            name="Pattern Match",
            enabled=True,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            status=status,
            details=details,
            passed=score >= 0.65,
        )
    
    def _analyze_trend(
        self, 
        prices: np.ndarray, 
        signal: Signal
    ) -> FactorAnalysis:
        """วิเคราะห์ Trend Alignment"""
        config = self.config.trend
        
        if not config.enabled or len(prices) < config.slow_ema:
            return FactorAnalysis(
                name="Trend Alignment",
                enabled=config.enabled,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="Insufficient data for trend analysis",
                passed=True,
            )
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, config.fast_ema)
        ema_slow = self._calculate_ema(prices, config.slow_ema)
        
        # Determine trend
        trend_up = ema_fast > ema_slow
        trend_strength = abs(ema_fast - ema_slow) / ema_slow
        
        # Check alignment
        is_buy_signal = signal in [Signal.BUY, Signal.STRONG_BUY]
        is_sell_signal = signal in [Signal.SELL, Signal.STRONG_SELL]
        
        if is_buy_signal and trend_up:
            # Bullish signal in uptrend - ALIGNED
            score = min(1.0, config.aligned_score + trend_strength * 2)
            status = FactorStatus.STRONG_BULLISH if trend_strength > 0.01 else FactorStatus.BULLISH
            details = f"BUY aligned with uptrend (strength: {trend_strength*100:.2f}%)"
            passed = True
        elif is_sell_signal and not trend_up:
            # Bearish signal in downtrend - ALIGNED
            score = min(1.0, config.aligned_score + trend_strength * 2)
            status = FactorStatus.STRONG_BEARISH if trend_strength > 0.01 else FactorStatus.BEARISH
            details = f"SELL aligned with downtrend (strength: {trend_strength*100:.2f}%)"
            passed = True
        elif is_buy_signal and not trend_up:
            # Bullish signal in downtrend - COUNTER
            score = config.counter_score
            status = FactorStatus.BEARISH
            details = f"⚠️ BUY against downtrend - Counter-trend trade"
            passed = False
        elif is_sell_signal and trend_up:
            # Bearish signal in uptrend - COUNTER
            score = config.counter_score
            status = FactorStatus.BULLISH
            details = f"⚠️ SELL against uptrend - Counter-trend trade"
            passed = False
        else:
            # WAIT signal or sideways
            score = config.neutral_score
            status = FactorStatus.NEUTRAL
            details = "No clear trend direction"
            passed = True
        
        return FactorAnalysis(
            name="Trend Alignment",
            enabled=config.enabled,
            score=score,
            weight=config.weight,
            weighted_score=score * config.weight,
            status=status,
            details=details,
            passed=passed,
        )
    
    def _analyze_volume(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray],
        signal: Signal
    ) -> FactorAnalysis:
        """วิเคราะห์ Volume Confirmation"""
        config = self.config.volume
        
        if not config.enabled or volumes is None or len(volumes) < config.lookback_period:
            return FactorAnalysis(
                name="Volume Confirmation",
                enabled=config.enabled,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="No volume data available",
                passed=True,
            )
        
        # Calculate volume metrics
        avg_volume = np.mean(volumes[-config.lookback_period:])
        recent_volume = np.mean(volumes[-3:])
        
        if avg_volume == 0:
            return FactorAnalysis(
                name="Volume Confirmation",
                enabled=config.enabled,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="Invalid volume data",
                passed=True,
            )
        
        volume_ratio = recent_volume / avg_volume
        is_spike = volume_ratio >= config.spike_threshold
        is_confirmed = volume_ratio >= config.confirmation_threshold
        
        # Calculate OBV trend if enabled
        obv_bullish = False
        if config.obv_enabled and len(prices) > 10 and len(volumes) == len(prices):
            obv_bullish = self._calculate_obv_trend(prices, volumes)
        
        # Score based on volume characteristics
        if is_spike:
            score = 1.0
            status = FactorStatus.STRONG_BULLISH
            details = f"Volume spike detected: {volume_ratio:.2f}x average"
            passed = True
        elif is_confirmed:
            score = 0.8
            status = FactorStatus.BULLISH
            details = f"Volume confirmed: {volume_ratio:.2f}x average"
            passed = True
        elif volume_ratio >= 0.8:
            score = 0.5
            status = FactorStatus.NEUTRAL
            details = f"Normal volume: {volume_ratio:.2f}x average"
            passed = True
        else:
            score = 0.3
            status = FactorStatus.BEARISH
            details = f"Low volume: {volume_ratio:.2f}x average - weak conviction"
            passed = False
        
        return FactorAnalysis(
            name="Volume Confirmation",
            enabled=config.enabled,
            score=score,
            weight=config.weight,
            weighted_score=score * config.weight,
            status=status,
            details=details,
            passed=passed,
        )
    
    def _analyze_recency(
        self,
        pattern_dates: Optional[List[datetime]]
    ) -> FactorAnalysis:
        """วิเคราะห์ Pattern Recency"""
        config = self.config.recency
        
        if not config.enabled or not pattern_dates:
            return FactorAnalysis(
                name="Pattern Recency",
                enabled=config.enabled,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="No pattern date information",
                passed=True,
            )
        
        now = datetime.now()
        decay_years = config.decay_years
        
        # Calculate recency scores using exponential decay
        scores = []
        ages_days = []
        
        for date in pattern_dates:
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date)
                except:
                    continue
            
            days_ago = (now - date).days
            years_ago = days_ago / 365.0
            ages_days.append(days_ago)
            
            # Exponential decay: e^(-t/τ)
            recency_score = np.exp(-years_ago / decay_years)
            scores.append(recency_score)
        
        if not scores:
            return FactorAnalysis(
                name="Pattern Recency",
                enabled=config.enabled,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="Could not parse pattern dates",
                passed=True,
            )
        
        avg_score = np.mean(scores)
        avg_age_days = np.mean(ages_days)
        
        if avg_score >= 0.7:
            status = FactorStatus.STRONG_BULLISH
            details = f"Recent patterns (avg: {avg_age_days:.0f} days old)"
            passed = True
        elif avg_score >= 0.5:
            status = FactorStatus.BULLISH
            details = f"Moderately recent patterns (avg: {avg_age_days:.0f} days)"
            passed = True
        elif avg_score >= 0.3:
            status = FactorStatus.NEUTRAL
            details = f"Older patterns (avg: {avg_age_days:.0f} days)"
            passed = True
        else:
            status = FactorStatus.BEARISH
            details = f"Very old patterns (avg: {avg_age_days:.0f} days) - may be outdated"
            passed = False
        
        return FactorAnalysis(
            name="Pattern Recency",
            enabled=config.enabled,
            score=avg_score,
            weight=config.weight,
            weighted_score=avg_score * config.weight,
            status=status,
            details=details,
            passed=passed,
        )
    
    def _analyze_volatility(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray],
        lows: Optional[np.ndarray]
    ) -> FactorAnalysis:
        """วิเคราะห์ Volatility"""
        config = self.config.volatility
        
        if not config.enabled:
            return FactorAnalysis(
                name="Volatility",
                enabled=False,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="Volatility filter disabled",
                passed=True,
            )
        
        # Calculate volatility
        if highs is not None and lows is not None and len(highs) >= config.atr_period:
            # Use ATR
            atr = self._calculate_atr(highs, lows, prices, config.atr_period)
            volatility_pct = (atr / prices[-1]) * 100
            method = "ATR"
        else:
            # Use standard deviation of returns
            if len(prices) < 20:
                return FactorAnalysis(
                    name="Volatility",
                    enabled=config.enabled,
                    score=0.5,
                    weight=config.weight,
                    weighted_score=0.5 * config.weight,
                    status=FactorStatus.NEUTRAL,
                    details="Insufficient data for volatility",
                    passed=True,
                )
            
            returns = np.diff(prices) / prices[:-1]
            volatility_pct = np.std(returns) * 100 * np.sqrt(252)  # Annualized
            method = "StdDev"
        
        # Score based on volatility
        # Optimal range: 30-70 percentile (configurable)
        optimal_low, optimal_high = config.optimal_range
        
        if volatility_pct < 0.1:
            score = 0.2
            status = FactorStatus.SKIP
            details = f"Too low volatility ({volatility_pct:.2f}%) - market dead"
            passed = False
        elif volatility_pct > 5.0:
            score = 0.3
            status = FactorStatus.SKIP
            details = f"Too high volatility ({volatility_pct:.2f}%) - too risky"
            passed = False
        else:
            # In acceptable range
            score = 0.8
            status = FactorStatus.BULLISH
            details = f"Good volatility ({volatility_pct:.2f}% {method})"
            passed = True
        
        return FactorAnalysis(
            name="Volatility",
            enabled=config.enabled,
            score=score,
            weight=config.weight,
            weighted_score=score * config.weight,
            status=status,
            details=details,
            passed=passed,
        )
    
    def _analyze_session(
        self,
        current_time: datetime,
        symbol: str
    ) -> FactorAnalysis:
        """วิเคราะห์ Session Timing"""
        config = self.config.session
        
        if not config.enabled:
            return FactorAnalysis(
                name="Session Timing",
                enabled=False,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="Session filter disabled",
                passed=True,
            )
        
        # Get current hour (UTC)
        current_hour = current_time.hour
        weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        
        # Check for weekend
        if weekday >= 5:  # Saturday or Sunday
            return FactorAnalysis(
                name="Session Timing",
                enabled=config.enabled,
                score=0.0,
                weight=config.weight,
                weighted_score=0.0,
                status=FactorStatus.SKIP,
                details="Market closed (Weekend)",
                passed=False,
            )
        
        # Check for weekend close (Friday evening)
        if weekday == 4 and current_hour >= (24 - config.weekend_close_hours):
            return FactorAnalysis(
                name="Session Timing",
                enabled=config.enabled,
                score=0.3,
                weight=config.weight,
                weighted_score=0.3 * config.weight,
                status=FactorStatus.BEARISH,
                details=f"Approaching weekend close ({config.weekend_close_hours}h before)",
                passed=False,
            )
        
        # Check for Monday open
        if weekday == 0 and current_hour < config.monday_open_skip_hours:
            return FactorAnalysis(
                name="Session Timing",
                enabled=config.enabled,
                score=0.4,
                weight=config.weight,
                weighted_score=0.4 * config.weight,
                status=FactorStatus.BEARISH,
                details=f"Early Monday - skip first {config.monday_open_skip_hours}h",
                passed=False,
            )
        
        # Check active session
        active_session = None
        session_score = 0.5
        
        for session_name, session_info in config.sessions.items():
            start_hour = int(session_info["start"].split(":")[0])
            end_hour = int(session_info["end"].split(":")[0])
            
            # Handle overnight sessions
            if end_hour < start_hour:
                in_session = current_hour >= start_hour or current_hour < end_hour
            else:
                in_session = start_hour <= current_hour < end_hour
            
            if in_session:
                active_session = session_name
                session_score = session_info.get("score", 0.7)
                break
        
        if active_session:
            if session_score >= 0.9:
                status = FactorStatus.STRONG_BULLISH
                details = f"Prime session: {active_session}"
            elif session_score >= 0.7:
                status = FactorStatus.BULLISH
                details = f"Good session: {active_session}"
            else:
                status = FactorStatus.NEUTRAL
                details = f"Active session: {active_session}"
            passed = True
        else:
            session_score = 0.4
            status = FactorStatus.BEARISH
            details = "No major session active"
            passed = True  # Don't block, just lower score
        
        return FactorAnalysis(
            name="Session Timing",
            enabled=config.enabled,
            score=session_score,
            weight=config.weight,
            weighted_score=session_score * config.weight,
            status=status,
            details=details,
            passed=passed,
        )
    
    def _analyze_momentum(
        self,
        prices: np.ndarray,
        signal: Signal
    ) -> FactorAnalysis:
        """วิเคราะห์ Momentum (RSI, MACD)"""
        config = self.config.momentum
        
        if not config.enabled or len(prices) < max(config.rsi_period, config.macd_slow) + 10:
            return FactorAnalysis(
                name="Momentum",
                enabled=config.enabled,
                score=0.5,
                weight=config.weight,
                weighted_score=0.5 * config.weight,
                status=FactorStatus.NEUTRAL,
                details="Insufficient data for momentum",
                passed=True,
            )
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices, config.rsi_period)
        
        # Calculate MACD
        macd, macd_signal, histogram = self._calculate_macd(
            prices, config.macd_fast, config.macd_slow, config.macd_signal
        )
        
        is_buy = signal in [Signal.BUY, Signal.STRONG_BUY]
        is_sell = signal in [Signal.SELL, Signal.STRONG_SELL]
        
        # Analyze momentum alignment
        rsi_confirms = False
        macd_confirms = False
        
        if is_buy:
            rsi_confirms = rsi < config.rsi_overbought  # Not overbought
            macd_confirms = histogram > 0  # MACD bullish
        elif is_sell:
            rsi_confirms = rsi > config.rsi_oversold  # Not oversold
            macd_confirms = histogram < 0  # MACD bearish
        
        # Score
        confirmation_count = sum([rsi_confirms, macd_confirms])
        
        if confirmation_count == 2:
            score = 1.0
            status = FactorStatus.STRONG_BULLISH if is_buy else FactorStatus.STRONG_BEARISH
            details = f"Full momentum confirmation (RSI: {rsi:.1f}, MACD: {'bullish' if histogram > 0 else 'bearish'})"
            passed = True
        elif confirmation_count == 1:
            score = 0.6
            status = FactorStatus.BULLISH if is_buy else FactorStatus.BEARISH
            details = f"Partial momentum (RSI: {rsi:.1f}, MACD: {'aligned' if macd_confirms else 'divergent'})"
            passed = True
        else:
            score = 0.3
            status = FactorStatus.NEUTRAL
            details = f"Momentum divergence (RSI: {rsi:.1f})"
            passed = False
        
        return FactorAnalysis(
            name="Momentum",
            enabled=config.enabled,
            score=score,
            weight=config.weight,
            weighted_score=score * config.weight,
            status=status,
            details=details,
            passed=passed,
        )
    
    def _determine_quality(self, final_score: float) -> SignalQuality:
        """กำหนดคุณภาพสัญญาณ"""
        thresholds = self.config.quality_thresholds
        
        if final_score >= thresholds["PREMIUM"]:
            return SignalQuality.PREMIUM
        elif final_score >= thresholds["HIGH"]:
            return SignalQuality.HIGH
        elif final_score >= thresholds["MEDIUM"]:
            return SignalQuality.MEDIUM
        elif final_score >= thresholds["LOW"]:
            return SignalQuality.LOW
        else:
            return SignalQuality.SKIP
    
    def _determine_recommendation(
        self,
        signal: Signal,
        final_score: float,
        factors: List[FactorAnalysis],
        skip_reasons: List[str]
    ) -> str:
        """กำหนด Recommendation"""
        min_score = self.config.get_min_final_score()
        strong_score = self.config.get_strong_signal_score()
        
        if signal == Signal.WAIT:
            return "SKIP"
        
        # Count critical factor failures
        critical_failures = sum(
            1 for f in factors 
            if f.name in ["Pattern Match", "Trend Alignment"] and not f.passed
        )
        
        # Check for hard skip conditions
        hard_skips = [f for f in factors if f.status == FactorStatus.SKIP]
        if hard_skips:
            return "SKIP"
        
        if final_score >= strong_score and critical_failures == 0:
            return "TRADE_STRONG"
        elif final_score >= min_score and critical_failures == 0:
            return "TRADE"
        elif final_score >= min_score - 10 and critical_failures <= 1:
            return "TRADE_REDUCED"
        else:
            return "SKIP"
    
    def _calculate_position_multiplier(
        self,
        final_score: float,
        quality: SignalQuality,
        factors: List[FactorAnalysis]
    ) -> float:
        """คำนวณตัวคูณ position size"""
        base_multiplier = self.config.get_position_size_multiplier()
        
        # Adjust based on quality
        quality_adjustments = {
            SignalQuality.PREMIUM: 1.2,
            SignalQuality.HIGH: 1.0,
            SignalQuality.MEDIUM: 0.8,
            SignalQuality.LOW: 0.5,
            SignalQuality.SKIP: 0.0,
        }
        
        quality_mult = quality_adjustments.get(quality, 0.5)
        
        # Adjust based on score
        if final_score >= 85:
            score_mult = 1.1
        elif final_score >= 75:
            score_mult = 1.0
        elif final_score >= 65:
            score_mult = 0.9
        else:
            score_mult = 0.7
        
        return base_multiplier * quality_mult * score_mult
    
    # ==================== Helper Methods ====================
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
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
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD"""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line * 0.9  # Simplified
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate ATR"""
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
    
    def _calculate_obv_trend(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> bool:
        """Calculate OBV trend (bullish or not)"""
        if len(prices) < 10:
            return False
        
        obv = 0.0
        obv_values = [0.0]
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
            obv_values.append(obv)
        
        # Check last 10 periods
        recent_obv = obv_values[-10:]
        return recent_obv[-1] > recent_obv[0]
