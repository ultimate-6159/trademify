"""
??? Peak Detection Module - Avoid Buying at Tops, Selling at Bottoms!

?????:
- ?????? BUY ????? ? ????????????????? ? ???????? ? PEAK ? ?????? ? ??? SL
- ???????????????? peak ????? signal ????????????????!

Solution: ???????????????????????????? peak ???? bottom

Techniques:
1. RSI Divergence - ???????? ??? RSI ?? = peak
2. Volume Exhaustion - ???????? ??? Volume ?? = buying exhaustion
3. ATR Extension - ??????????? mean ????????? = overextended
4. Candle Pattern - Shooting Star, Doji, Hammer ??? extreme
5. Price Extension - ??????? EMA ??????? = mean reversion soon
6. Multi-TF Conflict - H1 bullish ??? H4 bearish = risky
7. Momentum Exhaustion - MACD histogram ???? = momentum fading

Usage:
    detector = PeakDetector()
    result = await detector.analyze(symbol, df_h1, df_m15)
    
    if result.is_peak:
        # Don't BUY! Wait for pullback
    if result.is_bottom:
        # Don't SELL! Wait for bounce
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketExtreme(str, Enum):
    """Market extreme states"""
    PEAK = "PEAK"               # ??????????? peak - ?????? BUY
    NEAR_PEAK = "NEAR_PEAK"     # ???? peak - ????? BUY
    NEUTRAL = "NEUTRAL"         # ???? - ???????
    NEAR_BOTTOM = "NEAR_BOTTOM" # ???? bottom - ????? SELL
    BOTTOM = "BOTTOM"           # ??????????? bottom - ?????? SELL


@dataclass
class PeakDetectionResult:
    """Result from peak detection analysis"""
    symbol: str
    extreme: MarketExtreme
    is_peak: bool
    is_bottom: bool
    can_buy: bool               # True if safe to BUY
    can_sell: bool              # True if safe to SELL
    confidence: float           # 0-100% confidence in detection
    
    # Individual signals
    rsi_divergence: str         # "BULLISH_DIV", "BEARISH_DIV", "NONE"
    volume_exhaustion: str      # "BUYING_EXHAUSTED", "SELLING_EXHAUSTED", "NORMAL"
    price_extension: str        # "OVEREXTENDED_UP", "OVEREXTENDED_DOWN", "NORMAL"
    candle_pattern: str         # "SHOOTING_STAR", "HAMMER", "DOJI", "NONE"
    momentum_status: str        # "FADING", "STRONG", "NORMAL"
    atr_spike: bool             # True if ATR is spiking (volatility high)
    mtf_conflict: bool          # True if multi-timeframe conflict
    
    # Recommendations
    wait_for_pullback: bool     # True if should wait for pullback before BUY
    wait_for_bounce: bool       # True if should wait for bounce before SELL
    suggested_wait_pips: float  # How many pips/$ to wait for pullback
    
    # Debug info
    indicators: Dict[str, float]
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "extreme": self.extreme.value,
            "is_peak": self.is_peak,
            "is_bottom": self.is_bottom,
            "can_buy": self.can_buy,
            "can_sell": self.can_sell,
            "confidence": self.confidence,
            "rsi_divergence": self.rsi_divergence,
            "volume_exhaustion": self.volume_exhaustion,
            "price_extension": self.price_extension,
            "candle_pattern": self.candle_pattern,
            "momentum_status": self.momentum_status,
            "atr_spike": self.atr_spike,
            "mtf_conflict": self.mtf_conflict,
            "wait_for_pullback": self.wait_for_pullback,
            "wait_for_bounce": self.wait_for_bounce,
            "suggested_wait_pips": self.suggested_wait_pips,
            "indicators": self.indicators,
            "reasons": self.reasons,
        }


class PeakDetector:
    """
    ??? Peak Detection System
    
    Detects when price is at or near a peak/bottom to avoid bad entries.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Thresholds
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_extreme_high = self.config.get("rsi_extreme_high", 80)
        self.rsi_extreme_low = self.config.get("rsi_extreme_low", 20)
        
        self.extension_threshold_pct = self.config.get("extension_threshold_pct", 1.5)  # % from EMA
        self.atr_spike_multiplier = self.config.get("atr_spike_multiplier", 1.5)
        self.volume_exhaustion_threshold = self.config.get("volume_exhaustion_threshold", 0.5)  # 50% of avg
        
        self.enabled = self.config.get("enabled", True)
    
    async def analyze(
        self, 
        symbol: str, 
        df_main: pd.DataFrame,
        df_lower: Optional[pd.DataFrame] = None,
        current_signal: Optional[str] = None
    ) -> PeakDetectionResult:
        """
        Analyze if price is at peak or bottom
        
        Args:
            symbol: Trading symbol
            df_main: Main timeframe data (e.g., H1)
            df_lower: Lower timeframe data (e.g., M15) for confirmation
            current_signal: Current signal direction ("BUY" or "SELL")
        
        Returns:
            PeakDetectionResult with analysis
        """
        if not self.enabled or df_main is None or len(df_main) < 30:
            return self._neutral_result(symbol)
        
        try:
            indicators = {}
            reasons = []
            peak_score = 0  # Positive = peak, Negative = bottom
            
            close = df_main['close'].values.astype(np.float32)
            high = df_main['high'].values.astype(np.float32)
            low = df_main['low'].values.astype(np.float32)
            volume = df_main['volume'].values.astype(np.float32) if 'volume' in df_main.columns else None
            
            current_price = float(close[-1])
            
            # 1. RSI Analysis + Divergence
            rsi = self._calculate_rsi(close, 14)
            rsi_divergence = self._detect_rsi_divergence(close, rsi)
            indicators["rsi"] = float(rsi[-1]) if len(rsi) > 0 else 50
            
            if rsi[-1] >= self.rsi_extreme_high:
                peak_score += 30
                reasons.append(f"RSI extreme overbought ({rsi[-1]:.1f})")
            elif rsi[-1] >= self.rsi_overbought:
                peak_score += 15
                reasons.append(f"RSI overbought ({rsi[-1]:.1f})")
            elif rsi[-1] <= self.rsi_extreme_low:
                peak_score -= 30
                reasons.append(f"RSI extreme oversold ({rsi[-1]:.1f})")
            elif rsi[-1] <= self.rsi_oversold:
                peak_score -= 15
                reasons.append(f"RSI oversold ({rsi[-1]:.1f})")
            
            if rsi_divergence == "BEARISH_DIV":
                peak_score += 25
                reasons.append("Bearish RSI divergence (hidden peak)")
            elif rsi_divergence == "BULLISH_DIV":
                peak_score -= 25
                reasons.append("Bullish RSI divergence (hidden bottom)")
            
            # 2. Volume Exhaustion
            volume_status = "NORMAL"
            if volume is not None and len(volume) > 20:
                avg_volume = float(np.mean(volume[-20:-1]))
                current_volume = float(volume[-1])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                indicators["volume_ratio"] = volume_ratio
                
                # Rising price + falling volume = buying exhaustion
                price_rising = close[-1] > close[-5]
                volume_falling = volume[-1] < volume[-3] < volume[-5]
                
                if price_rising and volume_falling and volume_ratio < self.volume_exhaustion_threshold:
                    volume_status = "BUYING_EXHAUSTED"
                    peak_score += 20
                    reasons.append(f"Volume exhaustion on rise ({volume_ratio:.2f}x avg)")
                
                # Falling price + falling volume = selling exhaustion
                price_falling = close[-1] < close[-5]
                if price_falling and volume_falling and volume_ratio < self.volume_exhaustion_threshold:
                    volume_status = "SELLING_EXHAUSTED"
                    peak_score -= 20
                    reasons.append(f"Volume exhaustion on fall ({volume_ratio:.2f}x avg)")
            
            # 3. Price Extension from EMA
            ema20 = self._calculate_ema(close, 20)
            ema50 = self._calculate_ema(close, 50)
            extension_status = "NORMAL"
            
            if len(ema20) > 0 and len(ema50) > 0:
                ema20_current = float(ema20[-1])
                ema50_current = float(ema50[-1])
                
                extension_from_ema20 = ((current_price - ema20_current) / ema20_current) * 100
                extension_from_ema50 = ((current_price - ema50_current) / ema50_current) * 100
                
                indicators["extension_ema20_pct"] = extension_from_ema20
                indicators["extension_ema50_pct"] = extension_from_ema50
                
                if extension_from_ema20 > self.extension_threshold_pct:
                    extension_status = "OVEREXTENDED_UP"
                    peak_score += 20
                    reasons.append(f"Price overextended above EMA20 ({extension_from_ema20:.2f}%)")
                elif extension_from_ema20 < -self.extension_threshold_pct:
                    extension_status = "OVEREXTENDED_DOWN"
                    peak_score -= 20
                    reasons.append(f"Price overextended below EMA20 ({extension_from_ema20:.2f}%)")
            
            # 4. Candle Pattern Detection
            candle_pattern = self._detect_extreme_candle(high, low, close)
            
            if candle_pattern == "SHOOTING_STAR":
                peak_score += 15
                reasons.append("Shooting star pattern (bearish)")
            elif candle_pattern == "HAMMER":
                peak_score -= 15
                reasons.append("Hammer pattern (bullish)")
            elif candle_pattern == "DOJI":
                # Doji at extreme = indecision, possible reversal
                if rsi[-1] >= self.rsi_overbought:
                    peak_score += 10
                    reasons.append("Doji at overbought level")
                elif rsi[-1] <= self.rsi_oversold:
                    peak_score -= 10
                    reasons.append("Doji at oversold level")
            
            # 5. ATR Spike Detection
            atr = self._calculate_atr(high, low, close, 14)
            atr_spike = False
            
            if len(atr) > 5:
                avg_atr = float(np.mean(atr[-20:-1]))
                current_atr = float(atr[-1])
                atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
                indicators["atr_ratio"] = atr_ratio
                indicators["current_atr"] = current_atr
                
                if atr_ratio > self.atr_spike_multiplier:
                    atr_spike = True
                    reasons.append(f"ATR spike ({atr_ratio:.2f}x normal) - high volatility")
                    # ATR spike at extreme = more confidence in reversal
                    if peak_score > 0:
                        peak_score += 10
                    elif peak_score < 0:
                        peak_score -= 10
            
            # 6. MACD Momentum
            macd_line, signal_line, histogram = self._calculate_macd(close)
            momentum_status = "NORMAL"
            
            if len(histogram) >= 3:
                indicators["macd_histogram"] = float(histogram[-1])
                
                # Histogram shrinking = momentum fading
                if histogram[-1] > 0:
                    # Bullish but fading?
                    if histogram[-1] < histogram[-2] < histogram[-3]:
                        momentum_status = "FADING"
                        peak_score += 15
                        reasons.append("Bullish momentum fading (MACD histogram decreasing)")
                    else:
                        momentum_status = "STRONG"
                elif histogram[-1] < 0:
                    # Bearish but fading?
                    if histogram[-1] > histogram[-2] > histogram[-3]:
                        momentum_status = "FADING"
                        peak_score -= 15
                        reasons.append("Bearish momentum fading (MACD histogram increasing)")
                    else:
                        momentum_status = "STRONG"
            
            # 7. Multi-Timeframe Conflict
            mtf_conflict = False
            if df_lower is not None and len(df_lower) >= 30:
                lower_close = df_lower['close'].values.astype(np.float32)
                lower_rsi = self._calculate_rsi(lower_close, 14)
                lower_ema20 = self._calculate_ema(lower_close, 20)
                
                if len(lower_rsi) > 0 and len(lower_ema20) > 0:
                    # Main TF says buy, but lower TF is overbought
                    if current_signal == "BUY" and lower_rsi[-1] >= self.rsi_overbought:
                        mtf_conflict = True
                        peak_score += 15
                        reasons.append(f"MTF conflict: Lower TF RSI overbought ({lower_rsi[-1]:.1f})")
                    
                    # Main TF says sell, but lower TF is oversold
                    if current_signal == "SELL" and lower_rsi[-1] <= self.rsi_oversold:
                        mtf_conflict = True
                        peak_score -= 15
                        reasons.append(f"MTF conflict: Lower TF RSI oversold ({lower_rsi[-1]:.1f})")
            
            # 8. Recent Swing High/Low Detection
            swing_result = self._detect_recent_swing(high, low, close)
            if swing_result == "AT_SWING_HIGH":
                peak_score += 20
                reasons.append("Price at recent swing high")
            elif swing_result == "AT_SWING_LOW":
                peak_score -= 20
                reasons.append("Price at recent swing low")
            
            # Calculate final extreme state
            extreme = self._calculate_extreme(peak_score)
            is_peak = extreme in [MarketExtreme.PEAK, MarketExtreme.NEAR_PEAK]
            is_bottom = extreme in [MarketExtreme.BOTTOM, MarketExtreme.NEAR_BOTTOM]
            
            # Determine if can trade
            can_buy = not is_peak  # Don't BUY at peak
            can_sell = not is_bottom  # Don't SELL at bottom
            
            # If signal conflicts with extreme, block it
            if current_signal == "BUY" and is_peak:
                can_buy = False
                reasons.append("?? BLOCKED: BUY signal at PEAK")
            if current_signal == "SELL" and is_bottom:
                can_sell = False
                reasons.append("?? BLOCKED: SELL signal at BOTTOM")
            
            # Calculate pullback suggestion
            current_atr = indicators.get("current_atr", current_price * 0.002)
            suggested_wait = current_atr * 0.3  # Wait for 0.3 ATR pullback
            
            # Confidence based on how many signals agree
            signal_count = len(reasons)
            confidence = min(95, 50 + (abs(peak_score) / 2))
            
            return PeakDetectionResult(
                symbol=symbol,
                extreme=extreme,
                is_peak=is_peak,
                is_bottom=is_bottom,
                can_buy=can_buy,
                can_sell=can_sell,
                confidence=confidence,
                rsi_divergence=rsi_divergence,
                volume_exhaustion=volume_status,
                price_extension=extension_status,
                candle_pattern=candle_pattern,
                momentum_status=momentum_status,
                atr_spike=atr_spike,
                mtf_conflict=mtf_conflict,
                wait_for_pullback=is_peak,
                wait_for_bounce=is_bottom,
                suggested_wait_pips=round(suggested_wait, 2),
                indicators=indicators,
                reasons=reasons,
            )
            
        except Exception as e:
            logger.error(f"Peak detection error for {symbol}: {e}")
            return self._neutral_result(symbol)
    
    def _neutral_result(self, symbol: str) -> PeakDetectionResult:
        """Return neutral result when analysis not possible"""
        return PeakDetectionResult(
            symbol=symbol,
            extreme=MarketExtreme.NEUTRAL,
            is_peak=False,
            is_bottom=False,
            can_buy=True,
            can_sell=True,
            confidence=0,
            rsi_divergence="NONE",
            volume_exhaustion="NORMAL",
            price_extension="NORMAL",
            candle_pattern="NONE",
            momentum_status="NORMAL",
            atr_spike=False,
            mtf_conflict=False,
            wait_for_pullback=False,
            wait_for_bounce=False,
            suggested_wait_pips=0,
            indicators={},
            reasons=["Analysis not available"],
        )
    
    def _calculate_extreme(self, score: float) -> MarketExtreme:
        """Convert peak score to MarketExtreme"""
        if score >= 60:
            return MarketExtreme.PEAK
        elif score >= 35:
            return MarketExtreme.NEAR_PEAK
        elif score <= -60:
            return MarketExtreme.BOTTOM
        elif score <= -35:
            return MarketExtreme.NEAR_BOTTOM
        else:
            return MarketExtreme.NEUTRAL
    
    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        if len(close) < period + 1:
            return np.array([50.0])
        
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(gains))
        avg_loss = np.zeros(len(losses))
        
        avg_gain[period-1] = np.mean(gains[:period])
        avg_loss[period-1] = np.mean(losses[:period])
        
        for i in range(period, len(gains)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        if len(data) < period:
            return np.array([])
        
        multiplier = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[period-1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate ATR"""
        if len(close) < period + 1:
            return np.array([])
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    def _calculate_macd(self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        if len(close) < slow + signal:
            return np.array([]), np.array([]), np.array([])
        
        ema_fast = self._calculate_ema(close, fast)
        ema_slow = self._calculate_ema(close, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line[slow-1:], signal)
        
        # Pad signal_line to match macd_line length
        signal_padded = np.zeros(len(macd_line))
        signal_padded[slow-1+signal-1:] = signal_line[signal-1:]
        
        histogram = macd_line - signal_padded
        
        return macd_line, signal_padded, histogram
    
    def _detect_rsi_divergence(self, close: np.ndarray, rsi: np.ndarray, lookback: int = 10) -> str:
        """
        Detect RSI divergence
        
        Bearish divergence: Price makes higher high, RSI makes lower high = PEAK signal
        Bullish divergence: Price makes lower low, RSI makes higher low = BOTTOM signal
        """
        if len(close) < lookback or len(rsi) < lookback:
            return "NONE"
        
        price_recent = close[-lookback:]
        rsi_recent = rsi[-lookback:]
        
        # Find local highs/lows
        price_high_idx = np.argmax(price_recent)
        price_low_idx = np.argmin(price_recent)
        
        # Check bearish divergence (price up, RSI down)
        if price_high_idx > len(price_recent) // 2:  # Recent high
            if price_recent[-1] >= price_recent[0] * 0.999:  # Price still high
                if rsi_recent[-1] < rsi_recent[price_high_idx]:  # RSI falling
                    return "BEARISH_DIV"
        
        # Check bullish divergence (price down, RSI up)
        if price_low_idx > len(price_recent) // 2:  # Recent low
            if price_recent[-1] <= price_recent[0] * 1.001:  # Price still low
                if rsi_recent[-1] > rsi_recent[price_low_idx]:  # RSI rising
                    return "BULLISH_DIV"
        
        return "NONE"
    
    def _detect_extreme_candle(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> str:
        """Detect extreme candle patterns (last candle)"""
        if len(close) < 2:
            return "NONE"
        
        # Last candle
        h = high[-1]
        l = low[-1]
        c = close[-1]
        o = close[-2]  # Approximate open as previous close
        
        body = abs(c - o)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l
        total_range = h - l
        
        if total_range == 0:
            return "NONE"
        
        body_ratio = body / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        # Doji: Very small body
        if body_ratio < 0.1:
            return "DOJI"
        
        # Shooting Star: Small body at bottom, long upper shadow (bearish at top)
        if upper_shadow_ratio > 0.6 and body_ratio < 0.3 and lower_shadow_ratio < 0.1:
            return "SHOOTING_STAR"
        
        # Hammer: Small body at top, long lower shadow (bullish at bottom)
        if lower_shadow_ratio > 0.6 and body_ratio < 0.3 and upper_shadow_ratio < 0.1:
            return "HAMMER"
        
        return "NONE"
    
    def _detect_recent_swing(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 20) -> str:
        """Detect if price is at recent swing high or low"""
        if len(close) < lookback:
            return "NEUTRAL"
        
        recent_high = np.max(high[-lookback:])
        recent_low = np.min(low[-lookback:])
        current = close[-1]
        range_size = recent_high - recent_low
        
        if range_size == 0:
            return "NEUTRAL"
        
        # Within 5% of swing high
        if current >= recent_high * 0.995:
            return "AT_SWING_HIGH"
        
        # Within 5% of swing low
        if current <= recent_low * 1.005:
            return "AT_SWING_LOW"
        
        return "NEUTRAL"


# Singleton instance
_peak_detector: Optional[PeakDetector] = None


def get_peak_detector(config: Optional[Dict] = None) -> PeakDetector:
    """Get or create PeakDetector singleton"""
    global _peak_detector
    if _peak_detector is None:
        _peak_detector = PeakDetector(config)
    return _peak_detector


def create_peak_detector(config: Optional[Dict] = None) -> PeakDetector:
    """Create new PeakDetector instance"""
    return PeakDetector(config)
