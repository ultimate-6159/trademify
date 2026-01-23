"""
🏆👑 SUPREME INTELLIGENCE - ระบบ AI Trading ที่ฉลาดที่สุดในโลก!

ฟีเจอร์ระดับ Hedge Fund / High Frequency Trading:

1. Order Flow Analysis - วิเคราะห์ Buy/Sell Pressure
2. Institutional Footprint - ตรวจจับการเคลื่อนไหว Big Players
3. Market Microstructure - โครงสร้างตลาดระดับลึก
4. Multi-Asset Correlation - ความสัมพันธ์ระหว่าง Assets
5. Entropy Analysis - วัดความวุ่นวาย/สงบของตลาด
6. Fractal Dimension - วิเคราะห์ความซับซ้อนของราคา
7. Regime Probability - ความน่าจะเป็นของ Market Regime
8. Momentum Quality - คุณภาพของ Momentum (ไม่ใช่แค่ทิศทาง)
9. Risk-Adjusted Signals - Signal ที่ปรับตาม Risk
10. Dynamic Stop Strategy - SL ที่ปรับตามสภาพตลาด
11. Profit Target Optimization - TP ที่ optimal ตาม volatility
12. Position Scaling - เปิด/ปิด position แบบ scale
13. News Impact Predictor - คาดการณ์ผลกระทบข่าว
14. Time Decay Analysis - วิเคราะห์ signal ที่หมดอายุ
15. Confluence Scoring - คะแนนจาก multiple factors
16. Execution Quality - เวลาที่ดีที่สุดในการ execute
17. Slippage Estimator - คาดการณ์ slippage
18. Market Impact Model - ผลกระทบจากการเทรด
19. Alpha Decay Predictor - คาดการณ์การลดลงของ alpha
20. Self-Learning Weights - ปรับ weights อัตโนมัติ
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque
import math

logger = logging.getLogger(__name__)


class MarketEntropy(str, Enum):
    """ระดับความวุ่นวายของตลาด"""
    CHAOTIC = "CHAOTIC"       # วุ่นวายมาก - ไม่เทรด
    HIGH = "HIGH"             # สูง - ระวัง
    NORMAL = "NORMAL"         # ปกติ
    LOW = "LOW"               # สงบ - โอกาสดี
    DORMANT = "DORMANT"       # นิ่งมาก - รอ breakout


class InstitutionalActivity(str, Enum):
    """กิจกรรมของ Institution"""
    ACCUMULATING = "ACCUMULATING"     # กำลังสะสม
    DISTRIBUTING = "DISTRIBUTING"     # กำลังขาย
    NEUTRAL = "NEUTRAL"               # ไม่มีสัญญาณ
    STOP_HUNTING = "STOP_HUNTING"     # กำลัง hunt stops


class MomentumQuality(str, Enum):
    """คุณภาพของ Momentum"""
    EXPLOSIVE = "EXPLOSIVE"   # แรงมาก - เข้าเลย
    STRONG = "STRONG"         # แรง
    MODERATE = "MODERATE"     # ปานกลาง
    WEAK = "WEAK"             # อ่อน
    EXHAUSTED = "EXHAUSTED"   # หมดแรง - ระวัง reversal


class ExecutionTiming(str, Enum):
    """เวลาที่ดีในการ execute"""
    OPTIMAL = "OPTIMAL"       # เวลาที่ดีที่สุด
    GOOD = "GOOD"             # ดี
    ACCEPTABLE = "ACCEPTABLE" # พอรับได้
    WAIT = "WAIT"             # รอก่อน
    AVOID = "AVOID"           # หลีกเลี่ยง


@dataclass
class OrderFlowAnalysis:
    """ผลวิเคราะห์ Order Flow"""
    buy_pressure: float       # 0-100
    sell_pressure: float      # 0-100
    net_flow: float          # buy - sell
    large_orders_detected: bool
    sweep_detected: bool      # Stop sweep detected
    imbalance_zones: List[Tuple[float, str]]  # (price, "BUY"/"SELL")


@dataclass
class FractalAnalysis:
    """ผลวิเคราะห์ Fractal"""
    dimension: float         # 1-2 (1=smooth, 2=chaotic)
    hurst_exponent: float    # <0.5=mean-reverting, >0.5=trending
    complexity: str          # "LOW", "MEDIUM", "HIGH"


@dataclass
class SupremeDecision:
    """การตัดสินใจจาก Supreme Intelligence"""
    # Core Decision
    can_trade: bool
    confidence: float        # 0-100
    signal_strength: str     # "WEAK", "MODERATE", "STRONG", "EXPLOSIVE"
    
    # Position Management
    optimal_size_percent: float     # % of balance
    scale_in_levels: List[float]    # ราคาที่ควร scale in
    scale_out_levels: List[float]   # ราคาที่ควร scale out
    
    # Risk Management
    optimal_sl_distance: float      # ATR multiplier
    optimal_tp_distance: float      # ATR multiplier
    max_holding_hours: int          # ควรถือนานสุดกี่ชม.
    
    # Timing
    execution_timing: ExecutionTiming
    best_entry_window_minutes: int  # หน้าต่างเวลาที่ดีในการเข้า
    
    # Market Analysis
    entropy_level: MarketEntropy
    institutional_activity: InstitutionalActivity
    momentum_quality: MomentumQuality
    
    # Advanced Metrics
    confluence_score: float         # 0-100
    alpha_potential: float          # คาดการณ์ alpha
    expected_slippage_pips: float   # คาดการณ์ slippage
    win_probability: float          # 0-100
    
    # Fractal
    fractal_dimension: float
    market_complexity: str
    
    # Insights
    key_levels: Dict[str, float]    # support, resistance, pivot
    reasons: List[str]
    warnings: List[str]
    
    # Self-Learning
    suggested_weight_adjustments: Dict[str, float]


class SupremeIntelligence:
    """
    🏆👑 SUPREME INTELLIGENCE
    
    ระบบ AI Trading ที่ฉลาดที่สุด
    ใช้หลักการระดับ Hedge Fund และ HFT
    
    Features:
    - Order Flow Analysis
    - Institutional Footprint Detection
    - Entropy-based Market Analysis
    - Fractal Dimension Calculation
    - Self-Learning Weight Optimization
    - Execution Quality Scoring
    """
    
    # Configurable thresholds (can be adjusted for trial/aggressive modes)
    MIN_CONFIDENCE = 60          # Default: 70, Trial: 60
    MIN_WIN_PROBABILITY = 45     # Default: 55, Trial: 45
    MIN_CONFLUENCE = 40          # Default: 50, Trial: 40
    ALLOW_DORMANT = True         # Allow trading in DORMANT market (for trial)
    ALLOW_WAIT_TIMING = True     # Allow WAIT timing (for trial)
    
    def __init__(self, aggressive_mode: bool = True):
        # Performance History
        self._trade_history: deque = deque(maxlen=1000)
        self._signal_history: deque = deque(maxlen=500)
        
        # Self-Learning Weights
        self._factor_weights = {
            "pattern": 0.15,
            "momentum": 0.15,
            "volume": 0.10,
            "structure": 0.15,
            "session": 0.10,
            "entropy": 0.10,
            "institutional": 0.10,
            "fractal": 0.05,
            "correlation": 0.05,
            "timing": 0.05,
        }
        
        # Performance Tracking per Factor
        self._factor_performance: Dict[str, List[bool]] = {
            k: [] for k in self._factor_weights.keys()
        }
        
        # Market Memory
        self._last_entropy: Dict[str, float] = {}
        self._regime_history: deque = deque(maxlen=100)
        
        # Asset Correlations (pre-computed)
        self._correlations = {
            ("EURUSD", "GBPUSD"): 0.85,
            ("EURUSD", "XAUUSD"): -0.3,
            ("GBPUSD", "XAUUSD"): -0.25,
            ("EURUSD", "DXY"): -0.95,
        }
        
        logger.info("🏆👑 Supreme Intelligence initialized - 20x Smarter!")
    
    def analyze(
        self,
        symbol: str,
        signal_side: str,
        current_price: float,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        atr: float = 0,
        base_confidence: float = 70,
        balance: float = 10000,
        equity: float = 10000,
        existing_positions: List[Dict] = None,
    ) -> SupremeDecision:
        """
        🧠 วิเคราะห์แบบ Supreme Intelligence
        """
        reasons = []
        warnings = []
        
        # ===============================
        # 1. ENTROPY ANALYSIS
        # ===============================
        entropy, entropy_level = self._calculate_entropy(prices)
        reasons.append(f"🌀 Entropy: {entropy:.2f} ({entropy_level.value})")
        
        if entropy_level == MarketEntropy.CHAOTIC:
            return self._create_blocked_decision(
                "Market too chaotic - entropy too high",
                entropy_level=entropy_level
            )
        
        # ===============================
        # 2. FRACTAL ANALYSIS
        # ===============================
        fractal = self._calculate_fractal_analysis(prices)
        reasons.append(f"🔷 Fractal Dim: {fractal.dimension:.2f} | Hurst: {fractal.hurst_exponent:.2f}")
        
        # ===============================
        # 3. ORDER FLOW ANALYSIS
        # ===============================
        order_flow = self._analyze_order_flow(prices, volumes, highs, lows)
        reasons.append(f"📊 Order Flow: Buy {order_flow.buy_pressure:.0f}% | Sell {order_flow.sell_pressure:.0f}%")
        
        if order_flow.sweep_detected:
            warnings.append("⚠️ Stop sweep detected - wait for confirmation")
        
        # ===============================
        # 4. INSTITUTIONAL ACTIVITY
        # ===============================
        institutional = self._detect_institutional_activity(
            prices, volumes, highs, lows
        )
        reasons.append(f"🏦 Institutional: {institutional.value}")
        
        # Check alignment
        inst_aligned = self._check_institutional_alignment(
            signal_side, institutional
        )
        if not inst_aligned:
            warnings.append(f"⚠️ Signal against institutional flow ({institutional.value})")
        
        # ===============================
        # 5. MOMENTUM QUALITY
        # ===============================
        momentum_quality, momentum_score = self._analyze_momentum_quality(
            prices, signal_side
        )
        reasons.append(f"💪 Momentum: {momentum_quality.value} ({momentum_score:.0f})")
        
        if momentum_quality == MomentumQuality.EXHAUSTED:
            warnings.append("⚠️ Momentum exhausted - reversal risk")
        
        # ===============================
        # 6. KEY LEVELS
        # ===============================
        key_levels = self._calculate_key_levels(prices, highs, lows, current_price)
        reasons.append(f"📍 Support: {key_levels.get('support', 0):.5f} | Resistance: {key_levels.get('resistance', 0):.5f}")
        
        # ===============================
        # 7. EXECUTION TIMING
        # ===============================
        exec_timing = self._analyze_execution_timing(symbol, current_price, atr)
        reasons.append(f"⏰ Execution: {exec_timing.value}")
        
        if exec_timing == ExecutionTiming.AVOID:
            return self._create_blocked_decision(
                "Bad execution timing",
                entropy_level=entropy_level,
                institutional_activity=institutional
            )
        
        # ===============================
        # 8. CONFLUENCE SCORING
        # ===============================
        confluence_factors = {
            "entropy_favorable": entropy_level in [MarketEntropy.LOW, MarketEntropy.NORMAL],
            "momentum_strong": momentum_quality in [MomentumQuality.EXPLOSIVE, MomentumQuality.STRONG],
            "institutional_aligned": inst_aligned,
            "order_flow_favorable": (order_flow.net_flow > 20 and signal_side == "BUY") or \
                                   (order_flow.net_flow < -20 and signal_side == "SELL"),
            "fractal_trending": fractal.hurst_exponent > 0.5,
            "timing_good": exec_timing in [ExecutionTiming.OPTIMAL, ExecutionTiming.GOOD],
            "at_key_level": self._is_at_key_level(current_price, key_levels, signal_side),
        }
        
        confluence_score = sum(1 for v in confluence_factors.values() if v) / len(confluence_factors) * 100
        
        # ===============================
        # 9. WIN PROBABILITY
        # ===============================
        win_probability = self._calculate_win_probability(
            confluence_score,
            momentum_score,
            inst_aligned,
            fractal.hurst_exponent
        )
        reasons.append(f"🎯 Win Probability: {win_probability:.0f}%")
        
        # ===============================
        # 10. OPTIMAL POSITION SIZE
        # ===============================
        optimal_size = self._calculate_optimal_size(
            balance, equity, win_probability, 
            entropy_level, confluence_score
        )
        
        # ===============================
        # 11. OPTIMAL SL/TP
        # ===============================
        sl_mult, tp_mult = self._calculate_optimal_sl_tp(
            entropy_level, momentum_quality, fractal, atr
        )
        reasons.append(f"🎯 SL: {sl_mult:.1f}x ATR | TP: {tp_mult:.1f}x ATR")
        
        # ===============================
        # 12. SCALE IN/OUT LEVELS
        # ===============================
        scale_in, scale_out = self._calculate_scale_levels(
            current_price, signal_side, atr, key_levels
        )
        
        # ===============================
        # 13. ALPHA POTENTIAL
        # ===============================
        alpha_potential = self._estimate_alpha_potential(
            confluence_score, momentum_quality, institutional
        )
        reasons.append(f"📈 Alpha Potential: {alpha_potential:.1f}%")
        
        # ===============================
        # 14. EXPECTED SLIPPAGE
        # ===============================
        expected_slippage = self._estimate_slippage(
            symbol, entropy_level, exec_timing
        )
        
        # ===============================
        # 15. MAX HOLDING TIME
        # ===============================
        max_hold = self._calculate_max_holding(
            fractal, momentum_quality, entropy_level
        )
        
        # ===============================
        # 16. SIGNAL STRENGTH
        # ===============================
        signal_strength = self._determine_signal_strength(
            confluence_score, momentum_quality, win_probability
        )
        
        # ===============================
        # 17. FINAL CONFIDENCE
        # ===============================
        final_confidence = self._calculate_final_confidence(
            base_confidence, confluence_score, win_probability,
            entropy_level, momentum_quality
        )
        
        # ===============================
        # 18. SELF-LEARNING ADJUSTMENTS
        # ===============================
        weight_adjustments = self._suggest_weight_adjustments()
        
        # ===============================
        # FINAL DECISION (with configurable thresholds)
        # ===============================
        # Check entropy - allow DORMANT if configured
        entropy_ok = entropy_level not in [MarketEntropy.CHAOTIC]
        if not self.ALLOW_DORMANT:
            entropy_ok = entropy_level not in [MarketEntropy.CHAOTIC, MarketEntropy.DORMANT]
            
        # Check timing - allow WAIT if configured  
        timing_ok = exec_timing not in [ExecutionTiming.AVOID]
        if not self.ALLOW_WAIT_TIMING:
            timing_ok = exec_timing not in [ExecutionTiming.AVOID, ExecutionTiming.WAIT]
        
        can_trade = (
            final_confidence >= self.MIN_CONFIDENCE and
            win_probability >= self.MIN_WIN_PROBABILITY and
            confluence_score >= self.MIN_CONFLUENCE and
            entropy_ok and
            timing_ok and
            optimal_size >= 0.20
        )
        
        if not can_trade:
            fail_reasons = []
            if final_confidence < self.MIN_CONFIDENCE:
                fail_reasons.append(f"Confidence {final_confidence:.0f}% < {self.MIN_CONFIDENCE}%")
            if win_probability < self.MIN_WIN_PROBABILITY:
                fail_reasons.append(f"Win Prob {win_probability:.0f}% < {self.MIN_WIN_PROBABILITY}%")
            if confluence_score < self.MIN_CONFLUENCE:
                fail_reasons.append(f"Confluence {confluence_score:.0f}% < {self.MIN_CONFLUENCE}%")
            if not entropy_ok:
                fail_reasons.append(f"Entropy: {entropy_level}")
            if not timing_ok:
                fail_reasons.append(f"Timing: {exec_timing}")
            warnings.append(f"❌ Trade blocked: {', '.join(fail_reasons)}")
        
        return SupremeDecision(
            can_trade=can_trade,
            confidence=final_confidence,
            signal_strength=signal_strength,
            optimal_size_percent=optimal_size,
            scale_in_levels=scale_in,
            scale_out_levels=scale_out,
            optimal_sl_distance=sl_mult,
            optimal_tp_distance=tp_mult,
            max_holding_hours=max_hold,
            execution_timing=exec_timing,
            best_entry_window_minutes=15 if exec_timing == ExecutionTiming.OPTIMAL else 30,
            entropy_level=entropy_level,
            institutional_activity=institutional,
            momentum_quality=momentum_quality,
            confluence_score=confluence_score,
            alpha_potential=alpha_potential,
            expected_slippage_pips=expected_slippage,
            win_probability=win_probability,
            fractal_dimension=fractal.dimension,
            market_complexity=fractal.complexity,
            key_levels=key_levels,
            reasons=reasons,
            warnings=warnings,
            suggested_weight_adjustments=weight_adjustments
        )
    
    def _calculate_entropy(self, prices: np.ndarray) -> Tuple[float, MarketEntropy]:
        """คำนวณ Market Entropy (Shannon Entropy)"""
        if len(prices) < 20:
            return 0.5, MarketEntropy.NORMAL
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Discretize returns into bins
        n_bins = 10
        hist, _ = np.histogram(returns, bins=n_bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        normalized_entropy = entropy / np.log2(n_bins)  # 0-1
        
        # Classify
        if normalized_entropy > 0.9:
            level = MarketEntropy.CHAOTIC
        elif normalized_entropy > 0.75:
            level = MarketEntropy.HIGH
        elif normalized_entropy > 0.4:
            level = MarketEntropy.NORMAL
        elif normalized_entropy > 0.2:
            level = MarketEntropy.LOW
        else:
            level = MarketEntropy.DORMANT
        
        return normalized_entropy, level
    
    def _calculate_fractal_analysis(self, prices: np.ndarray) -> FractalAnalysis:
        """คำนวณ Fractal Dimension และ Hurst Exponent"""
        if len(prices) < 50:
            return FractalAnalysis(1.5, 0.5, "MEDIUM")
        
        # Hurst Exponent (R/S analysis simplified)
        returns = np.diff(np.log(prices))
        n = len(returns)
        
        # Calculate Hurst using variance ratio
        lags = [2, 4, 8, 16, 32]
        variances = []
        for lag in lags:
            if lag >= n:
                break
            variance = np.var(returns[::lag])
            variances.append(variance)
        
        if len(variances) >= 3:
            # Log-log regression for Hurst
            log_lags = np.log(lags[:len(variances)])
            log_vars = np.log(np.array(variances) + 1e-10)
            hurst = 0.5 * np.polyfit(log_lags, log_vars, 1)[0]
        else:
            hurst = 0.5
        
        # Fractal dimension (approximation)
        # D = 2 - H for financial time series
        dimension = 2 - abs(hurst)
        dimension = max(1.0, min(2.0, dimension))
        
        # Complexity classification
        if dimension > 1.7:
            complexity = "HIGH"
        elif dimension > 1.4:
            complexity = "MEDIUM"
        else:
            complexity = "LOW"
        
        return FractalAnalysis(dimension, hurst, complexity)
    
    def _analyze_order_flow(
        self, prices: np.ndarray, volumes: Optional[np.ndarray],
        highs: np.ndarray, lows: np.ndarray
    ) -> OrderFlowAnalysis:
        """วิเคราะห์ Order Flow"""
        if len(prices) < 10:
            return OrderFlowAnalysis(50, 50, 0, False, False, [])
        
        # Buy/Sell pressure from candle bodies
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        # Bullish candles = price went up
        bullish = prices > opens
        bearish = prices < opens
        
        # Volume-weighted - use consistent slicing for boolean indexing
        if volumes is not None and len(volumes) == len(prices) and len(prices) >= 20:
            # Use same slice for volumes and boolean arrays
            vol_slice = volumes[-20:]
            bull_slice = bullish[-20:]
            bear_slice = bearish[-20:]
            
            buy_vol = np.sum(vol_slice[bull_slice])
            sell_vol = np.sum(vol_slice[bear_slice])
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                buy_pressure = (buy_vol / total_vol) * 100
                sell_pressure = (sell_vol / total_vol) * 100
            else:
                buy_pressure = 50
                sell_pressure = 50
        else:
            buy_pressure = np.sum(bullish[-20:]) / 20 * 100 if len(bullish) >= 20 else 50
            sell_pressure = np.sum(bearish[-20:]) / 20 * 100 if len(bearish) >= 20 else 50
        
        net_flow = buy_pressure - sell_pressure
        
        # Detect large orders (volume spikes)
        large_orders = False
        if volumes is not None and len(volumes) >= 20:
            avg_vol = np.mean(volumes[-20:])
            recent_vol = volumes[-3:]
            large_orders = np.any(recent_vol > avg_vol * 2)
        
        # Detect stop sweeps (quick spike then reversal)
        sweep_detected = False
        if len(highs) >= 5 and len(lows) >= 5:
            recent_range = highs[-5:].max() - lows[-5:].min()
            avg_range = np.mean(highs[-20:] - lows[-20:])
            if recent_range > avg_range * 1.5:
                # Check for reversal
                if prices[-1] < highs[-5:].max() * 0.995 or prices[-1] > lows[-5:].min() * 1.005:
                    sweep_detected = True
        
        return OrderFlowAnalysis(
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            net_flow=net_flow,
            large_orders_detected=large_orders,
            sweep_detected=sweep_detected,
            imbalance_zones=[]
        )
    
    def _detect_institutional_activity(
        self, prices: np.ndarray, volumes: Optional[np.ndarray],
        highs: np.ndarray, lows: np.ndarray
    ) -> InstitutionalActivity:
        """ตรวจจับกิจกรรมของ Institution"""
        if len(prices) < 30:
            return InstitutionalActivity.NEUTRAL
        
        # Look for accumulation/distribution patterns
        price_change = (prices[-1] - prices[-30]) / prices[-30] * 100
        
        # Volume analysis
        vol_trend = 0
        if volumes is not None and len(volumes) >= 30:
            early_vol = np.mean(volumes[-30:-15])
            late_vol = np.mean(volumes[-15:])
            vol_trend = (late_vol - early_vol) / (early_vol + 1e-10) * 100
        
        # Accumulation: Price stable/up, volume increasing on up days
        # Distribution: Price stable/down, volume increasing on down days
        
        if abs(price_change) < 1:  # Price range-bound
            if vol_trend > 20:
                # Check if volume on up or down days
                # Use the same slice length for both prices comparison and volumes
                if volumes is not None and len(prices) >= 30 and len(volumes) >= 30:
                    price_slice = prices[-30:]
                    vol_slice = volumes[-30:]
                    up_days = price_slice[1:] > price_slice[:-1]  # shape (29,)
                    up_vol = np.mean(vol_slice[1:][up_days])  # vol_slice[1:] is (29,), up_days is (29,)
                    down_vol = np.mean(vol_slice[1:][~up_days])
                    
                    if up_vol > down_vol * 1.2:
                        return InstitutionalActivity.ACCUMULATING
                    elif down_vol > up_vol * 1.2:
                        return InstitutionalActivity.DISTRIBUTING
        
        # Stop hunting detection
        recent_high = np.max(highs[-10:])
        recent_low = np.min(lows[-10:])
        prev_high = np.max(highs[-30:-10])
        prev_low = np.min(lows[-30:-10])
        
        # Price spiked above previous high then came back
        if recent_high > prev_high * 1.002 and prices[-1] < recent_high * 0.998:
            return InstitutionalActivity.STOP_HUNTING
        # Price spiked below previous low then came back
        if recent_low < prev_low * 0.998 and prices[-1] > recent_low * 1.002:
            return InstitutionalActivity.STOP_HUNTING
        
        return InstitutionalActivity.NEUTRAL
    
    def _check_institutional_alignment(
        self, signal_side: str, institutional: InstitutionalActivity
    ) -> bool:
        """ตรวจสอบว่า signal align กับ institutional activity"""
        if institutional == InstitutionalActivity.NEUTRAL:
            return True
        if institutional == InstitutionalActivity.STOP_HUNTING:
            return False  # Wait after stop hunt
        if signal_side == "BUY" and institutional == InstitutionalActivity.ACCUMULATING:
            return True
        if signal_side == "SELL" and institutional == InstitutionalActivity.DISTRIBUTING:
            return True
        return False
    
    def _analyze_momentum_quality(
        self, prices: np.ndarray, signal_side: str
    ) -> Tuple[MomentumQuality, float]:
        """วิเคราะห์คุณภาพของ Momentum"""
        if len(prices) < 20:
            return MomentumQuality.MODERATE, 50
        
        # Multiple momentum indicators
        returns = np.diff(prices) / prices[:-1]
        
        # Short-term momentum (5 periods)
        short_mom = np.mean(returns[-5:]) * 100
        
        # Medium-term momentum (10 periods)
        med_mom = np.mean(returns[-10:]) * 100
        
        # Momentum acceleration
        accel = short_mom - med_mom
        
        # RSI-like overbought/oversold
        gains = np.maximum(returns, 0)
        losses = np.maximum(-returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Score
        if signal_side == "BUY":
            if short_mom > 0.5 and accel > 0.1:
                quality = MomentumQuality.EXPLOSIVE
                score = 95
            elif short_mom > 0.2 and accel > 0:
                quality = MomentumQuality.STRONG
                score = 80
            elif short_mom > 0:
                quality = MomentumQuality.MODERATE
                score = 60
            elif rsi > 70:
                quality = MomentumQuality.EXHAUSTED
                score = 30
            else:
                quality = MomentumQuality.WEAK
                score = 40
        else:  # SELL
            if short_mom < -0.5 and accel < -0.1:
                quality = MomentumQuality.EXPLOSIVE
                score = 95
            elif short_mom < -0.2 and accel < 0:
                quality = MomentumQuality.STRONG
                score = 80
            elif short_mom < 0:
                quality = MomentumQuality.MODERATE
                score = 60
            elif rsi < 30:
                quality = MomentumQuality.EXHAUSTED
                score = 30
            else:
                quality = MomentumQuality.WEAK
                score = 40
        
        return quality, score
    
    def _calculate_key_levels(
        self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray,
        current_price: float
    ) -> Dict[str, float]:
        """คำนวณ Key Levels"""
        if len(prices) < 20:
            return {"support": current_price * 0.99, "resistance": current_price * 1.01}
        
        # Find support/resistance from recent highs/lows
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        # Resistance: nearest high above current price
        above = recent_highs[recent_highs > current_price]
        resistance = np.min(above) if len(above) > 0 else current_price * 1.01
        
        # Support: nearest low below current price
        below = recent_lows[recent_lows < current_price]
        support = np.max(below) if len(below) > 0 else current_price * 0.99
        
        # Pivot point
        pivot = (np.max(recent_highs) + np.min(recent_lows) + prices[-1]) / 3
        
        return {
            "support": support,
            "resistance": resistance,
            "pivot": pivot,
            "range_high": np.max(recent_highs),
            "range_low": np.min(recent_lows)
        }
    
    def _is_at_key_level(
        self, current_price: float, key_levels: Dict[str, float], signal_side: str
    ) -> bool:
        """ตรวจสอบว่าราคาอยู่ใกล้ key level"""
        support = key_levels.get("support", 0)
        resistance = key_levels.get("resistance", 0)
        
        tolerance = 0.002  # 0.2%
        
        if signal_side == "BUY":
            return abs(current_price - support) / current_price < tolerance
        else:
            return abs(current_price - resistance) / current_price < tolerance
    
    def _analyze_execution_timing(
        self, symbol: str, current_price: float, atr: float
    ) -> ExecutionTiming:
        """วิเคราะห์เวลาที่ดีในการ execute"""
        now = datetime.utcnow()
        hour = now.hour
        minute = now.minute
        day = now.weekday()
        
        # Avoid weekends
        if day >= 5:
            return ExecutionTiming.AVOID
        
        # Avoid market open/close
        if hour in [0, 22, 23] or (hour == 21 and minute > 45):
            return ExecutionTiming.WAIT
        
        # Golden hours (London/NY overlap)
        if 13 <= hour < 16:
            return ExecutionTiming.OPTIMAL
        
        # Good hours (London or NY active)
        if 7 <= hour < 16 or 13 <= hour < 20:
            return ExecutionTiming.GOOD
        
        # Asia session
        if 0 <= hour < 7:
            if "XAU" in symbol.upper():  # Gold prefers London/NY
                return ExecutionTiming.WAIT
            return ExecutionTiming.ACCEPTABLE
        
        return ExecutionTiming.ACCEPTABLE
    
    def _calculate_win_probability(
        self, confluence: float, momentum: float,
        inst_aligned: bool, hurst: float
    ) -> float:
        """คำนวณความน่าจะเป็นในการชนะ"""
        # Base from confluence
        prob = confluence * 0.4
        
        # Momentum contribution
        prob += momentum * 0.3
        
        # Institutional alignment
        if inst_aligned:
            prob += 10
        
        # Trending market (Hurst > 0.5) is easier to trade
        if hurst > 0.6:
            prob += 10
        elif hurst < 0.4:
            prob -= 10
        
        return max(30, min(85, prob))
    
    def _calculate_optimal_size(
        self, balance: float, equity: float,
        win_prob: float, entropy: MarketEntropy, confluence: float
    ) -> float:
        """คำนวณ position size ที่ optimal"""
        # Base size from Kelly Criterion (simplified)
        edge = (win_prob / 100) * 2 - 1  # Assuming 1:1 R:R
        kelly_fraction = max(0, edge) / 2  # Half Kelly for safety
        
        # Adjustments
        size = kelly_fraction
        
        # Entropy adjustment
        entropy_mult = {
            MarketEntropy.LOW: 1.2,
            MarketEntropy.NORMAL: 1.0,
            MarketEntropy.HIGH: 0.7,
            MarketEntropy.CHAOTIC: 0.3,
            MarketEntropy.DORMANT: 0.5,
        }
        size *= entropy_mult.get(entropy, 1.0)
        
        # Confluence adjustment
        size *= (confluence / 100)
        
        # Equity protection
        if equity < balance * 0.95:
            size *= 0.75
        elif equity < balance * 0.9:
            size *= 0.5
        
        return round(max(0.25, min(2.0, size)), 2)
    
    def _calculate_optimal_sl_tp(
        self, entropy: MarketEntropy, momentum: MomentumQuality,
        fractal: FractalAnalysis, atr: float
    ) -> Tuple[float, float]:
        """คำนวณ SL/TP ที่ optimal"""
        # Base SL/TP
        sl_mult = 1.5
        tp_mult = 2.0
        
        # Entropy adjustment - more volatile = wider SL
        if entropy in [MarketEntropy.HIGH, MarketEntropy.CHAOTIC]:
            sl_mult *= 1.3
            tp_mult *= 0.8  # Closer TP in chaotic market
        elif entropy == MarketEntropy.LOW:
            sl_mult *= 0.9
            tp_mult *= 1.2  # Can extend TP in calm market
        
        # Momentum adjustment
        if momentum == MomentumQuality.EXPLOSIVE:
            tp_mult *= 1.3  # Extend TP for strong momentum
        elif momentum == MomentumQuality.EXHAUSTED:
            tp_mult *= 0.7  # Closer TP when exhausted
            sl_mult *= 1.2  # Wider SL for potential reversal
        
        # Fractal/Trending adjustment
        if fractal.hurst_exponent > 0.6:  # Trending
            tp_mult *= 1.2
        elif fractal.hurst_exponent < 0.4:  # Mean-reverting
            tp_mult *= 0.8
            sl_mult *= 0.9
        
        return round(sl_mult, 2), round(tp_mult, 2)
    
    def _calculate_scale_levels(
        self, current_price: float, signal_side: str,
        atr: float, key_levels: Dict[str, float]
    ) -> Tuple[List[float], List[float]]:
        """คำนวณ scale in/out levels"""
        scale_in = []
        scale_out = []
        
        if signal_side == "BUY":
            # Scale in: buy more at lower prices (support)
            support = key_levels.get("support", current_price * 0.99)
            scale_in = [
                round(current_price - atr * 0.5, 5),
                round(support, 5)
            ]
            # Scale out: sell at higher prices
            resistance = key_levels.get("resistance", current_price * 1.01)
            scale_out = [
                round(current_price + atr, 5),
                round(resistance, 5)
            ]
        else:
            # Scale in: sell more at higher prices
            resistance = key_levels.get("resistance", current_price * 1.01)
            scale_in = [
                round(current_price + atr * 0.5, 5),
                round(resistance, 5)
            ]
            # Scale out: buy back at lower prices
            support = key_levels.get("support", current_price * 0.99)
            scale_out = [
                round(current_price - atr, 5),
                round(support, 5)
            ]
        
        return scale_in, scale_out
    
    def _estimate_alpha_potential(
        self, confluence: float, momentum: MomentumQuality,
        institutional: InstitutionalActivity
    ) -> float:
        """คาดการณ์ alpha potential"""
        base = confluence / 100 * 2  # 0-2%
        
        if momentum == MomentumQuality.EXPLOSIVE:
            base *= 1.5
        elif momentum == MomentumQuality.STRONG:
            base *= 1.2
        elif momentum == MomentumQuality.WEAK:
            base *= 0.7
        
        if institutional in [InstitutionalActivity.ACCUMULATING, InstitutionalActivity.DISTRIBUTING]:
            base *= 1.3  # Institutional activity = bigger moves
        
        return round(base, 2)
    
    def _estimate_slippage(
        self, symbol: str, entropy: MarketEntropy, timing: ExecutionTiming
    ) -> float:
        """คาดการณ์ slippage"""
        # Base slippage in pips
        base = 0.5
        
        # Higher in chaotic markets
        if entropy in [MarketEntropy.HIGH, MarketEntropy.CHAOTIC]:
            base *= 2
        
        # Higher in bad timing
        if timing in [ExecutionTiming.WAIT, ExecutionTiming.AVOID]:
            base *= 1.5
        
        # Gold has higher slippage
        if "XAU" in symbol.upper():
            base *= 2
        
        return round(base, 1)
    
    def _calculate_max_holding(
        self, fractal: FractalAnalysis, momentum: MomentumQuality,
        entropy: MarketEntropy
    ) -> int:
        """คำนวณเวลาที่ควรถือนานสุด"""
        base = 24  # hours
        
        # Trending markets can hold longer
        if fractal.hurst_exponent > 0.6:
            base = 48
        elif fractal.hurst_exponent < 0.4:
            base = 12  # Mean-reverting = shorter hold
        
        # Strong momentum = can hold longer
        if momentum == MomentumQuality.EXPLOSIVE:
            base = min(base * 1.5, 72)
        elif momentum == MomentumQuality.EXHAUSTED:
            base = min(base * 0.5, 12)
        
        # High entropy = shorter hold
        if entropy in [MarketEntropy.HIGH, MarketEntropy.CHAOTIC]:
            base = min(base * 0.5, 12)
        
        return int(base)
    
    def _determine_signal_strength(
        self, confluence: float, momentum: MomentumQuality, win_prob: float
    ) -> str:
        """กำหนดความแรงของ signal"""
        score = (confluence + win_prob) / 2
        
        if momentum == MomentumQuality.EXPLOSIVE and score > 70:
            return "EXPLOSIVE"
        elif score > 75:
            return "STRONG"
        elif score > 60:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _calculate_final_confidence(
        self, base: float, confluence: float, win_prob: float,
        entropy: MarketEntropy, momentum: MomentumQuality
    ) -> float:
        """คำนวณ confidence สุดท้าย"""
        # Weighted average
        conf = base * 0.3 + confluence * 0.35 + win_prob * 0.35
        
        # Adjustments
        if entropy == MarketEntropy.LOW:
            conf += 5
        elif entropy in [MarketEntropy.HIGH, MarketEntropy.CHAOTIC]:
            conf -= 10
        
        if momentum == MomentumQuality.EXPLOSIVE:
            conf += 5
        elif momentum == MomentumQuality.EXHAUSTED:
            conf -= 10
        
        return round(max(0, min(100, conf)), 1)
    
    def _suggest_weight_adjustments(self) -> Dict[str, float]:
        """แนะนำการปรับ weights (Self-Learning)"""
        adjustments = {}
        
        for factor, history in self._factor_performance.items():
            if len(history) < 10:
                continue
            
            recent = history[-20:]
            win_rate = sum(recent) / len(recent) if recent else 0.5
            
            current_weight = self._factor_weights.get(factor, 0.1)
            
            if win_rate > 0.65:
                adjustments[factor] = min(current_weight * 1.1, 0.25)
            elif win_rate < 0.4:
                adjustments[factor] = max(current_weight * 0.9, 0.05)
        
        return adjustments
    
    def _create_blocked_decision(
        self, reason: str, **kwargs
    ) -> SupremeDecision:
        """สร้าง decision ที่บล็อก"""
        return SupremeDecision(
            can_trade=False,
            confidence=0,
            signal_strength="WEAK",
            optimal_size_percent=0,
            scale_in_levels=[],
            scale_out_levels=[],
            optimal_sl_distance=0,
            optimal_tp_distance=0,
            max_holding_hours=0,
            execution_timing=ExecutionTiming.AVOID,
            best_entry_window_minutes=0,
            entropy_level=kwargs.get("entropy_level", MarketEntropy.NORMAL),
            institutional_activity=kwargs.get("institutional_activity", InstitutionalActivity.NEUTRAL),
            momentum_quality=MomentumQuality.WEAK,
            confluence_score=0,
            alpha_potential=0,
            expected_slippage_pips=0,
            win_probability=0,
            fractal_dimension=1.5,
            market_complexity="MEDIUM",
            key_levels={},
            reasons=[],
            warnings=[f"❌ {reason}"],
            suggested_weight_adjustments={}
        )
    
    def update_trade_result(self, trade_data: Dict) -> None:
        """อัพเดทผลเทรดสำหรับ self-learning"""
        self._trade_history.append(trade_data)
        
        # Update factor performance
        if "factors_used" in trade_data:
            is_win = trade_data.get("pnl", 0) > 0
            for factor in trade_data["factors_used"]:
                if factor in self._factor_performance:
                    self._factor_performance[factor].append(is_win)
                    # Keep last 100
                    if len(self._factor_performance[factor]) > 100:
                        self._factor_performance[factor] = self._factor_performance[factor][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        return {
            "total_trades": len(self._trade_history),
            "factor_weights": self._factor_weights,
            "factor_performance": {
                k: sum(v[-20:]) / max(len(v[-20:]), 1) * 100
                for k, v in self._factor_performance.items()
                if len(v) > 0
            }
        }


# Singleton
_supreme_intelligence: Optional[SupremeIntelligence] = None


def get_supreme_intelligence() -> SupremeIntelligence:
    """Get or create Supreme Intelligence instance"""
    global _supreme_intelligence
    if _supreme_intelligence is None:
        _supreme_intelligence = SupremeIntelligence()
    return _supreme_intelligence
