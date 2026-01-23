"""
🧠⚡ ULTRA INTELLIGENCE - ระบบ AI Trading ที่ฉลาดที่สุด

ฟีเจอร์ระดับ Institutional:
1. Dynamic R:R - ปรับ Risk:Reward ตามสภาพตลาด
2. Adaptive Position Sizing - ปรับ lot ตาม win/loss streak
3. Smart Session Filter - เทรดเฉพาะช่วงเวลาทอง
4. Volatility Scaling - ปรับทุกอย่างตาม volatility
5. Partial Profit Taking - ปิดกำไรบางส่วน ปล่อยที่เหลือวิ่ง
6. Market Structure - ดู Break of Structure, CHoCH
7. Liquidity Zones - หา Stop Hunt zones
8. News Protection - หยุดเทรดช่วงข่าว
9. Momentum Filter - เทรดตาม momentum เท่านั้น
10. Smart Re-entry - กลับเข้าเมื่อ pullback
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MarketPhase(str, Enum):
    """สถานะตลาด"""
    ACCUMULATION = "ACCUMULATION"     # สะสม (ก่อนขึ้น)
    MARKUP = "MARKUP"                 # ขาขึ้น
    DISTRIBUTION = "DISTRIBUTION"     # กระจาย (ก่อนลง)
    MARKDOWN = "MARKDOWN"             # ขาลง
    RANGING = "RANGING"               # sideways


class SessionQuality(str, Enum):
    """คุณภาพช่วงเวลา"""
    GOLDEN = "GOLDEN"       # ช่วงทอง (London/NY overlap)
    GOOD = "GOOD"           # ช่วงดี (London, NY)
    AVERAGE = "AVERAGE"     # ปานกลาง (Asia)
    POOR = "POOR"           # ไม่ดี (ช่วงเปลี่ยน session)
    AVOID = "AVOID"         # หลีกเลี่ยง (ก่อนข่าว, weekend)


class VolatilityState(str, Enum):
    """สถานะ Volatility"""
    EXTREME = "EXTREME"     # สูงมาก (>2x ATR) - ลด size
    HIGH = "HIGH"           # สูง (1.5-2x ATR)
    NORMAL = "NORMAL"       # ปกติ
    LOW = "LOW"             # ต่ำ (<0.5x ATR) - อาจรอ breakout
    DEAD = "DEAD"           # นิ่งมาก - ไม่เทรด


@dataclass
class LiquidityZone:
    """โซน Liquidity (Stop Hunt)"""
    price: float
    strength: float  # 0-100
    type: str  # "ABOVE" or "BELOW"
    touched: bool = False
    

@dataclass
class MarketStructure:
    """โครงสร้างตลาด"""
    trend: str  # "BULLISH", "BEARISH", "RANGING"
    last_high: float
    last_low: float
    bos_level: Optional[float] = None  # Break of Structure
    choch_detected: bool = False  # Change of Character
    swing_points: List[Tuple[float, str]] = field(default_factory=list)


@dataclass
class UltraDecision:
    """การตัดสินใจจาก Ultra Intelligence"""
    can_trade: bool
    confidence: float  # 0-100
    
    # Position Management
    position_size_multiplier: float  # 0.25 - 2.0
    optimal_rr: float  # Risk:Reward ที่เหมาะสม
    
    # Entry
    entry_type: str  # "MARKET", "LIMIT", "STOP"
    entry_price: Optional[float] = None
    
    # Exit Strategy
    use_partial_tp: bool = False
    tp1_percent: float = 50  # ปิด 50% ที่ TP1
    tp1_rr: float = 1.0      # TP1 ที่ R:R = 1
    tp2_rr: float = 2.0      # TP2 ที่ R:R = 2
    
    # Filters
    session_quality: SessionQuality = SessionQuality.AVERAGE
    volatility_state: VolatilityState = VolatilityState.NORMAL
    market_phase: MarketPhase = MarketPhase.RANGING
    
    # Insights
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Advanced
    liquidity_above: Optional[float] = None
    liquidity_below: Optional[float] = None
    market_structure: Optional[MarketStructure] = None


class UltraIntelligence:
    """
    🧠⚡ ULTRA INTELLIGENCE
    ระบบ AI ที่ฉลาดที่สุดสำหรับ Trading
    
    ใช้หลักการ:
    - Smart Money Concepts (SMC)
    - Institutional Order Flow
    - Multi-dimensional Analysis
    - Adaptive Risk Management
    """
    
    def __init__(self):
        # Performance Tracking
        self._win_streak = 0
        self._loss_streak = 0
        self._recent_trades: List[Dict] = []
        self._daily_pnl = 0.0
        
        # Session Times (UTC)
        self._sessions = {
            "ASIA": (0, 8),       # 00:00 - 08:00 UTC
            "LONDON": (7, 16),    # 07:00 - 16:00 UTC
            "NY": (13, 22),       # 13:00 - 22:00 UTC
            "OVERLAP": (13, 16),  # London/NY overlap - GOLDEN
        }
        
        # High Impact News Times (check externally)
        self._news_times: List[datetime] = []
        
        # Symbol-specific settings
        self._symbol_profiles = {
            "XAUUSD": {"atr_mult": 1.5, "session_pref": ["LONDON", "NY"]},
            "XAUUSDm": {"atr_mult": 1.5, "session_pref": ["LONDON", "NY"]},
            "EURUSD": {"atr_mult": 1.2, "session_pref": ["LONDON", "OVERLAP"]},
            "EURUSDm": {"atr_mult": 1.2, "session_pref": ["LONDON", "OVERLAP"]},
            "GBPUSD": {"atr_mult": 1.3, "session_pref": ["LONDON", "OVERLAP"]},
            "GBPUSDm": {"atr_mult": 1.3, "session_pref": ["LONDON", "OVERLAP"]},
        }
        
        logger.info("🧠⚡ Ultra Intelligence initialized")
    
    def analyze(
        self,
        symbol: str,
        signal_side: str,  # "BUY" or "SELL"
        current_price: float,
        prices: np.ndarray,  # Historical close prices
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        atr: float = 0,
        base_confidence: float = 70,
        current_balance: float = 10000,
        account_equity: float = 10000,
    ) -> UltraDecision:
        """
        🧠 วิเคราะห์แบบ Ultra Intelligence
        
        Returns:
            UltraDecision พร้อมคำแนะนำครบถ้วน
        """
        reasons = []
        warnings = []
        
        # 1. Session Analysis
        session_quality = self._analyze_session(symbol)
        reasons.append(f"📅 Session: {session_quality.value}")
        
        if session_quality == SessionQuality.AVOID:
            return UltraDecision(
                can_trade=False,
                confidence=0,
                position_size_multiplier=0,
                optimal_rr=0,
                entry_type="NONE",
                session_quality=session_quality,
                reasons=reasons,
                warnings=["❌ Bad session - avoid trading"]
            )
        
        # 2. Volatility Analysis
        volatility_state, vol_multiplier = self._analyze_volatility(prices, atr)
        reasons.append(f"📊 Volatility: {volatility_state.value}")
        
        if volatility_state == VolatilityState.DEAD:
            warnings.append("⚠️ Market is dead - low probability")
        elif volatility_state == VolatilityState.EXTREME:
            warnings.append("⚠️ Extreme volatility - reduced size")
        
        # 3. Market Structure Analysis
        market_structure = self._analyze_market_structure(prices, highs, lows)
        market_phase = self._determine_market_phase(prices, volumes)
        reasons.append(f"🏗️ Structure: {market_structure.trend}")
        reasons.append(f"📈 Phase: {market_phase.value}")
        
        # 4. Check Signal Alignment with Structure
        structure_aligned = self._check_structure_alignment(
            signal_side, market_structure, market_phase
        )
        if not structure_aligned:
            warnings.append("⚠️ Signal against market structure")
        
        # 5. Find Liquidity Zones
        liq_above, liq_below = self._find_liquidity_zones(
            current_price, highs, lows
        )
        
        # 6. Momentum Check
        momentum_ok, momentum_msg = self._check_momentum(prices, signal_side)
        if momentum_ok:
            reasons.append(f"💪 {momentum_msg}")
        else:
            warnings.append(f"⚠️ {momentum_msg}")
        
        # 7. Calculate Position Size Multiplier
        size_mult = self._calculate_size_multiplier(
            vol_multiplier,
            session_quality,
            structure_aligned,
            momentum_ok,
            current_balance,
            account_equity
        )
        
        # 8. Calculate Optimal R:R
        optimal_rr = self._calculate_optimal_rr(
            volatility_state,
            session_quality,
            market_phase,
            structure_aligned
        )
        
        # 9. Determine Entry Type
        entry_type, entry_price = self._determine_entry(
            signal_side, current_price, atr, market_structure
        )
        reasons.append(f"🎯 Entry: {entry_type}")
        
        # 10. Partial TP Strategy
        use_partial = optimal_rr >= 1.5 and structure_aligned
        
        # 11. Calculate Final Confidence
        confidence = self._calculate_confidence(
            base_confidence,
            session_quality,
            volatility_state,
            structure_aligned,
            momentum_ok,
            market_phase
        )
        
        # 12. Final Decision
        can_trade = (
            confidence >= 65 and
            session_quality not in [SessionQuality.AVOID, SessionQuality.POOR] and
            volatility_state != VolatilityState.DEAD and
            size_mult >= 0.25
        )
        
        if not can_trade:
            warnings.append("❌ Conditions not met for trading")
        
        return UltraDecision(
            can_trade=can_trade,
            confidence=confidence,
            position_size_multiplier=size_mult,
            optimal_rr=optimal_rr,
            entry_type=entry_type,
            entry_price=entry_price,
            use_partial_tp=use_partial,
            tp1_percent=50 if use_partial else 100,
            tp1_rr=1.0,
            tp2_rr=optimal_rr,
            session_quality=session_quality,
            volatility_state=volatility_state,
            market_phase=market_phase,
            reasons=reasons,
            warnings=warnings,
            liquidity_above=liq_above,
            liquidity_below=liq_below,
            market_structure=market_structure
        )
    
    def _analyze_session(self, symbol: str) -> SessionQuality:
        """วิเคราะห์คุณภาพช่วงเวลา"""
        now = datetime.utcnow()
        hour = now.hour
        day = now.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend
        if day >= 5:
            return SessionQuality.AVOID
        
        # Friday after 20:00 UTC
        if day == 4 and hour >= 20:
            return SessionQuality.AVOID
        
        # Check sessions
        overlap_start, overlap_end = self._sessions["OVERLAP"]
        london_start, london_end = self._sessions["LONDON"]
        ny_start, ny_end = self._sessions["NY"]
        asia_start, asia_end = self._sessions["ASIA"]
        
        # Golden hour (London/NY overlap)
        if overlap_start <= hour < overlap_end:
            return SessionQuality.GOLDEN
        
        # London session
        if london_start <= hour < london_end:
            return SessionQuality.GOOD
        
        # NY session
        if ny_start <= hour < ny_end:
            return SessionQuality.GOOD
        
        # Asia session
        if asia_start <= hour < asia_end:
            # Gold prefers London/NY
            if "XAU" in symbol.upper():
                return SessionQuality.AVERAGE
            return SessionQuality.AVERAGE
        
        # Between sessions
        return SessionQuality.POOR
    
    def _analyze_volatility(
        self, prices: np.ndarray, atr: float
    ) -> Tuple[VolatilityState, float]:
        """วิเคราะห์ volatility"""
        if len(prices) < 20:
            return VolatilityState.NORMAL, 1.0
        
        # Calculate recent volatility
        returns = np.diff(prices) / prices[:-1]
        recent_vol = np.std(returns[-20:]) * np.sqrt(252)  # Annualized
        avg_vol = np.std(returns) * np.sqrt(252)
        
        if avg_vol == 0:
            return VolatilityState.DEAD, 0.0
        
        vol_ratio = recent_vol / avg_vol
        
        if vol_ratio > 2.0:
            return VolatilityState.EXTREME, 0.5
        elif vol_ratio > 1.5:
            return VolatilityState.HIGH, 0.75
        elif vol_ratio < 0.3:
            return VolatilityState.DEAD, 0.0
        elif vol_ratio < 0.5:
            return VolatilityState.LOW, 0.8
        else:
            return VolatilityState.NORMAL, 1.0
    
    def _analyze_market_structure(
        self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ) -> MarketStructure:
        """วิเคราะห์โครงสร้างตลาด (Smart Money Concepts)"""
        if len(prices) < 20:
            return MarketStructure(
                trend="RANGING",
                last_high=highs[-1] if len(highs) > 0 else 0,
                last_low=lows[-1] if len(lows) > 0 else 0
            )
        
        # Find swing points
        swing_points = []
        for i in range(2, len(prices) - 2):
            # Swing High
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_points.append((highs[i], "HIGH"))
            # Swing Low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_points.append((lows[i], "LOW"))
        
        if len(swing_points) < 4:
            return MarketStructure(
                trend="RANGING",
                last_high=max(highs[-20:]),
                last_low=min(lows[-20:]),
                swing_points=swing_points
            )
        
        # Analyze trend from swing points
        recent_highs = [p for p, t in swing_points[-6:] if t == "HIGH"]
        recent_lows = [p for p, t in swing_points[-6:] if t == "LOW"]
        
        trend = "RANGING"
        bos_level = None
        choch = False
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Higher Highs and Higher Lows = Bullish
            hh = recent_highs[-1] > recent_highs[-2] if len(recent_highs) >= 2 else False
            hl = recent_lows[-1] > recent_lows[-2] if len(recent_lows) >= 2 else False
            
            # Lower Highs and Lower Lows = Bearish
            lh = recent_highs[-1] < recent_highs[-2] if len(recent_highs) >= 2 else False
            ll = recent_lows[-1] < recent_lows[-2] if len(recent_lows) >= 2 else False
            
            if hh and hl:
                trend = "BULLISH"
                bos_level = recent_highs[-2]  # Break of Structure level
            elif lh and ll:
                trend = "BEARISH"
                bos_level = recent_lows[-2]
            
            # Check for Change of Character (CHoCH)
            if trend == "BULLISH" and ll:
                choch = True
            elif trend == "BEARISH" and hh:
                choch = True
        
        return MarketStructure(
            trend=trend,
            last_high=max(highs[-20:]),
            last_low=min(lows[-20:]),
            bos_level=bos_level,
            choch_detected=choch,
            swing_points=swing_points[-10:]
        )
    
    def _determine_market_phase(
        self, prices: np.ndarray, volumes: Optional[np.ndarray]
    ) -> MarketPhase:
        """กำหนด Market Phase"""
        if len(prices) < 50:
            return MarketPhase.RANGING
        
        # Calculate trend
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        current = prices[-1]
        
        # Price momentum
        momentum = (prices[-1] - prices[-20]) / prices[-20] * 100
        
        # Volume analysis (if available)
        vol_increasing = True
        if volumes is not None and len(volumes) >= 20:
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes[-20:])
            vol_increasing = recent_vol > avg_vol
        
        # Determine phase
        if current > sma_20 > sma_50:
            if momentum > 2 and vol_increasing:
                return MarketPhase.MARKUP
            elif momentum < 0.5:
                return MarketPhase.DISTRIBUTION
            return MarketPhase.MARKUP
        
        elif current < sma_20 < sma_50:
            if momentum < -2 and vol_increasing:
                return MarketPhase.MARKDOWN
            elif momentum > -0.5:
                return MarketPhase.ACCUMULATION
            return MarketPhase.MARKDOWN
        
        else:
            # Check for accumulation/distribution
            if abs(momentum) < 1:
                if sma_20 < sma_50:
                    return MarketPhase.ACCUMULATION
                else:
                    return MarketPhase.DISTRIBUTION
            return MarketPhase.RANGING
    
    def _check_structure_alignment(
        self,
        signal_side: str,
        structure: MarketStructure,
        phase: MarketPhase
    ) -> bool:
        """ตรวจสอบว่า signal align กับ structure"""
        if signal_side == "BUY":
            # BUY should align with bullish structure
            if structure.trend == "BULLISH":
                return True
            if phase in [MarketPhase.ACCUMULATION, MarketPhase.MARKUP]:
                return True
            if structure.choch_detected and structure.trend == "BEARISH":
                return True  # Potential reversal
            return False
        else:
            # SELL should align with bearish structure
            if structure.trend == "BEARISH":
                return True
            if phase in [MarketPhase.DISTRIBUTION, MarketPhase.MARKDOWN]:
                return True
            if structure.choch_detected and structure.trend == "BULLISH":
                return True  # Potential reversal
            return False
    
    def _find_liquidity_zones(
        self, current_price: float, highs: np.ndarray, lows: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """หา Liquidity Zones (Stop Hunt areas)"""
        if len(highs) < 20:
            return None, None
        
        # Equal highs (liquidity above)
        recent_highs = highs[-20:]
        high_clusters = []
        for i, h in enumerate(recent_highs):
            similar = [h2 for h2 in recent_highs if abs(h2 - h) / h < 0.001]
            if len(similar) >= 2:
                high_clusters.append(np.mean(similar))
        
        liq_above = None
        if high_clusters:
            # Find nearest liquidity above current price
            above = [h for h in high_clusters if h > current_price]
            if above:
                liq_above = min(above)
        
        # Equal lows (liquidity below)
        recent_lows = lows[-20:]
        low_clusters = []
        for i, l in enumerate(recent_lows):
            similar = [l2 for l2 in recent_lows if abs(l2 - l) / l < 0.001]
            if len(similar) >= 2:
                low_clusters.append(np.mean(similar))
        
        liq_below = None
        if low_clusters:
            # Find nearest liquidity below current price
            below = [l for l in low_clusters if l < current_price]
            if below:
                liq_below = max(below)
        
        return liq_above, liq_below
    
    def _check_momentum(
        self, prices: np.ndarray, signal_side: str
    ) -> Tuple[bool, str]:
        """ตรวจสอบ momentum"""
        if len(prices) < 21:
            return True, "Insufficient data for momentum"
        
        # Calculate momentum indicators
        # np.diff(prices[-20:]) returns (19,) elements, so we need prices[-20:-1] which is also (19,)
        returns = np.diff(prices[-20:]) / prices[-20:-1]
        recent_returns = returns[-5:] if len(returns) >= 5 else returns
        
        avg_momentum = np.mean(recent_returns) * 100
        
        if signal_side == "BUY":
            if avg_momentum > 0.1:
                return True, f"Strong bullish momentum (+{avg_momentum:.2f}%)"
            elif avg_momentum > -0.05:
                return True, f"Neutral momentum ({avg_momentum:.2f}%)"
            else:
                return False, f"Bearish momentum ({avg_momentum:.2f}%)"
        else:
            if avg_momentum < -0.1:
                return True, f"Strong bearish momentum ({avg_momentum:.2f}%)"
            elif avg_momentum < 0.05:
                return True, f"Neutral momentum ({avg_momentum:.2f}%)"
            else:
                return False, f"Bullish momentum ({avg_momentum:.2f}%)"
    
    def _calculate_size_multiplier(
        self,
        vol_mult: float,
        session: SessionQuality,
        structure_aligned: bool,
        momentum_ok: bool,
        balance: float,
        equity: float
    ) -> float:
        """คำนวณ position size multiplier"""
        base = 1.0
        
        # Volatility adjustment
        base *= vol_mult
        
        # Session adjustment
        session_mult = {
            SessionQuality.GOLDEN: 1.2,
            SessionQuality.GOOD: 1.0,
            SessionQuality.AVERAGE: 0.75,
            SessionQuality.POOR: 0.5,
            SessionQuality.AVOID: 0.0
        }
        base *= session_mult.get(session, 1.0)
        
        # Structure alignment
        if not structure_aligned:
            base *= 0.5
        
        # Momentum
        if not momentum_ok:
            base *= 0.7
        
        # Win/Loss streak adjustment
        if self._win_streak >= 3:
            base *= 1.25  # Increase on win streak
        elif self._loss_streak >= 2:
            base *= 0.5   # Decrease on loss streak
        
        # Equity protection
        if equity < balance * 0.95:
            base *= 0.75  # Reduce if in drawdown
        elif equity < balance * 0.9:
            base *= 0.5
        
        return round(max(0.25, min(2.0, base)), 2)
    
    def _calculate_optimal_rr(
        self,
        volatility: VolatilityState,
        session: SessionQuality,
        phase: MarketPhase,
        structure_aligned: bool
    ) -> float:
        """คำนวณ Risk:Reward ที่เหมาะสม"""
        base_rr = 1.5
        
        # Session adjustment
        if session == SessionQuality.GOLDEN:
            base_rr = 2.0  # Can aim higher in golden session
        elif session == SessionQuality.POOR:
            base_rr = 1.0  # Conservative
        
        # Phase adjustment
        if phase in [MarketPhase.MARKUP, MarketPhase.MARKDOWN]:
            base_rr += 0.5  # Trending = extend targets
        elif phase == MarketPhase.RANGING:
            base_rr = min(base_rr, 1.5)  # Ranging = closer targets
        
        # Volatility adjustment
        if volatility == VolatilityState.HIGH:
            base_rr = min(base_rr, 1.5)  # Don't overextend in high vol
        elif volatility == VolatilityState.LOW:
            base_rr = min(base_rr, 1.2)  # Low vol = smaller moves
        
        # Structure
        if structure_aligned:
            base_rr += 0.25
        
        return round(max(1.0, min(3.0, base_rr)), 1)
    
    def _determine_entry(
        self,
        signal_side: str,
        current_price: float,
        atr: float,
        structure: MarketStructure
    ) -> Tuple[str, Optional[float]]:
        """กำหนด entry type และ price"""
        # Default: Market order
        entry_type = "MARKET"
        entry_price = None
        
        # Check if we should wait for pullback
        if structure.bos_level:
            distance_to_bos = abs(current_price - structure.bos_level) / current_price
            
            if distance_to_bos < 0.002:  # Very close to BOS
                entry_type = "MARKET"  # Enter now
            elif distance_to_bos < 0.005:  # Moderate distance
                # Suggest limit order for better entry
                if signal_side == "BUY":
                    entry_price = current_price - (atr * 0.3)  # Small pullback
                    entry_type = "LIMIT"
                else:
                    entry_price = current_price + (atr * 0.3)
                    entry_type = "LIMIT"
        
        return entry_type, entry_price
    
    def _calculate_confidence(
        self,
        base_confidence: float,
        session: SessionQuality,
        volatility: VolatilityState,
        structure_aligned: bool,
        momentum_ok: bool,
        phase: MarketPhase
    ) -> float:
        """คำนวณ confidence รวม"""
        conf = base_confidence
        
        # Session boost/penalty
        session_adj = {
            SessionQuality.GOLDEN: 10,
            SessionQuality.GOOD: 5,
            SessionQuality.AVERAGE: 0,
            SessionQuality.POOR: -10,
            SessionQuality.AVOID: -30
        }
        conf += session_adj.get(session, 0)
        
        # Volatility
        if volatility == VolatilityState.NORMAL:
            conf += 5
        elif volatility in [VolatilityState.EXTREME, VolatilityState.DEAD]:
            conf -= 15
        
        # Structure
        if structure_aligned:
            conf += 10
        else:
            conf -= 15
        
        # Momentum
        if momentum_ok:
            conf += 5
        else:
            conf -= 10
        
        # Phase bonus
        if phase in [MarketPhase.MARKUP, MarketPhase.MARKDOWN]:
            conf += 5  # Trending bonus
        
        return round(max(0, min(100, conf)), 1)
    
    def update_performance(self, pnl: float, is_win: bool):
        """อัพเดท performance tracking"""
        if is_win:
            self._win_streak += 1
            self._loss_streak = 0
        else:
            self._loss_streak += 1
            self._win_streak = 0
        
        self._daily_pnl += pnl
        self._recent_trades.append({
            "pnl": pnl,
            "is_win": is_win,
            "timestamp": datetime.now()
        })
        
        # Keep last 100 trades
        if len(self._recent_trades) > 100:
            self._recent_trades = self._recent_trades[-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        if not self._recent_trades:
            return {
                "win_streak": 0,
                "loss_streak": 0,
                "daily_pnl": 0,
                "recent_win_rate": 0,
                "total_trades": 0
            }
        
        wins = sum(1 for t in self._recent_trades if t["is_win"])
        total = len(self._recent_trades)
        
        return {
            "win_streak": self._win_streak,
            "loss_streak": self._loss_streak,
            "daily_pnl": self._daily_pnl,
            "recent_win_rate": (wins / total * 100) if total > 0 else 0,
            "total_trades": total
        }
    
    def reset_daily(self):
        """Reset daily stats"""
        self._daily_pnl = 0.0


# Singleton instance
_ultra_intelligence: Optional[UltraIntelligence] = None


def get_ultra_intelligence() -> UltraIntelligence:
    """Get or create Ultra Intelligence instance"""
    global _ultra_intelligence
    if _ultra_intelligence is None:
        _ultra_intelligence = UltraIntelligence()
    return _ultra_intelligence
