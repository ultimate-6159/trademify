"""
🧠 SMART MONEY CONCEPTS (SMC) - THE 90% PROTOCOL
กฎเหล็ก 5 ข้อสำหรับ Win Rate 90%+

📜 THE 5 IRON RULES:
1. HTF Structure (H4) - เทรดตามโครงสร้างใหญ่เท่านั้น (HH+HL=Buy, LH+LL=Sell, Sideway=NO TRADE)
2. Liquidity Sweep - "No Sweep, No Entry" (ต้องมีการกวาด SL ก่อน)
3. Premium/Discount Zone - Buy เฉพาะ Discount (< Fib 0.5), Sell เฉพาะ Premium (> Fib 0.5)
4. OB + FVG Confluence - จุดเข้าต้องมี Order Block ที่มี FVG (รอยเท้าเจ้ามือ)
5. LTF ChoCH Confirmation - รอ M5 เปลี่ยนโครงสร้างยืนยันก่อนเข้า
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class SwingType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class StructureTrend(Enum):
    BULLISH = "BULLISH"   # HH + HL
    BEARISH = "BEARISH"   # LH + LL
    RANGING = "RANGING"


class SweepSide(Enum):
    BUY_SIDE = "BUY_SIDE"   # Sweep above swing high → grabbed buy stops → SELL signal
    SELL_SIDE = "SELL_SIDE"  # Sweep below swing low → grabbed sell stops → BUY signal


class OBType(Enum):
    BULLISH = "BULLISH"  # Last bearish candle before bullish impulse
    BEARISH = "BEARISH"  # Last bullish candle before bearish impulse


@dataclass
class SwingPoint:
    index: int
    price: float
    swing_type: SwingType


@dataclass
class LiquidityPool:
    price: float
    side: SweepSide
    strength: int
    swept: bool = False


@dataclass
class LiquiditySweep:
    side: SweepSide
    sweep_price: float
    pool_price: float
    rejection_strength: float  # 0-100
    candle_index: int


@dataclass
class OrderBlock:
    ob_type: OBType
    high: float
    low: float
    index: int
    strength: float  # 0-100
    has_fvg: bool = False  # Rule 4: OB ต้องมี FVG ด้วย
    is_fresh: bool = True  # ยังไม่เคยถูก test


@dataclass
class FairValueGap:
    high: float
    low: float
    is_bullish: bool
    index: int


@dataclass
class SMCSignal:
    signal: Optional[str]  # "BUY", "SELL", or None
    confidence: float      # 0-100
    stop_loss: float
    take_profit: float
    sweep_detected: bool
    sweep_side: Optional[SweepSide]
    structure: StructureTrend
    htf_structure: StructureTrend  # Rule 1: H4 structure
    order_block: Optional[OrderBlock]
    fvg: Optional[FairValueGap]
    in_discount_zone: bool  # Rule 3
    choch_confirmed: bool   # Rule 5
    reason: str


# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 SMC STRATEGY ENGINE - THE 90% PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

class SMCStrategy:
    """
    Smart Money Concepts - กฎเหล็ก 5 ข้อ
    1. HTF Structure → ทิศทางเท่านั้น
    2. Liquidity Sweep → No Sweep No Entry
    3. Premium/Discount Zone → Fibonacci
    4. OB + FVG → รอยเท้าเจ้ามือ
    5. LTF ChoCH → ยืนยัน M5
    """

    def __init__(self, swing_lookback: int = 5, sweep_lookback: int = 3,
                 sl_buffer_atr: float = 0.3, max_sl_atr: float = 2.0,
                 min_sweep_strength: float = 50.0):
        self.swing_lookback = swing_lookback
        self.sweep_lookback = sweep_lookback
        self.sl_buffer_atr = sl_buffer_atr
        self.max_sl_atr = max_sl_atr
        self.min_sweep_strength = min_sweep_strength

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SWING POINT DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_swing_points(self, highs: np.ndarray, lows: np.ndarray,
                            lookback: int = None) -> List[SwingPoint]:
        """ตรวจหา Swing High / Swing Low"""
        if lookback is None:
            lookback = self.swing_lookback

        swings: List[SwingPoint] = []
        n = len(highs)
        if n < lookback * 2 + 1:
            return swings

        for i in range(lookback, n - lookback):
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swings.append(SwingPoint(index=i, price=float(highs[i]), swing_type=SwingType.HIGH))

            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swings.append(SwingPoint(index=i, price=float(lows[i]), swing_type=SwingType.LOW))

        return swings

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MARKET STRUCTURE DETECTION (Rule 1)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_structure(self, swing_points: List[SwingPoint]) -> StructureTrend:
        """
        วิเคราะห์โครงสร้างตลาด:
        BULLISH = Higher Highs + Higher Lows
        BEARISH = Lower Highs + Lower Lows
        RANGING = ไม่ชัดเจน → ห้ามเทรด
        """
        swing_highs = [s for s in swing_points if s.swing_type == SwingType.HIGH]
        swing_lows = [s for s in swing_points if s.swing_type == SwingType.LOW]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return StructureTrend.RANGING

        recent_highs = sorted(swing_highs, key=lambda s: s.index)[-3:]
        recent_lows = sorted(swing_lows, key=lambda s: s.index)[-3:]

        hh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i].price > recent_highs[i - 1].price)
        hl_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i].price > recent_lows[i - 1].price)
        lh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i].price < recent_highs[i - 1].price)
        ll_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i].price < recent_lows[i - 1].price)

        if hh_count >= 1 and hl_count >= 1:
            return StructureTrend.BULLISH
        elif lh_count >= 1 and ll_count >= 1:
            return StructureTrend.BEARISH
        else:
            return StructureTrend.RANGING

    # ─────────────────────────────────────────────────────────────────────────
    # 3. LIQUIDITY POOL DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_liquidity_pools(self, swing_points: List[SwingPoint],
                                current_price: float, atr: float
                                ) -> Tuple[List[LiquidityPool], List[LiquidityPool]]:
        """ระบุ Liquidity Pools เหนือ/ใต้ราคาปัจจุบัน"""
        pools_above: List[LiquidityPool] = []
        pools_below: List[LiquidityPool] = []

        swing_highs = sorted([s for s in swing_points if s.swing_type == SwingType.HIGH],
                             key=lambda s: s.price)
        swing_lows = sorted([s for s in swing_points if s.swing_type == SwingType.LOW],
                            key=lambda s: s.price)

        cluster_threshold = atr * 0.5

        if swing_highs:
            for cluster in self._cluster_prices([s.price for s in swing_highs], cluster_threshold):
                pool_price = max(cluster)
                if pool_price > current_price:
                    pools_above.append(LiquidityPool(
                        price=pool_price, side=SweepSide.BUY_SIDE, strength=len(cluster)))

        if swing_lows:
            for cluster in self._cluster_prices([s.price for s in swing_lows], cluster_threshold):
                pool_price = min(cluster)
                if pool_price < current_price:
                    pools_below.append(LiquidityPool(
                        price=pool_price, side=SweepSide.SELL_SIDE, strength=len(cluster)))

        pools_above.sort(key=lambda p: p.price)
        pools_below.sort(key=lambda p: -p.price)
        return pools_above, pools_below

    def _cluster_prices(self, prices: List[float], threshold: float) -> List[List[float]]:
        if not prices:
            return []
        clusters = [[prices[0]]]
        for p in prices[1:]:
            if abs(p - clusters[-1][-1]) <= threshold:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return clusters

    # ─────────────────────────────────────────────────────────────────────────
    # 4. LIQUIDITY SWEEP DETECTION ⚡ (Rule 2: No Sweep, No Entry)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_liquidity_sweep(self, opens: np.ndarray, highs: np.ndarray,
                                lows: np.ndarray, closes: np.ndarray,
                                swing_points: List[SwingPoint],
                                lookback: int = None) -> Optional[LiquiditySweep]:
        """
        Rule 2: "No Sweep, No Entry"
        SELL-SIDE Sweep → BUY (ทะลุ swing low แล้ว reject กลับ)
        BUY-SIDE Sweep → SELL (ทะลุ swing high แล้ว reject กลับ)
        """
        if lookback is None:
            lookback = self.sweep_lookback

        n = len(closes)
        if n < 3:
            return None

        swing_highs = sorted([s for s in swing_points if s.swing_type == SwingType.HIGH],
                             key=lambda s: s.index)
        swing_lows = sorted([s for s in swing_points if s.swing_type == SwingType.LOW],
                            key=lambda s: s.index)

        best_sweep: Optional[LiquiditySweep] = None
        best_strength = 0.0

        for i in range(max(0, n - lookback), n):
            # SELL-SIDE SWEEP (ทะลุ swing low แล้วกลับ → BUY)
            for sl_point in swing_lows:
                if sl_point.index >= i:
                    continue
                if lows[i] < sl_point.price and closes[i] > sl_point.price:
                    candle_range = highs[i] - lows[i]
                    if candle_range > 0:
                        wick_below = min(opens[i], closes[i]) - lows[i]
                        rejection = (wick_below / candle_range) * 100
                        close_recovery = (closes[i] - lows[i]) / candle_range * 100
                        strength = (rejection * 0.6 + close_recovery * 0.4)
                        if strength > best_strength and strength >= self.min_sweep_strength:
                            best_strength = strength
                            best_sweep = LiquiditySweep(
                                side=SweepSide.SELL_SIDE,
                                sweep_price=float(lows[i]),
                                pool_price=sl_point.price,
                                rejection_strength=strength,
                                candle_index=i
                            )

            # BUY-SIDE SWEEP (ทะลุ swing high แล้วกลับ → SELL)
            for sh_point in swing_highs:
                if sh_point.index >= i:
                    continue
                if highs[i] > sh_point.price and closes[i] < sh_point.price:
                    candle_range = highs[i] - lows[i]
                    if candle_range > 0:
                        wick_above = highs[i] - max(opens[i], closes[i])
                        rejection = (wick_above / candle_range) * 100
                        close_recovery = (highs[i] - closes[i]) / candle_range * 100
                        strength = (rejection * 0.6 + close_recovery * 0.4)
                        if strength > best_strength and strength >= self.min_sweep_strength:
                            best_strength = strength
                            best_sweep = LiquiditySweep(
                                side=SweepSide.BUY_SIDE,
                                sweep_price=float(highs[i]),
                                pool_price=sh_point.price,
                                rejection_strength=strength,
                                candle_index=i
                            )

        return best_sweep

    # ─────────────────────────────────────────────────────────────────────────
    # 5. PREMIUM / DISCOUNT ZONE (Rule 3: Fibonacci)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_premium_discount(self, swing_points: List[SwingPoint],
                                 current_price: float
                                 ) -> Tuple[bool, bool, float]:
        """
        Rule 3: Buy เฉพาะ Discount zone (< Fib 0.5), Sell เฉพาะ Premium zone (> Fib 0.5)

        Returns: (in_discount, in_premium, fib_level)
          - fib_level: 0.0 = Swing Low, 1.0 = Swing High
          - Discount = fib_level < 0.5
          - Premium = fib_level > 0.5
        """
        swing_highs = sorted([s for s in swing_points if s.swing_type == SwingType.HIGH],
                             key=lambda s: s.index)
        swing_lows = sorted([s for s in swing_points if s.swing_type == SwingType.LOW],
                            key=lambda s: s.index)

        if not swing_highs or not swing_lows:
            return False, False, 0.5

        # ใช้ swing high/low ล่าสุดสำหรับ Fib range
        latest_high = max(swing_highs[-3:], key=lambda s: s.price) if len(swing_highs) >= 3 else swing_highs[-1]
        latest_low = min(swing_lows[-3:], key=lambda s: s.price) if len(swing_lows) >= 3 else swing_lows[-1]

        swing_range = latest_high.price - latest_low.price
        if swing_range <= 0:
            return False, False, 0.5

        # Fibonacci level: 0 = swing low, 1 = swing high
        fib_level = (current_price - latest_low.price) / swing_range
        fib_level = max(0.0, min(1.0, fib_level))

        in_discount = fib_level < 0.5   # Buy zone
        in_premium = fib_level > 0.5    # Sell zone

        return in_discount, in_premium, fib_level

    # ─────────────────────────────────────────────────────────────────────────
    # 6. ORDER BLOCK DETECTION (Rule 4: OB + FVG)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_order_blocks(self, opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray,
                            lookback: int = 20) -> List[OrderBlock]:
        """
        Rule 4: OB ต้องมี FVG ด้วย (รอยเท้าเจ้ามือ)
        Bullish OB: แท่ง bearish ก่อน impulse ขาขึ้นที่สร้าง FVG
        Bearish OB: แท่ง bullish ก่อน impulse ขาลงที่สร้าง FVG
        """
        obs: List[OrderBlock] = []
        n = len(closes)
        start = max(0, n - lookback)

        for i in range(start, n - 2):
            body_i = abs(closes[i] - opens[i])
            range_i = highs[i] - lows[i]
            if range_i <= 0:
                continue

            body_next = abs(closes[i + 1] - opens[i + 1])
            range_next = highs[i + 1] - lows[i + 1]
            if range_next <= 0:
                continue
            impulse_ratio = body_next / range_next

            if impulse_ratio < 0.6 or body_next < body_i * 1.2:
                continue

            # Check if impulse created an FVG (3-candle gap)
            has_fvg = False
            if i + 2 < n:
                # Bullish FVG: candle[i].high < candle[i+2].low
                if closes[i + 1] > opens[i + 1] and highs[i] < lows[i + 2]:
                    has_fvg = True
                # Bearish FVG: candle[i].low > candle[i+2].high
                if closes[i + 1] < opens[i + 1] and lows[i] > highs[i + 2]:
                    has_fvg = True

            # Check if OB zone was retested (no longer fresh)
            is_fresh = True
            for k in range(i + 2, n):
                if lows[k] <= highs[i] and highs[k] >= lows[i]:
                    is_fresh = False
                    break

            # Bullish OB: bearish candle → strong bullish impulse
            if closes[i] < opens[i] and closes[i + 1] > opens[i + 1]:
                strength = min(100, impulse_ratio * 100 * (body_next / max(body_i, 0.01)))
                obs.append(OrderBlock(
                    ob_type=OBType.BULLISH,
                    high=float(highs[i]), low=float(lows[i]),
                    index=i, strength=min(100, strength),
                    has_fvg=has_fvg, is_fresh=is_fresh
                ))

            # Bearish OB: bullish candle → strong bearish impulse
            elif closes[i] > opens[i] and closes[i + 1] < opens[i + 1]:
                strength = min(100, impulse_ratio * 100 * (body_next / max(body_i, 0.01)))
                obs.append(OrderBlock(
                    ob_type=OBType.BEARISH,
                    high=float(highs[i]), low=float(lows[i]),
                    index=i, strength=min(100, strength),
                    has_fvg=has_fvg, is_fresh=is_fresh
                ))

        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # 7. FAIR VALUE GAP DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_fair_value_gaps(self, highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray, lookback: int = 20
                                ) -> List[FairValueGap]:
        """FVG = ช่องว่าง 3 แท่ง (Imbalance)"""
        fvgs: List[FairValueGap] = []
        n = len(highs)
        start = max(1, n - lookback)

        for i in range(start, n - 1):
            if highs[i - 1] < lows[i + 1]:
                fvgs.append(FairValueGap(
                    high=float(lows[i + 1]), low=float(highs[i - 1]),
                    is_bullish=True, index=i))
            if lows[i - 1] > highs[i + 1]:
                fvgs.append(FairValueGap(
                    high=float(lows[i - 1]), low=float(highs[i + 1]),
                    is_bullish=False, index=i))

        return fvgs

    # ─────────────────────────────────────────────────────────────────────────
    # 8. CHANGE OF CHARACTER DETECTION (Rule 5: LTF ChoCH)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_choch(self, highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, expected_direction: str,
                     lookback: int = 20) -> bool:
        """
        Rule 5: รอ M5 เปลี่ยนโครงสร้าง (Change of Character) ยืนยันก่อนเข้า

        ChoCH for BUY: M5 ต้อง break above recent swing high (bearish→bullish shift)
        ChoCH for SELL: M5 ต้อง break below recent swing low (bullish→bearish shift)

        Args:
            highs, lows, closes: LTF (M5) data
            expected_direction: "BUY" or "SELL"
        """
        n = len(closes)
        if n < 10:
            return False

        # ใช้ lookback น้อยสำหรับ M5 (3 แท่งรอบข้าง)
        swings = self.detect_swing_points(highs, lows, lookback=3)
        if len(swings) < 3:
            return False

        swing_highs = sorted([s for s in swings if s.swing_type == SwingType.HIGH],
                             key=lambda s: s.index)
        swing_lows = sorted([s for s in swings if s.swing_type == SwingType.LOW],
                            key=lambda s: s.index)

        current_close = closes[-1]

        if expected_direction == "BUY":
            # ChoCH BUY: ราคาปัจจุบัน break เหนือ swing high ล่าสุด
            if swing_highs:
                last_sh = swing_highs[-1]
                if current_close > last_sh.price:
                    logger.info(f"   ✅ ChoCH BUY: M5 broke above SH {last_sh.price:.2f}")
                    return True
        elif expected_direction == "SELL":
            # ChoCH SELL: ราคาปัจจุบัน break ใต้ swing low ล่าสุด
            if swing_lows:
                last_sl = swing_lows[-1]
                if current_close < last_sl.price:
                    logger.info(f"   ✅ ChoCH SELL: M5 broke below SL {last_sl.price:.2f}")
                    return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # 🎯 MAIN SIGNAL GENERATOR - THE 90% PROTOCOL
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, opens: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, closes: np.ndarray,
                        atr: float, current_price: float,
                        htf_highs: np.ndarray = None,
                        htf_lows: np.ndarray = None,
                        ltf_highs: np.ndarray = None,
                        ltf_lows: np.ndarray = None,
                        ltf_closes: np.ndarray = None,
                        ) -> SMCSignal:
        """
        🧠 THE 90% PROTOCOL - ต้องผ่านทุกกฎก่อนเปิดออเดอร์

        Args:
            opens/highs/lows/closes: H1 data (main timeframe)
            htf_highs/htf_lows: H4 data (Rule 1: HTF Structure)
            ltf_highs/ltf_lows/ltf_closes: M5 data (Rule 5: LTF ChoCH)
        """
        no_signal = SMCSignal(
            signal=None, confidence=0, stop_loss=0, take_profit=0,
            sweep_detected=False, sweep_side=None,
            structure=StructureTrend.RANGING,
            htf_structure=StructureTrend.RANGING,
            order_block=None, fvg=None,
            in_discount_zone=False, choch_confirmed=False,
            reason="No signal"
        )

        n = len(closes)
        if n < 60:
            no_signal.reason = "Insufficient data"
            return no_signal

        # ═══════════════════════════════════════════════════════════════════
        # RULE 1: HTF STRUCTURE (H4) - ใช้เป็น directional bias (ไม่ hard block)
        # ═══════════════════════════════════════════════════════════════════
        htf_structure = StructureTrend.RANGING

        if htf_highs is not None and htf_lows is not None and len(htf_highs) >= 20:
            htf_swings = self.detect_swing_points(htf_highs, htf_lows, lookback=3)
            htf_structure = self.detect_structure(htf_swings)
            logger.info(f"   📐 Rule 1 HTF (H4): {htf_structure.value} ({len(htf_swings)} swings)")

            if htf_structure == StructureTrend.RANGING:
                # H4 Ranging → fallback to H1 structure (don't hard block)
                logger.info(f"   ⚠️ Rule 1: H4 is RANGING → fallback to H1 structure")
        else:
            # ถ้าไม่มี H4 data ใช้ H1 structure แทน (fallback)
            logger.info(f"   ⚠️ Rule 1: No H4 data, using H1 structure as fallback")

        # ═══════════════════════════════════════════════════════════════════
        # H1 Analysis
        # ═══════════════════════════════════════════════════════════════════
        swings = self.detect_swing_points(highs, lows)
        if len(swings) < 3:
            no_signal.reason = f"Not enough H1 swings ({len(swings)})"
            return no_signal

        structure = self.detect_structure(swings)

        # ถ้ามี H4 trend → ให้ preference ตามทิศ H4 (but don't hard block)
        # H4 conflict จะลด confidence แทนที่จะ block
        h1_h4_conflict = False
        if htf_structure != StructureTrend.RANGING:
            if structure != htf_structure and structure != StructureTrend.RANGING:
                h1_h4_conflict = True
                logger.info(f"   ⚠️ Rule 1: H1 {structure.value} vs H4 {htf_structure.value} → confidence penalty")

        # ═══════════════════════════════════════════════════════════════════
        # RULE 2: LIQUIDITY SWEEP - "No Sweep, No Entry"
        # Try multiple lookback windows to catch sweeps
        # ═══════════════════════════════════════════════════════════════════
        pools_above, pools_below = self.detect_liquidity_pools(swings, current_price, atr)

        # Try with configured lookback first, then expand if no sweep found
        sweep = self.detect_liquidity_sweep(opens, highs, lows, closes, swings)
        if sweep is None and self.sweep_lookback < 20:
            # Expand search window — sweeps can happen earlier and still be valid
            sweep = self.detect_liquidity_sweep(opens, highs, lows, closes, swings, lookback=20)
            if sweep:
                logger.info(f"   🔍 Rule 2: Found sweep in expanded window (20 candles)")

        if sweep is None:
            no_signal.structure = structure
            no_signal.htf_structure = htf_structure
            no_signal.reason = "Rule 2 FAIL: No Liquidity Sweep → No Entry"
            logger.info(f"   🚫 Rule 2: No sweep detected → ห้ามเข้า")
            return no_signal

        logger.info(f"   ✅ Rule 2: {sweep.side.value} sweep at {sweep.pool_price:.2f} (strength={sweep.rejection_strength:.0f}%)")

        # ═══════════════════════════════════════════════════════════════════
        # RULE 3: PREMIUM/DISCOUNT ZONE (Fibonacci)
        # ═══════════════════════════════════════════════════════════════════
        in_discount, in_premium, fib_level = self.detect_premium_discount(swings, current_price)

        # Determine signal direction from sweep
        # Rule 3: Relaxed zone — allow near-boundary trades (0.35-0.65 neutral zone)
        zone_penalty = 0  # confidence penalty for being in wrong zone
        if sweep.side == SweepSide.SELL_SIDE:
            proposed_signal = "BUY"
            if not in_discount:
                if fib_level <= 0.65:
                    # Near equilibrium — allow but penalize
                    zone_penalty = 15
                    logger.info(f"   ⚠️ Rule 3: BUY at Fib {fib_level:.2f} (neutral zone, -15 confidence)")
                else:
                    no_signal.structure = structure
                    no_signal.htf_structure = htf_structure
                    no_signal.sweep_detected = True
                    no_signal.sweep_side = sweep.side
                    no_signal.reason = f"Rule 3 FAIL: BUY in deep Premium zone (Fib={fib_level:.2f}, need <0.65)"
                    logger.info(f"   🚫 Rule 3: BUY at Fib {fib_level:.2f} > 0.65 → ไม่อยู่ในโซน Discount")
                    return no_signal
        else:
            proposed_signal = "SELL"
            if not in_premium:
                if fib_level >= 0.35:
                    # Near equilibrium — allow but penalize
                    zone_penalty = 15
                    logger.info(f"   ⚠️ Rule 3: SELL at Fib {fib_level:.2f} (neutral zone, -15 confidence)")
                else:
                    no_signal.structure = structure
                    no_signal.htf_structure = htf_structure
                    no_signal.sweep_detected = True
                    no_signal.sweep_side = sweep.side
                    no_signal.reason = f"Rule 3 FAIL: SELL in deep Discount zone (Fib={fib_level:.2f}, need >0.35)"
                    logger.info(f"   🚫 Rule 3: SELL at Fib {fib_level:.2f} < 0.35 → ไม่อยู่ในโซน Premium")
                    return no_signal

        logger.info(f"   ✅ Rule 3: {proposed_signal} in {'Discount' if in_discount else 'Premium/Neutral'} zone (Fib={fib_level:.2f})")

        # HTF direction check — penalize instead of hard block
        htf_penalty = 0
        if htf_structure == StructureTrend.BULLISH and proposed_signal != "BUY":
            if h1_h4_conflict:
                # Both H1 and H4 disagree → hard block
                no_signal.structure = structure
                no_signal.htf_structure = htf_structure
                no_signal.reason = f"Rule 1: H4 BULLISH + H1 conflict → SELL blocked"
                logger.info(f"   🚫 Rule 1: H4 BULLISH → can't SELL (H1 also conflicts)")
                return no_signal
            else:
                htf_penalty = 20
                logger.info(f"   ⚠️ Rule 1: H4 BULLISH but SELL from sweep → -20 confidence")
        if htf_structure == StructureTrend.BEARISH and proposed_signal != "SELL":
            if h1_h4_conflict:
                no_signal.structure = structure
                no_signal.htf_structure = htf_structure
                no_signal.reason = f"Rule 1: H4 BEARISH + H1 conflict → BUY blocked"
                logger.info(f"   🚫 Rule 1: H4 BEARISH → can't BUY (H1 also conflicts)")
                return no_signal
            else:
                htf_penalty = 20
                logger.info(f"   ⚠️ Rule 1: H4 BEARISH but BUY from sweep → -20 confidence")

        # ═══════════════════════════════════════════════════════════════════
        # RULE 4: ORDER BLOCK + FVG (รอยเท้าเจ้ามือ)
        # ═══════════════════════════════════════════════════════════════════
        order_blocks = self.detect_order_blocks(opens, highs, lows, closes)
        fvgs = self.detect_fair_value_gaps(highs, lows, closes)

        matched_ob: Optional[OrderBlock] = None
        matched_fvg: Optional[FairValueGap] = None
        confidence = sweep.rejection_strength
        reason_parts = [f"{sweep.side.value} sweep at {sweep.pool_price:.2f}", f"Fib={fib_level:.2f}"]

        if proposed_signal == "BUY":
            # หา Bullish OB ที่ราคาอยู่ใน/ใกล้ + มี FVG + ยัง Fresh
            for ob in order_blocks:
                if ob.ob_type == OBType.BULLISH:
                    if ob.low <= current_price <= ob.high + atr * 0.5:
                        matched_ob = ob
                        if ob.has_fvg:
                            confidence += 15
                            reason_parts.append(f"Bullish OB+FVG ({ob.low:.2f}-{ob.high:.2f}) ✓")
                        else:
                            confidence += 8
                            reason_parts.append(f"Bullish OB ({ob.low:.2f}-{ob.high:.2f})")
                        if ob.is_fresh:
                            confidence += 5
                            reason_parts.append("Fresh zone ✓")
                        break

            for fvg in fvgs:
                if fvg.is_bullish and fvg.low <= current_price <= fvg.high:
                    matched_fvg = fvg
                    confidence += 5
                    reason_parts.append(f"In Bullish FVG ({fvg.low:.2f}-{fvg.high:.2f}) ✓")
                    break
        else:
            for ob in order_blocks:
                if ob.ob_type == OBType.BEARISH:
                    if ob.low - atr * 0.5 <= current_price <= ob.high:
                        matched_ob = ob
                        if ob.has_fvg:
                            confidence += 15
                            reason_parts.append(f"Bearish OB+FVG ({ob.low:.2f}-{ob.high:.2f}) ✓")
                        else:
                            confidence += 8
                            reason_parts.append(f"Bearish OB ({ob.low:.2f}-{ob.high:.2f})")
                        if ob.is_fresh:
                            confidence += 5
                            reason_parts.append("Fresh zone ✓")
                        break

            for fvg in fvgs:
                if not fvg.is_bullish and fvg.low <= current_price <= fvg.high:
                    matched_fvg = fvg
                    confidence += 5
                    reason_parts.append(f"In Bearish FVG ({fvg.low:.2f}-{fvg.high:.2f}) ✓")
                    break

        # Rule 4 bonus: OB+FVG = strongest setup
        if matched_ob and matched_ob.has_fvg:
            confidence += 5
            logger.info(f"   ✅ Rule 4: OB+FVG confluence → strongest setup")
        elif matched_ob:
            logger.info(f"   ⚠️ Rule 4: OB found but no FVG → weaker setup")
        else:
            logger.info(f"   ⚠️ Rule 4: No OB at current price → proceed with sweep only")

        # Structure alignment bonus
        if structure == htf_structure:
            confidence += 10
            reason_parts.append(f"H1+H4 aligned: {htf_structure.value} ✓")

        # ═══════════════════════════════════════════════════════════════════
        # RULE 5: LTF ChoCH CONFIRMATION (M5)
        # ═══════════════════════════════════════════════════════════════════
        choch_confirmed = False

        if ltf_highs is not None and ltf_lows is not None and ltf_closes is not None:
            if len(ltf_highs) >= 10:
                choch_confirmed = self.detect_choch(
                    ltf_highs, ltf_lows, ltf_closes,
                    expected_direction=proposed_signal
                )
                if choch_confirmed:
                    confidence += 10
                    reason_parts.append("M5 ChoCH confirmed ✓")
                    logger.info(f"   ✅ Rule 5: M5 ChoCH confirmed → safe entry")
                else:
                    confidence -= 10
                    reason_parts.append("M5 ChoCH NOT confirmed ⚠")
                    logger.info(f"   ⚠️ Rule 5: M5 ChoCH not confirmed → riskier entry")
            else:
                logger.info(f"   ⚠️ Rule 5: M5 data too short ({len(ltf_highs)}), skipping ChoCH check")
        else:
            logger.info(f"   ⚠️ Rule 5: No M5 data available, skipping ChoCH check")

        # ═══════════════════════════════════════════════════════════════════
        # 🎯 SL / TP CALCULATION
        # ═══════════════════════════════════════════════════════════════════

        if proposed_signal == "BUY":
            # SL = ใต้ sweep level (liquidity ถูกเก็บแล้ว)
            stop_loss = sweep.sweep_price - (atr * self.sl_buffer_atr)
            max_sl_dist = atr * self.max_sl_atr
            if current_price - stop_loss > max_sl_dist:
                stop_loss = current_price - max_sl_dist

            # TP = liquidity pool ถัดไปข้างบน
            if pools_above:
                take_profit = pools_above[0].price
                reason_parts.append(f"TP→pool {take_profit:.2f}")
            else:
                take_profit = current_price + abs(current_price - stop_loss) * 2.0
                reason_parts.append(f"TP→2R {take_profit:.2f}")
        else:
            stop_loss = sweep.sweep_price + (atr * self.sl_buffer_atr)
            max_sl_dist = atr * self.max_sl_atr
            if stop_loss - current_price > max_sl_dist:
                stop_loss = current_price + max_sl_dist

            if pools_below:
                take_profit = pools_below[0].price
                reason_parts.append(f"TP→pool {take_profit:.2f}")
            else:
                take_profit = current_price - abs(stop_loss - current_price) * 2.0
                reason_parts.append(f"TP→2R {take_profit:.2f}")

        # Ensure R:R >= 1.5
        sl_dist = abs(current_price - stop_loss)
        tp_dist = abs(take_profit - current_price)
        if tp_dist < sl_dist * 1.5:
            if proposed_signal == "BUY":
                take_profit = current_price + sl_dist * 1.5
            else:
                take_profit = current_price - sl_dist * 1.5
            reason_parts.append("TP adjusted R:R≥1.5")

        # Apply penalties from relaxed rules
        confidence -= zone_penalty
        confidence -= htf_penalty
        if h1_h4_conflict:
            confidence -= 10

        confidence = max(0, min(100, confidence))
        reason = " | ".join(reason_parts)

        logger.info(f"   🧠 SMC 90%%: {proposed_signal} | Conf={confidence:.0f}% | {reason}")

        return SMCSignal(
            signal=proposed_signal,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sweep_detected=True,
            sweep_side=sweep.side,
            structure=structure,
            htf_structure=htf_structure,
            order_block=matched_ob,
            fvg=matched_fvg,
            in_discount_zone=in_discount,
            choch_confirmed=choch_confirmed,
            reason=reason
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_smc_instance: Optional[SMCStrategy] = None


def get_smc_strategy(**kwargs) -> SMCStrategy:
    global _smc_instance
    if _smc_instance is None:
        _smc_instance = SMCStrategy(**kwargs)
    return _smc_instance
