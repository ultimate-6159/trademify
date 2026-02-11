"""
🧠 SMART MONEY CONCEPTS (SMC) STRATEGY ENGINE
หลักการเทรดแบบ Institutional - เข้าหลัง Stop Hunt

📖 แนวคิดหลัก:
- Retail traders ใช้ EMA/RSI → เข้าพร้อมฝูงชน → Smart Money ล่า Stop Loss
- SMC: รอ Smart Money sweep liquidity ก่อน → เข้าหลัง sweep → SL หลัง sweep (ปลอดภัย)

🔑 Core Concepts:
1. Swing Structure (HH/HL/LH/LL) - โครงสร้างตลาด
2. Liquidity Pools - กลุ่ม Stop Loss ที่สะสมเหนือ Swing High / ใต้ Swing Low
3. Liquidity Sweep - ราคาแทงทะลุ Swing แล้วกลับ (จับ stop แล้ว)
4. Order Blocks - โซนที่สถาบันเข้าซื้อ/ขาย
5. Fair Value Gaps (FVG) - ช่องว่างราคาที่ต้องถูกเติมเต็ม
6. Break of Structure (BOS) - โครงสร้างเปลี่ยนทิศ
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
    side: SweepSide  # BUY_SIDE = above, SELL_SIDE = below
    strength: int     # จำนวน swing points ที่ซ้อนทับ
    swept: bool = False


@dataclass
class LiquiditySweep:
    side: SweepSide
    sweep_price: float       # ราคาที่ sweep ไปถึง
    pool_price: float        # ราคา liquidity pool ที่ถูก sweep
    rejection_strength: float  # 0-100, ความแรงของการ reject กลับ
    candle_index: int


@dataclass
class OrderBlock:
    ob_type: OBType
    high: float
    low: float
    index: int
    strength: float  # 0-100


@dataclass
class FairValueGap:
    high: float     # ขอบบนของ gap
    low: float      # ขอบล่างของ gap
    is_bullish: bool  # True = bullish FVG (gap up), False = bearish FVG (gap down)
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
    order_block: Optional[OrderBlock]
    fvg: Optional[FairValueGap]
    reason: str


# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 SMC STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SMCStrategy:
    """
    Smart Money Concepts Strategy
    หลักการ: เข้าหลัง Liquidity Sweep + Order Block + Structure Alignment
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
        """
        ตรวจหา Swing High / Swing Low
        Swing High = จุดที่สูงกว่าแท่งรอบข้าง lookback แท่ง
        Swing Low = จุดที่ต่ำกว่าแท่งรอบข้าง lookback แท่ง
        """
        if lookback is None:
            lookback = self.swing_lookback

        swings: List[SwingPoint] = []
        n = len(highs)
        if n < lookback * 2 + 1:
            return swings

        for i in range(lookback, n - lookback):
            # Swing High: high[i] > all highs within lookback on both sides
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                swings.append(SwingPoint(index=i, price=float(highs[i]), swing_type=SwingType.HIGH))

            # Swing Low: low[i] < all lows within lookback on both sides
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swings.append(SwingPoint(index=i, price=float(lows[i]), swing_type=SwingType.LOW))

        return swings

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MARKET STRUCTURE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_structure(self, swing_points: List[SwingPoint]) -> StructureTrend:
        """
        วิเคราะห์โครงสร้างตลาด:
        BULLISH = Higher Highs + Higher Lows
        BEARISH = Lower Highs + Lower Lows
        """
        swing_highs = [s for s in swing_points if s.swing_type == SwingType.HIGH]
        swing_lows = [s for s in swing_points if s.swing_type == SwingType.LOW]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return StructureTrend.RANGING

        # ดู 3 swing highs/lows ล่าสุด
        recent_highs = sorted(swing_highs, key=lambda s: s.index)[-3:]
        recent_lows = sorted(swing_lows, key=lambda s: s.index)[-3:]

        # Higher Highs?
        hh_count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price > recent_highs[i - 1].price:
                hh_count += 1

        # Higher Lows?
        hl_count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price > recent_lows[i - 1].price:
                hl_count += 1

        # Lower Highs?
        lh_count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price < recent_highs[i - 1].price:
                lh_count += 1

        # Lower Lows?
        ll_count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price < recent_lows[i - 1].price:
                ll_count += 1

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
        """
        ระบุ Liquidity Pools:
        - เหนือ Swing Highs = Buy-side liquidity (buy stops สะสม)
        - ใต้ Swing Lows = Sell-side liquidity (sell stops สะสม)
        
        Returns: (pools_above, pools_below)
        """
        pools_above: List[LiquidityPool] = []
        pools_below: List[LiquidityPool] = []

        # กลุ่ม swing highs ที่ใกล้กัน (within 0.5 ATR) = strong liquidity zone
        swing_highs = sorted([s for s in swing_points if s.swing_type == SwingType.HIGH],
                             key=lambda s: s.price)
        swing_lows = sorted([s for s in swing_points if s.swing_type == SwingType.LOW],
                            key=lambda s: s.price)

        cluster_threshold = atr * 0.5

        # Cluster swing highs
        if swing_highs:
            clusters_h = self._cluster_prices([s.price for s in swing_highs], cluster_threshold)
            for cluster in clusters_h:
                pool_price = max(cluster)  # ใช้จุดสูงสุดของ cluster
                if pool_price > current_price:
                    pools_above.append(LiquidityPool(
                        price=pool_price,
                        side=SweepSide.BUY_SIDE,
                        strength=len(cluster)
                    ))

        # Cluster swing lows
        if swing_lows:
            clusters_l = self._cluster_prices([s.price for s in swing_lows], cluster_threshold)
            for cluster in clusters_l:
                pool_price = min(cluster)  # ใช้จุดต่ำสุดของ cluster
                if pool_price < current_price:
                    pools_below.append(LiquidityPool(
                        price=pool_price,
                        side=SweepSide.SELL_SIDE,
                        strength=len(cluster)
                    ))

        # เรียงตามใกล้ราคาปัจจุบันที่สุด
        pools_above.sort(key=lambda p: p.price)
        pools_below.sort(key=lambda p: -p.price)

        return pools_above, pools_below

    def _cluster_prices(self, prices: List[float], threshold: float) -> List[List[float]]:
        """จับกลุ่มราคาที่อยู่ใกล้กัน"""
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
    # 4. LIQUIDITY SWEEP DETECTION ⚡ (หัวใจสำคัญ)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_liquidity_sweep(self, opens: np.ndarray, highs: np.ndarray,
                                lows: np.ndarray, closes: np.ndarray,
                                swing_points: List[SwingPoint],
                                lookback: int = None) -> Optional[LiquiditySweep]:
        """
        ตรวจหา Liquidity Sweep:
        
        SELL-SIDE Sweep (→ BUY signal):
          - ราคา low ทะลุต่ำกว่า swing low (grab sell stops)
          - แต่ close กลับมาอยู่เหนือ swing low (rejection)
          - = Smart Money เก็บ liquidity แล้ว กำลังจะดัน price ขึ้น
        
        BUY-SIDE Sweep (→ SELL signal):
          - ราคา high ทะลุสูงกว่า swing high (grab buy stops)
          - แต่ close กลับมาอยู่ต่ำกว่า swing high (rejection)
          - = Smart Money เก็บ liquidity แล้ว กำลังจะดัน price ลง
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

        # ตรวจแท่งล่าสุด lookback แท่ง
        for i in range(max(0, n - lookback), n):
            # ─── SELL-SIDE SWEEP (ทะลุ swing low แล้วกลับ → BUY) ───
            for sl_point in swing_lows:
                if sl_point.index >= i:
                    continue  # ต้องเป็น swing ก่อนแท่งปัจจุบัน

                # Low ทะลุต่ำกว่า swing low?
                if lows[i] < sl_point.price:
                    # Close กลับมาเหนือ swing low? (rejection)
                    if closes[i] > sl_point.price:
                        # คำนวณ rejection strength
                        sweep_depth = sl_point.price - lows[i]
                        candle_range = highs[i] - lows[i]
                        if candle_range > 0:
                            # wick ratio: ยิ่ง wick ยาว ยิ่ง reject แรง
                            wick_below = min(opens[i], closes[i]) - lows[i]
                            rejection = (wick_below / candle_range) * 100

                            # Close ยิ่งห่างจาก low ยิ่งดี
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

            # ─── BUY-SIDE SWEEP (ทะลุ swing high แล้วกลับ → SELL) ───
            for sh_point in swing_highs:
                if sh_point.index >= i:
                    continue

                # High ทะลุสูงกว่า swing high?
                if highs[i] > sh_point.price:
                    # Close กลับมาต่ำกว่า swing high? (rejection)
                    if closes[i] < sh_point.price:
                        sweep_depth = highs[i] - sh_point.price
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
    # 5. ORDER BLOCK DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_order_blocks(self, opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray,
                            lookback: int = 20) -> List[OrderBlock]:
        """
        ตรวจหา Order Blocks:
        
        Bullish OB: แท่งขาลง (bearish) ก่อน impulse ขาขึ้นที่แรง
          → สถาบันซื้อตรงนี้ ถ้าราคากลับมาจะเด้งขึ้นอีก
        
        Bearish OB: แท่งขาขึ้น (bullish) ก่อน impulse ขาลงที่แรง
          → สถาบันขายตรงนี้ ถ้าราคากลับมาจะเด้งลงอีก
        """
        obs: List[OrderBlock] = []
        n = len(closes)
        start = max(0, n - lookback)

        for i in range(start, n - 2):
            body_i = abs(closes[i] - opens[i])
            range_i = highs[i] - lows[i]
            if range_i <= 0:
                continue

            # Impulse candle (แท่งถัดไปต้องแรง)
            body_next = abs(closes[i + 1] - opens[i + 1])
            range_next = highs[i + 1] - lows[i + 1]
            if range_next <= 0:
                continue
            impulse_ratio = body_next / range_next

            # ต้องเป็น impulse (body > 60% ของ range) และใหญ่กว่า 1.5x แท่งก่อน
            if impulse_ratio < 0.6 or body_next < body_i * 1.2:
                continue

            # Bullish OB: bearish candle → strong bullish impulse
            if closes[i] < opens[i] and closes[i + 1] > opens[i + 1]:
                strength = min(100, impulse_ratio * 100 * (body_next / max(body_i, 0.01)))
                obs.append(OrderBlock(
                    ob_type=OBType.BULLISH,
                    high=float(highs[i]),
                    low=float(lows[i]),
                    index=i,
                    strength=min(100, strength)
                ))

            # Bearish OB: bullish candle → strong bearish impulse
            elif closes[i] > opens[i] and closes[i + 1] < opens[i + 1]:
                strength = min(100, impulse_ratio * 100 * (body_next / max(body_i, 0.01)))
                obs.append(OrderBlock(
                    ob_type=OBType.BEARISH,
                    high=float(highs[i]),
                    low=float(lows[i]),
                    index=i,
                    strength=min(100, strength)
                ))

        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # 6. FAIR VALUE GAP DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_fair_value_gaps(self, highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray, lookback: int = 20
                                ) -> List[FairValueGap]:
        """
        ตรวจหา Fair Value Gaps (FVG):
        
        Bullish FVG: candle[i-1].high < candle[i+1].low  (gap ว่างระหว่าง 3 แท่ง → ราคาจะกลับมาเติม)
        Bearish FVG: candle[i-1].low > candle[i+1].high
        """
        fvgs: List[FairValueGap] = []
        n = len(highs)
        start = max(1, n - lookback)

        for i in range(start, n - 1):
            # Bullish FVG: gap up (ราคาจะดึงกลับลงมาเติม → support zone)
            if highs[i - 1] < lows[i + 1]:
                fvgs.append(FairValueGap(
                    high=float(lows[i + 1]),  # ขอบบน = low ของแท่ง i+1
                    low=float(highs[i - 1]),   # ขอบล่าง = high ของแท่ง i-1
                    is_bullish=True,
                    index=i
                ))

            # Bearish FVG: gap down (ราคาจะดึงกลับขึ้นมาเติม → resistance zone)
            if lows[i - 1] > highs[i + 1]:
                fvgs.append(FairValueGap(
                    high=float(lows[i - 1]),   # ขอบบน = low ของแท่ง i-1
                    low=float(highs[i + 1]),    # ขอบล่าง = high ของแท่ง i+1
                    is_bullish=False,
                    index=i
                ))

        return fvgs

    # ─────────────────────────────────────────────────────────────────────────
    # 🎯 MAIN SIGNAL GENERATOR
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, opens: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, closes: np.ndarray,
                        atr: float, current_price: float
                        ) -> SMCSignal:
        """
        🧠 SMC Signal Generator
        
        Logic:
        1. ตรวจ Swing Structure → กำหนดทิศทางตลาด
        2. ตรวจ Liquidity Pools → หา stop clusters
        3. ตรวจ Liquidity Sweep → หาจุดที่ Smart Money เก็บ liquidity
        4. ตรวจ Order Blocks → ยืนยันโซนสถาบัน
        5. ตรวจ FVG → ยืนยันช่องว่างราคา
        
        Signal:
        - SELL_SIDE sweep + bullish structure → BUY (เข้าหลัง sweep ลง)
        - BUY_SIDE sweep + bearish structure → SELL (เข้าหลัง sweep ขึ้น)
        """
        no_signal = SMCSignal(
            signal=None, confidence=0, stop_loss=0, take_profit=0,
            sweep_detected=False, sweep_side=None,
            structure=StructureTrend.RANGING,
            order_block=None, fvg=None, reason="No sweep detected"
        )

        n = len(closes)
        if n < 60:
            no_signal.reason = "Insufficient data"
            return no_signal

        # 1. Detect swing points
        swings = self.detect_swing_points(highs, lows)
        if len(swings) < 4:
            no_signal.reason = f"Not enough swings ({len(swings)})"
            return no_signal

        # 2. Detect market structure
        structure = self.detect_structure(swings)

        # 3. Detect liquidity pools
        pools_above, pools_below = self.detect_liquidity_pools(swings, current_price, atr)

        # 4. Detect liquidity sweep ⚡
        sweep = self.detect_liquidity_sweep(opens, highs, lows, closes, swings)

        if sweep is None:
            no_signal.structure = structure
            no_signal.reason = "No liquidity sweep detected in recent candles"
            return no_signal

        # 5. Detect order blocks & FVGs for confluence
        order_blocks = self.detect_order_blocks(opens, highs, lows, closes)
        fvgs = self.detect_fair_value_gaps(highs, lows, closes)

        # ═══════════════════════════════════════════════════════════════════
        # 🎯 SIGNAL LOGIC
        # ═══════════════════════════════════════════════════════════════════

        signal = None
        confidence = sweep.rejection_strength  # Base confidence from sweep strength
        stop_loss = 0.0
        take_profit = 0.0
        matched_ob: Optional[OrderBlock] = None
        matched_fvg: Optional[FairValueGap] = None
        reason_parts = []

        if sweep.side == SweepSide.SELL_SIDE:
            # Sweep ด้านล่าง (เก็บ sell stops) → BUY
            signal = "BUY"
            reason_parts.append(f"Sell-side sweep at {sweep.pool_price:.2f}")

            # SL = ใต้จุด sweep (liquidity ถูกเก็บแล้ว ราคาไม่ควรกลับมา)
            stop_loss = sweep.sweep_price - (atr * self.sl_buffer_atr)

            # Max SL distance
            max_sl_dist = atr * self.max_sl_atr
            if current_price - stop_loss > max_sl_dist:
                stop_loss = current_price - max_sl_dist

            # TP = liquidity pool ถัดไปข้างบน (ที่ราคากำลังจะวิ่งไป)
            if pools_above:
                take_profit = pools_above[0].price  # Pool ที่ใกล้ที่สุด
                reason_parts.append(f"TP at next buy-side pool {take_profit:.2f}")
            else:
                # Fallback: ใช้ recent high
                recent_high = float(np.max(highs[-20:]))
                take_profit = recent_high
                reason_parts.append(f"TP at recent high {take_profit:.2f}")

            # ──── CONFLUENCE CHECKS (เพิ่ม confidence) ────

            # Structure alignment: Bullish structure + sell-side sweep = strong
            if structure == StructureTrend.BULLISH:
                confidence += 15
                reason_parts.append("Structure: BULLISH ✓")
            elif structure == StructureTrend.RANGING:
                confidence += 5
                reason_parts.append("Structure: RANGING (neutral)")
            else:
                confidence -= 10
                reason_parts.append("Structure: BEARISH ⚠ (counter-trend)")

            # Order Block: ราคาอยู่ใน/ใกล้ bullish OB?
            for ob in order_blocks:
                if ob.ob_type == OBType.BULLISH:
                    if ob.low <= current_price <= ob.high + atr * 0.3:
                        matched_ob = ob
                        confidence += 10
                        reason_parts.append(f"In Bullish OB ({ob.low:.2f}-{ob.high:.2f}) ✓")
                        break

            # FVG: ราคาอยู่ใน bullish FVG? (price filling the gap)
            for fvg in fvgs:
                if fvg.is_bullish and fvg.low <= current_price <= fvg.high:
                    matched_fvg = fvg
                    confidence += 5
                    reason_parts.append(f"In Bullish FVG ({fvg.low:.2f}-{fvg.high:.2f}) ✓")
                    break

        elif sweep.side == SweepSide.BUY_SIDE:
            # Sweep ด้านบน (เก็บ buy stops) → SELL
            signal = "SELL"
            reason_parts.append(f"Buy-side sweep at {sweep.pool_price:.2f}")

            # SL = เหนือจุด sweep
            stop_loss = sweep.sweep_price + (atr * self.sl_buffer_atr)

            max_sl_dist = atr * self.max_sl_atr
            if stop_loss - current_price > max_sl_dist:
                stop_loss = current_price + max_sl_dist

            # TP = liquidity pool ถัดไปข้างล่าง
            if pools_below:
                take_profit = pools_below[0].price
                reason_parts.append(f"TP at next sell-side pool {take_profit:.2f}")
            else:
                recent_low = float(np.min(lows[-20:]))
                take_profit = recent_low
                reason_parts.append(f"TP at recent low {take_profit:.2f}")

            # Structure alignment
            if structure == StructureTrend.BEARISH:
                confidence += 15
                reason_parts.append("Structure: BEARISH ✓")
            elif structure == StructureTrend.RANGING:
                confidence += 5
                reason_parts.append("Structure: RANGING (neutral)")
            else:
                confidence -= 10
                reason_parts.append("Structure: BULLISH ⚠ (counter-trend)")

            # Order Block
            for ob in order_blocks:
                if ob.ob_type == OBType.BEARISH:
                    if ob.low - atr * 0.3 <= current_price <= ob.high:
                        matched_ob = ob
                        confidence += 10
                        reason_parts.append(f"In Bearish OB ({ob.low:.2f}-{ob.high:.2f}) ✓")
                        break

            # FVG
            for fvg in fvgs:
                if not fvg.is_bullish and fvg.low <= current_price <= fvg.high:
                    matched_fvg = fvg
                    confidence += 5
                    reason_parts.append(f"In Bearish FVG ({fvg.low:.2f}-{fvg.high:.2f}) ✓")
                    break

        if signal is None:
            no_signal.structure = structure
            no_signal.reason = "Sweep detected but no valid signal"
            return no_signal

        # ──── VALIDATE TP > SL DISTANCE ────
        sl_distance = abs(current_price - stop_loss)
        tp_distance = abs(take_profit - current_price)

        # TP ต้องมากกว่า SL (R:R >= 1.0)
        if tp_distance < sl_distance * 0.8:
            # ขยาย TP อีก
            if signal == "BUY":
                take_profit = current_price + sl_distance * 1.5
            else:
                take_profit = current_price - sl_distance * 1.5
            tp_distance = abs(take_profit - current_price)
            reason_parts.append(f"TP adjusted for R:R (min 1.5:1)")

        confidence = max(0, min(100, confidence))

        reason = " | ".join(reason_parts)
        logger.info(f"   🧠 SMC: {signal} | Confidence={confidence:.0f}% | {reason}")

        return SMCSignal(
            signal=signal,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sweep_detected=True,
            sweep_side=sweep.side,
            structure=structure,
            order_block=matched_ob,
            fvg=matched_fvg,
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
