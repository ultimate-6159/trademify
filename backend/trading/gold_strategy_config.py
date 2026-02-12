"""
🥇 GOLD STRATEGY CONFIGURATION
ค่า Config ที่ใช้จริงในการเทรด Gold (XAUUSDm)

⚠️ สำคัญ: ค่าเหล่านี้จะถูกใช้จริงใน ai_trading_bot.py
แก้ไขที่นี่เพื่อปรับกลยุทธ์

📊 PROVEN SETTINGS (จาก Backtest):
- Win Rate Target: 80%+
- R:R Ratio: 1.5:1 (ได้มากกว่าเสีย)
"""
from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class GoldH1Config:
    """
    🥇 Gold H1 Strategy Configuration
    แก้ไขค่าเหล่านี้เพื่อปรับกลยุทธ์
    """
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 📊 TREND FILTER - ควบคุมว่าจะเทรดตาม Trend แบบไหน
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # True = ต้อง Strong Trend เท่านั้น (EMA Fast > Mid > Slow > Trend)
    # False = อนุญาต Moderate Trend ด้วย (เทรดบ่อยขึ้น — SMC handles structure)
    require_strong_trend: bool = False
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎯 SIGNAL SCORING - ควบคุมว่าสัญญาณต้องแข็งแค่ไหน
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # จำนวน conditions ขั้นต่ำที่ต้องผ่าน (จาก 5 conditions)
    # SMC handles entry precision → technical เป็นแค่ basic filter
    min_conditions: int = 2  # 2/5 = 40% (SMC is primary signal generator)

    # Score Gap ขั้นต่ำ (ความแตกต่างระหว่าง BUY vs SELL score)
    # ลดลงเพราะ SMC กำหนดทิศทาง
    min_score_gap: int = 1  # BUY vs SELL ต้องห่างกัน >= 1 (SMC overrides direction)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 📈 RSI FILTER - ควบคุม RSI Range ที่อนุญาต
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # RSI Range สำหรับ BUY (ไม่ซื้อตอน Overbought)
    rsi_buy_min: float = 35.0  # ไม่ซื้อถ้า RSI < 35 (Oversold มาก)
    rsi_buy_max: float = 58.0  # ไม่ซื้อถ้า RSI > 58 (ใกล้ Overbought)
    
    # RSI Range สำหรับ SELL (ไม่ขายตอน Oversold)
    rsi_sell_min: float = 42.0  # ไม่ขายถ้า RSI < 42
    rsi_sell_max: float = 65.0  # ไม่ขายถ้า RSI > 65 (Overbought มาก)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🕐 SESSION FILTER - ควบคุมช่วงเวลาที่เทรด
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # London Session (UTC)
    london_start_hour: int = 7
    london_end_hour: int = 16
    
    # New York Session (UTC)
    ny_start_hour: int = 13
    ny_end_hour: int = 21
    
    # Block Asian Session? (ช่วงที่สัญญาณผิดบ่อย)
    block_asian_session: bool = True
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎯 SL/TP SETTINGS - ควบคุม Risk:Reward
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # ATR Multiplier สำหรับ SL
    # ค่าน้อย = SL แคบ (เสียบ่อย แต่เสียน้อย)
    # ค่ามาก = SL กว้าง (เสียยาก แต่เสียมาก)
    sl_atr_multiplier: float = 0.6  # SL = ATR × 0.6 (~$10-15)
    
    # ATR Multiplier สำหรับ TP
    tp_atr_multiplier: float = 0.9  # TP = ATR × 0.9 (~$15-22)
    
    # R:R Ratio (TP / SL)
    # ค่านี้จะ override tp_atr_multiplier ถ้าตั้งไว้
    # 1.5 = ได้ $1.5 ต่อทุก $1 ที่เสี่ยง
    rr_ratio: float = 1.5
    
    # Minimum SL Distance (USD) - ป้องกัน SL แคบเกินไป
    min_sl_distance: float = 8.0  # ขั้นต่ำ $8
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🛡️ TRAILING STOP - ปกป้องกำไรโดยขยับ SL ตามราคา
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # เปิด/ปิด Trailing Stop
    trailing_stop_enabled: bool = True
    
    # Trailing Stop เริ่มทำงานเมื่อกำไรถึง % นี้
    trailing_stop_trigger_pct: float = 0.5  # 0.5% (~$10-15 สำหรับ Gold)
    
    # ระยะห่าง Trailing Stop จากราคาปัจจุบัน (%)
    trailing_stop_distance_pct: float = 0.3  # 0.3% (~$7-10)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🔒 BREAK-EVEN - ย้าย SL มาจุดเข้าเมื่อกำไรพอ
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # เปิด/ปิด Break-Even Protection
    break_even_enabled: bool = True
    
    # Break-Even เริ่มทำงานเมื่อกำไรถึง % นี้
    break_even_trigger_pct: float = 0.3  # 0.3% (~$7-10)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🕯️ CANDLE FILTER - ควบคุมรูปแบบแท่งเทียน
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Body Ratio ขั้นต่ำ (Body / Total Range)
    # ค่าสูง = ต้องมีแท่งเทียน Strong body
    min_body_ratio: float = 0.4  # 40% ของแท่ง
    
    # Body Ratio สำหรับ Confirmation (Strong Candle)
    confirmation_body_ratio: float = 0.45  # 45% = Strong confirmation
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 📊 VOLUME FILTER - ควบคุม Volume ขั้นต่ำ
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Volume Ratio ขั้นต่ำ (Current / Avg20)
    # ค่าต่ำ = อนุญาต volume ต่ำกว่าค่าเฉลี่ย (MT5 tick_volume ไม่เชื่อถือได้มาก)
    min_volume_ratio: float = 0.8  # 80% ของค่าเฉลี่ย (SMC handles confirmation)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎚️ VOLATILITY FILTER - ควบคุม ATR% สูงสุด
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # ATR% สูงสุดที่ยอมรับ (ATR / Price × 100)
    # ค่าน้อย = ไม่เทรดตอน Volatile มาก
    max_volatility_pct: float = 2.5  # ไม่เทรดถ้า ATR% > 2.5%
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 📐 PULLBACK ZONE - ควบคุมระยะห่างจาก EMA
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Pullback Zone = Distance to EMA <= ATR × multiplier
    pullback_atr_multiplier: float = 2.0  # ต้องอยู่ภายใน 2x ATR จาก EMA
    
    # Support/Resistance Zone (% of range)
    sr_zone_pct: float = 0.25  # 25% จาก High/Low
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🚫 ADVANCED FILTERS - ป้องกันช่องโหว่เพิ่มเติม
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # 🎯 Peak Detection Hard Block (ไม่ใช่แค่ลด position)
    peak_detection_hard_block: bool = False  # Disabled: SMC sweep detection replaces peak/bottom detection
    
    # 📉 Momentum Filter (MACD Histogram)
    momentum_filter_enabled: bool = False  # Disabled: SMC structure analysis replaces MACD
    momentum_weakening_threshold: float = 0.3
    
    # 🔴 Consecutive Loss Protection
    consecutive_loss_pause: bool = True
    max_consecutive_losses: int = 2  # Pause หลังเสีย 2 ครั้งติด
    pause_duration_hours: int = 4  # พัก 4 ชม.
    
    # 📊 Volume Spike Detection (ข่าว/Event)
    volume_spike_block: bool = True
    volume_spike_threshold: float = 3.0  # Block ถ้า Volume > 3x avg
    
    # 📈 ATR Expansion Check (Volatility Spike)
    atr_expansion_block: bool = True
    atr_expansion_threshold: float = 1.5  # Block ถ้า ATR ปัจจุบัน > 1.5x ATR avg
    
    # 🗓️ Friday Late Block
    friday_late_block: bool = True
    friday_cutoff_hour: int = 19  # บล็อกหลัง 19:00 UTC ในวันศุกร์

    # 🌅 Monday Gap Check
    monday_gap_skip: bool = True
    monday_gap_threshold_pct: float = 0.5  # Skip ถ้า Gap > 0.5%

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🧠 SMART MONEY CONCEPTS (SMC) - เข้าหลัง Stop Hunt
    # ═══════════════════════════════════════════════════════════════════════════════

    # เปิด/ปิด SMC Filter (True = ต้องมี Liquidity Sweep ก่อนเข้าเทรด)
    smc_enabled: bool = True

    # ต้องมี Sweep ก่อนเข้าเทรด (True = บล็อกถ้าไม่มี sweep)
    smc_require_sweep: bool = True

    # Swing Point Detection: จำนวนแท่งแต่ละด้านในการหา Swing High/Low
    smc_swing_lookback: int = 5

    # จำนวนแท่งล่าสุดที่ตรวจ Sweep
    smc_sweep_lookback_candles: int = 20

    # SL Buffer: ระยะห่าง SL จากจุด sweep (ATR multiplier)
    smc_sl_buffer_atr: float = 0.3

    # Max SL Distance (ATR multiplier) - ป้องกัน SL กว้างเกินไป
    smc_max_sl_atr: float = 2.0

    # Minimum Sweep Strength (0-100) - ความแรงขั้นต่ำของ sweep
    smc_min_sweep_strength: float = 40.0

    # 🆕 SMC IS PRIMARY - SMC สามารถสร้าง signal โดยไม่ต้องรอ technical conditions ผ่าน
    # True = SMC runs INDEPENDENTLY (ไม่ถูก gate โดย technical min_conditions)
    # False = technical ต้องผ่าน min_conditions ก่อน SMC จึงจะรัน
    smc_is_primary: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dict สำหรับ logging/debugging"""
        return {
            "trend": {
                "require_strong_trend": self.require_strong_trend,
            },
            "scoring": {
                "min_conditions": self.min_conditions,
                "min_score_gap": self.min_score_gap,
            },
            "rsi": {
                "buy_range": f"{self.rsi_buy_min}-{self.rsi_buy_max}",
                "sell_range": f"{self.rsi_sell_min}-{self.rsi_sell_max}",
            },
            "sl_tp": {
                "sl_atr_mult": self.sl_atr_multiplier,
                "tp_atr_mult": self.tp_atr_multiplier,
                "rr_ratio": self.rr_ratio,
                "min_sl": self.min_sl_distance,
            },
            "trailing_stop": {
                "enabled": self.trailing_stop_enabled,
                "trigger_pct": self.trailing_stop_trigger_pct,
                "distance_pct": self.trailing_stop_distance_pct,
            },
            "break_even": {
                "enabled": self.break_even_enabled,
                "trigger_pct": self.break_even_trigger_pct,
            },
            "filters": {
                "min_body_ratio": self.min_body_ratio,
                "min_volume_ratio": self.min_volume_ratio,
                "max_volatility_pct": self.max_volatility_pct,
            },
            "advanced_filters": {
                "peak_hard_block": self.peak_detection_hard_block,
                "momentum_filter": self.momentum_filter_enabled,
                "consecutive_loss_pause": self.consecutive_loss_pause,
                "volume_spike_block": self.volume_spike_block,
                "atr_expansion_block": self.atr_expansion_block,
                "friday_late_block": self.friday_late_block,
                "monday_gap_skip": self.monday_gap_skip,
            },
            "smc": {
                "enabled": self.smc_enabled,
                "require_sweep": self.smc_require_sweep,
                "is_primary": self.smc_is_primary,
                "swing_lookback": self.smc_swing_lookback,
                "sweep_lookback": self.smc_sweep_lookback_candles,
                "sl_buffer_atr": self.smc_sl_buffer_atr,
                "max_sl_atr": self.smc_max_sl_atr,
                "min_sweep_strength": self.smc_min_sweep_strength,
            },
        }


@dataclass  
class GoldM15Config(GoldH1Config):
    """
    🥇 Gold M15 Scalping Configuration
    สืบทอดจาก H1 แต่ปรับค่าสำหรับ Scalping
    """
    
    # M15 ใช้ Moderate Trend ได้ (เทรดบ่อยขึ้น)
    require_strong_trend: bool = False
    
    # M15 ต้องการ conditions น้อยกว่า
    min_conditions: int = 4  # 4/10
    min_score_gap: int = 3
    
    # RSI กว้างขึ้นสำหรับ Scalping
    rsi_buy_min: float = 32.0
    rsi_buy_max: float = 65.0
    rsi_sell_min: float = 35.0
    rsi_sell_max: float = 68.0
    
    # SL/TP สำหรับ Scalping (แคบกว่า)
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 0.6
    rr_ratio: float = 0.6  # R:R 0.6:1 ต้อง WR > 63%
    min_sl_distance: float = 0.5
    
    # Candle filter ผ่อนลงสำหรับ M15
    min_body_ratio: float = 0.3
    
    # Volatility สูงกว่าได้
    max_volatility_pct: float = 3.5


# ═══════════════════════════════════════════════════════════════════════════════
# 💱 FOREX H1 CONFIG - EURUSDm & GBPUSDm (SMC สำหรับพอร์ต $500)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ForexH1Config(GoldH1Config):
    """
    💱 Forex H1 Strategy Configuration (EURUSDm, GBPUSDm)
    กฎเหล็ก SMC สำหรับพอร์ต $500:

    1. Risk 1-2% ($5-$10) ต่อไม้
    2. R:R ขั้นต่ำ 1:3 (เสี่ยง $5 ต้องหวังกำไร $15+)
    3. SL แคบ 5-15 pips (ถ้าเกินนี้ห้ามเข้า)
    4. Kill Zone ONLY (London 07-09 UTC, NY 12-15 UTC)
    5. ต้องเห็น Liquidity Sweep ก่อนเข้าเสมอ
    6. ChoCH M5 ยืนยันก่อนเข้า
    7. TP1 = Prior swing (50% close + BE), TP2 = Run with H4
    """

    # ═══════════════════════════════════════════════════════════════════════════════
    # 📊 TREND FILTER - ใช้ Moderate Trend ก็พอ (SMC handles structure)
    # ═══════════════════════════════════════════════════════════════════════════════
    require_strong_trend: bool = False

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎯 SIGNAL SCORING - SMC handles precision, basic trend filter only
    # ═══════════════════════════════════════════════════════════════════════════════
    min_conditions: int = 2   # 2/5 basic conditions (SMC is primary)
    min_score_gap: int = 1    # BUY vs SELL gap (SMC overrides direction)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 📈 RSI FILTER - Forex ranges
    # ═══════════════════════════════════════════════════════════════════════════════
    rsi_buy_min: float = 35.0
    rsi_buy_max: float = 60.0
    rsi_sell_min: float = 40.0
    rsi_sell_max: float = 65.0

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🕐 KILL ZONE SESSION FILTER - เทรดเฉพาะเวลาเจ้ามือตื่น
    # London Kill Zone: 07:00 - 09:00 UTC (14:00-16:00 Thai)
    # New York Kill Zone: 12:00 - 15:00 UTC (19:00-22:00 Thai)
    # นอกเวลาเหล่านี้ ห้ามเปิดออเดอร์เด็ดขาด
    # ═══════════════════════════════════════════════════════════════════════════════
    london_start_hour: int = 7
    london_end_hour: int = 12    # EXPANDED: 07-12 UTC (was 09) — 5 hours London
    ny_start_hour: int = 12
    ny_end_hour: int = 18        # EXPANDED: 12-18 UTC (was 15) — 6 hours NY
    block_asian_session: bool = True  # ห้ามเทรด Asian session

    # 🆕 KILL ZONE HARD BLOCK - ห้ามเทรดนอก Kill Zone เด็ดขาด
    kill_zone_hard_block: bool = True

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎯 SL/TP SETTINGS - R:R ขั้นต่ำ 1:3 / SL แคบ 5-15 pips
    # ═══════════════════════════════════════════════════════════════════════════════
    sl_atr_multiplier: float = 0.8   # SL = ATR × 0.8 (tight)
    tp_atr_multiplier: float = 2.4   # TP = ATR × 2.4
    rr_ratio: float = 3.0            # 🔴 R:R 1:3 ขั้นต่ำ (เสี่ยง $5 ต้องหวัง $15+)
    min_sl_distance: float = 0.0005  # Min SL = 5 pips (0.0005 for EURUSD)
    max_sl_distance: float = 0.0015  # Max SL = 15 pips (ถ้าเกินห้ามเข้า)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🛡️ TRAILING STOP
    # ═══════════════════════════════════════════════════════════════════════════════
    trailing_stop_enabled: bool = True
    trailing_stop_trigger_pct: float = 0.3
    trailing_stop_distance_pct: float = 0.2

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🔒 BREAK-EVEN
    # ═══════════════════════════════════════════════════════════════════════════════
    break_even_enabled: bool = True
    break_even_trigger_pct: float = 0.2   # BE เร็วขึ้นสำหรับพอร์ตเล็ก

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🕯️ CANDLE FILTER
    # ═══════════════════════════════════════════════════════════════════════════════
    min_body_ratio: float = 0.4
    confirmation_body_ratio: float = 0.45

    # ═══════════════════════════════════════════════════════════════════════════════
    # 📊 VOLUME FILTER
    # ═══════════════════════════════════════════════════════════════════════════════
    min_volume_ratio: float = 0.8  # 80% (MT5 tick_volume ไม่เชื่อถือ — SMC handles confirmation)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎚️ VOLATILITY FILTER
    # ═══════════════════════════════════════════════════════════════════════════════
    max_volatility_pct: float = 2.0   # Forex = ต่ำกว่า Gold

    # ═══════════════════════════════════════════════════════════════════════════════
    # 📐 PULLBACK ZONE
    # ═══════════════════════════════════════════════════════════════════════════════
    pullback_atr_multiplier: float = 2.0
    sr_zone_pct: float = 0.30

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🚫 ADVANCED FILTERS
    # ═══════════════════════════════════════════════════════════════════════════════
    peak_detection_hard_block: bool = False
    momentum_filter_enabled: bool = False
    consecutive_loss_pause: bool = True
    max_consecutive_losses: int = 2
    pause_duration_hours: int = 4
    volume_spike_block: bool = True
    volume_spike_threshold: float = 3.0
    atr_expansion_block: bool = True
    atr_expansion_threshold: float = 1.5
    friday_late_block: bool = True
    friday_cutoff_hour: int = 19
    monday_gap_skip: bool = True
    monday_gap_threshold_pct: float = 0.3   # Forex: tighter gap threshold

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🧠 SMART MONEY CONCEPTS (SMC) - กฎเหล็ก 5 ข้อ
    # ═══════════════════════════════════════════════════════════════════════════════
    smc_enabled: bool = True
    smc_require_sweep: bool = True          # ห้ามเข้าถ้าไม่มี Liquidity Sweep
    smc_swing_lookback: int = 5
    smc_sweep_lookback_candles: int = 20    # 20 candles (H1 = 20 ชม.)
    smc_sl_buffer_atr: float = 0.2          # Forex: tighter SL buffer
    smc_max_sl_atr: float = 1.5             # Forex: max SL tighter than Gold
    smc_min_sweep_strength: float = 40.0

    # 🆕 SMC IS PRIMARY (inherited from GoldH1Config, confirmed for Forex)
    smc_is_primary: bool = True


@dataclass
class GBPUSDConfig(ForexH1Config):
    """
    🇬🇧 GBPUSDm (Cable) - Specific Configuration

    นิสัยคู่เงิน:
    - ผันผวนกว่า EURUSD, วิ่งแรง
    - ชอบ "Deep Retracement" (ย่อลึก)
    - London Open Strategy: Judas Swing (ทางหลอก) → V-Shape Reversal

    กลยุทธ์:
    - จับตาดู London Kill Zone 07:00-09:00 UTC
    - ถ้ากราฟพุ่งไปทางไหนแรงๆ = Judas Swing (ทางหลอก)
    - รอมันพุ่งหลอกเสร็จ แล้วตบกลับทาง V-Shape ใน M15
    - Entry ที่ไหล่ขวา (Retest) ของการกลับตัว
    """

    # GBP: ผันผวนมากกว่า → ยอม SL กว้างกว่านิดหน่อย
    sl_atr_multiplier: float = 0.9
    max_sl_distance: float = 0.0018   # Max 18 pips (GBP volatile)
    min_sl_distance: float = 0.0006   # Min 6 pips

    # GBP: ATR สูงกว่า → ยอม volatility มากขึ้น
    max_volatility_pct: float = 2.5

    # GBP: ลด sweep strength ลงเล็กน้อย (มี fake moves บ่อย)
    smc_min_sweep_strength: float = 45.0


@dataclass
class EURUSDConfig(ForexH1Config):
    """
    🇪🇺 EURUSDm (Euro) - Specific Configuration

    นิสัยคู่เงิน:
    - ชอบเคารพโครงสร้าง H4
    - ชอบทำ Fakeout หลอกกิน SL ที่ Swing High/Low ก่อนกลับตัว
    - Imbalance (FVG) ใน H1 ที่ยังไม่ถูกเติมเต็ม = จุดโฟกัส

    กลยุทธ์:
    - ดู H4 เทรนด์หลัก
    - H4 ขาลง: รอราคาดีดขึ้นกวาด Asian High / Previous Daily High → ChoCH M5 → SELL
    - H4 ขาขึ้น: รอราคาลงกวาด Asian Low / Previous Daily Low → ChoCH M5 → BUY
    """

    # EUR: เคารพโครงสร้างดี → SL แคบกว่า GBP ได้
    sl_atr_multiplier: float = 0.7
    max_sl_distance: float = 0.0012   # Max 12 pips
    min_sl_distance: float = 0.0005   # Min 5 pips

    # EUR: น้อยผันผวนที่สุดใน Major → volatility ต่ำ
    max_volatility_pct: float = 1.8


@dataclass
class USDJPYConfig(ForexH1Config):
    """
    🇯🇵 USDJPYm (Dollar-Yen) - Specific Configuration

    นิสัยคู่เงิน:
    - JPY pip_value = 0.01 (ไม่ใช่ 0.0001)
    - ชอบวิ่งแรงตอน Tokyo-London overlap + NY session
    - สัมพันธ์กับ US Treasury Yields / BOJ policy
    - ชอบ Range-bound ช่วง Asian → Breakout ช่วง London/NY

    กลยุทธ์:
    - ดู H4 trend (yield-driven)
    - London Kill Zone: รอ Sweep ของ Asian Range High/Low
    - NY Kill Zone: รอ Sweep ของ London Range High/Low
    - JPY SL/TP ใช้หน่วยต่างจาก EUR/GBP (pip = 0.01)
    """

    # JPY: pip = 0.01 → SL/TP distances ต้อง 100x ของ EUR/GBP
    sl_atr_multiplier: float = 0.8
    min_sl_distance: float = 0.05    # Min 5 pips (0.05 for JPY)
    max_sl_distance: float = 0.15    # Max 15 pips (0.15 for JPY)

    # JPY: ผันผวนปานกลาง แต่ช่วง News BOJ วิ่งแรงมาก
    max_volatility_pct: float = 2.0

    # JPY: sweep strength ปกติ
    smc_min_sweep_strength: float = 50.0


@dataclass
class USDCADConfig(ForexH1Config):
    """
    🇨🇦 USDCADm (Loonie) - Specific Configuration

    นิสัยคู่เงิน:
    - สัมพันธ์กับราคาน้ำมัน (Oil up → CAD strong → USDCAD down)
    - ผันผวนน้อยกว่า GBP แต่มากกว่า EUR
    - ชอบ Consolidate ยาวแล้ว Breakout
    - NY session เคลื่อนไหวมากที่สุด (ทั้ง USD+CAD อยู่ timezone เดียวกัน)

    กลยุทธ์:
    - ดู H4 trend (oil correlation)
    - London Kill Zone: setup เตรียมตัว
    - NY Kill Zone: entry หลัก (ทั้ง USD+CAD active)
    - ระวัง Oil inventory report (พุธ) + CAD employment (ศุกร์แรกเดือน)
    """

    # CAD: pip = 0.0001 เหมือน EUR/GBP
    sl_atr_multiplier: float = 0.8
    min_sl_distance: float = 0.0005  # Min 5 pips
    max_sl_distance: float = 0.0015  # Max 15 pips

    # CAD: ผันผวนปานกลาง
    max_volatility_pct: float = 2.0

    # CAD: sweep strength ปกติ
    smc_min_sweep_strength: float = 50.0


# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 GLOBAL CONFIG INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

# Config instances ที่ใช้จริง
GOLD_H1_CONFIG = GoldH1Config()
GOLD_M15_CONFIG = GoldM15Config()
EURUSD_H1_CONFIG = EURUSDConfig()
GBPUSD_H1_CONFIG = GBPUSDConfig()
USDJPY_H1_CONFIG = USDJPYConfig()
USDCAD_H1_CONFIG = USDCADConfig()
FOREX_H1_CONFIG = ForexH1Config()  # Generic Forex fallback


def get_gold_config(timeframe: str = "H1") -> GoldH1Config:
    """
    ดึง Config สำหรับ Gold ตาม timeframe

    Usage:
        config = get_gold_config("H1")
        if score >= config.min_conditions:
            # Execute trade
    """
    if timeframe.upper() in ["M15", "M5", "M30"]:
        return GOLD_M15_CONFIG
    return GOLD_H1_CONFIG


def get_forex_config(symbol: str, timeframe: str = "H1") -> ForexH1Config:
    """
    💱 ดึง Config สำหรับ Forex ตาม symbol

    Usage:
        config = get_forex_config("EURUSDm", "H1")
        if config.smc_enabled:
            # Use SMC strategy
    """
    symbol_upper = symbol.upper()
    if "EURUSD" in symbol_upper:
        return EURUSD_H1_CONFIG
    elif "GBPUSD" in symbol_upper:
        return GBPUSD_H1_CONFIG
    elif "USDJPY" in symbol_upper:
        return USDJPY_H1_CONFIG
    elif "USDCAD" in symbol_upper:
        return USDCAD_H1_CONFIG
    return FOREX_H1_CONFIG


def get_strategy_config(symbol: str, timeframe: str = "H1"):
    """
    🌐 Universal Config - ดึง Config ที่ถูกต้องสำหรับทุก symbol

    Usage:
        config = get_strategy_config("XAUUSDm", "H1")  → GoldH1Config
        config = get_strategy_config("EURUSDm", "H1")  → EURUSDConfig
        config = get_strategy_config("GBPUSDm", "H1")  → GBPUSDConfig
        config = get_strategy_config("USDJPYm", "H1")  → USDJPYConfig
        config = get_strategy_config("USDCADm", "H1")  → USDCADConfig
    """
    symbol_upper = symbol.upper()
    if "XAU" in symbol_upper or "GOLD" in symbol_upper:
        return get_gold_config(timeframe)
    return get_forex_config(symbol, timeframe)


def load_gold_config_from_env() -> GoldH1Config:
    """
    โหลด Config จาก Environment Variables (ถ้ามี)
    
    ตัวอย่าง .env:
        GOLD_MIN_CONDITIONS=7
        GOLD_MIN_SCORE_GAP=5
        GOLD_SL_ATR_MULT=0.6
        GOLD_TP_ATR_MULT=0.9
        GOLD_RR_RATIO=1.5
        GOLD_TRAILING_STOP_ENABLED=true
        GOLD_TRAILING_STOP_TRIGGER_PCT=0.5
        GOLD_BREAK_EVEN_ENABLED=true
    """
    config = GoldH1Config()
    
    # Override จาก env ถ้ามี
    if os.getenv("GOLD_MIN_CONDITIONS"):
        config.min_conditions = int(os.getenv("GOLD_MIN_CONDITIONS"))
    if os.getenv("GOLD_MIN_SCORE_GAP"):
        config.min_score_gap = int(os.getenv("GOLD_MIN_SCORE_GAP"))
    if os.getenv("GOLD_SL_ATR_MULT"):
        config.sl_atr_multiplier = float(os.getenv("GOLD_SL_ATR_MULT"))
    if os.getenv("GOLD_TP_ATR_MULT"):
        config.tp_atr_multiplier = float(os.getenv("GOLD_TP_ATR_MULT"))
    if os.getenv("GOLD_RR_RATIO"):
        config.rr_ratio = float(os.getenv("GOLD_RR_RATIO"))
    if os.getenv("GOLD_MIN_BODY_RATIO"):
        config.min_body_ratio = float(os.getenv("GOLD_MIN_BODY_RATIO"))
    if os.getenv("GOLD_MIN_VOLUME_RATIO"):
        config.min_volume_ratio = float(os.getenv("GOLD_MIN_VOLUME_RATIO"))
    if os.getenv("GOLD_REQUIRE_STRONG_TREND"):
        config.require_strong_trend = os.getenv("GOLD_REQUIRE_STRONG_TREND").lower() == "true"
    
    # 🆕 Trailing Stop settings
    if os.getenv("GOLD_TRAILING_STOP_ENABLED"):
        config.trailing_stop_enabled = os.getenv("GOLD_TRAILING_STOP_ENABLED").lower() == "true"
    if os.getenv("GOLD_TRAILING_STOP_TRIGGER_PCT"):
        config.trailing_stop_trigger_pct = float(os.getenv("GOLD_TRAILING_STOP_TRIGGER_PCT"))
    if os.getenv("GOLD_TRAILING_STOP_DISTANCE_PCT"):
        config.trailing_stop_distance_pct = float(os.getenv("GOLD_TRAILING_STOP_DISTANCE_PCT"))
    
    # 🆕 Break-Even settings
    if os.getenv("GOLD_BREAK_EVEN_ENABLED"):
        config.break_even_enabled = os.getenv("GOLD_BREAK_EVEN_ENABLED").lower() == "true"
    if os.getenv("GOLD_BREAK_EVEN_TRIGGER_PCT"):
        config.break_even_trigger_pct = float(os.getenv("GOLD_BREAK_EVEN_TRIGGER_PCT"))

    # 🧠 SMC settings
    if os.getenv("GOLD_SMC_ENABLED"):
        config.smc_enabled = os.getenv("GOLD_SMC_ENABLED").lower() == "true"
    if os.getenv("GOLD_SMC_REQUIRE_SWEEP"):
        config.smc_require_sweep = os.getenv("GOLD_SMC_REQUIRE_SWEEP").lower() == "true"
    if os.getenv("GOLD_SMC_SWING_LOOKBACK"):
        config.smc_swing_lookback = int(os.getenv("GOLD_SMC_SWING_LOOKBACK"))
    if os.getenv("GOLD_SMC_SWEEP_LOOKBACK"):
        config.smc_sweep_lookback_candles = int(os.getenv("GOLD_SMC_SWEEP_LOOKBACK"))
    if os.getenv("GOLD_SMC_MIN_SWEEP_STRENGTH"):
        config.smc_min_sweep_strength = float(os.getenv("GOLD_SMC_MIN_SWEEP_STRENGTH"))

    return config
