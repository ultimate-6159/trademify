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
    # False = อนุญาต Moderate Trend ด้วย (เทรดบ่อยขึ้น แต่เสี่ยงขึ้น)
    require_strong_trend: bool = True
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎯 SIGNAL SCORING - ควบคุมว่าสัญญาณต้องแข็งแค่ไหน
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # จำนวน conditions ขั้นต่ำที่ต้องผ่าน (จาก 10-12 conditions)
    # สูงขึ้น = เทรดน้อยลง แต่แม่นยำขึ้น
    min_conditions: int = 3  # 3/5 = 60% (SMC handles entry precision)
    
    # Score Gap ขั้นต่ำ (ความแตกต่างระหว่าง BUY vs SELL score)
    # สูงขึ้น = สัญญาณชัดเจนขึ้น
    min_score_gap: int = 2  # BUY vs SELL ต้องห่างกัน >= 2 (fewer conditions)
    
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
    # ค่าสูง = ต้องมี Volume สูงกว่าค่าเฉลี่ย
    min_volume_ratio: float = 1.15  # 115% ของค่าเฉลี่ย
    
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
    smc_sweep_lookback_candles: int = 3

    # SL Buffer: ระยะห่าง SL จากจุด sweep (ATR multiplier)
    smc_sl_buffer_atr: float = 0.3

    # Max SL Distance (ATR multiplier) - ป้องกัน SL กว้างเกินไป
    smc_max_sl_atr: float = 2.0

    # Minimum Sweep Strength (0-100) - ความแรงขั้นต่ำของ sweep
    smc_min_sweep_strength: float = 50.0

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
# 🌐 GLOBAL CONFIG INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

# Config instances ที่ใช้จริง
GOLD_H1_CONFIG = GoldH1Config()
GOLD_M15_CONFIG = GoldM15Config()


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
