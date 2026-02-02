"""
Unified Bot API - Single Source of Truth
==========================================

🔥 ENTERPRISE GRADE - 10 Year Stability System

Architecture:
- ONE bot instance (_bot) 
- ONE trading engine (from trading_routes)
- ALL views read from same source
- AUTO-RESTART on crash
- WATCHDOG monitoring
- MEMORY CLEANUP
- STATE PERSISTENCE

Endpoints:
- GET  /api/v1/unified/status      - Bot status + signal + account
- POST /api/v1/unified/start       - Start bot (auto or manual mode)
- POST /api/v1/unified/stop        - Stop bot
- POST /api/v1/unified/switch-mode - Switch between modes (stops other first)
- GET  /api/v1/unified/signal/{symbol} - Current signal for symbol
- GET  /api/v1/unified/layers/{symbol} - Layer status for symbol
- POST /api/v1/unified/execute     - Execute a trade manually
"""

import asyncio
import logging
import gc
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/unified", tags=["unified"])


# =====================
# 🔥 ULTRA STABILITY CONFIG - 10 Year Runtime
# =====================
_stability_config = {
    "auto_restart_enabled": True,           # 🔄 Auto-restart เมื่อ crash
    "max_restart_attempts": 0,              # 🔥 0 = UNLIMITED restarts (10 year mode!)
    "restart_cooldown_seconds": 5,          # 🔄 ลดเหลือ 5 วินาที (เร็วสุด!)
    "watchdog_interval_seconds": 15,        # 🔄 ลดเหลือ 15 วินาที (เช็คบ่อยสุด!)
    "memory_cleanup_interval": 300,         # ทำความสะอาด memory ทุก 5 นาที
    "max_memory_mb": 2048,                  # ถ้า memory > 2GB ให้ cleanup
    "state_persistence_enabled": True,      # เก็บ state เพื่อ restore
    "state_file_path": "bot_state.json",    # ไฟล์เก็บ state
    "heartbeat_timeout_seconds": 60,        # 🔄 ลดเหลือ 60 วินาที (ตรวจจับเร็วสุด!)
    "auto_start_on_api_init": True,         # 🔥 เปิด! เริ่ม bot อัตโนมัติเมื่อ API start
    "daily_restart_count_reset": True,      # 🔥 Reset restart count ทุกวัน
    "auto_start_symbols": "XAUUSDm",        # 🆕 Symbols ที่จะ auto-start
    "auto_start_mode": "auto",              # 🆕 Mode: auto หรือ manual
}

# 🔥 RUNTIME STATISTICS
_runtime_stats = {
    "total_uptime_seconds": 0,
    "restart_count": 0,
    "restart_count_today": 0,               # 🔥 Restart count วันนี้
    "last_restart_time": None,
    "last_heartbeat": None,
    "last_daily_reset": None,               # 🔥 วันที่ reset ล่าสุด
    "errors_count": 0,
    "recoveries_count": 0,
    "memory_cleanups": 0,
    "network_errors": 0,                    # 🆕 Track network errors
    "mt5_reconnects": 0,                    # 🆕 Track MT5 reconnects
    "started_at": datetime.now().isoformat(),
}

# 🔥 WATCHDOG STATE
_watchdog_task = None
_last_successful_cycle = None

# 🆕 CIRCUIT BREAKER - ป้องกันระบบพัง
_circuit_breaker = {
    "state": "CLOSED",                      # CLOSED (ปกติ), OPEN (หยุด), HALF_OPEN (ทดสอบ)
    "failure_count": 0,
    "failure_threshold": 5,                 # 5 failures = เปิด circuit
    "success_count": 0,
    "success_threshold": 3,                 # 3 successes = ปิด circuit
    "last_failure_time": None,
    "cooldown_seconds": 30,                 # รอ 30 วินาทีก่อน half-open
}


# =====================
# JSON SERIALIZATION HELPER
# =====================

def _convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-friendly types"""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    return obj


# =====================
# BOT MODES
# =====================

class BotMode(str, Enum):
    """Bot operation modes - mutual exclusive"""
    STOPPED = "stopped"         # Bot not running
    AUTO = "auto"               # Auto analysis + auto trade
    MANUAL = "manual"           # Auto analysis only, manual trade




# =====================
# SINGLE BOT INSTANCE
# =====================
_bot = None
_bot_task = None
_bot_status = {
    "mode": BotMode.STOPPED.value,  # Current mode
    "running": False,
    "initialized": False,
    "symbols": [],
    "timeframe": "H1",
    "signal_mode": "technical",     # technical or pattern (FAISS)
    "quality": "MEDIUM",
    "interval": 60,
    "auto_trade": False,            # Whether to auto-execute trades
    "last_analysis": {},            # Latest analysis per symbol
    "last_signal": {},              # Latest signal per symbol  
    "layer_status": {},             # Layer status per symbol
    "daily_stats": {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "last_reset_date": None     # 🔥 Track when stats were last reset
    },
    "error": None,
    "started_at": None
}


def _check_and_reset_daily_stats():
    """🔥 Reset daily_stats ทุกวันใหม่ (เที่ยงคืน)"""
    global _bot_status
    
    today = datetime.now().date().isoformat()
    last_reset = _bot_status["daily_stats"].get("last_reset_date")
    
    if last_reset != today:
        old_stats = dict(_bot_status["daily_stats"])
        _bot_status["daily_stats"] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "last_reset_date": today
        }
        if last_reset:  # ไม่ใช่ครั้งแรก
            logger.info(f"📊 DAILY RESET: Cleared stats for new day")
            logger.info(f"   Yesterday: {old_stats['trades']} trades, W:{old_stats['wins']} L:{old_stats['losses']}, PnL:${old_stats['pnl']:.2f}")


# 🔒 TRADE LOCK - ป้องกัน Race Condition!
_trade_locks = {}  # {symbol: timestamp} - เวลาที่เริ่มเทรด
_trade_lock_timeout = 10  # 10 วินาที timeout


def _acquire_trade_lock(symbol: str) -> bool:
    """
    🔒 ขอ lock ก่อนเทรด - ป้องกัน duplicate orders!
    
    Returns: True if lock acquired, False if already locked
    """
    global _trade_locks, _trade_lock_timeout
    
    now = datetime.now()
    symbol_upper = symbol.upper()
    
    # Check if lock exists and not expired
    if symbol_upper in _trade_locks:
        lock_time = _trade_locks[symbol_upper]
        elapsed = (now - lock_time).total_seconds()
        
        if elapsed < _trade_lock_timeout:
            logger.warning(f"🔒 TRADE LOCKED: {symbol} (locked {elapsed:.1f}s ago)")
            return False
        else:
            # Lock expired, release it
            logger.info(f"🔓 Lock expired for {symbol}, acquiring new lock")
    
    # Acquire lock
    _trade_locks[symbol_upper] = now
    logger.debug(f"🔒 Lock acquired for {symbol}")
    return True


def _release_trade_lock(symbol: str):
    """🔓 ปล่อย lock หลังเทรดเสร็จ"""
    global _trade_locks
    
    symbol_upper = symbol.upper()
    if symbol_upper in _trade_locks:
        del _trade_locks[symbol_upper]
        logger.debug(f"🔓 Lock released for {symbol}")


# 🔓 DUPLICATE TRADE PREVENTION
_last_traded_signal = {}      # {symbol: {"signal": "BUY", "timestamp": datetime, "signal_id": "hash"}}
_open_positions = {}          # {symbol: True/False}
_trade_cooldown_seconds = 300  # 🔥 5 นาที cooldown! (เทรดบ่อยขึ้น!)

# 🚨 MAX TRADES PER DAY - ป้องกันเทรดมากเกินไป!
_daily_trade_limit = {
    "enabled": True,
    "max_trades_per_day": 20,           # 🔥 เพิ่มเป็น 20 เทรด/วัน!
    "max_losing_streak": 3,             # 🔥 แพ้ติดต่อกัน 3 ครั้ง = หยุด!
    "max_daily_loss_percent": 5.0,      # 🔥 ขาดทุน 5% ของ balance = หยุด!
    "pause_after_loss_minutes": 30,     # 🔥 หยุด 30 นาทีหลังแพ้ติดต่อกัน
}
_consecutive_losses = 0
_daily_trade_count = 0
_last_trade_date = None

# 🏗️ 20-LAYER GATE - ต้องผ่านกี่ layer ถึงจะเทรด
_layer_gate_config = {
    "enabled": True,                     # ✅ เปิดใช้งาน 20 Layer!
    "min_layers_passed": 12,             # 🔥 ต้องผ่านอย่างน้อย 12/20 layers (60%)
    "min_pass_rate": 60,                 # 🔥 Pass rate >= 60%
    "required_layers": [1, 2, 3, 4],     # 🔥 Layer 1-4 (Base) ต้องผ่านทั้งหมด
}

# 📊 Loss Streak Tracking
_loss_streak_tracker = {
    "current_streak": 0,
    "last_loss_time": None,
    "paused_until": None,
}

# 🥇 SYMBOL WHITELIST - เทรดเฉพาะ Gold เท่านั้น!
_symbol_whitelist = {
    "enabled": True,                         # ✅ เปิด! Block Forex
    "allowed_symbols": ["XAUUSDm", "XAUUSD", "GOLD"],  # 🥇 Gold only!
    "block_forex": True,                     # ❌ Block all Forex pairs
}

# 🔄 REVERSE SIGNAL CLOSE - ปิด position เมื่อสัญญาณตรงข้าม (ต้องกำไรก่อน!)
_enable_reverse_signal_close = True    # ✅ เปิด! แต่ต้องกำไรก่อน
_open_new_after_close = True           # ✅ ปิดแล้วเปิดใหม่ (รอ pullback)
_reverse_signal_min_profit_percent = 10.0  # 🔥 ต้องมีกำไร >= 10% ของ balance ถึงจะปิดตาม reverse signal

# ⚡ SIGNAL MOMENTUM TRACKER - ตรวจสอบว่าสัญญาณกำลังอ่อนตัว (ต้องกำไรก่อน!)
# 🔥 ปิดชั่วคราว! เพราะ trigger บ่อยเกินไปทำให้ระบบไม่เสถียร
_signal_history = {}  # {symbol: [{"signal": "BUY", "quality": "HIGH", "confidence": 75, "timestamp": datetime}, ...]}
_signal_weakening_config = {
    "enabled": False,                       # 🔥 ปิด! ไม่เสถียร - ใช้ SL/TP แทน
    "history_size": 5,                      # เก็บ signal ย้อนหลัง 5 รายการ
    "close_on_quality_drop": False,         # 🔥 ปิด! ไม่เสถียร
    "close_on_confidence_drop": False,      # 🔥 ปิด! ไม่เสถียร
    "quality_drop_threshold": 3,            # 🔥 เพิ่มเป็น 3 (PREMIUM→LOW = 3 levels)
    "confidence_drop_threshold": 25,        # 🔥 เพิ่มเป็น 25%
    "min_profit_to_exit_early_percent": 15.0,  # 🔥 กำไร >= 15% ของ balance ถึงจะ early exit
}

# =====================
# 🔔 SIGNAL FADE ALERT - Early Warning System (NEW!)
# =====================
# ตรวจจับเมื่อ confidence เริ่มลดลง ก่อนที่สัญญาณจะหายไป!

_signal_fade_config = {
    "enabled": True,                              # ✅ เปิด Early Warning
    "alert_on_confidence_drop": True,             # ✅ Alert เมื่อ confidence ลด
    "confidence_drop_threshold_percent": 10,     # 📉 Alert เมื่อลดลง >= 10% จาก peak
    "alert_on_quality_drop": True,                # ✅ Alert เมื่อ quality ลด
    "peak_tracking_enabled": True,                # 📊 Track peak confidence
    "momentum_window_size": 5,                    # 📈 ใช้ 5 readings ล่าสุดคำนวณ momentum
    "trend_detection_enabled": True,              # 🔄 ตรวจจับ trend ของ confidence
    
    # 🤖 AUTO-ACTION SETTINGS - จัดการ position อัตโนมัติเมื่อ signal fading!
    "auto_action_enabled": True,                  # ✅ เปิด auto-action
    "block_new_trades_on_warning": True,          # ❌ Block เปิด position ใหม่เมื่อ WARNING
    "block_new_trades_on_danger": True,           # ❌ Block เปิด position ใหม่เมื่อ DANGER
    "move_sl_to_breakeven_on_warning": True,      # 🔄 ย้าย SL มา break-even เมื่อ WARNING (ต้องมีกำไร)
    "close_profitable_on_danger": True,           # 💰 ปิด position ที่มีกำไรเมื่อ DANGER
    "min_profit_percent_to_close_on_danger": 5.0, # 💰 ต้องกำไร >= 5% ของ balance ถึงจะปิด
    "min_profit_to_move_sl": 0.5,                 # 🔄 ต้องกำไร >= 0.5% ของ balance ถึงจะย้าย SL
}

# 📊 Signal Health Tracker - เก็บ peak confidence และ momentum
_signal_health = {}  # {symbol: {"peak_confidence": 85, "current_confidence": 75, "momentum": "FALLING", "trend": "DOWN", "alert_level": "WARNING"}}

# 🎯 SCORE GAP FILTER - ป้องกันสัญญาณไม่ชัดเจน (NEW!)
# =====================
# ถ้า Buy Score กับ Sell Score ใกล้กันเกินไป = ไม่เทรด!
# ช่วยลด false signals และเพิ่ม win rate

_score_gap_config = {
    "enabled": True,                              # ✅ เปิด Score Gap Filter
    "min_score_gap_gold": 2,                      # 🥇 Gold: ต้องต่างกันอย่างน้อย 2 points
    "min_score_gap_forex": 3,                     # 💱 Forex: ต้องต่างกันอย่างน้อย 3 points
    "min_dominant_score_gold": 7,                 # 🥇 Gold: Score ที่ชนะต้อง >= 7/12
    "min_dominant_score_forex": 7,                # 💱 Forex: Score ที่ชนะต้อง >= 7/12
    "confidence_bonus_gap_5": 10,                 # 📈 Gap >= 5 = +10% confidence
    "confidence_bonus_gap_4": 5,                  # 📈 Gap 4 = +5% confidence  
    "confidence_bonus_gap_3": 2,                  # 📈 Gap 3 = +2% confidence
    "confidence_penalty_gap_2": -5,               # 📉 Gap 2 = -5% confidence penalty
}


# 🔀 CONTRARIAN MODE - กลับสัญญาณ
# ❌ ปิดถาวร! ใช้สัญญาณปกติ (BUY=BUY, SELL=SELL)
_contrarian_mode = {
    "enabled": False,
    "reverse_signal": False,
    "reverse_strong_signal": False,
}

# 🎯 PULLBACK ENTRY STRATEGY - รอ pullback ก่อนเข้าเทรด
# ❌ ปิดชั่วคราว! ทำให้พลาดโอกาสเทรดเมื่อราคาพุ่งตรงๆ
_pullback_config = {
    "enabled": False,                        # ❌ ปิด! เข้าเทรดทันทีเมื่อมีสัญญาณ
    "min_pullback_percent": 0.03,            # 🔥 Gold: ราคาต้องย่อ >= 0.03% (~$1.5)
    "max_pullback_percent": 0.50,            # 🔥 Gold: ย่อไม่เกิน 0.50% (~$25) - เพิ่มจาก 0.30%!
    "wait_for_stabilization": True,          # ✅ รอให้ราคานิ่งก่อนเข้า
    "stabilization_candles": 2,              # 🔥 รอ 2 แท่งเด้งกลับ (M5)
    "max_wait_minutes": 15,                  # 🔥 รอสูงสุด 15 นาที
    "require_signal_still_valid": True,      # ✅ Signal ต้องยังเป็นทิศทางเดิม
    "use_multi_timeframe": True,             # 🆕 ใช้ M5 ยืนยัน pullback
    "confirmation_timeframe": "M5",          # 🆕 Timeframe สำหรับยืนยัน pullback
}
_pending_signals = {}  # {symbol: {"signal": "BUY", "price_at_signal": 2750, "timestamp": datetime, "pullback_detected": False}}

# =====================
# 🛡️ UNIVERSAL LOT SIZING - $200 to $2,000,000,000!
# =====================
# 🔥 สูตรสากลที่ปลอดภัยสำหรับทุกขนาดบัญชี
# 
# ปัญหาที่พบ:
# - บัญชี $3,181 ใช้ Gold 0.8 lot (SL $95 = $7,600 risk = 239%!)
# - บัญชี $3,181 ใช้ EUR 10 lot (จะล้างพอร์ตทันที!)
# 
# Solution: สูตร Risk-Based + Hard Limit ที่ scale ตาม balance

_anti_wipeout_config = {
    "enabled": True,                          # ✅ เปิด! ป้องกันล้างพอร์ต
    
    # =====================
    # 🎯 UNIVERSAL LOT FORMULA - Risk-Based Calculation
    # =====================
    # Lot = (Balance × Risk%) / (SL_distance × Point_Value)
    #
    # 🏆 Gold ($5000, SL $15 = 150 pts):
    #   Point Value = $100 per $1 move per lot (100 oz contract)
    #   Risk per lot with $15 SL = $15 × 100 = $1,500
    #
    # Formula: lot = (balance × 0.01) / 1500
    #
    "max_risk_per_trade_percent": 1.0,        # 🔥 Risk 1% ต่อเทรด (เหมาะทุกขนาด)
    
    # =====================
    # 🔒 HARD LIMITS BY BALANCE TIER
    # =====================
    # เพิ่มความปลอดภัยด้วย absolute max ต่อ tier
    #
    "gold_lot_formula_divisor": 50000,        # 🔥 Gold: lot = balance / 50000
    "forex_lot_formula_divisor": 20000,       # 🔥 Forex: lot = balance / 20000
    
    # =====================
    # 📊 BALANCE TIER LIMITS (Absolute Max)
    # =====================
    # 🔴 UPDATED: Lower max lots to account for wider ATR-based SL
    "balance_tiers": {
        200: {"gold_max": 0.01, "forex_max": 0.01},      # $200-$499
        500: {"gold_max": 0.01, "forex_max": 0.01},      # $500-$999  (reduced!)
        1000: {"gold_max": 0.01, "forex_max": 0.02},     # $1,000-$2,999 (reduced!)
        3000: {"gold_max": 0.02, "forex_max": 0.05},     # $3,000-$4,999 (reduced!)
        5000: {"gold_max": 0.03, "forex_max": 0.08},     # $5,000-$9,999 (reduced!)
        10000: {"gold_max": 0.05, "forex_max": 0.15},    # $10,000-$24,999 (reduced!)
        25000: {"gold_max": 0.10, "forex_max": 0.30},    # $25,000-$49,999 (reduced!)
        50000: {"gold_max": 0.20, "forex_max": 0.50},    # $50,000-$99,999 (reduced!)
        100000: {"gold_max": 0.50, "forex_max": 1.00},   # $100,000-$499,999 (reduced!)
        500000: {"gold_max": 2.00, "forex_max": 5.00},   # $500,000-$999,999
        1000000: {"gold_max": 5.00, "forex_max": 10.0},  # $1M-$9.99M
        10000000: {"gold_max": 50.0, "forex_max": 100.0}, # $10M-$99.99M
        100000000: {"gold_max": 500.0, "forex_max": 1000.0}, # $100M-$1.99B
        2000000000: {"gold_max": 10000.0, "forex_max": 20000.0}, # $2B
    },
    
    # =====================
    # 📏 SL DISTANCE (% of Price) - UPDATED FOR ATR-BASED!
    # =====================
    # 🔴 UPDATED: Wider range to match ATR-based SL calculation
    "gold_sl_percent_min": 0.25,               # Gold SL >= 0.25% (~$12 ที่ $4700)
    "gold_sl_percent_max": 1.0,                # Gold SL <= 1.0% (~$47 ที่ $4700)
    "forex_sl_percent_min": 0.15,              # Forex SL >= 0.15%
    "forex_sl_percent_max": 0.5,               # Forex SL <= 0.5%
    
    # =====================
    # 🚫 TREND PROTECTION
    # =====================
    "check_higher_timeframe": True,
    "block_counter_trend": True,
    "min_trend_strength": 60,
}


def _get_max_lot_for_balance(balance: float, is_gold: bool) -> float:
    """
    🔒 ดึง absolute max lot จาก balance tier
    
    ใช้ tier ที่ balance >= tier_threshold
    """
    global _anti_wipeout_config
    
    tiers = _anti_wipeout_config.get("balance_tiers", {})
    key = "gold_max" if is_gold else "forex_max"
    
    # Sort tiers descending to find highest matching tier
    sorted_tiers = sorted(tiers.items(), key=lambda x: x[0], reverse=True)
    
    for tier_balance, limits in sorted_tiers:
        if balance >= tier_balance:
            return limits.get(key, 0.01)
    
    # Default for very small accounts
    return 0.01


def _calculate_safe_lot_size(balance: float, symbol: str, sl_price_distance: float = 0, current_price: float = 0) -> float:
    """
    🛡️ UNIVERSAL LOT CALCULATOR - $200 to $2,000,000,000!
    
    ใช้ 3 วิธีคำนวณแล้วเลือกค่าต่ำสุด:
    1. Risk-Based: (balance × 1%) / (SL × point_value)
    2. Formula-Based: balance / divisor
    3. Tier-Based: absolute max ตาม balance tier
    
    🔥 EXAMPLES (Gold, 1% risk, $15 SL):
    ┌──────────────┬───────────┬───────────┬───────────┬───────────┐
    │ Balance      │ Risk-Based│ Formula   │ Tier Max  │ Final Lot │
    ├──────────────┼───────────┼───────────┼───────────┼───────────┤
    │ $200         │ 0.013     │ 0.004     │ 0.01      │ 0.01      │
    │ $500         │ 0.033     │ 0.01      │ 0.01      │ 0.01      │
    │ $1,000       │ 0.067     │ 0.02      │ 0.02      │ 0.02      │
    │ $3,000       │ 0.20      │ 0.06      │ 0.06      │ 0.06      │
    │ $5,000       │ 0.33      │ 0.10      │ 0.10      │ 0.10      │
    │ $10,000      │ 0.67      │ 0.20      │ 0.20      │ 0.20      │
    │ $50,000      │ 3.33      │ 1.00      │ 1.00      │ 1.00      │
    │ $100,000     │ 6.67      │ 2.00      │ 2.00      │ 2.00      │
    │ $1,000,000   │ 66.67     │ 20.0      │ 20.0      │ 20.0      │
    │ $10,000,000  │ 666.67    │ 200.0     │ 200.0     │ 200.0     │
    │ $100,000,000 │ 6666.67   │ 2000.0    │ 2000.0    │ 2000.0    │
    │ $2,000,000,000│ 133333   │ 40000.0   │ 40000.0   │ 40000.0   │
    └──────────────┴───────────┴───────────┴───────────┴───────────┘
    """
    global _anti_wipeout_config
    
    if not _anti_wipeout_config.get("enabled", True):
        return 0.01
    
    is_gold = 'XAU' in symbol.upper() or 'GOLD' in symbol.upper()
    
    # =====================
    # 1. RISK-BASED CALCULATION
    # =====================
    max_risk_percent = _anti_wipeout_config.get("max_risk_per_trade_percent", 1.0)
    risk_amount = balance * (max_risk_percent / 100.0)
    
    if is_gold:
        # Gold: $100 per $1 price move per lot (contract = 100 oz)
        # SL of $15 = $1500 risk per lot
        point_value_per_lot = 100.0  # $100 per $1 move per lot
        
        # Use provided SL distance or default
        if sl_price_distance > 0:
            sl_dollar = sl_price_distance
        else:
            # Default SL = 0.3% of price
            min_sl_pct = _anti_wipeout_config.get("gold_sl_percent_min", 0.3)
            sl_dollar = (current_price if current_price > 0 else 5000) * (min_sl_pct / 100.0)
        
        risk_per_lot = sl_dollar * point_value_per_lot
        risk_based_lot = risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01
        
    else:
        # Forex: ~$10 per pip per lot (varies by pair)
        pip_value_per_lot = 10.0
        
        if sl_price_distance > 0:
            # Convert price distance to pips (assuming 4/5 digit pricing)
            sl_pips = sl_price_distance * 10000
        else:
            sl_pips = 30  # Default 30 pips
        
        risk_per_lot = sl_pips * pip_value_per_lot
        risk_based_lot = risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01
    
    # =====================
    # 2. FORMULA-BASED CALCULATION
    # =====================
    divisor = _anti_wipeout_config.get(
        "gold_lot_formula_divisor" if is_gold else "forex_lot_formula_divisor",
        50000 if is_gold else 20000
    )
    formula_lot = balance / divisor
    
    # =====================
    # 3. TIER-BASED MAX
    # =====================
    tier_max = _get_max_lot_for_balance(balance, is_gold)
    
    # =====================
    # FINAL: Take minimum of all 3 methods
    # =====================
    safe_lot = min(risk_based_lot, formula_lot, tier_max)
    
    # Apply min/max constraints
    safe_lot = max(0.01, safe_lot)  # Minimum 0.01
    safe_lot = round(safe_lot, 2)   # Round to 2 decimals
    
    # =====================
    # LOG FOR DEBUG
    # =====================
    logger.info(f"🛡️ UNIVERSAL LOT CALC: {symbol}")
    logger.info(f"   Balance: ${balance:,.0f} | Risk: {max_risk_percent}%")
    logger.info(f"   Risk-Based: {risk_based_lot:.4f} | Formula: {formula_lot:.4f} | Tier Max: {tier_max:.2f}")
    logger.info(f"   ✅ FINAL LOT: {safe_lot}")
    
    return safe_lot


def _validate_sl_distance(symbol: str, entry_price: float, sl_price: float, side: str) -> tuple[float, str]:
    """
    📏 ตรวจสอบและปรับ SL ให้อยู่ในช่วงที่ปลอดภัย
    
    🔥 PERCENT BASED - รองรับทุก price level!
    - Gold $5000: min SL 0.3% = $15, max SL 1.0% = $50
    - Gold $10000: min SL 0.3% = $30, max SL 1.0% = $100
    
    Returns: (adjusted_sl, message)
    """
    global _anti_wipeout_config
    
    if not _anti_wipeout_config.get("enabled", True):
        return sl_price, "SL validation disabled"
    
    is_gold = 'XAU' in symbol.upper() or 'GOLD' in symbol.upper()
    
    # 🔥 USE PERCENT OF PRICE instead of fixed points!
    if is_gold:
        min_sl_percent = _anti_wipeout_config.get("gold_sl_percent_min", 0.3)  # 0.3% of price
        max_sl_percent = _anti_wipeout_config.get("gold_sl_percent_max", 1.0)  # 1.0% of price
    else:
        min_sl_percent = _anti_wipeout_config.get("forex_sl_percent_min", 0.15)
        max_sl_percent = _anti_wipeout_config.get("forex_sl_percent_max", 0.5)
    
    # Calculate min/max SL distance based on % of entry price
    min_distance = entry_price * (min_sl_percent / 100.0)
    max_distance = entry_price * (max_sl_percent / 100.0)
    
    # Calculate current SL distance
    if side.upper() == "BUY":
        current_distance = entry_price - sl_price
    else:  # SELL
        current_distance = sl_price - entry_price
    
    current_percent = abs(current_distance) / entry_price * 100
    
    adjusted_sl = sl_price
    message = "SL OK"
    
    # Check if SL too tight
    if current_percent < min_sl_percent:
        if side.upper() == "BUY":
            adjusted_sl = entry_price - min_distance
        else:
            adjusted_sl = entry_price + min_distance
        message = f"⚠️ SL too tight ({current_percent:.2f}%), adjusted to {min_sl_percent}% (${min_distance:.2f})"
        logger.warning(f"📏 {symbol}: {message}")
    
    # Check if SL too wide
    elif current_percent > max_sl_percent:
        if side.upper() == "BUY":
            adjusted_sl = entry_price - max_distance
        else:
            adjusted_sl = entry_price + max_distance
        message = f"⚠️ SL too wide ({current_percent:.2f}%), adjusted to {max_sl_percent}% (${max_distance:.2f})"
        logger.warning(f"📏 {symbol}: {message}")
    
    return round(adjusted_sl, 2 if is_gold else 5), message


async def _check_trend_alignment(symbol: str, signal: str) -> tuple[bool, str]:
    """
    🔍 เช็คว่าสัญญาณตรงกับเทรนด์บน timeframe ใหญ่หรือไม่
    
    ใช้ H4 เป็น reference - ถ้าสวนเทรนด์รุนแรง → ไม่เทรด
    
    Returns: (is_aligned, reason)
    """
    global _bot, _anti_wipeout_config
    
    if not _anti_wipeout_config.get("check_higher_timeframe", True):
        return True, "Higher timeframe check disabled"
    
    if not _anti_wipeout_config.get("block_counter_trend", True):
        return True, "Counter-trend blocking disabled"
    
    if not _bot or not _bot.trading_engine or not _bot.trading_engine.broker:
        return True, "Cannot check - bot not ready"
    
    try:
        broker = _bot.trading_engine.broker
        if not hasattr(broker, '_mt5') or not broker._mt5:
            return True, "MT5 not available"
        
        mt5 = broker._mt5
        
        # Get H4 data for trend check
        timeframe = mt5.TIMEFRAME_H4
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 20)
        
        if rates is None or len(rates) < 20:
            return True, "Cannot get H4 data"
        
        # Simple trend detection using EMA
        closes = [r['close'] for r in rates]
        ema_fast = sum(closes[-5:]) / 5
        ema_slow = sum(closes[-20:]) / 20
        
        # Calculate trend strength (0-100)
        trend_diff = (ema_fast - ema_slow) / ema_slow * 100
        trend_strength = min(100, abs(trend_diff) * 10)
        
        is_uptrend = ema_fast > ema_slow
        is_downtrend = ema_fast < ema_slow
        
        is_buy_signal = "BUY" in signal.upper()
        is_sell_signal = "SELL" in signal.upper()
        
        min_strength = _anti_wipeout_config.get("min_trend_strength", 60)
        
        # Check alignment
        if is_buy_signal and is_downtrend and trend_strength >= min_strength:
            return False, f"❌ BLOCKED: BUY signal against strong DOWNTREND (H4 strength={trend_strength:.0f}%)"
        
        if is_sell_signal and is_uptrend and trend_strength >= min_strength:
            return False, f"❌ BLOCKED: SELL signal against strong UPTREND (H4 strength={trend_strength:.0f}%)"
        
        # Aligned or weak trend
        trend_dir = "UP" if is_uptrend else "DOWN"
        return True, f"✅ Trend aligned: H4 {trend_dir} (strength={trend_strength:.0f}%)"
        
    except Exception as e:
        logger.warning(f"Trend check error: {e}")
        return True, f"Trend check error: {e}"

# =====================
# 🎯 UNIVERSAL DYNAMIC CONFIG - รองรับ $100 ถึง $200,000,000!
# =====================
# ใช้ % แทน $ เพื่อ scale อัตโนมัติตามขนาด port

def _calc_dynamic(balance: float, percent: float, min_val: float = 1.0) -> float:
    """คำนวณค่า dynamic จาก % ของ balance"""
    return max(balance * (percent / 100.0), min_val)

def _is_micro_account(balance: float) -> bool:
    """🆕 ตรวจสอบว่าเป็นบัญชีเล็กมากหรือไม่ ($100-$300)"""
    threshold = _small_account_config.get("micro_account_threshold", 300)
    return balance < threshold

def _get_adjusted_lot_for_micro(balance: float, base_lot: float) -> float:
    """🆕 ปรับ lot size สำหรับบัญชีเล็กมาก"""
    if not _is_micro_account(balance):
        return base_lot
    
    max_lot = _small_account_config.get("micro_account_max_lot", 0.01)
    return min(base_lot, max_lot)

def _get_adjusted_sl_for_micro(balance: float, sl_percent: float) -> float:
    """🆕 ปรับ SL ให้กว้างขึ้นสำหรับบัญชีเล็กมาก (ลด risk)"""
    if not _is_micro_account(balance):
        return sl_percent
    
    multiplier = _small_account_config.get("micro_account_sl_multiplier", 1.5)
    return sl_percent * multiplier

# 📈 AUTO TRAILING STOP - PERCENT BASED!
_trailing_stop_config = {
    "enabled": True,
    "trigger_profit_percent": 5.0,           # 🎯 เริ่ม trail เมื่อกำไร >= 5% ของ balance
    "trail_distance_percent": 2.5,           # 🎯 ยก SL ห่าง 2.5% ของ balance
    "step_size_percent": 0.5,                # 🎯 ยก SL ทีละ 0.5%
    "lock_profit_percent": 10.0,             # 🎯 ล็อกกำไรเมื่อ >= 10%
}

# 🎯 SMART TRADING CONFIG - PERCENT BASED!
_aggressive_config = {
    "enabled": True,
    "min_confidence_to_trade": 75,
    "min_quality": "HIGH",
    "signal_window_minutes": 5,
    "allow_same_direction_reentry": True,
    "min_profit_for_wait_close_percent": 5.0,  # 🎯 ปิดเมื่อ WAIT + กำไร >= 5% ของ balance
    "quick_scalp_mode": False,
}

# 📈 SMART DCA (Dollar Cost Averaging) - เข้าซ้ำเมื่อราคาย่อ
# ❌ ปิดถาวร! DCA เสี่ยงเกินไปสำหรับพอร์ตเล็ก
_dca_config = {
    "enabled": False,                        # ❌ ปิดถาวร! DCA เพิ่ม risk
    "max_dca_entries": 0,                    # ❌ ไม่อนุญาต DCA
    "min_retracement_percent": 0.20,         # 🔥 ราคาต้องย่อ >= 0.20% (~$10 Gold)
    "wait_for_reversal": True,               # ✅ รอให้ราคากลับตัวก่อนเข้าซ้ำ
    "reversal_candles": 1,                   # 🔥 รอ 1 candle ที่กลับตัว
    "signal_must_persist": True,             # ✅ สัญญาณต้องยังคงเป็นทิศทางเดิม (สำคัญมาก!)
    "min_time_between_dca": 300,             # 🔥 ห่างกัน 5 นาที
    "lot_multiplier": 0.5,                   # 🔥 Lot size 0.5x (ลด risk)
    "max_loss_percent_before_dca": 5.0,      # 🎯 ถ้าขาดทุน > 5% ของ balance ไม่เข้าซ้ำ
    "require_strong_signal": True,
    "min_confidence_for_dca": 75,
    "check_signal_trend": True,
    "min_balance_for_dca": 999999,           # ❌ ตั้งสูงมากเพื่อ block DCA ทุกกรณี
}
_dca_tracking = {}

# 📊 SIGNAL STRENGTH TRACKER - ตรวจสอบความแข็งแรงของสัญญาณ
# ใช้ตรวจจับว่าสัญญาณกำลังอ่อนตัวหรือเปลี่ยนทิศทาง
_signal_strength_tracker = {}  # {symbol: {"confidence_history": [80, 78, 75], "quality_history": ["HIGH", "HIGH", "MEDIUM"], "direction_changes": 0}}

# 💰 SMART PROFIT PROTECTION - PERCENT BASED!
_profit_protection_config = {
    "enabled": True,
    "profit_drawdown_percent": 25,           # ปิดเมื่อกำไรลดลง 25% จาก peak
    "min_profit_to_protect_percent": 2.0,    # 🎯 เริ่ม protect เมื่อกำไร >= 2% ของ balance
    "trailing_trigger_percent": 5.0,         # 🎯 เริ่ม trailing เมื่อกำไร >= 5%
    "trailing_distance_percent": 2.0,        # 🎯 trailing stop ห่าง 2%
}
_peak_profit_by_position = {}

# 🛡️ SMALL ACCOUNT PROTECTION - รองรับ $100-$500!
_small_account_config = {
    "enabled": True,
    "min_balance_warning": 100,              # ⚠️ แจ้งเตือนเมื่อ balance < $100
    "min_balance_stop_trading": 50,          # 🛑 หยุดเทรดเมื่อ balance < $50
    "disable_dca_below": 500,                # ปิด DCA ถ้า balance < $500
    "micro_account_threshold": 300,          # 🆕 บัญชีเล็กมาก < $300
    "micro_account_max_lot": 0.01,           # 🆕 Lot สูงสุดสำหรับบัญชีเล็ก
    "micro_account_sl_multiplier": 1.5,      # 🆕 SL กว้างขึ้น 1.5x สำหรับบัญชีเล็ก
}

# 🧮 DYNAMIC VALUE HELPERS
def _get_trailing_trigger(balance: float) -> float:
    return _calc_dynamic(balance, _trailing_stop_config["trigger_profit_percent"], 5)

def _get_trailing_distance(balance: float) -> float:
    return _calc_dynamic(balance, _trailing_stop_config["trail_distance_percent"], 2)

def _get_trailing_step(balance: float) -> float:
    return _calc_dynamic(balance, _trailing_stop_config["step_size_percent"], 1)

def _get_trailing_lock(balance: float) -> float:
    return _calc_dynamic(balance, _trailing_stop_config["lock_profit_percent"], 10)

def _get_wait_close_profit(balance: float) -> float:
    return _calc_dynamic(balance, _aggressive_config["min_profit_for_wait_close_percent"], 5)

def _get_max_dca_loss(balance: float) -> float:
    return _calc_dynamic(balance, _dca_config["max_loss_percent_before_dca"], 10)

def _get_min_profit_to_protect(balance: float) -> float:
    return _calc_dynamic(balance, _profit_protection_config["min_profit_to_protect_percent"], 2)

def _should_allow_dca(balance: float) -> bool:
    return balance >= _dca_config.get("min_balance_for_dca", 500)

def _should_stop_trading(balance: float) -> bool:
    return balance < _small_account_config.get("min_balance_stop_trading", 50)

def _get_reverse_signal_min_profit(balance: float) -> float:
    """🔄 Get minimum profit for reverse signal close (% based)"""
    return _calc_dynamic(balance, _reverse_signal_min_profit_percent, 5)

def _get_early_exit_min_profit(balance: float) -> float:
    """⚡ Get minimum profit for early exit on weakening signal (% based)"""
    return _calc_dynamic(balance, _signal_weakening_config.get("min_profit_to_exit_early_percent", 15), 10)





# =====================
# 🔌 CIRCUIT BREAKER FUNCTIONS
# =====================

def _circuit_breaker_record_failure():
    """บันทึก failure - เพิ่ม failure count"""
    global _circuit_breaker
    
    _circuit_breaker["failure_count"] += 1
    _circuit_breaker["last_failure_time"] = datetime.now()
    _circuit_breaker["success_count"] = 0
    
    # Check if should open circuit
    if _circuit_breaker["failure_count"] >= _circuit_breaker["failure_threshold"]:
        if _circuit_breaker["state"] != "OPEN":
            _circuit_breaker["state"] = "OPEN"
            logger.warning(f"⚡ CIRCUIT BREAKER OPEN! Failures: {_circuit_breaker['failure_count']}")


def _circuit_breaker_record_success():
    """บันทึก success - รีเซ็ต failure count"""
    global _circuit_breaker
    
    if _circuit_breaker["state"] == "HALF_OPEN":
        _circuit_breaker["success_count"] += 1
        
        if _circuit_breaker["success_count"] >= _circuit_breaker["success_threshold"]:
            _circuit_breaker["state"] = "CLOSED"
            _circuit_breaker["failure_count"] = 0
            logger.info("✅ CIRCUIT BREAKER CLOSED - System recovered!")
    else:
        _circuit_breaker["failure_count"] = 0
        _circuit_breaker["success_count"] = 0


def _circuit_breaker_can_proceed() -> bool:
    """ตรวจสอบว่าสามารถทำงานต่อได้หรือไม่"""
    global _circuit_breaker
    
    if _circuit_breaker["state"] == "CLOSED":
        return True
    
    if _circuit_breaker["state"] == "OPEN":
        # Check if cooldown passed
        last_failure = _circuit_breaker["last_failure_time"]
        if last_failure:
            elapsed = (datetime.now() - last_failure).total_seconds()
            if elapsed >= _circuit_breaker["cooldown_seconds"]:
                _circuit_breaker["state"] = "HALF_OPEN"
                logger.info(f"⚡ CIRCUIT BREAKER HALF-OPEN - Testing connection...")
                return True
        return False
    
    # HALF_OPEN - allow request to test
    return True


def _get_circuit_breaker_status() -> Dict:
    """Get circuit breaker status"""
    return {
        "state": _circuit_breaker["state"],
        "failure_count": _circuit_breaker["failure_count"],
        "success_count": _circuit_breaker["success_count"],
        "last_failure": _circuit_breaker["last_failure_time"].isoformat() if _circuit_breaker["last_failure_time"] else None,
    }

# 🚨 MAX LOSS PROTECTION - บังคับปิดเมื่อขาดทุนเกินกำหนด
# 🔥 100% PERCENT BASED - รองรับ $100 ถึง $2,000,000,000!
_max_loss_config = {
    "enabled": True,
    "max_loss_percent_per_position": 2.0,   # 🔥 Max loss 2% ต่อ position (ไม่ใช่ fixed $!)
    "max_loss_percent_daily": 5.0,          # 🔥 Max loss 5% ต่อวัน
    "max_drawdown_percent": 10.0,           # 🔥 Max drawdown 10% ของ balance
    "close_on_reverse_signal": True,        # ✅ ปิดทันทีเมื่อสัญญาณตรงข้าม (แม้ขาดทุน)
}


# =====================
# 🔥 STABILITY FUNCTIONS - 10 Year Runtime
# =====================

def _save_state():
    """💾 Save bot state to file for recovery after restart"""
    global _bot_status, _stability_config, _runtime_stats
    
    
    if not _stability_config.get("state_persistence_enabled", True):
        return
    
    try:
        state = {
            "bot_status": {
                "mode": _bot_status.get("mode"),
                "symbols": _bot_status.get("symbols", []),
                "timeframe": _bot_status.get("timeframe", "H1"),
                "signal_mode": _bot_status.get("signal_mode", "technical"),
                "quality": _bot_status.get("quality", "MEDIUM"),
                "interval": _bot_status.get("interval", 60),
                "auto_trade": _bot_status.get("auto_trade", False),
                "daily_stats": _bot_status.get("daily_stats", {}),
            },
            "runtime_stats": _runtime_stats,
            "saved_at": datetime.now().isoformat(),
        }
        
        state_file = _stability_config.get("state_file_path", "bot_state.json")
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.debug(f"💾 State saved to {state_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save state: {e}")


def _load_state() -> Optional[Dict]:
    """📂 Load bot state from file for recovery"""
    global _stability_config
    
    if not _stability_config.get("state_persistence_enabled", True):
        return None
    
    try:
        state_file = _stability_config.get("state_file_path", "bot_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"📂 State loaded from {state_file}")
            return state
    except Exception as e:
        logger.warning(f"Failed to load state: {e}")
    
    return None


def _cleanup_memory():
    """🧹 Force garbage collection to prevent memory leaks"""
    global _runtime_stats, _bot_status, _signal_history
    
    try:
        # Clean up old signal history (keep only last 10)
        for symbol in list(_signal_history.keys()):
            if len(_signal_history[symbol]) > 10:
                _signal_history[symbol] = _signal_history[symbol][-10:]
        
        # Clean up old analysis data (keep only last 2 per symbol)
        if len(_bot_status.get("last_analysis", {})) > 20:
            # Keep only tracked symbols
            for sym in list(_bot_status["last_analysis"].keys()):
                if sym not in _bot_status.get("symbols", []):
                    del _bot_status["last_analysis"][sym]
        
        # Force garbage collection
        collected = gc.collect()
        
        _runtime_stats["memory_cleanups"] += 1
        logger.debug(f"🧹 Memory cleanup: collected {collected} objects")
        
    except Exception as e:
        logger.warning(f"Memory cleanup error: {e}")


def _get_memory_usage_mb() -> float:
    """📊 Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0
    except Exception:
        return 0


async def _watchdog_loop():
    """
    🐕 WATCHDOG - ตรวจสอบ health และ auto-restart
    
    ทำงาน:
    1. ตรวจสอบว่า bot ยังทำงานอยู่
    2. ตรวจสอบ memory usage
    3. Auto-restart ถ้า crash
    4. Save state periodically
    """
    global _bot_status, _bot_task, _runtime_stats, _stability_config, _last_successful_cycle
    
    logger.info("🐕 Watchdog started - monitoring bot health")
    
    watchdog_interval = _stability_config.get("watchdog_interval_seconds", 60)
    memory_cleanup_interval = _stability_config.get("memory_cleanup_interval", 300)
    last_memory_cleanup = datetime.now()
    last_state_save = datetime.now()
    
    while True:
        try:
            await asyncio.sleep(watchdog_interval)
            
            # Update heartbeat
            _runtime_stats["last_heartbeat"] = datetime.now().isoformat()
            
            # 1. Check if bot should be running but isn't
            if _bot_status.get("running") and (_bot_task is None or _bot_task.done()):
                logger.warning("🐕 WATCHDOG: Bot task died! Attempting restart...")
                _runtime_stats["errors_count"] += 1
                
                if _stability_config.get("auto_restart_enabled", True):
                    await _auto_restart_bot()
            
            # 2. Check memory usage
            memory_mb = _get_memory_usage_mb()
            max_memory = _stability_config.get("max_memory_mb", 2048)
            
            if memory_mb > max_memory:
                logger.warning(f"🐕 WATCHDOG: High memory usage ({memory_mb:.1f}MB > {max_memory}MB) - forcing cleanup")
                _cleanup_memory()
            
            # 3. Periodic memory cleanup
            if (datetime.now() - last_memory_cleanup).total_seconds() > memory_cleanup_interval:
                _cleanup_memory()
                last_memory_cleanup = datetime.now()
            
            # 4. Save state periodically (every 5 minutes)
            if (datetime.now() - last_state_save).total_seconds() > 300:
                _save_state()
                last_state_save = datetime.now()
            
            # 5. Check heartbeat timeout
            if _bot_status.get("running") and _last_successful_cycle:
                last_cycle_age = (datetime.now() - _last_successful_cycle).total_seconds()
                timeout = _stability_config.get("heartbeat_timeout_seconds", 120)
                
                if last_cycle_age > timeout:
                    logger.warning(f"🐕 WATCHDOG: No successful cycle for {last_cycle_age:.0f}s - restarting...")
                    _runtime_stats["errors_count"] += 1
                    await _auto_restart_bot()
            
            # 6. 🔥 Daily restart count reset
            _check_daily_restart_reset()
            
            # Update uptime
            started = _runtime_stats.get("started_at")
            if started:
                try:
                    start_dt = datetime.fromisoformat(started)
                    _runtime_stats["total_uptime_seconds"] = int((datetime.now() - start_dt).total_seconds())
                except:
                    pass
            
        except asyncio.CancelledError:
            logger.info("🐕 Watchdog stopped")
            break
        except Exception as e:
            logger.error(f"🐕 Watchdog error: {e}")
            await asyncio.sleep(10)


def _check_daily_restart_reset():
    """🔥 Reset restart count ทุกวัน (เที่ยงคืน)"""
    global _runtime_stats, _stability_config
    
    if not _stability_config.get("daily_restart_count_reset", True):
        return
    
    today = datetime.now().date().isoformat()
    last_reset = _runtime_stats.get("last_daily_reset")
    
    if last_reset != today:
        old_count = _runtime_stats.get("restart_count_today", 0)
        _runtime_stats["restart_count_today"] = 0
        _runtime_stats["last_daily_reset"] = today
        if old_count > 0:
            logger.info(f"🔄 Daily reset: Cleared {old_count} restart count for new day")


async def _auto_restart_bot():
    """
    🔄 AUTO-RESTART - เปิด bot ใหม่อัตโนมัติเมื่อ crash
    
    🔥 UNLIMITED MODE: max_restart_attempts = 0 หมายถึงไม่จำกัด
    """
    global _bot, _bot_task, _bot_status, _runtime_stats, _stability_config
    
    max_attempts = _stability_config.get("max_restart_attempts", 0)
    cooldown = _stability_config.get("restart_cooldown_seconds", 30)
    
    # 🔥 UNLIMITED MODE: 0 = no limit
    if max_attempts > 0 and _runtime_stats["restart_count"] >= max_attempts:
        logger.error(f"🔄 AUTO-RESTART: Max attempts ({max_attempts}) reached - giving up")
        return False
    
    # Show restart info
    if max_attempts == 0:
        logger.info(f"🔄 AUTO-RESTART: Attempt #{_runtime_stats['restart_count'] + 1} (UNLIMITED mode)")
    else:
        logger.info(f"🔄 AUTO-RESTART: Attempt {_runtime_stats['restart_count'] + 1}/{max_attempts}")
    
    
    # Wait cooldown
    logger.info(f"🔄 Waiting {cooldown}s before restart...")
    await asyncio.sleep(cooldown)
    
    try:
        # Stop old task if exists
        if _bot_task and not _bot_task.done():
            _bot_task.cancel()
            try:
                await _bot_task
            except asyncio.CancelledError:
                pass
        
        # Get saved settings
        symbols = _bot_status.get("symbols", ["XAUUSDm"])
        timeframe = _bot_status.get("timeframe", "H1")
        signal_mode = _bot_status.get("signal_mode", "technical")
        quality = _bot_status.get("quality", "MEDIUM")
        interval = _bot_status.get("interval", 60)
        auto_trade = _bot_status.get("auto_trade", False)
        
        # Reinitialize bot
        from ai_trading_bot import AITradingBot, SignalQuality
        quality_map = {
            "LOW": SignalQuality.LOW,
            "MEDIUM": SignalQuality.MEDIUM,
            "HIGH": SignalQuality.HIGH,
            "PREMIUM": SignalQuality.PREMIUM
        }
        quality_enum = quality_map.get(quality.upper(), SignalQuality.MEDIUM)
        
        
        _bot = AITradingBot(
            symbols=symbols,
            timeframe=timeframe,
            min_quality=quality_enum,
            broker_type="MT5",
            signal_mode=signal_mode
        )
        
        await _bot.initialize()
        
        # Restart loop
        _bot_task = asyncio.create_task(
            _run_bot_loop(interval, auto_trade)
        )
        
        _bot_status["running"] = True
        _bot_status["initialized"] = True
        _bot_status["error"] = None
        
        _runtime_stats["restart_count"] += 1
        _runtime_stats["restart_count_today"] = _runtime_stats.get("restart_count_today", 0) + 1
        _runtime_stats["last_restart_time"] = datetime.now().isoformat()
        _runtime_stats["recoveries_count"] += 1
        
        logger.info(f"✅ AUTO-RESTART successful! (Total: {_runtime_stats['restart_count']}, Today: {_runtime_stats['restart_count_today']})")
        
        # Save state
        _save_state()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AUTO-RESTART failed: {e}")
        _bot_status["error"] = f"Auto-restart failed: {e}"
        _runtime_stats["restart_count"] += 1
        _runtime_stats["restart_count_today"] = _runtime_stats.get("restart_count_today", 0) + 1
        return False


def _start_watchdog():
    """🐕 Start the watchdog task"""
    global _watchdog_task
    
    if _watchdog_task is None or _watchdog_task.done():
        _watchdog_task = asyncio.create_task(_watchdog_loop())
        logger.info("🐕 Watchdog task started")


def _stop_watchdog():
    """🐕 Stop the watchdog task"""
    global _watchdog_task
    
    if _watchdog_task and not _watchdog_task.done():
        _watchdog_task.cancel()
        logger.info("🐕 Watchdog task stopped")


# =====================
# REQUEST MODELS
# =====================

class StartBotRequest(BaseModel):
    """Request to start unified bot"""
    mode: str = Field(default="manual", description="'auto' for auto-trade, 'manual' for analysis only")
    symbols: str = Field(default="XAUUSDm", description="Comma-separated symbols")
    timeframe: str = Field(default="H1")
    signal_mode: str = Field(default="technical", description="'technical' or 'pattern' (FAISS)")
    quality: str = Field(default="MEDIUM", description="LOW, MEDIUM, HIGH, PREMIUM")
    interval: int = Field(default=60, ge=10, le=3600, description="Analysis interval in seconds")


class SwitchModeRequest(BaseModel):
    """Request to switch bot mode"""
    mode: str = Field(..., description="'auto' or 'manual'")


class ManualTradeRequest(BaseModel):
    """Request for manual trade execution"""
    symbol: str
    side: str  # BUY or SELL
    lot_size: float = Field(default=0.01, ge=0.01, le=10.0)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


# =====================
# HELPER FUNCTIONS
# =====================

# 🔄 Track positions for sync detection
_known_positions = {}  # {ticket: {"symbol": "XAUUSDm", "side": "BUY", "open_price": 5100, "open_time": datetime}}

# 📊 Trade history for accurate tracking
_trade_history = []  # [{"ticket": 123, "symbol": "XAUUSDm", "side": "BUY", "pnl": 150.0, "close_time": datetime, "close_reason": "TP"}]


async def _get_closed_deal_pnl(ticket: int) -> Optional[Dict]:
    """
    📊 ดึง PnL ของ deal ที่ปิดไปแล้วจาก MT5 history
    
    Returns: {"pnl": 150.0, "close_price": 5320, "close_reason": "TP/SL"}
    """
    global _bot
    
    if not _bot or not _bot.trading_engine or not _bot.trading_engine.broker:
        return None
    
    try:
        broker = _bot.trading_engine.broker
        if not hasattr(broker, '_mt5') or not broker._mt5:
            return None
        
        mt5 = broker._mt5
        
        # Get deals from history (last 24 hours)
        from datetime import timedelta
        from_date = datetime.now() - timedelta(days=1)
        to_date = datetime.now() + timedelta(hours=1)
        
        deals = mt5.history_deals_get(from_date, to_date, position=ticket)
        
        if not deals:
            # Try getting by ticket directly
            deals = mt5.history_deals_get(from_date, to_date)
            if deals:
                deals = [d for d in deals if d.position_id == ticket]
        
        if not deals:
            return None
        
        # Find the closing deal (entry=0 for open, entry=1 for close)
        close_deals = [d for d in deals if d.entry == 1]  # 1 = Deal out (close)
        
        if close_deals:
            close_deal = close_deals[-1]  # Latest close deal
            
            # Determine close reason
            close_reason = "UNKNOWN"
            comment = close_deal.comment.lower() if close_deal.comment else ""
            
            if "tp" in comment or "take profit" in comment:
                close_reason = "TP"
            elif "sl" in comment or "stop loss" in comment:
                close_reason = "SL"
            elif "close" in comment:
                close_reason = "MANUAL"
            else:
                close_reason = "EXTERNAL"
            
            return {
                "pnl": close_deal.profit,
                "close_price": close_deal.price,
                "close_reason": close_reason,
                "commission": close_deal.commission,
                "swap": close_deal.swap,
                "close_time": datetime.fromtimestamp(close_deal.time),
            }
        
        # If no close deal found, sum up all deals for this position
        total_pnl = sum(d.profit for d in deals)
        return {
            "pnl": total_pnl,
            "close_price": deals[-1].price if deals else 0,
            "close_reason": "CALCULATED",
            "close_time": datetime.now(),
        }
        
    except Exception as e:
        logger.warning(f"Error getting deal PnL for ticket {ticket}: {e}")
        return None


async def _sync_positions_with_mt5():
    """
    🔄 SYNC WITH MT5 - ตรวจสอบว่า position ถูกปิดไปแล้วหรือยัง
    
    🔥 ENHANCED: ดึง PnL จริงจาก MT5 history เพื่อ track win/loss อย่างแม่นยำ!
    
    Logic:
    - SL hit แต่ได้กำไร (trailing stop) → WIN ✅
    - TP hit → WIN ✅
    - SL hit และขาดทุน → LOSS ❌
    - Manual close กำไร → WIN ✅
    - Manual close ขาดทุน → LOSS ❌
    """
    global _bot, _known_positions, _bot_status, _peak_profit_by_position, _last_traded_signal, _trade_history
    
    if not _bot or not _bot.trading_engine:
        return
    
    try:
        # Get actual positions from MT5 (FRESH DATA)
        positions = await _bot.trading_engine.broker.get_positions()
        
        # Build set of current position tickets AND symbols
        current_tickets = set()
        current_symbols = set()
        
        for pos in (positions or []):
            if isinstance(pos, dict):
                ticket = pos.get("ticket") or pos.get("id")
                symbol = pos.get("symbol", "")
            else:
                ticket = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                symbol = getattr(pos, "symbol", "")
            
            if ticket:
                current_tickets.add(str(ticket))
            if symbol:
                current_symbols.add(symbol.upper())
        
        # 🔥 DEBUG: Log current MT5 state
        logger.debug(f"🔄 MT5 SYNC: {len(positions or [])} positions, tickets: {current_tickets}, symbols: {current_symbols}")
        
        # Check for positions that were closed externally
        closed_externally = []
        for ticket, info in list(_known_positions.items()):
            if str(ticket) not in current_tickets:
                # Position was closed externally (SL/TP hit)
                closed_symbol = info.get("symbol", "")
                closed_side = info.get("side", "")
                
                # 🔥 NEW: Get actual PnL from MT5 history!
                deal_info = await _get_closed_deal_pnl(int(ticket))
                actual_pnl = 0.0
                close_reason = "EXTERNAL"
                
                if deal_info:
                    actual_pnl = deal_info.get("pnl", 0)
                    close_reason = deal_info.get("close_reason", "EXTERNAL")
                    
                    # 📊 UPDATE DAILY STATS BASED ON PnL (NOT SL/TP!)
                    _bot_status["daily_stats"]["trades"] += 1
                    _bot_status["daily_stats"]["pnl"] += actual_pnl
                    
                    # ✅ WIN = กำไร (ไม่ว่าจะปิดด้วยอะไร!)
                    # ❌ LOSS = ขาดทุน
                    if actual_pnl > 0:
                        _bot_status["daily_stats"]["wins"] += 1
                        _update_loss_streak(is_win=True)  # 🆕 Reset loss streak
                        logger.info(f"✅ WIN: #{ticket} {closed_symbol} {closed_side} | PnL: +${actual_pnl:.2f} | Reason: {close_reason}")
                    elif actual_pnl < 0:
                        _bot_status["daily_stats"]["losses"] += 1
                        _update_loss_streak(is_win=False)  # 🆕 Increment loss streak
                        logger.info(f"❌ LOSS: #{ticket} {closed_symbol} {closed_side} | PnL: ${actual_pnl:.2f} | Reason: {close_reason}")
                    else:
                        logger.info(f"➖ BREAKEVEN: #{ticket} {closed_symbol} {closed_side} | PnL: $0.00 | Reason: {close_reason}")
                    
                    # 📝 Save to trade history
                    _trade_history.append({
                        "ticket": ticket,
                        "symbol": closed_symbol,
                        "side": closed_side,
                        "pnl": actual_pnl,
                        "close_reason": close_reason,
                        "close_time": deal_info.get("close_time", datetime.now()).isoformat(),
                        "is_win": actual_pnl > 0,
                    })
                    
                    # Keep only last 100 trades
                    if len(_trade_history) > 100:
                        _trade_history.pop(0)
                else:
                    logger.warning(f"⚠️ Could not get PnL for closed position #{ticket} - stats not updated")
                
                closed_externally.append({
                    "ticket": ticket,
                    "symbol": closed_symbol,
                    "side": closed_side,
                    "pnl": actual_pnl,
                    "close_reason": close_reason,
                })
                
                # Clean up tracking
                del _known_positions[ticket]
                
                # Clean up peak profit tracking
                if ticket in _peak_profit_by_position:
                    del _peak_profit_by_position[ticket]
                
                # 🔥 CRITICAL: Reset cooldown for this symbol so bot can trade again
                if closed_symbol and closed_symbol in _last_traded_signal:
                    del _last_traded_signal[closed_symbol]
                    logger.info(f"🔓 Reset cooldown for {closed_symbol} - position closed externally")
                
                # Also try uppercase version
                if closed_symbol and closed_symbol.upper() in _last_traded_signal:
                    del _last_traded_signal[closed_symbol.upper()]
        
        # Log summary of closed positions
        if closed_externally:
            total_pnl = sum(p.get("pnl", 0) for p in closed_externally)
            wins = sum(1 for p in closed_externally if p.get("pnl", 0) > 0)
            losses = sum(1 for p in closed_externally if p.get("pnl", 0) < 0)
            logger.info(f"📊 SYNC SUMMARY: {len(closed_externally)} positions closed | W:{wins} L:{losses} | PnL: ${total_pnl:.2f}")
        
        # 🔥 NEW: Also check if _known_positions has symbols that MT5 doesn't have
        # This catches cases where position was opened by bot but closed externally
        symbols_in_known = set(info.get("symbol", "").upper() for info in _known_positions.values())
        orphan_symbols = symbols_in_known - current_symbols
        
        for orphan_symbol in orphan_symbols:
            logger.warning(f"⚠️ ORPHAN DETECTED: {orphan_symbol} in _known_positions but not in MT5!")
            # Find and remove orphan entries
            tickets_to_remove = []
            for ticket, info in _known_positions.items():
                if info.get("symbol", "").upper() == orphan_symbol:
                    tickets_to_remove.append(ticket)
            for ticket in tickets_to_remove:
                del _known_positions[ticket]
                logger.info(f"🧹 Removed orphan position #{ticket} ({orphan_symbol})")
            # Reset cooldown
            if orphan_symbol in _last_traded_signal:
                del _last_traded_signal[orphan_symbol]
            if orphan_symbol.upper() in _last_traded_signal:
                del _last_traded_signal[orphan_symbol.upper()]
        
        # 🔥 CRITICAL FIX: If MT5 has 0 positions, clear ALL tracking data
        if len(positions or []) == 0 and len(_known_positions) > 0:
            logger.warning(f"🧹 MT5 has 0 positions but _known_positions has {len(_known_positions)} - CLEARING ALL!")
            
            # Clear all known positions
            _known_positions.clear()
            
            # Clear all cooldowns so bot can trade
            cleared_symbols = list(_last_traded_signal.keys())
            _last_traded_signal.clear()
            
            # Clear peak profits
            _peak_profit_by_position.clear()
            
            logger.info(f"✅ Cleared tracking data: known_positions, cooldowns ({cleared_symbols}), peak_profits")
        
        # Update known positions with current positions
        for pos in (positions or []):
            if isinstance(pos, dict):
                ticket = pos.get("ticket") or pos.get("id")
                symbol = pos.get("symbol", "")
                side = pos.get("side", "")
            else:
                ticket = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                symbol = getattr(pos, "symbol", "")
                side = getattr(pos, "side", "")
                if hasattr(side, "value"):
                    side = side.value
            
            if ticket and str(ticket) not in _known_positions:
                _known_positions[str(ticket)] = {
                    "symbol": symbol,
                    "side": str(side).upper(),
                }
                logger.info(f"📥 New position tracked: #{ticket} ({symbol} {side})")
        
        # 🔥 CRITICAL: Also sync trading_engine.positions cache
        # This ensures ai_trading_bot.py sees the correct position state
        if _bot and _bot.trading_engine and hasattr(_bot.trading_engine, 'sync_with_broker'):
            try:
                sync_result = await _bot.trading_engine.sync_with_broker()
                if sync_result.get("removed"):
                    for removed in sync_result["removed"]:
                        logger.info(f"🔄 TradingEngine sync: Removed {removed['symbol']} from cache")
                if sync_result.get("added"):
                    for added in sync_result["added"]:
                        logger.info(f"🔄 TradingEngine sync: Added {added['symbol']} to cache")
            except Exception as sync_err:
                logger.debug(f"TradingEngine sync: {sync_err}")
        
    except Exception as e:
        logger.warning(f"Sync with MT5 failed: {e}")


def _get_bot_instance():
    """Get or create bot instance"""
    global _bot
    if _bot is None:
        from ai_trading_bot import AITradingBot, SignalQuality
        _bot = AITradingBot(
            symbols=["XAUUSDm"],
            timeframe="H1",
            min_quality=SignalQuality.MEDIUM,
            broker_type="MT5",
            signal_mode="technical"
        )
    return _bot


async def _reinitialize_bot():
    """Reinitialize bot when MT5 connection fails repeatedly"""
    global _bot, _bot_status
    
    logger.info("Reinitializing bot...")
    
    try:
        symbols = _bot_status.get("symbols", ["XAUUSDm"])
        timeframe = _bot_status.get("timeframe", "H1")
        signal_mode = _bot_status.get("signal_mode", "technical")
        quality = _bot_status.get("quality", "MEDIUM")
        
        if _bot:
            try:
                await _bot.stop()
            except:
                pass
        
        from ai_trading_bot import AITradingBot, SignalQuality
        quality_map = {
            "LOW": SignalQuality.LOW,
            "MEDIUM": SignalQuality.MEDIUM,
            "HIGH": SignalQuality.HIGH,
            "PREMIUM": SignalQuality.PREMIUM
        }
        quality_enum = quality_map.get(quality.upper(), SignalQuality.MEDIUM)
        
        _bot = AITradingBot(
            symbols=symbols,
            timeframe=timeframe,
            min_quality=quality_enum,
            broker_type="MT5",
            signal_mode=signal_mode
        )
        
        await _bot.initialize()
        _bot_status["initialized"] = True
        _bot_status["error"] = None
        
        logger.info("Bot reinitialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to reinitialize bot: {e}")
        _bot_status["error"] = f"Reinitialization failed: {e}"


async def _analyze_single_symbol(symbol: str, auto_trade: bool) -> Optional[Dict]:
    """
    🚀 PARALLEL ANALYSIS - วิเคราะห์ symbol เดียว (ใช้กับ asyncio.gather)
    
    Returns: analysis result หรือ None
    """
    global _bot, _bot_status
    
    try:
        # Run analysis
        analysis = await _bot.analyze_symbol(symbol)
        
        if not analysis:
            return None
        
        # Store analysis
        _bot_status["last_analysis"][symbol] = analysis
        
        # Extract signal - try multiple confidence fields
        raw_confidence = analysis.get("enhanced_confidence", 0) or analysis.get("base_confidence", 0) or analysis.get("confidence", 0)
        
        # 🔥 Get current_price from analysis or market_data
        current_price = analysis.get("current_price", 0)
        if current_price == 0 and "market_data" in analysis:
            current_price = analysis["market_data"].get("close", 0)
        
        # 🔥 Extract buy_score and sell_score from analysis
        scores = analysis.get("scores", {})
        
        
        # 🔥 PRIORITY: Try to get values directly from analysis first (new format)
        buy_score = analysis.get("buy_score", 0)
        sell_score = analysis.get("sell_score", 0)
        session = analysis.get("session", "N/A")
        trend = analysis.get("market_regime", "UNKNOWN")
        
        # Fallback: Try to extract from factors (old format)
        factors = analysis.get("factors", {})
        if buy_score == 0 and sell_score == 0:
            if factors.get("bullish"):
                for f in factors["bullish"]:
                    if "Buy Score" in str(f):
                        try:
                            match = str(f).split(":")[1].split("/")[0].strip()
                            buy_score = int(match)
                        except:
                            pass
                    if session == "N/A" and "Session" in str(f):
                        try:
                            session = str(f).split(":")[1].strip()
                        except:
                            pass
            
            if factors.get("bearish"):
                for f in factors["bearish"]:
                    if "Sell Score" in str(f):
                        try:
                            match = str(f).split(":")[1].split("/")[0].strip()
                            sell_score = int(match)
                        except:
                            pass
        
        # Last fallback: calculate from pattern score
        if buy_score == 0 and sell_score == 0:
            pattern_score = scores.get("pattern", 0)
            signal_type = analysis.get("signal", "WAIT")
            if signal_type in ["BUY", "STRONG_BUY"]:
                buy_score = max(1, pattern_score // 10) if pattern_score > 0 else 0
            elif signal_type in ["SELL", "STRONG_SELL"]:
                sell_score = max(1, pattern_score // 10) if pattern_score > 0 else 0
        
        # Get session from market_data if still N/A
        if session == "N/A":
            session = analysis.get("market_data", {}).get("session", "N/A")
        
        
        signal_data = {
            "symbol": symbol,
            "signal": analysis.get("signal", "WAIT"),
            "confidence": raw_confidence,
            "quality": analysis.get("quality", "SKIP"),
            "current_price": current_price,
            "stop_loss": analysis.get("risk_management", {}).get("stop_loss") or 0,
            "take_profit": analysis.get("risk_management", {}).get("take_profit") or 0,
            "scores": scores,
            "indicators": analysis.get("indicators", {}),
            "market_regime": trend,
            "market_data": analysis.get("market_data", {}),
            "timestamp": datetime.now().isoformat(),
            # 🔥 NEW: Add explicit fields for frontend
            "buy_score": buy_score,
            "sell_score": sell_score,
            "session": session,
            "trend": trend,
            "factors": factors,  # 🔥 Pass factors to frontend
        }
        _bot_status["last_signal"][symbol] = signal_data
        
        # ⚡ TRACK SIGNAL HISTORY for momentum detection
        _track_signal_history(symbol, signal_data)
        
        # 📊 TRACK SIGNAL STRENGTH for DCA safety
        _track_signal_strength(symbol, signal_data)
        
        # 🔔 CALCULATE SIGNAL HEALTH & MOMENTUM (NEW!)
        signal_health = _calculate_signal_momentum(symbol, raw_confidence)
        
        # Add health info to signal_data
        signal_data["health"] = {
            "momentum": signal_health.get("momentum", "UNKNOWN"),
            "trend": signal_health.get("trend", "UNKNOWN"),
            "alert_level": signal_health.get("alert_level", "OK"),
            "peak_confidence": signal_health.get("peak_confidence", raw_confidence),
            "peak_drop_percent": signal_health.get("peak_drop_percent", 0),
            "is_fading": signal_health.get("momentum") in ["FALLING", "FADING"],
        }
        _bot_status["last_signal"][symbol] = signal_data  # Update with health
        
        # 🔔 LOG WARNING if signal is fading
        if signal_health.get("alert_level") == "DANGER":
            logger.warning(f"⚠️ SIGNAL DANGER: {symbol} confidence dropped {signal_health['peak_drop_percent']:.1f}% from peak!")
        elif signal_health.get("alert_level") == "WARNING":
            logger.warning(f"📉 SIGNAL WARNING: {symbol} - {signal_health.get('momentum', 'UNKNOWN')} momentum, trend: {signal_health.get('trend', 'UNKNOWN')}")
        
        # 🤖 AUTO-ACTION: จัดการ position อัตโนมัติเมื่อ signal fading!
        if signal_health.get("alert_level") in ["WARNING", "DANGER"]:
            auto_action_result = await _handle_signal_fade_auto_action(symbol, signal_health)
            if auto_action_result.get("actions_taken"):
                for action in auto_action_result["actions_taken"]:
                    logger.info(f"🤖 AUTO-ACTION: {action.get('action')} - {action.get('reason')}")
        
        # 📊 LOG SIGNAL STRENGTH SCORE
        strength = _get_signal_strength_score(symbol, signal_data)
        if strength["score"] < 50:
            logger.warning(f"⚠️ {symbol}: WEAK SIGNAL! Score={strength['score']}/100 - {strength['recommendation']}")
        
        # ⚡ CHECK SIGNAL WEAKENING - ปิด position ก่อนสัญญาณกลับทิศ
        await _check_and_close_weakening_positions(symbol, signal_data)
        
        # Extract layer status
        _bot_status["layer_status"][symbol] = _extract_layer_status(symbol)
        
        logger.info(f"📊 {symbol}: {signal_data['signal']} @ {signal_data['confidence']:.1f}% ({_bot_status['mode']} mode)")
        
        # 🔄 REVERSE SIGNAL CLOSE + OPEN NEW - ปิด position เดิม + เปิดใหม่ตามสัญญาณ
        closed_opposite = False
        if signal_data["signal"] in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
            closed_opposite = await _check_and_close_opposite_positions(symbol, signal_data["signal"])
            if closed_opposite:
                _bot_status["last_signal"][symbol]["trade_status"] = "REVERSED"
                logger.info(f"   🔄 {symbol}: Opposite position closed due to reverse signal")
                
                # 🔥 NEW: Wait a moment then open new position in new direction
                if _open_new_after_close and auto_trade and _bot_status["mode"] == BotMode.AUTO.value:
                    await asyncio.sleep(1)  # รอ 1 วินาทีให้ MT5 update
                    logger.info(f"   🎯 {symbol}: Opening NEW position in direction {signal_data['signal']}")
                    # Skip position check because we just closed it!
                    await _execute_signal_trade(symbol, signal_data, skip_position_check=True)
                    _bot_status["last_signal"][symbol]["trade_status"] = "REVERSED_AND_OPENED"
        
        # 🚨 WAIT SIGNAL = CLOSE PROFITABLE - ถ้าสัญญาณเป็น WAIT และมีกำไร → ปิดทันที
        elif signal_data["signal"] in ["WAIT", "SKIP"]:
            closed = await _close_profitable_on_wait_signal(symbol)
            if closed:
                _bot_status["last_signal"][symbol]["trade_status"] = "CLOSED_ON_WAIT"
                logger.info(f"   🚨 {symbol}: Profitable position closed due to WAIT signal")
        
        
        # Auto trade ONLY if mode is AUTO (and not already handled by reverse)
        if auto_trade and _bot_status["mode"] == BotMode.AUTO.value and not closed_opposite:
            if signal_data["signal"] in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
                # 🔍 CHECK SIGNAL HEALTH before trading!
                health_ok, health_reason = _check_signal_health_allows_trading(symbol)
                if not health_ok:
                    logger.warning(f"   🤖 {symbol}: Trade blocked by Signal Health - {health_reason}")
                    _bot_status["last_signal"][symbol]["trade_status"] = f"BLOCKED: {health_reason}"
                    return signal_data  # Skip trade
                
                # Check if can trade before attempting
                can_trade, reason = await _can_trade_signal(symbol, signal_data)
                
                # 🔥 DEBUG: Log why trade is blocked
                logger.info(f"   🔍 {symbol}: can_trade={can_trade}, reason={reason}")
                
                if can_trade:
                    await _execute_signal_trade(symbol, signal_data)
                    # Update signal status
                    _bot_status["last_signal"][symbol]["trade_status"] = "EXECUTED"
                else:
                    # DCA ถูกปิดแล้ว ไม่ต้องเช็ค
                    logger.info(f"   ❌ {symbol}: Trade blocked - {reason}")
                    _bot_status["last_signal"][symbol]["trade_status"] = f"BLOCKED: {reason}"
            else:
                _bot_status["last_signal"][symbol]["trade_status"] = "NO_SIGNAL"
        elif signal_data["signal"] not in ["WAIT", "SKIP"] and not closed_opposite:
            logger.info(f"   📋 Signal available but mode is MANUAL - not auto-trading")
            _bot_status["last_signal"][symbol]["trade_status"] = "MANUAL_MODE"
        
        return signal_data
        
    except Exception as e:
        logger.error(f"❌ Error analyzing {symbol}: {e}")
        return None


async def _run_bot_loop(interval: int, auto_trade: bool):
    """
    🚀 ENTERPRISE GRADE Bot Analysis Loop - PARALLEL MODE!
    
    Features:
    - 🔥 PARALLEL ANALYSIS - วิเคราะห์ทุก symbol พร้อมกัน!
    - Auto-reconnect on MT5 disconnect
    - Heartbeat tracking for watchdog
    - Error recovery
    - Memory-efficient
    """
    global _bot, _bot_status, _last_successful_cycle, _runtime_stats
    
    mode_str = "AUTO" if auto_trade else "MANUAL"
    logger.info(f"🚀 Unified bot loop starting (mode={mode_str}, interval={interval}s) - PARALLEL MODE!")
    
    consecutive_failures = 0
    max_failures = 5
    cycle_count = 0
    
    # Start watchdog
    _start_watchdog()
    
    while _bot_status["running"]:
        cycle_count += 1
        cycle_start = datetime.now()
        
        try:
            # 🔥 CHECK DAILY RESET - Reset stats ทุกวันใหม่
            _check_and_reset_daily_stats()
            
            # Check MT5 connection before each cycle
            mt5_ok = True
            if _bot and _bot.trading_engine and _bot.trading_engine.broker:
                broker = _bot.trading_engine.broker
                if hasattr(broker, 'ensure_connected'):
                    mt5_ok = broker.ensure_connected()
                    if not mt5_ok:
                        logger.warning("MT5 not connected - waiting for reconnect...")
                        _bot_status["error"] = "MT5 disconnected - attempting reconnect"
                        consecutive_failures += 1
                        
                        if consecutive_failures >= max_failures:
                            logger.error(f"{max_failures} consecutive failures - reinitializing bot...")
                            await _reinitialize_bot()
                            consecutive_failures = 0
                        
                        await asyncio.sleep(10)
                        continue
                    else:
                        if _bot_status.get("error") == "MT5 disconnected - attempting reconnect":
                            _bot_status["error"] = None
                            logger.info("MT5 reconnected successfully!")
                        consecutive_failures = 0
            
            # 🔄 SYNC WITH MT5 - ตรวจสอบว่า position ถูกปิดไปแล้วหรือยัง (SL/TP hit externally)
            await _sync_positions_with_mt5()
            
            # 🔥 VALIDATE TRACKING DATA - ตรวจสอบว่า tracking ตรงกับ MT5 จริง
            await _validate_and_cleanup_tracking()
            
            # 🚀 PARALLEL ANALYSIS - วิเคราะห์ทุก symbol พร้อมกัน!
            analysis_tasks = [
                _analyze_single_symbol(symbol, auto_trade) 
                for symbol in _bot_status["symbols"]
            ]
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Log any errors from parallel analysis
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Analysis error for {_bot_status['symbols'][i]}: {result}")
            
            # 📈 UPDATE DCA TRACKING - ล้าง tracking ของ symbols ที่ไม่มี position แล้ว
            await _update_dca_tracking_from_positions()
            
            # 📈 AUTO TRAILING STOP - ยก SL ตามราคาอัตโนมัติ!
            await _auto_trailing_stop()
            
            # 💰 SMART PROFIT PROTECTION - ตรวจสอบทุก cycle
            closed = await _check_profit_protection()
            if closed:
                for pos in closed:
                    logger.info(f"🛡️ Profit protected: {pos['symbol']} locked ${pos['locked_profit']:.2f}")
            
            # ✅ SUCCESSFUL CYCLE - Update heartbeat
            _last_successful_cycle = datetime.now()
            consecutive_failures = 0  # Reset on success
            _circuit_breaker_record_success()  # 🆕 Circuit breaker success
            
            # 📊 Log cycle stats periodically (every 10 cycles)
            if cycle_count % 10 == 0:
                uptime = _runtime_stats.get("total_uptime_seconds", 0)
                restarts = _runtime_stats.get("restart_count", 0)
                cb_state = _circuit_breaker.get("state", "CLOSED")
                logger.info(f"📊 Cycle #{cycle_count} | Uptime: {uptime//3600}h {(uptime%3600)//60}m | Restarts: {restarts} | Circuit: {cb_state}")
            
            # Wait for next cycle
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            logger.info("🛑 Bot loop cancelled")
            break
        except OSError as e:
            # 🔥 Network error - รอแล้วลองใหม่
            consecutive_failures += 1
            _runtime_stats["network_errors"] = _runtime_stats.get("network_errors", 0) + 1
            _circuit_breaker_record_failure()  # 🆕 Circuit breaker
            
            logger.warning(f"⚠️ Network error ({consecutive_failures}/{max_failures}): {e}")
            _bot_status["error"] = f"Network error: {e}"
            _runtime_stats["errors_count"] += 1
            
            if consecutive_failures >= max_failures:
                logger.error(f"🔥 Too many failures - triggering watchdog restart")
                break
            
            # 🆕 Smart wait: ถ้า circuit open รอนานกว่า
            wait_time = 10 if _circuit_breaker_can_proceed() else 30
            await asyncio.sleep(wait_time)
            
        except ConnectionError as e:
            # 🔥 Connection lost - รอแล้วลองใหม่
            consecutive_failures += 1
            _circuit_breaker_record_failure()  # 🆕 Circuit breaker
            
            logger.warning(f"⚠️ Connection error ({consecutive_failures}/{max_failures}): {e}")
            _bot_status["error"] = f"Connection error: {e}"
            _runtime_stats["errors_count"] += 1
            
            if consecutive_failures >= max_failures:
                logger.error(f"🔥 Too many failures - triggering watchdog restart")
                break
            
            wait_time = 10 if _circuit_breaker_can_proceed() else 30
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            # 🔥 Unexpected error
            consecutive_failures += 1
            error_type = type(e).__name__
            _circuit_breaker_record_failure()  # 🆕 Circuit breaker
            
            logger.error(f"❌ Bot loop error ({error_type}) [{consecutive_failures}/{max_failures}]: {e}")
            logger.error(traceback.format_exc())
            _bot_status["error"] = f"{error_type}: {e}"
            _runtime_stats["errors_count"] += 1
            
            if consecutive_failures >= max_failures:
                logger.error(f"🔥 Too many failures - triggering watchdog restart")
                break
            
            wait_time = 5 if _circuit_breaker_can_proceed() else 15
            await asyncio.sleep(wait_time)
    
    # Save state before exit
    _save_state()
    logger.info("🔴 Unified bot loop stopped")




def _extract_layer_status(symbol: str) -> Dict:
    """Extract 20-layer status from bot - now includes results from _run_20_layer_analysis"""
    global _bot, _bot_status
    
    if not _bot:
        return {"layers": [], "passed": 0, "total": 20, "pass_rate": 0}
    
    # 🔥 First, check if analysis has layer_results (from TECHNICAL mode with 20-layer)
    analysis = _bot_status.get("last_analysis", {}).get(symbol, {})
    if "layer_results" in analysis:
        return analysis["layer_results"]
    
    # Fallback: Build layer status from bot attributes
    layers = []
    passed = 0
    total = 20
    
    # Layer 1-4: Base layers
    base_configs = [
        ("data_lake", "Data Lake", 1),
        ("pattern_matcher", "Pattern Matcher", 2),
        ("voting", "Voting System", 3),
        ("enhanced", "Enhanced Analyzer", 4),
    ]
    
    for attr, name, num in base_configs:
        status = "READY" if hasattr(_bot, attr) and getattr(_bot, attr) else "N/A"
        score = 100 if status == "READY" else 0
        layers.append({"layer": num, "name": name, "status": status, "score": score})
        if status == "READY":
            passed += 1
    
    # Layer 5-16: Intelligence modules
    intel_configs = [
        ("_last_intel_result_by_symbol", "Advanced Intelligence", 5),
        ("_last_smart_result_by_symbol", "Smart Brain", 6),
        ("_last_neural_result_by_symbol", "Neural Brain", 7),
        ("_last_deep_result_by_symbol", "Deep Intelligence", 8),
        ("_last_quantum_result_by_symbol", "Quantum Strategy", 9),
        ("_last_alpha_result_by_symbol", "Alpha Engine", 10),
        ("_last_omega_result_by_symbol", "Omega Brain", 11),
        ("_last_titan_decision_by_symbol", "Titan Core", 12),
        ("_last_pro_result_by_symbol", "Pro Features", 13),
        (None, "Risk Guardian", 14),
        (None, "Smart Features", 15),
        (None, "Correlation", 16),
    ]
    
    for attr, name, num in intel_configs:
        if attr and hasattr(_bot, attr):
            result = getattr(_bot, attr, {}).get(symbol, {})
            score = result.get("confidence", result.get("score", 0)) if result else 0
            can_trade = result.get("can_trade", result.get("should_trade", True)) if result else True
            status = "PASS" if can_trade and score > 50 else "FAIL" if not can_trade else "N/A"
        else:
            status = "N/A"
            score = 0
        
        layers.append({"layer": num, "name": name, "status": status, "score": score})
        if status == "PASS":
            passed += 1
    
    # Layer 17-20: Adaptive layers
    adaptive_configs = [
        ("_last_ultra_decision_by_symbol", "Ultra Intelligence", 17),
        ("_last_supreme_decision_by_symbol", "Supreme Intelligence", 18),
        ("_last_transcendent_decision_by_symbol", "Transcendent", 19),
        ("_last_omniscient_decision_by_symbol", "Omniscient", 20),
    ]
    
    for attr, name, num in adaptive_configs:
        if hasattr(_bot, attr):
            result = getattr(_bot, attr, {}).get(symbol, {})
            score = result.get("confidence", 0) if result else 0
            can_trade = result.get("can_trade", True) if result else True
            status = "PASS" if can_trade and score > 50 else "FAIL" if not can_trade else "N/A"
        else:
            status = "N/A"
            score = 0
        
        layers.append({"layer": num, "name": name, "status": status, "score": score})
        if status == "PASS":
            passed += 1
    
    return {
        "layers": layers,
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total * 100) if total > 0 else 0
    }


async def _validate_and_cleanup_tracking():
    """
    🔥 VALIDATE & CLEANUP TRACKING DATA
    
    ตรวจสอบและ cleanup tracking data ให้ตรงกับ MT5 จริง:
    1. ลบ cooldown สำหรับ symbol ที่ไม่มี position
    2. ลบ known_positions ที่ไม่มีใน MT5
    3. Reset ข้อมูลที่ไม่ตรงกัน
    
    เรียกทุก cycle เพื่อให้ระบบ sync ตลอดเวลา
    """
    global _bot, _known_positions, _last_traded_signal, _peak_profit_by_position
    
    if not _bot or not _bot.trading_engine:
        return
    
    try:
        # Get fresh positions from MT5
        positions = await _bot.trading_engine.broker.get_positions()
        
        # Build set of current MT5 data
        mt5_tickets = set()
        mt5_symbols = set()
        
        for pos in (positions or []):
            if isinstance(pos, dict):
                ticket = pos.get("ticket") or pos.get("id")
                symbol = pos.get("symbol", "")
            else:
                ticket = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                symbol = getattr(pos, "symbol", "")
            
            if ticket:
                mt5_tickets.add(str(ticket))
            if symbol:
                mt5_symbols.add(symbol.upper())
        
        # 1. Clean up known_positions not in MT5
        orphan_tickets = []
        for ticket in list(_known_positions.keys()):
            if str(ticket) not in mt5_tickets:
                orphan_tickets.append(ticket)
                symbol = _known_positions[ticket].get("symbol", "")
                del _known_positions[ticket]
                logger.info(f"🧹 Cleaned orphan position #{ticket} ({symbol})")
        
        # 2. Clean up cooldowns for symbols without positions
        # 🔥 FIX: ลด buffer จาก 120 วินาที เป็น 5 วินาที เพื่อให้เทรดได้เร็วขึ้น!
        for symbol in list(_last_traded_signal.keys()):
            if symbol.upper() not in mt5_symbols:
                # Check if cooldown has expired (allow 5s buffer after position close)
                last_time = _last_traded_signal[symbol].get("timestamp")
                if last_time:
                    elapsed = (datetime.now() - last_time).total_seconds()
                    if elapsed > 5:  # 🔥 5 seconds - ให้เทรดได้เร็ว!
                        del _last_traded_signal[symbol]
                        logger.info(f"🧹 Cleaned expired cooldown for {symbol} (no MT5 position)")
                else:
                    # No timestamp - ลบทันที
                    del _last_traded_signal[symbol]
                    logger.info(f"🧹 Cleaned stale cooldown for {symbol} (no timestamp, no MT5 position)")
        
        # 3. Clean up peak profits for closed positions
        for ticket in list(_peak_profit_by_position.keys()):
            if str(ticket) not in mt5_tickets:
                del _peak_profit_by_position[ticket]
        
        # Log if we did any cleanup
        if orphan_tickets:
            logger.info(f"🔄 MT5 SYNC: Cleaned {len(orphan_tickets)} orphan tracking entries")
            
    except Exception as e:
        logger.warning(f"Validation error (non-critical): {e}")


async def _check_profit_protection() -> List[Dict]:
    """
    💰 SMART PROFIT PROTECTION - ล็อกกำไรอัตโนมัติ
    
    Logic:
    1. Monitor ทุก position ที่มีกำไร >= min_profit_to_protect
    2. เก็บ peak profit ของแต่ละ position
    3. ถ้ากำไรลดลง >= profit_drawdown_percent จาก peak → ปิดทันที
    
    Example:
    - min_profit_to_protect = $100
    - profit_drawdown_percent = 30%
    - Position กำไรขึ้นไป $1000 (peak)
    - กำไรลดลงมาเหลือ $700 (drawdown 30%) → ปิด! Lock กำไร $700
    
    Returns: List of closed positions
    """
    global _bot, _profit_protection_config, _peak_profit_by_position, _bot_status
    
    if not _profit_protection_config.get("enabled", False):
        return []
    
    
    if not _bot or not _bot.trading_engine:
        return []
    
    closed_positions = []
    drawdown_pct = _profit_protection_config.get("profit_drawdown_percent", 30)
    
    # 🔥 DYNAMIC: Get balance for dynamic min_profit
    try:
        balance = await _bot.trading_engine.broker.get_balance() or 1000
    except:
        balance = 1000
    min_profit = _get_min_profit_to_protect(balance)
    
    try:
        positions = await _bot.trading_engine.broker.get_positions()
        if not positions:
            return []
        
        for pos in positions:
            # Extract position info
            if isinstance(pos, dict):
                pos_id = pos.get("ticket") or pos.get("id") or pos.get("position_id")
                pos_symbol = pos.get("symbol", "")
                pos_pnl = float(pos.get("profit", 0) or pos.get("pnl", 0))
                pos_side = pos.get("side", "").upper()
            else:
                pos_id = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                pos_symbol = getattr(pos, "symbol", "")
                pos_pnl = float(getattr(pos, "profit", 0) or getattr(pos, "pnl", 0))
                pos_side = getattr(pos, "side", "")
                if hasattr(pos_side, "value"):
                    pos_side = pos_side.value.upper()
            
            if not pos_id:
                continue
            
            # Skip if profit < minimum
            if pos_pnl < min_profit:
                # Clear peak if profit dropped below minimum
                if pos_id in _peak_profit_by_position:
                    del _peak_profit_by_position[pos_id]
                continue
            
            # Update peak profit
            current_peak = _peak_profit_by_position.get(pos_id, pos_pnl)
            if pos_pnl > current_peak:
                _peak_profit_by_position[pos_id] = pos_pnl
                current_peak = pos_pnl
                logger.info(f"📈 {pos_symbol} #{pos_id}: New peak profit ${current_peak:.2f}")
            
            # Check drawdown from peak
            if current_peak > 0:
                drawdown = ((current_peak - pos_pnl) / current_peak) * 100
                
                if drawdown >= drawdown_pct:
                    # PROFIT PROTECTION TRIGGERED!
                    logger.warning(f"🛡️ PROFIT PROTECTION: {pos_symbol} #{pos_id}")
                    logger.warning(f"   Peak: ${current_peak:.2f} → Current: ${pos_pnl:.2f} (Drawdown: {drawdown:.1f}%)")
                    logger.warning(f"   Closing to lock profit ${pos_pnl:.2f}!")
                    
                    try:
                        result = await _bot.trading_engine.broker.close_position(pos_id)
                        if result:
                            logger.info(f"✅ Position #{pos_id} closed! Locked profit: ${pos_pnl:.2f}")
                            
                            # Update stats
                            _bot_status["daily_stats"]["trades"] += 1
                            _bot_status["daily_stats"]["pnl"] += pos_pnl
                            _bot_status["daily_stats"]["wins"] += 1
                            
                            
                            # Clean up peak tracking
                            if pos_id in _peak_profit_by_position:
                                del _peak_profit_by_position[pos_id]
                            
                            closed_positions.append({
                                "ticket": pos_id,
                                "symbol": pos_symbol,
                                "side": pos_side,
                                "peak_profit": current_peak,
                                "locked_profit": pos_pnl,
                                "drawdown_percent": drawdown,
                                "reason": "profit_protection"
                            })
                        else:
                            logger.error(f"❌ Failed to close position #{pos_id}")
                    except Exception as e:
                        logger.error(f"❌ Error closing position #{pos_id}: {e}")
        
        
        return closed_positions
        
    except Exception as e:
        logger.error(f"Error in profit protection check: {e}")
        return []


# Track last SL position for each position (to avoid moving SL backwards)
_last_trailing_sl = {}  # {position_id: last_sl_price}


async def _auto_trailing_stop():
    """
    📈 AUTO TRAILING STOP - ยก SL ตามราคาอัตโนมัติ!
    
    Logic:
    1. เมื่อกำไร >= trigger_profit_usd → เริ่ม trail
    2. ยก SL ให้ห่างจากราคาปัจจุบัน trail_distance_usd
    3. ยกทีละ step_size_usd (ไม่ยกย้อนกลับ!)
    4. เมื่อกำไร >= lock_profit_at_usd → ยก SL เข้ามากำไร (lock profit)
    
    🔥 GOLD Example:
    - trigger: $100, distance: $50, step: $10, lock: $200
    - Entry BUY @ 5500, SL @ 5400
    - Price → 5600 (กำไร $100) → SL ยกเป็น 5550 (ห่าง $50)
    - Price → 5650 (กำไร $150) → SL ยกเป็น 5600 (ห่าง $50)
    - Price → 5700 (กำไร $200) → SL ยกเป็น 5650 (lock กำไร $150!)
    """
    global _bot, _trailing_stop_config, _last_trailing_sl
    
    if not _trailing_stop_config.get("enabled", False):
        return
    
    if not _bot or not _bot.trading_engine:
        return
    
    try:
        positions = await _bot.trading_engine.broker.get_positions()
        if not positions:
            return
        
        # 🔥 DYNAMIC: Get balance for dynamic values
        try:
            balance = await _bot.trading_engine.broker.get_balance() or 1000
        except:
            balance = 1000
        
        trigger_profit = _get_trailing_trigger(balance)
        trail_distance = _get_trailing_distance(balance)
        step_size = _get_trailing_step(balance)
        lock_profit_at = _get_trailing_lock(balance)
        
        for pos in positions:
            try:
                # Extract position info
                if isinstance(pos, dict):
                    pos_id = pos.get("ticket") or pos.get("id")
                    pos_symbol = pos.get("symbol", "")
                    pos_side = pos.get("side", "").upper()
                    pos_pnl = float(pos.get("profit", 0) or 0)
                    pos_sl = float(pos.get("sl", 0) or 0)
                    pos_entry = float(pos.get("open_price", 0) or pos.get("price_open", 0) or 0)
                    current_price = float(pos.get("price_current", 0) or 0)
                else:
                    pos_id = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                    pos_symbol = getattr(pos, "symbol", "")
                    pos_side = getattr(pos, "side", "")
                    if hasattr(pos_side, "value"):
                        pos_side = pos_side.value.upper()
                    pos_pnl = float(getattr(pos, "profit", 0) or 0)
                    pos_sl = float(getattr(pos, "sl", 0) or 0)
                    pos_entry = float(getattr(pos, "open_price", 0) or getattr(pos, "price_open", 0) or 0)
                    current_price = float(getattr(pos, "price_current", 0) or 0)
                
                if not pos_id or not pos_symbol or pos_pnl <= 0:
                    continue
                
                # Skip if profit not enough to trigger
                if pos_pnl < trigger_profit:
                    continue
                
                # Get current price from MT5 if not in position
                if current_price <= 0:
                    current_price = await _bot.trading_engine.broker.get_current_price(pos_symbol)
                    if current_price <= 0:
                        continue
                
                # Determine point value for this symbol (Gold = ~$1 per point)
                point_value = 1.0  # Default
                if 'XAU' in pos_symbol.upper() or 'GOLD' in pos_symbol.upper():
                    point_value = 1.0  # Gold: $1 per point (0.01 lot = $0.01)
                
                # Calculate new SL based on trailing distance
                distance_points = trail_distance / point_value
                
                if pos_side == "BUY":
                    # BUY: SL ต่ำกว่าราคาปัจจุบัน
                    new_sl = current_price - distance_points
                    
                    # ถ้ากำไร >= lock_profit_at → ยก SL เข้ามากำไร
                    if pos_pnl >= lock_profit_at:
                        # Lock กำไรขั้นต่ำ 50% ของ lock_profit_at
                        min_lock_profit = lock_profit_at * 0.5 / point_value
                        new_sl = max(new_sl, pos_entry + min_lock_profit)
                    
                    # ไม่ยก SL ย้อนกลับ!
                    last_sl = _last_trailing_sl.get(pos_id, 0)
                    if new_sl <= last_sl + step_size:
                        continue  # ยังไม่ถึง step size
                    
                    # ต้องยกขึ้นเท่านั้น (ไม่ลง)
                    if pos_sl > 0 and new_sl <= pos_sl:
                        continue
                    
                else:  # SELL
                    # SELL: SL สูงกว่าราคาปัจจุบัน
                    new_sl = current_price + distance_points
                    
                    # ถ้ากำไร >= lock_profit_at → ยก SL เข้ามากำไร
                    if pos_pnl >= lock_profit_at:
                        min_lock_profit = lock_profit_at * 0.5 / point_value
                        new_sl = min(new_sl, pos_entry - min_lock_profit)
                    
                    # ไม่ยก SL ย้อนกลับ!
                    last_sl = _last_trailing_sl.get(pos_id, float('inf'))
                    if new_sl >= last_sl - step_size:
                        continue  # ยังไม่ถึง step size
                    
                    # ต้องยกลงเท่านั้น (ไม่ขึ้น)
                    if pos_sl > 0 and new_sl >= pos_sl:
                        continue
                
                # Round SL to proper precision
                new_sl = round(new_sl, 2)
                
                # Modify position
                logger.info(f"📈 TRAILING STOP: {pos_symbol} #{pos_id} | Profit: ${pos_pnl:.2f}")
                logger.info(f"   Current SL: {pos_sl:.2f} → New SL: {new_sl:.2f} | Price: {current_price:.2f}")
                
                result = await _bot.trading_engine.broker.modify_position(
                    str(pos_id),
                    stop_loss=new_sl
                )
                
                if result and result.success:
                    _last_trailing_sl[pos_id] = new_sl
                    logger.info(f"✅ SL moved: {pos_symbol} #{pos_id} → SL={new_sl:.2f}")
                else:
                    error = result.error if result else "Unknown"
                    logger.warning(f"⚠️ Failed to move SL: {error}")
                    
            except Exception as e:
                logger.warning(f"Error trailing position {pos_id}: {e}")
                
    except Exception as e:
        logger.error(f"Error in auto trailing stop: {e}")


async def _close_profitable_on_wait_signal(symbol: str) -> bool:
    """
    🚨 WAIT SIGNAL = CLOSE PROFITABLE
    
    เมื่อสัญญาณเปลี่ยนเป็น WAIT (ไม่แน่ใจทิศทาง) → ปิด position ที่มีกำไรทันที
    
    Logic:
    - สัญญาณ WAIT = ตลาดไม่มีทิศทางชัดเจน
    - ถ้ามี position ที่กำไร → ปิดทันที เพื่อ lock กำไร
    - ถ้าขาดทุน → ไม่ปิด รอ SL/TP
    
    Returns: True if position was closed, False otherwise
    """
    global _bot, _bot_status, _peak_profit_by_position
    
    if not _bot or not _bot.trading_engine:
        return False
    
    try:
        positions = await _bot.trading_engine.broker.get_positions()
        if not positions:
            return False
        
        closed_any = False
        
        for pos in positions:
            # Extract position info
            if isinstance(pos, dict):
                pos_id = pos.get("ticket") or pos.get("id") or pos.get("position_id")
                pos_symbol = pos.get("symbol", "")
                pos_pnl = float(pos.get("profit", 0) or pos.get("pnl", 0))
                pos_side = pos.get("side", "").upper()
            else:
                pos_id = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                pos_symbol = getattr(pos, "symbol", "")
                pos_pnl = float(getattr(pos, "profit", 0) or getattr(pos, "pnl", 0))
                pos_side = getattr(pos, "side", "")
                if hasattr(pos_side, "value"):
                    pos_side = pos_side.value.upper()
            
            # Check if this position is for the target symbol
            if pos_symbol.upper() != symbol.upper():
                continue
            
            # 🔥 DYNAMIC: Get balance and calculate min profit
            try:
                balance = await _bot.trading_engine.broker.get_balance() or 1000
            except:
                balance = 1000
            min_profit_for_wait = _get_wait_close_profit(balance)
            
            # Only close if profitable AND profit >= minimum
            if pos_pnl <= 0:
                logger.info(f"🚨 WAIT SIGNAL: {symbol} {pos_side} PnL=${pos_pnl:.2f} (loss) → NOT closing")
                continue
            
            if pos_pnl < min_profit_for_wait:
                logger.info(f"🚨 WAIT SIGNAL: {symbol} {pos_side} PnL=${pos_pnl:.2f} < ${min_profit_for_wait:.0f} (5% of ${balance:.0f}) → NOT closing")
                continue
            
            # Close profitable position only if >= minimum
            logger.warning(f"🚨 WAIT SIGNAL CLOSE: {symbol} #{pos_id} | {pos_side} | Profit: ${pos_pnl:.2f} >= ${min_profit_for_wait:.0f}")
            logger.warning(f"   Signal changed to WAIT + High profit → Closing to lock!")
            
            try:
                result = await _bot.trading_engine.broker.close_position(pos_id)
                if result:
                    logger.info(f"✅ Position #{pos_id} closed! Locked profit: ${pos_pnl:.2f}")
                    
                    # Update daily stats
                    _bot_status["daily_stats"]["trades"] += 1
                    _bot_status["daily_stats"]["pnl"] += pos_pnl
                    _bot_status["daily_stats"]["wins"] += 1
                    
                    # Clean up peak tracking
                    if pos_id in _peak_profit_by_position:
                        del _peak_profit_by_position[pos_id]
                    
                    closed_any = True
                else:
                    logger.error(f"❌ Failed to close position #{pos_id}")
            except Exception as e:
                logger.error(f"❌ Error closing position #{pos_id}: {e}")
        
        return closed_any
        
        
    except Exception as e:
        logger.error(f"Error in WAIT signal close: {e}")
        return False




async def _check_and_close_opposite_positions(symbol: str, new_signal: str) -> bool:
    """
    🔄 REVERSE SIGNAL CLOSE - ปิด position เมื่อสัญญาณมาตรงข้าม
    
    Logic (ใหม่):
    - มี SELL position อยู่ + สัญญาณ BUY มา → ปิด SELL ทันที (ไม่ว่ากำไรหรือขาดทุน)
    - มี BUY position อยู่ + สัญญาณ SELL มา → ปิด BUY ทันที (ไม่ว่ากำไรหรือขาดทุน)
    
    🚨 สำคัญ: สัญญาณตรงข้าม = ตลาดเปลี่ยนทิศทาง → ต้องปิดทันที!
    
    Returns: True if position was closed, False otherwise
    """
    global _bot, _enable_reverse_signal_close, _max_loss_config
    
    if not _enable_reverse_signal_close:
        return False
    
    if not _bot or not _bot.trading_engine:
        return False
    
    # Determine signal direction
    is_buy_signal = new_signal in ["BUY", "STRONG_BUY"]
    is_sell_signal = new_signal in ["SELL", "STRONG_SELL"]
    
    if not is_buy_signal and not is_sell_signal:
        return False
    
    # Check if we should close losing positions on reverse signal
    close_on_reverse = _max_loss_config.get("close_on_reverse_signal", True)
    
    try:
        # Get current positions
        positions = await _bot.trading_engine.broker.get_positions()
        if not positions:
            return False
        
        for pos in positions:
            # Handle both dict and Position objects
            if isinstance(pos, dict):
                pos_symbol = pos.get("symbol", "")
                pos_side = pos.get("side", "").upper()
                pos_id = pos.get("ticket") or pos.get("id") or pos.get("position_id")
                pos_pnl = pos.get("profit", 0) or pos.get("pnl", 0)
            else:
                pos_symbol = getattr(pos, "symbol", "")
                pos_side = getattr(pos, "side", "")
                if hasattr(pos_side, "value"):
                    pos_side = pos_side.value.upper()
                else:
                    pos_side = str(pos_side).upper()
                pos_id = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                pos_pnl = getattr(pos, "profit", 0) or getattr(pos, "pnl", 0)
            
            # Check if this position is for the same symbol
            if pos_symbol.upper() != symbol.upper():
                continue
            
            # Check if signal is opposite to position
            is_opposite = False
            
            if pos_side == "BUY" and is_sell_signal:
                is_opposite = True
            elif pos_side == "SELL" and is_buy_signal:
                is_opposite = True
            
            if not is_opposite:
                continue
            
            # 🔥 DYNAMIC: Get minimum profit based on balance (% based)
            try:
                balance = await _bot.trading_engine.broker.get_balance() or 300
            except:
                balance = 300
            min_profit_for_reverse = _get_reverse_signal_min_profit(balance)
            
            # Determine if we should close
            should_close = False
            close_reason = ""
            
            if pos_pnl >= min_profit_for_reverse:
                # กำไร >= min → ปิดเลย ล็อกกำไร!
                should_close = True
                close_reason = f"PROFIT ${pos_pnl:.2f} >= ${min_profit_for_reverse:.0f} (10% of ${balance:.0f}) + reverse signal"
                logger.info(f"✅ REVERSE SIGNAL PROFIT: {symbol} {pos_side} PROFIT ${pos_pnl:.2f} + {new_signal} → CLOSE & LOCK PROFIT!")
            elif pos_pnl > 0 and pos_pnl < min_profit_for_reverse:
                # กำไรน้อย → ไม่ปิด รอกำไรเพิ่ม
                logger.info(f"⏳ REVERSE SIGNAL: {symbol} {pos_side} profit ${pos_pnl:.2f} < ${min_profit_for_reverse:.0f} → HOLD (wait for more profit)")
                continue
            elif pos_pnl <= 0:
                # ❌ ขาดทุน → ไม่ปิด! รอกลับมากำไรก่อน
                logger.info(f"🛑 REVERSE SIGNAL: {symbol} {pos_side} LOSS ${pos_pnl:.2f} + {new_signal} → NOT closing (waiting for profit)")
                continue
            
            if should_close and pos_id:
                logger.info(f"🔄 Closing position #{pos_id} | Reason: {close_reason}")
                
                # Close the position
                try:
                    result = await _bot.trading_engine.broker.close_position(pos_id)
                    if result:
                        logger.info(f"✅ Position #{pos_id} closed! PnL: ${pos_pnl:.2f}")
                        
                        # Update daily stats
                        _bot_status["daily_stats"]["trades"] += 1
                        _bot_status["daily_stats"]["pnl"] += float(pos_pnl)
                        if pos_pnl > 0:
                            _bot_status["daily_stats"]["wins"] += 1
                        else:
                            _bot_status["daily_stats"]["losses"] += 1
                        
                        return True
                    else:
                        logger.warning(f"⚠️ Failed to close position #{pos_id}")
                except Exception as e:
                    logger.error(f"❌ Error closing position #{pos_id}: {e}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking opposite positions: {e}")
        return False


def _get_trade_protection_info() -> Dict:
    """Get trade protection info safely"""
    global _last_traded_signal, _trade_cooldown_seconds
    
    last_trades = {}
    for symbol, data in _last_traded_signal.items():
        try:
            ts = data.get("timestamp")
            if ts and isinstance(ts, datetime):
                elapsed = int((datetime.now() - ts).total_seconds())
                can_trade = elapsed >= _trade_cooldown_seconds
            else:
                elapsed = 0
                can_trade = True
            
            last_trades[symbol] = {
                "signal_id": data.get("signal_id", ""),
                "elapsed": elapsed,
                "can_trade": can_trade
            }
        except Exception as e:
            logger.warning(f"Error getting trade protection for {symbol}: {e}")
            last_trades[symbol] = {"signal_id": "", "elapsed": 0, "can_trade": True}
    
    return {
        "cooldown_seconds": _trade_cooldown_seconds,
        "last_trades": last_trades
    }


def _apply_contrarian_mode(signal: str) -> str:
    """
    🔀 CONTRARIAN MODE - กลับสัญญาณ
    
    ถ้าสัญญาณเดิมผิดบ่อย → กลับสัญญาณ!
    - BUY → SELL
    - SELL → BUY
    - STRONG_BUY → STRONG_SELL
    - STRONG_SELL → STRONG_BUY
    
    Returns: Reversed signal or original signal
    """
    global _contrarian_mode
    
    if not _contrarian_mode.get("enabled", False):
        return signal
    
    # Signal mapping
    signal_map = {
        "BUY": "SELL",
        "SELL": "BUY",
        "STRONG_BUY": "STRONG_SELL",
        "STRONG_SELL": "STRONG_BUY",
    }
    
    # Check if we should reverse this signal
    if signal in ["BUY", "SELL"] and _contrarian_mode.get("reverse_signal", True):
        reversed_signal = signal_map.get(signal, signal)
        logger.info(f"🔀 CONTRARIAN: {signal} → {reversed_signal}")
        return reversed_signal
    
    if signal in ["STRONG_BUY", "STRONG_SELL"] and _contrarian_mode.get("reverse_strong_signal", True):
        reversed_signal = signal_map.get(signal, signal)
        logger.info(f"🔀 CONTRARIAN: {signal} → {reversed_signal}")
        return reversed_signal
    
    return signal


# =====================
# 🎯 PULLBACK ENTRY FUNCTIONS
# =====================

def _check_pullback_entry(symbol: str, signal_data: Dict, current_price: float) -> tuple[bool, str]:
    """
    🎯 PULLBACK ENTRY STRATEGY
    
    สัญญาณมา → รอราคา pullback → รอนิ่ง → ค่อยเข้า
    
    Logic:
    1. BUY signal มา ที่ราคา $2750
    2. รอราคาลง (pullback) เช่น ลงมา $2745 (0.18%)
    3. รอราคาเริ่มนิ่ง/กลับขึ้น
    4. เข้า BUY ที่ราคาดีกว่า
    
    Returns: (can_enter: bool, reason: str)
    """
    global _pullback_config, _pending_signals
    
    if not _pullback_config.get("enabled", False):
        return True, "Pullback disabled - enter immediately"
    
    signal = signal_data.get("signal", "WAIT")
    if signal not in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
        return False, "No valid signal"
    
    is_buy = "BUY" in signal
    pending = _pending_signals.get(symbol)
    
    # First time seeing this signal? Store it and wait
    if not pending or pending.get("signal") != signal:
        _pending_signals[symbol] = {
            "signal": signal,
            "price_at_signal": current_price,
            "timestamp": datetime.now(),
            "pullback_detected": False,
            "lowest_price": current_price if is_buy else current_price,
            "highest_price": current_price if not is_buy else current_price,
            "stable_count": 0
        }
        logger.info(f"🎯 PULLBACK: {symbol} {signal} detected @ {current_price:.2f} - WAITING for pullback...")
        return False, f"New signal - waiting for pullback"
    
    # Check if signal expired
    signal_age = (datetime.now() - pending["timestamp"]).total_seconds() / 60
    max_wait = _pullback_config.get("max_wait_minutes", 30)
    if signal_age > max_wait:
        del _pending_signals[symbol]
        logger.info(f"🎯 PULLBACK: {symbol} signal expired after {max_wait} minutes")
        return False, "Signal expired"
    
    signal_price = pending["price_at_signal"]
    min_pullback_pct = _pullback_config.get("min_pullback_percent", 0.15)
    max_pullback_pct = _pullback_config.get("max_pullback_percent", 1.0)
    
    if is_buy:
        # For BUY: we want price to go DOWN first, then stabilize
        pending["lowest_price"] = min(pending["lowest_price"], current_price)
        pullback_pct = ((signal_price - pending["lowest_price"]) / signal_price) * 100
        
        # Check if pullback exceeded max (signal might be wrong)
        if pullback_pct > max_pullback_pct:
            del _pending_signals[symbol]
            logger.warning(f"🎯 PULLBACK: {symbol} pullback too large ({pullback_pct:.2f}%) - cancelling signal")
            return False, "Pullback too large - signal cancelled"
        
        # Check if minimum pullback achieved
        if pullback_pct < min_pullback_pct:
            return False, f"Waiting for pullback ({pullback_pct:.2f}% < {min_pullback_pct}%)"
        
        # Pullback detected!
        if not pending["pullback_detected"]:
            pending["pullback_detected"] = True
            logger.info(f"🎯 PULLBACK: {symbol} pullback detected ({pullback_pct:.2f}%) - waiting for stabilization")
        
        # Check if price stabilizing (going back up)
        if current_price > pending["lowest_price"]:
            pending["stable_count"] += 1
            required_stable = _pullback_config.get("stabilization_candles", 2)
            
            if pending["stable_count"] >= required_stable:
                logger.info(f"✅ PULLBACK ENTRY: {symbol} {signal} - price stabilized after {pullback_pct:.2f}% pullback")
                del _pending_signals[symbol]
                return True, f"Pullback complete ({pullback_pct:.2f}%)"
            else:
                return False, f"Waiting for stabilization ({pending['stable_count']}/{required_stable})"
        else:
            pending["stable_count"] = 0
            return False, "Price still falling"
    
    else:  # SELL
        # For SELL: we want price to go UP first, then stabilize
        pending["highest_price"] = max(pending["highest_price"], current_price)
        pullback_pct = ((pending["highest_price"] - signal_price) / signal_price) * 100
        
        if pullback_pct > max_pullback_pct:
            del _pending_signals[symbol]
            logger.warning(f"🎯 PULLBACK: {symbol} pullback too large ({pullback_pct:.2f}%) - cancelling signal")
            return False, "Pullback too large - signal cancelled"
        
        if pullback_pct < min_pullback_pct:
            return False, f"Waiting for pullback ({pullback_pct:.2f}% < {min_pullback_pct}%)"
        
        if not pending["pullback_detected"]:
            pending["pullback_detected"] = True
            logger.info(f"🎯 PULLBACK: {symbol} pullback detected ({pullback_pct:.2f}%) - waiting for stabilization")
        
        if current_price < pending["highest_price"]:
            pending["stable_count"] += 1
            required_stable = _pullback_config.get("stabilization_candles", 2)
            
            if pending["stable_count"] >= required_stable:
                logger.info(f"✅ PULLBACK ENTRY: {symbol} {signal} - price stabilized after {pullback_pct:.2f}% pullback")
                del _pending_signals[symbol]
                return True, f"Pullback complete ({pullback_pct:.2f}%)"
            else:
                return False, f"Waiting for stabilization ({pending['stable_count']}/{required_stable})"
        else:
            pending["stable_count"] = 0
            return False, "Price still rising"


def _generate_signal_id(symbol: str, signal: str, confidence: float) -> str:
    """Generate unique signal ID to prevent duplicate trades - AGGRESSIVE VERSION"""
    import hashlib
    global _aggressive_config
    
    # Signal ID based on: symbol + signal direction + confidence band + X-min window
    confidence_band = int(confidence // 5) * 5  # 🔥 Round to 5s (more granular: 65, 70, 75, etc.)
    
    # 🔥 Use configurable window (default 5 minutes for more trades)
    window_minutes = _aggressive_config.get("signal_window_minutes", 5)
    now = datetime.now()
    time_window = f"{now.strftime('%Y%m%d%H')}{now.minute // window_minutes}"
    
    raw = f"{symbol}_{signal}_{confidence_band}_{time_window}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


async def _check_open_positions(symbol: str) -> bool:
    """
    Check if there's already an open position for this symbol
    
    🔥 CRITICAL: ต้อง query MT5 ใหม่ทุกครั้งเพื่อให้ได้ข้อมูลล่าสุด
    🔥 FIX: Force refresh และ log รายละเอียดเพื่อ debug
    🔥 FIX2: Use MT5 directly for most accurate data
    """
    global _bot, _known_positions, _last_traded_signal
    
    if not _bot or not _bot.trading_engine:
        logger.warning(f"📊 _check_open_positions({symbol}): Bot not ready")
        return False
    
    try:
        # 🔥 FORCE REFRESH: Ensure MT5 connection is fresh
        broker = _bot.trading_engine.broker
        if hasattr(broker, 'ensure_connected'):
            broker.ensure_connected()
        
        # 🔥 DIRECT MT5 QUERY: Bypass any caching - get REAL positions
        if hasattr(broker, '_mt5') and broker._mt5:
            mt5 = broker._mt5
            
            # Query symbol-specific positions for accuracy
            symbol_positions = mt5.positions_get(symbol=symbol)
            
            if symbol_positions and len(symbol_positions) > 0:
                pos_count = len(symbol_positions)
                logger.info(f"🔍 MT5 DIRECT QUERY: {symbol} - Found {pos_count} position(s) for this symbol")
                for pos in symbol_positions:
                    logger.info(f"   📍 #{pos.ticket} {pos.symbol} vol={pos.volume} profit={pos.profit:.2f}")
                return True
            else:
                # No positions for this symbol - clear tracking
                logger.info(f"📊 MT5 DIRECT: NO POSITION for {symbol}")
                # Clear stale entries
                tickets_to_remove = [k for k, v in _known_positions.items() if v.get("symbol", "").upper() == symbol.upper()]
                for ticket in tickets_to_remove:
                    logger.info(f"🧹 Clearing stale tracking #{ticket} for {symbol}")
                    del _known_positions[ticket]
                # Clear cooldown
                if symbol in _last_traded_signal:
                    logger.info(f"🔓 Clearing cooldown for {symbol}")
                    del _last_traded_signal[symbol]
                if symbol.upper() in _last_traded_signal:
                    del _last_traded_signal[symbol.upper()]
                return False
        
        # Fallback to broker method if direct MT5 access fails
        positions = await broker.get_positions()
        
        # 🔥 DEBUG: Log what MT5 returns
        pos_count = len(positions) if positions else 0
        logger.info(f"🔍 MT5 QUERY (fallback): {symbol} - MT5 returns {pos_count} total positions")
        
        if not positions:
            # No positions at all - clear any stale tracking for this symbol
            if symbol in _known_positions or symbol.upper() in [v.get("symbol", "").upper() for v in _known_positions.values()]:
                logger.warning(f"🧹 _check_open_positions({symbol}): MT5=0 but found in tracking - CLEARING STALE DATA!")
                # Clear stale entries
                tickets_to_remove = [k for k, v in _known_positions.items() if v.get("symbol", "").upper() == symbol.upper()]
                for ticket in tickets_to_remove:
                    del _known_positions[ticket]
                # Clear cooldown
                if symbol in _last_traded_signal:
                    del _last_traded_signal[symbol]
                if symbol.upper() in _last_traded_signal:
                    del _last_traded_signal[symbol.upper()]
            return False
        
        # Check if any position matches this symbol
        mt5_has_position = False
        for pos in positions:
            # Handle both dict and Position objects
            if isinstance(pos, dict):
                pos_symbol = pos.get("symbol", "")
                pos_ticket = pos.get("ticket") or pos.get("id")
                pos_profit = pos.get("profit", 0)
            else:
                pos_symbol = getattr(pos, "symbol", "")
                pos_ticket = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                pos_profit = getattr(pos, "profit", 0)
            
            # 🔥 DEBUG: Log each position
            logger.debug(f"   📍 Position: #{pos_ticket} {pos_symbol} profit={pos_profit}")
            
            if pos_symbol.upper() == symbol.upper():
                logger.info(f"✅ MT5 HAS POSITION: {symbol} #{pos_ticket} (profit={pos_profit})")
                mt5_has_position = True
                break
        
        if not mt5_has_position:
            # No position for this symbol - clear any stale tracking
            logger.info(f"📊 MT5 NO POSITION for {symbol} (total {pos_count} other positions)")
            # Clear stale entries for this symbol
            tickets_to_remove = [k for k, v in _known_positions.items() if v.get("symbol", "").upper() == symbol.upper()]
            for ticket in tickets_to_remove:
                logger.warning(f"🧹 Clearing stale tracking #{ticket} for {symbol}")
                del _known_positions[ticket]
            # 🔥 IMMEDIATE COOLDOWN RESET - ให้เทรดได้ทันที!
            if symbol in _last_traded_signal:
                logger.info(f"🔓 IMMEDIATE RESET: Clearing cooldown for {symbol} - MT5 has no position!")
                del _last_traded_signal[symbol]
            if symbol.upper() in _last_traded_signal:
                logger.info(f"🔓 IMMEDIATE RESET: Clearing cooldown for {symbol.upper()} - MT5 has no position!")
                del _last_traded_signal[symbol.upper()]
            # 🔥 Also clear any known_positions tracking
            if symbol in _known_positions:
                del _known_positions[symbol]
        
        return mt5_has_position
        
    except Exception as e:
        logger.error(f"❌ Failed to check positions for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False  # Assume no position if check fails


# =====================
# 📊 SIGNAL STRENGTH DETECTION - ตรวจจับสัญญาณกำลังเปลี่ยน
# =====================

def _track_signal_strength(symbol: str, signal_data: Dict):
    """
    📊 Track signal strength over time
    
    เก็บ history ของ confidence และ quality เพื่อตรวจจับว่าสัญญาณกำลังอ่อนตัว
    """
    global _signal_strength_tracker
    
    if symbol not in _signal_strength_tracker:
        _signal_strength_tracker[symbol] = {
            "confidence_history": [],
            "quality_history": [],
            "signal_history": [],
            "direction_changes": 0,
            "last_signal": None,
        }
    
    tracker = _signal_strength_tracker[symbol]
    current_signal = signal_data.get("signal", "WAIT")
    current_confidence = signal_data.get("confidence", 0)
    current_quality = signal_data.get("quality", "SKIP")
    
    # Track direction changes
    if tracker["last_signal"]:
        last_is_buy = "BUY" in tracker["last_signal"]
        current_is_buy = "BUY" in current_signal
        last_is_sell = "SELL" in tracker["last_signal"]
        current_is_sell = "SELL" in current_signal
        
        if (last_is_buy and current_is_sell) or (last_is_sell and current_is_buy):
            tracker["direction_changes"] += 1
            logger.warning(f"📊 SIGNAL DIRECTION CHANGE: {symbol} {tracker['last_signal']} → {current_signal} (changes: {tracker['direction_changes']})")
    
    # Add to history (keep last 10)
    tracker["confidence_history"].append(current_confidence)
    tracker["quality_history"].append(current_quality)
    tracker["signal_history"].append(current_signal)
    tracker["last_signal"] = current_signal
    
    # Keep only last 10
    if len(tracker["confidence_history"]) > 10:
        tracker["confidence_history"] = tracker["confidence_history"][-10:]
        tracker["quality_history"] = tracker["quality_history"][-10:]
        tracker["signal_history"] = tracker["signal_history"][-10:]


def _check_signal_weakening_for_dca(symbol: str, signal_data: Dict) -> bool:
    """
    📊 Check if signal is weakening (should NOT DCA)
    
    Detects:
    1. Confidence dropping consistently
    2. Quality dropping
    3. Signal direction becoming unstable
    4. Moving toward WAIT
    
    Returns: True if signal is weakening (DO NOT DCA), False if signal is strong
    """
    global _signal_strength_tracker
    
    tracker = _signal_strength_tracker.get(symbol)
    if not tracker:
        return False  # No history = assume OK
    
    conf_history = tracker.get("confidence_history", [])
    quality_history = tracker.get("quality_history", [])
    signal_history = tracker.get("signal_history", [])
    
    # Need at least 3 data points
    if len(conf_history) < 3:
        return False
    
    current_signal = signal_data.get("signal", "WAIT")
    current_confidence = signal_data.get("confidence", 0)
    current_quality = signal_data.get("quality", "SKIP")
    
    quality_order = {"SKIP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "PREMIUM": 4}
    
    # 1. CHECK: Confidence dropping consistently
    recent_conf = conf_history[-3:]
    if len(recent_conf) >= 3:
        # Check if confidence has been dropping
        if recent_conf[-1] < recent_conf[-2] < recent_conf[-3]:
            drop = recent_conf[-3] - recent_conf[-1]
            if drop >= 10:  # Dropped 10%+ in last 3 checks
                logger.warning(f"📊 WEAKENING: {symbol} confidence dropping! {recent_conf[-3]:.1f}% → {recent_conf[-1]:.1f}% (-{drop:.1f}%)")
                return True
    
    # 2. CHECK: Quality dropped
    recent_quality = quality_history[-3:]
    if len(recent_quality) >= 3:
        peak_quality = max(quality_order.get(q, 0) for q in recent_quality)
        current_quality_idx = quality_order.get(current_quality, 0)
        if peak_quality - current_quality_idx >= 2:  # Dropped 2+ levels (e.g., PREMIUM → MEDIUM)
            peak_name = [k for k, v in quality_order.items() if v == peak_quality][0]
            logger.warning(f"📊 WEAKENING: {symbol} quality dropped! {peak_name} → {current_quality}")
            return True
    
    # 3. CHECK: Signal direction unstable (multiple changes)
    if tracker.get("direction_changes", 0) >= 2:
        logger.warning(f"📊 WEAKENING: {symbol} direction unstable! {tracker['direction_changes']} changes")
        return True
    
    # 4. CHECK: Mixed signals (BUY and SELL in recent history)
    recent_signals = signal_history[-5:]
    has_buy = any("BUY" in s for s in recent_signals)
    has_sell = any("SELL" in s for s in recent_signals)
    has_wait = any(s in ["WAIT", "SKIP"] for s in recent_signals)
    
    if has_buy and has_sell:
        logger.warning(f"📊 WEAKENING: {symbol} mixed signals! BUY and SELL both in recent history")
        return True
    
    if has_wait and (has_buy or has_sell):
        # Had a direction but now seeing WAIT = weakening
        wait_count = sum(1 for s in recent_signals if s in ["WAIT", "SKIP"])
        if wait_count >= 2:
            logger.warning(f"📊 WEAKENING: {symbol} fading to WAIT! ({wait_count}/5 signals are WAIT)")
            return True
    
    # 5. CHECK: Low confidence in general
    avg_conf = sum(conf_history[-5:]) / min(5, len(conf_history))
    if avg_conf < 70:
        logger.info(f"📊 WEAKENING: {symbol} avg confidence {avg_conf:.1f}% < 70%")
        return True
    
    return False


def _get_signal_strength_score(symbol: str, signal_data: Dict) -> Dict:
    """
    📊 Get signal strength score
    
    Returns a score 0-100 indicating how strong/stable the signal is
    """
    global _signal_strength_tracker
    
    tracker = _signal_strength_tracker.get(symbol, {})
    conf_history = tracker.get("confidence_history", [])
    quality_history = tracker.get("quality_history", [])
    
    current_confidence = signal_data.get("confidence", 0)
    current_quality = signal_data.get("quality", "SKIP")
    current_signal = signal_data.get("signal", "WAIT")
    
    score = 50  # Start at neutral
    reasons = []
    
    # 1. Base confidence score
    if current_confidence >= 85:
        score += 20
        reasons.append("High confidence (85%+)")
    elif current_confidence >= 75:
        score += 10
        reasons.append("Good confidence (75%+)")
    elif current_confidence < 65:
        score -= 20
        reasons.append("Low confidence (<65%)")
    
    # 2. Quality score
    quality_order = {"SKIP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "PREMIUM": 4}
    q_idx = quality_order.get(current_quality, 0)
    if q_idx >= 4:
        score += 15
        reasons.append("PREMIUM quality")
    elif q_idx >= 3:
        score += 10
        reasons.append("HIGH quality")
    elif q_idx <= 1:
        score -= 15
        reasons.append("LOW/SKIP quality")
    
    # 3. Confidence trend
    if len(conf_history) >= 3:
        recent = conf_history[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            score += 10
            reasons.append("Confidence rising")
        elif recent[-1] < recent[-2] < recent[-3]:
            score -= 15
            reasons.append("Confidence dropping!")
    
    # 4. Direction stability
    direction_changes = tracker.get("direction_changes", 0)
    if direction_changes == 0:
        score += 10
        reasons.append("Direction stable")
    elif direction_changes >= 2:
        score -= 20
        reasons.append("Direction unstable!")
    
    # 5. Signal type
    if "STRONG" in current_signal:
        score += 10
        reasons.append("STRONG signal")
    elif current_signal in ["WAIT", "SKIP"]:
        score -= 20
        reasons.append("No direction")
    
    # Clamp score
    score = max(0, min(100, score))
    
    # Determine recommendation
    if score >= 80:
        recommendation = "STRONG - Safe to trade/DCA"
    elif score >= 60:
        recommendation = "OK - Proceed with caution"
    elif score >= 40:
        recommendation = "WEAK - Avoid DCA, consider closing"
    else:
        recommendation = "DANGER - Do not trade, consider exit"
    
    return {
        "score": score,
        "recommendation": recommendation,
        "reasons": reasons,
        "confidence": current_confidence,
        "quality": current_quality,
        "signal": current_signal,
        "direction_changes": direction_changes,
    }


# =====================
# 🔔 SIGNAL FADE ALERT FUNCTIONS - Early Warning System
# =====================

def _calculate_signal_momentum(symbol: str, current_confidence: float) -> Dict:
    """
    📈 Calculate Signal Momentum - ตรวจจับว่า confidence กำลังขึ้นหรือลง
    
    Returns:
    - momentum: RISING, STABLE, FALLING, FADING
    - trend: UP, FLAT, DOWN
    - alert_level: OK, WARNING, DANGER
    - peak_drop_percent: % ที่ลดลงจาก peak
    """
    global _signal_strength_tracker, _signal_health, _signal_fade_config
    
    tracker = _signal_strength_tracker.get(symbol, {})
    conf_history = tracker.get("confidence_history", [])
    quality_history = tracker.get("quality_history", [])
    
    # Initialize health tracking for this symbol
    if symbol not in _signal_health:
        _signal_health[symbol] = {
            "peak_confidence": current_confidence,
            "peak_quality": "SKIP",
            "current_confidence": current_confidence,
            "momentum": "STABLE",
            "trend": "FLAT",
            "alert_level": "OK",
            "peak_drop_percent": 0,
            "consecutive_drops": 0,
            "last_alert_time": None,
        }
    
    health = _signal_health[symbol]
    
    # Update current confidence
    health["current_confidence"] = current_confidence
    
    # Track peak confidence
    if current_confidence > health["peak_confidence"]:
        health["peak_confidence"] = current_confidence
        health["consecutive_drops"] = 0
    
    # Calculate drop from peak
    peak = health["peak_confidence"]
    if peak > 0:
        drop_percent = ((peak - current_confidence) / peak) * 100
        health["peak_drop_percent"] = round(drop_percent, 1)
    else:
        drop_percent = 0
        health["peak_drop_percent"] = 0
    
    # Need at least 3 data points to calculate momentum
    if len(conf_history) < 3:
        return health
    
    # Calculate momentum from recent history
    window = _signal_fade_config.get("momentum_window_size", 5)
    recent = conf_history[-window:] if len(conf_history) >= window else conf_history
    
    # Trend detection: compare first half to second half
    if len(recent) >= 4:
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        diff = second_half - first_half
        
        if diff > 5:
            health["trend"] = "UP"
            health["momentum"] = "RISING"
        elif diff < -5:
            health["trend"] = "DOWN"
            health["momentum"] = "FALLING"
        else:
            health["trend"] = "FLAT"
            health["momentum"] = "STABLE"
    
    # Check for consecutive drops
    if len(recent) >= 3:
        if recent[-1] < recent[-2] < recent[-3]:
            health["consecutive_drops"] = 3
            health["momentum"] = "FADING"
        elif recent[-1] < recent[-2]:
            health["consecutive_drops"] = max(1, health.get("consecutive_drops", 0))
    
    # Determine alert level
    threshold = _signal_fade_config.get("confidence_drop_threshold_percent", 10)
    
    if drop_percent >= threshold * 2:  # 20%+ drop from peak
        health["alert_level"] = "DANGER"
    elif drop_percent >= threshold:     # 10%+ drop from peak
        health["alert_level"] = "WARNING"
    elif health["momentum"] == "FADING":
        health["alert_level"] = "WARNING"
    else:
        health["alert_level"] = "OK"
    
    # Quality drop detection
    if quality_history and len(quality_history) >= 2:
        quality_order = {"SKIP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "PREMIUM": 4}
        current_quality = quality_history[-1]
        peak_quality_idx = max(quality_order.get(q, 0) for q in quality_history)
        current_quality_idx = quality_order.get(current_quality, 0)
        
        if peak_quality_idx - current_quality_idx >= 2:
            health["alert_level"] = "DANGER"
        elif peak_quality_idx - current_quality_idx >= 1:
            if health["alert_level"] == "OK":
                health["alert_level"] = "WARNING"
        
        # Store peak quality
        health["peak_quality"] = [k for k, v in quality_order.items() if v == peak_quality_idx][0] if peak_quality_idx > 0 else "SKIP"
    
    return health


def _get_signal_health_summary(symbol: str) -> Dict:
    """
    📊 Get comprehensive signal health summary
    """
    global _signal_health, _bot_status
    
    signal = _bot_status.get("last_signal", {}).get(symbol, {})
    health = _signal_health.get(symbol, {})
    
    return {
        "symbol": symbol,
        "current_confidence": signal.get("confidence", 0),
        "peak_confidence": health.get("peak_confidence", 0),
        "peak_drop_percent": health.get("peak_drop_percent", 0),
        "current_quality": signal.get("quality", "SKIP"),
        "peak_quality": health.get("peak_quality", "SKIP"),
        "momentum": health.get("momentum", "UNKNOWN"),
        "trend": health.get("trend", "UNKNOWN"),
        "alert_level": health.get("alert_level", "OK"),
        "consecutive_drops": health.get("consecutive_drops", 0),
        "signal": signal.get("signal", "WAIT"),
        "is_fading": health.get("momentum") in ["FALLING", "FADING"],
        "warning_message": _get_warning_message(health),
    }


def _get_warning_message(health: Dict) -> Optional[str]:
    """Generate warning message based on health status"""
    alert = health.get("alert_level", "OK")
    momentum = health.get("momentum", "STABLE")
    drop = health.get("peak_drop_percent", 0)
    
    if alert == "DANGER":
        return f"⚠️ DANGER: Confidence dropped {drop:.1f}% from peak! Signal may reverse soon!"
    elif alert == "WARNING":
        if momentum == "FADING":
            return f"📉 WARNING: Signal fading! {health.get('consecutive_drops', 0)} consecutive drops detected."
        else:
            return f"📉 WARNING: Confidence dropped {drop:.1f}% from peak. Monitor closely."
    elif momentum == "FALLING":
        return f"📊 NOTE: Confidence trending down. Watch for reversal."
    
    return None


def _get_health_recommendations(health: Dict) -> List[str]:
    """Generate actionable recommendations based on signal health"""
    recommendations = []
    
    alert = health.get("alert_level", "OK")
    momentum = health.get("momentum", "STABLE")
    is_fading = health.get("is_fading", False)
    
    if alert == "DANGER":
        recommendations.append("🛑 Consider closing profitable positions")
        recommendations.append("⛔ Do NOT open new positions")
        recommendations.append("👀 Wait for signal to stabilize")
    elif alert == "WARNING":
        recommendations.append("⚠️ Tighten stop loss on existing positions")
        recommendations.append("🔍 Monitor closely for reversal")
        recommendations.append("⏸️ Pause new entries until stabilized")
    elif momentum == "FALLING":
        recommendations.append("📊 Signal weakening - be cautious")
        recommendations.append("💰 Consider taking partial profits")
    elif momentum == "RISING":
        recommendations.append("✅ Signal strengthening - good for entries")
        recommendations.append("📈 Consider adding to position")
    else:
        recommendations.append("👍 Signal stable - normal trading")
    
    return recommendations


# =====================
# 🤖 SIGNAL FADE AUTO-ACTION - จัดการ position อัตโนมัติ!
# =====================

async def _handle_signal_fade_auto_action(symbol: str, signal_health: Dict):
    """
    🤖 AUTO-ACTION เมื่อ Signal Fading
    
    Actions:
    1. WARNING → ย้าย SL มา break-even (ถ้ามีกำไร)
    2. DANGER → ปิด position ที่มีกำไร >= X%
    
    Returns: Dict with actions taken
    """
    global _bot, _signal_fade_config, _bot_status
    
    if not _signal_fade_config.get("auto_action_enabled", True):
        return {"action": "disabled"}
    
    if not _bot or not _bot.trading_engine:
        return {"action": "bot_not_ready"}
    
    alert_level = signal_health.get("alert_level", "OK")
    
    if alert_level == "OK":
        return {"action": "none", "reason": "Signal healthy"}
    
    actions_taken = []
    
    try:
        positions = await _bot.trading_engine.broker.get_positions()
        if not positions:
            return {"action": "no_positions"}
        
        # Get balance for % calculations
        try:
            balance = await _bot.trading_engine.broker.get_balance() or 1000
        except:
            balance = 1000
        
        for pos in positions:
            # Extract position info
            if isinstance(pos, dict):
                pos_id = pos.get("ticket") or pos.get("id")
                pos_symbol = pos.get("symbol", "")
                pos_side = pos.get("side", "").upper()
                pos_pnl = float(pos.get("profit", 0) or 0)
                pos_entry = float(pos.get("open_price", 0) or pos.get("price_open", 0) or 0)
                pos_sl = float(pos.get("sl", 0) or 0)
            else:
                pos_id = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                pos_symbol = getattr(pos, "symbol", "")
                pos_side = getattr(pos, "side", "")
                if hasattr(pos_side, "value"):
                    pos_side = pos_side.value.upper()
                pos_pnl = float(getattr(pos, "profit", 0) or 0)
                pos_entry = float(getattr(pos, "open_price", 0) or getattr(pos, "price_open", 0) or 0)
                pos_sl = float(getattr(pos, "sl", 0) or 0)
            
            # Only handle positions for this symbol
            if pos_symbol.upper() != symbol.upper():
                continue
            
            # Calculate profit % of balance
            profit_percent = (pos_pnl / balance) * 100 if balance > 0 else 0
            
            # =====================
            # 🔴 DANGER → Close profitable positions
            # =====================
            if alert_level == "DANGER" and _signal_fade_config.get("close_profitable_on_danger", True):
                min_profit_percent = _signal_fade_config.get("min_profit_percent_to_close_on_danger", 5.0)
                
                if profit_percent >= min_profit_percent:
                    logger.warning(f"🤖 AUTO-ACTION DANGER: {symbol} closing position #{pos_id}")
                    logger.warning(f"   Profit: ${pos_pnl:.2f} ({profit_percent:.1f}% of balance)")
                    logger.warning(f"   Reason: Signal DANGER - confidence dropped significantly!")
                    
                    try:
                        result = await _bot.trading_engine.broker.close_position(pos_id)
                        if result:
                            logger.info(f"✅ Position #{pos_id} closed! Locked profit: ${pos_pnl:.2f}")
                            
                            # Update stats
                            _bot_status["daily_stats"]["trades"] += 1
                            _bot_status["daily_stats"]["pnl"] += pos_pnl
                            _bot_status["daily_stats"]["wins"] += 1
                            
                            actions_taken.append({
                                "action": "closed",
                                "ticket": pos_id,
                                "symbol": pos_symbol,
                                "pnl": pos_pnl,
                                "reason": "DANGER - signal fading",
                            })
                        else:
                            logger.error(f"❌ Failed to close position #{pos_id}")
                    except Exception as e:
                        logger.error(f"❌ Error closing position: {e}")
                else:
                    logger.info(f"📊 DANGER but profit {profit_percent:.1f}% < {min_profit_percent}% - NOT closing yet")
            
            # =====================
            # 🟡 WARNING → Move SL to break-even
            # =====================
            elif alert_level == "WARNING" and _signal_fade_config.get("move_sl_to_breakeven_on_warning", True):
                min_profit_to_move = _signal_fade_config.get("min_profit_to_move_sl", 0.5)
                
                if profit_percent >= min_profit_to_move and pos_entry > 0:
                    # Calculate break-even SL (entry price + small buffer)
                    is_gold = 'XAU' in pos_symbol.upper() or 'GOLD' in pos_symbol.upper()
                    buffer = 1.0 if is_gold else 0.0005  # $1 for gold, 0.5 pips for forex
                    
                    if pos_side == "BUY":
                        new_sl = pos_entry + buffer
                        # Only move if new SL is better (higher for BUY)
                        if pos_sl > 0 and new_sl <= pos_sl:
                            continue  # Already better SL
                    else:  # SELL
                        new_sl = pos_entry - buffer
                        # Only move if new SL is better (lower for SELL)
                        if pos_sl > 0 and new_sl >= pos_sl:
                            continue  # Already better SL
                    
                    new_sl = round(new_sl, 2 if is_gold else 5)
                    
                    logger.warning(f"🤖 AUTO-ACTION WARNING: {symbol} moving SL to break-even")
                    logger.warning(f"   Position: #{pos_id} {pos_side} @ {pos_entry:.2f}")
                    logger.warning(f"   Current SL: {pos_sl:.2f} → New SL: {new_sl:.2f}")
                    logger.warning(f"   Reason: Signal WARNING - protecting profit ${pos_pnl:.2f}")
                    
                    try:
                        result = await _bot.trading_engine.broker.modify_position(
                            str(pos_id),
                            stop_loss=new_sl
                        )
                        
                        if result and result.success:
                            logger.info(f"✅ SL moved to break-even: #{pos_id} SL={new_sl:.2f}")
                            actions_taken.append({
                                "action": "sl_moved",
                                "ticket": pos_id,
                                "symbol": pos_symbol,
                                "old_sl": pos_sl,
                                "new_sl": new_sl,
                                "reason": "WARNING - break-even protection",
                            })
                        else:
                            error = result.error if result else "Unknown"
                            logger.warning(f"⚠️ Failed to move SL: {error}")
                    except Exception as e:
                        logger.error(f"❌ Error moving SL: {e}")
        
        return {
            "action": "completed",
            "alert_level": alert_level,
            "actions_taken": actions_taken,
        }
        
    except Exception as e:
        logger.error(f"❌ Error in signal fade auto-action: {e}")
        return {"action": "error", "error": str(e)}


def _check_signal_health_allows_trading(symbol: str) -> tuple[bool, str]:
    """
    🔍 Check if signal health allows opening new trades
    
    Returns: (can_trade: bool, reason: str)
    """
    global _signal_health, _signal_fade_config
    
    if not _signal_fade_config.get("auto_action_enabled", True):
        return True, "Auto-action disabled"
    
    health = _signal_health.get(symbol, {})
    alert_level = health.get("alert_level", "OK")
    
    # DANGER → Block new trades
    if alert_level == "DANGER" and _signal_fade_config.get("block_new_trades_on_danger", True):
        drop = health.get("peak_drop_percent", 0)
        return False, f"🔴 BLOCKED: Signal DANGER (confidence dropped {drop:.1f}% from peak)"
    
    # WARNING → Block new trades
    if alert_level == "WARNING" and _signal_fade_config.get("block_new_trades_on_warning", True):
        momentum = health.get("momentum", "UNKNOWN")
        return False, f"🟡 BLOCKED: Signal WARNING ({momentum} momentum)"
    
    return True, "✅ Signal health OK"


# =====================
# 📈 SMART DCA FUNCTIONS - เข้าซ้ำเมื่อราคาย่อ
# =====================

async def _check_dca_opportunity(symbol: str, signal_data: Dict, current_price: float) -> bool:
    """
    📈 SMART DCA - Check if we should add to position (Dollar Cost Averaging)
    
    🚨 SAFETY CHECKS ADDED:
    - ตรวจสอบว่าสัญญาณไม่ได้อ่อนตัว
    - ตรวจสอบ confidence ยังสูง
    - ตรวจสอบว่าไม่มี reversal signals
    
    Returns: True if DCA executed, False otherwise
    """
    global _bot, _dca_config, _dca_tracking, _bot_status, _signal_strength_tracker
    
    
    if not _dca_config.get("enabled", False):
        return False
    
    if not _bot or not _bot.trading_engine:
        return False
    
    signal = signal_data.get("signal", "WAIT")
    confidence = signal_data.get("confidence", 0)
    quality = signal_data.get("quality", "SKIP")
    
    if signal not in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
        return False
    
    # 🆕 SIGNAL STRENGTH CHECK - ต้องเป็น signal แข็งแรง
    if _dca_config.get("require_strong_signal", True):
        min_conf = _dca_config.get("min_confidence_for_dca", 80)
        if confidence < min_conf:
            logger.info(f"📈 DCA BLOCKED: {symbol} confidence {confidence:.1f}% < {min_conf}%")
            return False
        
        quality_order = {"SKIP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "PREMIUM": 4}
        if quality_order.get(quality, 0) < quality_order.get("HIGH", 3):
            logger.info(f"📈 DCA BLOCKED: {symbol} quality {quality} < HIGH")
            return False
    
    # 🆕 SIGNAL TREND CHECK - ตรวจสอบว่าสัญญาณไม่ได้อ่อนตัว
    if _dca_config.get("check_signal_trend", True):
        is_weakening = _check_signal_weakening_for_dca(symbol, signal_data)
        if is_weakening:
            logger.warning(f"📈 DCA BLOCKED: {symbol} signal is WEAKENING - DO NOT ADD!")
            return False
    
    is_buy_signal = "BUY" in signal
    
    try:
        # Get current positions
        positions = await _bot.trading_engine.broker.get_positions()
        if not positions:
            return False
        
        # Find position for this symbol
        symbol_position = None
        total_pnl = 0
        position_count = 0
        
        for pos in positions:
            if isinstance(pos, dict):
                pos_symbol = pos.get("symbol", "")
                pos_side = pos.get("side", "").upper()
                pos_pnl = float(pos.get("profit", 0) or 0)
                pos_price = float(pos.get("open_price", 0) or pos.get("price_open", 0) or 0)
            else:
                pos_symbol = getattr(pos, "symbol", "")
                pos_side = getattr(pos, "side", "")
                if hasattr(pos_side, "value"):
                    pos_side = pos_side.value.upper()
                pos_pnl = float(getattr(pos, "profit", 0) or 0)
                pos_price = float(getattr(pos, "open_price", 0) or getattr(pos, "price_open", 0) or 0)
            
            if pos_symbol.upper() == symbol.upper():
                symbol_position = pos
                total_pnl += pos_pnl
                position_count += 1
                
                # Store first entry info
                if symbol not in _dca_tracking:
                    _dca_tracking[symbol] = {
                        "entries": position_count,
                        "first_entry_price": pos_price,
                        "last_dca_time": None,
                        "peak_adverse": current_price,
                        "side": pos_side,
                    }
        
        if not symbol_position:
            # No position = clear tracking
            if symbol in _dca_tracking:
                del _dca_tracking[symbol]
            return False
        
        # Get tracking data
        tracking = _dca_tracking.get(symbol)
        if not tracking:
            return False
        
        first_entry_price = tracking.get("first_entry_price", current_price)
        peak_adverse = tracking.get("peak_adverse", current_price)
        entries = tracking.get("entries", 1)
        last_dca_time = tracking.get("last_dca_time")
        position_side = tracking.get("side", "").upper()
        
        # 1. Check max DCA entries
        max_entries = _dca_config.get("max_dca_entries", 2)
        if entries > max_entries:
            return False
        
        # 2. Check time between DCA
        min_time = _dca_config.get("min_time_between_dca", 300)
        if last_dca_time:
            elapsed = (datetime.now() - last_dca_time).total_seconds()
            if elapsed < min_time:
                return False
        
        # 3. Check total loss before DCA (DYNAMIC)
        try:
            balance = await _bot.trading_engine.broker.get_balance() or 1000
        except:
            balance = 1000
        
        # 🔥 Check if balance allows DCA
        if not _should_allow_dca(balance):
            logger.info(f"📈 DCA BLOCKED: Balance ${balance:.0f} < ${_dca_config.get('min_balance_for_dca', 500)} minimum")
            return False
        
        max_loss = _get_max_dca_loss(balance)
        if total_pnl < -max_loss:
            logger.info(f"📈 DCA BLOCKED: {symbol} total loss ${total_pnl:.2f} > ${max_loss:.0f} (5% of ${balance:.0f})")
            return False
        
        # 4. Check signal consistency
        if _dca_config.get("signal_must_persist", True):
            # Signal must match position direction
            if position_side == "BUY" and not is_buy_signal:
                return False
            if position_side == "SELL" and is_buy_signal:
                return False
        
        # 5. Calculate retracement
        min_retracement = _dca_config.get("min_retracement_percent", 0.15)
        
        # 🔥 FIX: Prevent division by zero
        if first_entry_price <= 0:
            logger.warning(f"📈 DCA BLOCKED: {symbol} first_entry_price is 0 or negative")
            return False
        
        if position_side == "BUY":
            # For BUY: price going DOWN is adverse → track lowest (peak_adverse = lowest)
            if current_price < peak_adverse:
                tracking["peak_adverse"] = current_price
                peak_adverse = current_price
            
            # Retracement = how much price dropped from entry
            retracement_pct = ((first_entry_price - peak_adverse) / first_entry_price) * 100
            
            # Check if price is now reversing UP
            price_reversing = current_price > peak_adverse
            
        else:  # SELL
            # For SELL: price going UP is adverse → track highest (peak_adverse = highest)
            if current_price > peak_adverse:
                tracking["peak_adverse"] = current_price
                peak_adverse = current_price
            
            # Retracement = how much price rose from entry
            retracement_pct = ((peak_adverse - first_entry_price) / first_entry_price) * 100
            
            # Check if price is now reversing DOWN
            price_reversing = current_price < peak_adverse
        
        # Check if retracement is enough
        if retracement_pct < min_retracement:
            return False
        
        # 6. Check reversal
        if _dca_config.get("wait_for_reversal", True):
            if not price_reversing:
                logger.debug(f"📈 DCA WAIT: {symbol} retracement {retracement_pct:.2f}% but no reversal yet")
                return False
        
        # 🎯 ALL CONDITIONS MET - EXECUTE DCA!
        logger.info(f"📈 DCA OPPORTUNITY: {symbol}")
        logger.info(f"   Position: {position_side} @ {first_entry_price:.2f}")
        logger.info(f"   Peak adverse: {peak_adverse:.2f} (retracement: {retracement_pct:.2f}%)")
        logger.info(f"   Current: {current_price:.2f} (reversing: {price_reversing})")
        logger.info(f"   Entries: {entries}/{max_entries+1}")
        
        # Execute DCA trade
        lot_multiplier = _dca_config.get("lot_multiplier", 1.0)
        
        # Use same analysis with DCA flag
        analysis = _bot_status["last_analysis"].get(symbol)
        if not analysis:
            return False
        
        # Mark as DCA trade
        dca_analysis = analysis.copy()
        dca_analysis["is_dca"] = True
        dca_analysis["dca_entry_number"] = entries + 1
        dca_analysis["lot_multiplier"] = lot_multiplier
        
        result = await _bot.execute_trade(dca_analysis)
        
        if result and result.get("success"):
            # Update tracking
            tracking["entries"] = entries + 1
            tracking["last_dca_time"] = datetime.now()
            tracking["peak_adverse"] = current_price  # Reset peak after DCA
            
            logger.info(f"✅ DCA #{entries + 1} executed: {symbol} {position_side} @ {current_price:.2f}")
            _bot_status["daily_stats"]["trades"] += 1
            
            return True
        else:
            reason = result.get("reason", "Unknown") if result else "No result"
            logger.warning(f"⚠️ DCA trade failed: {reason}")
            return False
        
    except Exception as e:
        logger.error(f"Error checking DCA opportunity: {e}")
        return False


def _reset_dca_tracking(symbol: str):
    """Reset DCA tracking for a symbol when position is closed"""
    global _dca_tracking
    
    if symbol in _dca_tracking:
        del _dca_tracking[symbol]
        logger.info(f"📈 DCA tracking reset for {symbol}")


async def _update_dca_tracking_from_positions():
    """Update DCA tracking based on current MT5 positions"""
    global _bot, _dca_tracking
    
    if not _bot or not _bot.trading_engine:
        return
    
    try:
        positions = await _bot.trading_engine.broker.get_positions()
        
        # Get current symbols with positions
        current_symbols = set()
        for pos in (positions or []):
            if isinstance(pos, dict):
                symbol = pos.get("symbol", "")
            else:
                symbol = getattr(pos, "symbol", "")
            if symbol:
                current_symbols.add(symbol.upper())
        
        # Clear tracking for symbols without positions
        for symbol in list(_dca_tracking.keys()):
            if symbol.upper() not in current_symbols:
                del _dca_tracking[symbol]
                logger.info(f"📈 DCA tracking cleared for {symbol} (no position)")
                
    except Exception as e:
        logger.warning(f"Error updating DCA tracking: {e}")


# ⚡ SIGNAL MOMENTUM FUNCTIONS
def _track_signal_history(symbol: str, signal_data: Dict):
    """Track signal history for momentum detection"""
    global _signal_history, _signal_weakening_config
    
    if not _signal_weakening_config.get("enabled", True):
        return
    
    history_size = _signal_weakening_config.get("history_size", 5)
    
    if symbol not in _signal_history:
        _signal_history[symbol] = []
    
    # Add new signal to history
    _signal_history[symbol].append({
        "signal": signal_data.get("signal", "WAIT"),
        "quality": signal_data.get("quality", "SKIP"),
        "confidence": signal_data.get("confidence", 0),
        "timestamp": datetime.now()
    })
    
    # Keep only last N signals
    if len(_signal_history[symbol]) > history_size:
        _signal_history[symbol] = _signal_history[symbol][-history_size:]


def _detect_signal_weakening(symbol: str, current_signal: Dict, position_side: str) -> tuple[bool, str]:
    """
    ⚡ Detect if signal is weakening (should close position early)
    
    Returns: (should_close: bool, reason: str)
    
    Detects:
    1. Quality dropping: PREMIUM → HIGH → MEDIUM
    2. Confidence dropping: 88% → 76% → 65%
    3. Signal direction weakening: BUY → WAIT (แต่ยังไม่ SELL)
    """
    global _signal_history, _signal_weakening_config
    
    if not _signal_weakening_config.get("enabled", True):
        return False, "Weakening detection disabled"
    
    history = _signal_history.get(symbol, [])
    if len(history) < 3:  # Need at least 3 signals to detect trend
        return False, "Not enough history"
    
    current_signal_type = current_signal.get("signal", "WAIT")
    current_quality = current_signal.get("quality", "SKIP")
    current_confidence = current_signal.get("confidence", 0)
    
    quality_order = {"SKIP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "PREMIUM": 4}
    
    # 1. Check if signal direction changed (BUY → WAIT/SELL while holding BUY)
    if position_side == "BUY" and current_signal_type in ["SELL", "STRONG_SELL"]:
        return True, f"⚠️ Signal reversed to {current_signal_type} - CLOSE IMMEDIATELY"
    if position_side == "SELL" and current_signal_type in ["BUY", "STRONG_BUY"]:
        return True, f"⚠️ Signal reversed to {current_signal_type} - CLOSE IMMEDIATELY"
    
    # 2. Check quality drop (e.g., PREMIUM → HIGH → MEDIUM)
    if _signal_weakening_config.get("close_on_quality_drop", True):
        # Find peak quality in recent history
        peak_quality_idx = 0
        for h in history:
            q_idx = quality_order.get(h.get("quality", "SKIP"), 0)
            peak_quality_idx = max(peak_quality_idx, q_idx)
        
        current_quality_idx = quality_order.get(current_quality, 0)
        quality_drop = peak_quality_idx - current_quality_idx
        threshold = _signal_weakening_config.get("quality_drop_threshold", 2)
        
        if quality_drop >= threshold:
            peak_quality_name = [k for k, v in quality_order.items() if v == peak_quality_idx][0]
            return True, f"⚠️ Quality dropped {quality_drop} levels: {peak_quality_name} → {current_quality}"
    
    # 3. Check confidence drop
    if _signal_weakening_config.get("close_on_confidence_drop", True):
        # Find peak confidence in recent history
        peak_confidence = max(h.get("confidence", 0) for h in history)
        confidence_drop = peak_confidence - current_confidence
        threshold = _signal_weakening_config.get("confidence_drop_threshold", 15)
        
        if confidence_drop >= threshold:
            return True, f"⚠️ Confidence dropped {confidence_drop:.1f}%: {peak_confidence:.1f}% → {current_confidence:.1f}%"
    
    # 4. Check if signal is becoming WAIT (momentum fading)
    if position_side in ["BUY", "SELL"] and current_signal_type == "WAIT":
        # Check if we had strong signal before
        recent_strong = any(
            h.get("signal") in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"] 
            for h in history[-3:]
        )
        if recent_strong:
            return True, f"⚠️ Signal faded to WAIT - momentum lost"
    
    return False, "Signal stable"


async def _check_and_close_weakening_positions(symbol: str, signal_data: Dict):
    """
    ⚡ Check if any positions should be closed due to weakening signal
    
    🔥 NOTE: ปิดการทำงานชั่วคราว เพราะ trigger บ่อยเกินไปทำให้ไม่เสถียร
    ใช้ SL/TP ปกติแทน
    """
    global _bot, _signal_weakening_config, _bot_status
    
    # 🔥 ปิดการทำงานถ้า disabled
    if not _signal_weakening_config.get("enabled", False):
        return
    
    if not _bot or not _bot.trading_engine:
        return
    
    try:
        positions = await _bot.trading_engine.broker.get_positions()
        if not positions:
            return
        
        for pos in positions:
            # Get position details
            if isinstance(pos, dict):
                pos_symbol = pos.get("symbol", "")
                pos_side = pos.get("side", "")
                pos_pnl = pos.get("profit", pos.get("pnl", 0))
                pos_ticket = pos.get("ticket", pos.get("id", ""))
            else:
                pos_symbol = getattr(pos, "symbol", "")
                pos_side = getattr(pos, "side", "")
                pos_pnl = getattr(pos, "profit", getattr(pos, "pnl", 0))
                pos_ticket = getattr(pos, "ticket", getattr(pos, "id", ""))
            
            if pos_symbol.upper() != symbol.upper():
                continue
            
            # Normalize side
            if hasattr(pos_side, 'value'):
                pos_side = pos_side.value
            pos_side = str(pos_side).upper()
            if pos_side in ["0", "ORDER_TYPE_BUY"]:
                pos_side = "BUY"
            elif pos_side in ["1", "ORDER_TYPE_SELL"]:
                pos_side = "SELL"
            
            # Check if signal is weakening
            should_close, reason = _detect_signal_weakening(symbol, signal_data, pos_side)
            
            if should_close:
                # 🔥 DYNAMIC: Get min profit based on balance (% based)
                try:
                    balance = await _bot.trading_engine.broker.get_balance() or 300
                except:
                    balance = 300
                min_profit = _get_early_exit_min_profit(balance)
                
                # ต้องมีกำไร >= min_profit ถึงจะปิด
                if pos_pnl >= min_profit:
                    logger.warning(f"⚡ SIGNAL WEAKENING: {symbol} - {reason}")
                    logger.warning(f"   Position: {pos_side} | PnL: ${pos_pnl:.2f} (>= ${min_profit:.0f} = 15% of ${balance:.0f})")
                    logger.warning(f"   ACTION: Closing to LOCK PROFIT!")
                    
                    # 🔥 ใช้ broker interface แทน MT5 โดยตรง - เสถียรกว่า!
                    try:
                        result = await _bot.trading_engine.broker.close_position(pos_ticket)
                        if result:
                            logger.info(f"✅ Position closed early: {symbol} | Reason: {reason}")
                            # Update daily stats
                            _bot_status["daily_stats"]["trades"] += 1
                            if pos_pnl > 0:
                                _bot_status["daily_stats"]["wins"] += 1
                            else:
                                _bot_status["daily_stats"]["losses"] += 1
                            _bot_status["daily_stats"]["pnl"] += float(pos_pnl)
                        else:
                            logger.error(f"❌ Failed to close position #{pos_ticket}")
                    except Exception as e:
                        logger.error(f"Error closing weakening position: {e}")
                # 🔥 ไม่ต้อง log ทุกครั้ง - ลด noise
                        
    except Exception as e:
        logger.error(f"Error checking weakening positions: {e}")


# =====================
# 🏗️ 20-LAYER GATE + DAILY LIMIT FUNCTIONS
# =====================

def _check_layer_gate(symbol: str) -> tuple[bool, str]:
    """
    🏗️ 20-LAYER GATE - ตรวจสอบว่าผ่าน layer เพียงพอหรือไม่
    
    🔥 ต้องผ่าน 12/20 layers (60%) ถึงจะเทรดได้!
    
    Returns: (can_trade: bool, reason: str)
    """
    global _bot_status, _layer_gate_config
    
    if not _layer_gate_config.get("enabled", True):
        return True, "Layer gate disabled"
    
    layer_status = _bot_status.get("layer_status", {}).get(symbol)
    
    if not layer_status:
        return False, "No layer data available"
    
    passed = layer_status.get("passed", 0)
    total = layer_status.get("total", 20)
    pass_rate = layer_status.get("pass_rate", 0)
    layers = layer_status.get("layers", [])
    
    min_passed = _layer_gate_config.get("min_layers_passed", 12)
    min_rate = _layer_gate_config.get("min_pass_rate", 60)
    required = _layer_gate_config.get("required_layers", [1, 2, 3, 4])
    
    # 1. Check minimum layers passed
    if passed < min_passed:
        return False, f"Only {passed}/{total} layers passed (need {min_passed}+)"
    
    # 2. Check pass rate
    if pass_rate < min_rate:
        return False, f"Pass rate {pass_rate:.1f}% < {min_rate}% required"
    
    # 3. Check required layers (Base layers 1-4)
    for layer_num in required:
        layer = next((l for l in layers if l.get("layer") == layer_num), None)
        if layer and layer.get("status") not in ["PASS", "READY"]:
            layer_name = layer.get("name", f"Layer {layer_num}")
            return False, f"Required layer {layer_num} ({layer_name}) not passed"
    
    logger.info(f"🏗️ LAYER GATE PASSED: {symbol} - {passed}/{total} ({pass_rate:.1f}%)")
    return True, f"✅ Layer gate passed: {passed}/{total} ({pass_rate:.1f}%)"


def _check_daily_limit(symbol: str) -> tuple[bool, str]:
    """
    🚨 CHECK DAILY LIMIT - ตรวจสอบว่าเทรดเกินลิมิตหรือยัง
    
    🔥 Features:
    - Max 20 trades per day
    - Pause 30 mins after 3 consecutive losses
    - Stop if daily loss > 5% of balance
    
    Returns: (can_trade: bool, reason: str)
    """
    global _bot_status, _daily_trade_limit, _loss_streak_tracker
    
    if not _daily_trade_limit.get("enabled", True):
        return True, "Daily limit disabled"
    
    daily_stats = _bot_status.get("daily_stats", {})
    trades_today = daily_stats.get("trades", 0)
    pnl_today = daily_stats.get("pnl", 0)
    
    max_trades = _daily_trade_limit.get("max_trades_per_day", 20)
    max_loss_streak = _daily_trade_limit.get("max_losing_streak", 3)
    pause_minutes = _daily_trade_limit.get("pause_after_loss_minutes", 30)
    
    # 1. Check max trades per day
    if trades_today >= max_trades:
        return False, f"Daily limit reached: {trades_today}/{max_trades} trades"
    
    # 2. Check if paused after loss streak
    paused_until = _loss_streak_tracker.get("paused_until")
    if paused_until and datetime.now() < paused_until:
        remaining = int((paused_until - datetime.now()).total_seconds() / 60)
        return False, f"Paused for {remaining}m after loss streak"
    
    # 3. Check current loss streak
    current_streak = _loss_streak_tracker.get("current_streak", 0)
    if current_streak >= max_loss_streak:
        # Pause trading
        _loss_streak_tracker["paused_until"] = datetime.now() + timedelta(minutes=pause_minutes)
        _loss_streak_tracker["current_streak"] = 0  # Reset after pause
        logger.warning(f"🚨 LOSS STREAK {current_streak} >= {max_loss_streak} - PAUSING {pause_minutes} minutes!")
        return False, f"Loss streak {current_streak} >= {max_loss_streak} - pausing {pause_minutes}m"
    
    # 4. Check daily loss limit (% of balance) - async so we skip here
    # Will be checked in trade execution
    
    logger.debug(f"📊 Daily limit OK: {trades_today}/{max_trades} trades, streak: {current_streak}")
    return True, f"OK: {trades_today}/{max_trades} trades, streak: {current_streak}"


def _update_loss_streak(is_win: bool):
    """
    📊 Update loss streak after trade closes
    
    - Win: Reset streak to 0
    - Loss: Increment streak
    """
    global _loss_streak_tracker
    
    if is_win:
        if _loss_streak_tracker.get("current_streak", 0) > 0:
            logger.info(f"✅ WIN! Loss streak reset from {_loss_streak_tracker['current_streak']} to 0")
        _loss_streak_tracker["current_streak"] = 0
    else:
        _loss_streak_tracker["current_streak"] = _loss_streak_tracker.get("current_streak", 0) + 1
        _loss_streak_tracker["last_loss_time"] = datetime.now()
        logger.warning(f"❌ LOSS! Streak now: {_loss_streak_tracker['current_streak']}")


async def _can_trade_signal(symbol: str, signal_data: Dict) -> tuple[bool, str]:
    """
    🎯 SMART TRADE FILTER - ENHANCED!
    
    🔥 NEW CHECKS ADDED:
    1. 🏗️ 20-LAYER GATE - ต้องผ่าน 12/20 layers!
    2. 🚨 DAILY LIMIT - Max 20 trades, loss streak protection!
    3. 🥇 Symbol whitelist (Gold only)
    4. 📊 Quality + Confidence filter
    5. 🛡️ Trend alignment check
    6. 📍 Open position check
    7. ⏱️ Cooldown check
    
    Returns: (can_trade: bool, reason: str)
    """
    global _last_traded_signal, _open_positions, _trade_cooldown_seconds, _aggressive_config, _symbol_whitelist
    
    signal = signal_data.get("signal", "WAIT")
    confidence = signal_data.get("confidence", 0)
    quality = signal_data.get("quality", "SKIP")
    
    # 0. 🥇 SYMBOL WHITELIST CHECK - Block non-Gold symbols!
    if _symbol_whitelist.get("enabled", True):
        is_gold = 'XAU' in symbol.upper() or 'GOLD' in symbol.upper()
        
        if not is_gold and _symbol_whitelist.get("block_forex", True):
            logger.info(f"🚫 BLOCKED: {symbol} is FOREX - only GOLD trading allowed!")
            return False, f"FOREX BLOCKED: {symbol} - Only GOLD trading enabled"
        
        # Also check explicit whitelist
        allowed = _symbol_whitelist.get("allowed_symbols", [])
        if allowed and symbol.upper() not in [s.upper() for s in allowed]:
            is_in_whitelist = any(sym.upper() in symbol.upper() for sym in allowed)
            if not is_in_whitelist:
                logger.info(f"🚫 BLOCKED: {symbol} not in whitelist {allowed}")
                return False, f"Symbol {symbol} not in whitelist"
    
    # 1. Check if signal is tradeable
    if signal in ["WAIT", "SKIP"]:
        return False, "Signal is WAIT/SKIP"
    
    # 🆕 2. 🚨 CHECK DAILY LIMIT - เทรดเกินลิมิตหรือยัง?
    can_trade_daily, daily_reason = _check_daily_limit(symbol)
    if not can_trade_daily:
        return False, f"DAILY LIMIT: {daily_reason}"
    
    # 🆕 3. 🏗️ CHECK 20-LAYER GATE - ผ่าน layer เพียงพอหรือไม่?
    layer_passed, layer_reason = _check_layer_gate(symbol)
    if not layer_passed:
        return False, f"LAYER GATE: {layer_reason}"
    
    # 4. 🎯 SYMBOL-SPECIFIC QUALITY FILTER (Gold only now)
    is_gold = 'XAU' in symbol.upper() or 'GOLD' in symbol.upper()
    
    # 🔥 Gold-focused settings
    if is_gold:
        min_quality = "HIGH"      # 🔥 Gold ต้อง HIGH ขึ้นไป
        min_confidence = 75       # 🔥 Gold ต้อง 75%+
    else:
        # Forex blocked above, but just in case
        min_quality = "PREMIUM"   # 🔥 Forex needs PREMIUM (very strict)
        min_confidence = 90       # 🔥 Forex needs 90%+ (almost never)
    
    quality_order = {"SKIP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "PREMIUM": 4}
    if quality_order.get(quality, 0) < quality_order.get(min_quality, 2):
        return False, f"Quality {quality} < minimum {min_quality} (for {'Gold' if is_gold else 'Forex'})"
    
    # 5. 🎯 Confidence Filter
    if confidence < min_confidence:
        return False, f"Confidence {confidence:.1f}% < minimum {min_confidence}% (for {'Gold' if is_gold else 'Forex'})"
    
    # 6. 🛡️ TREND ALIGNMENT CHECK - ห้ามสวนเทรนด์รุนแรง!
    trend_aligned, trend_reason = await _check_trend_alignment(symbol, signal)
    if not trend_aligned:
        logger.warning(f"🛡️ ANTI-WIPEOUT: {symbol} - {trend_reason}")
        return False, trend_reason
    
    # 7. 🎯 PULLBACK ENTRY CHECK - รอ pullback ก่อนเข้า (ถ้าเปิดใช้งาน)
    if _pullback_config.get("enabled", False):
        current_price = signal_data.get("current_price", 0)
        if current_price > 0:
            can_enter_pullback, pullback_reason = _check_pullback_entry(symbol, signal_data, current_price)
            if not can_enter_pullback:
                return False, f"PULLBACK: {pullback_reason}"
    
    # 8. Check for open positions
    has_position = await _check_open_positions(symbol)
    if has_position:
        return False, f"Already have open position for {symbol}"
    
    # 9. Generate signal ID
    signal_id = _generate_signal_id(symbol, signal, confidence)
    
    # 10. Check if we already traded this signal / Cooldown
    last_trade = _last_traded_signal.get(symbol)
    if last_trade:
        last_signal_id = last_trade.get("signal_id")
        last_time = last_trade.get("timestamp")
        
        # Same signal ID = duplicate
        if last_signal_id == signal_id:
            return False, f"Already traded this signal (ID: {signal_id})"
        
        # Check cooldown (5 minutes)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < _trade_cooldown_seconds:
                remaining = int(_trade_cooldown_seconds - elapsed)
                return False, f"Cooldown active ({remaining}s remaining)"
    
    return True, "OK"








async def _execute_signal_trade(symbol: str, signal_data: Dict, skip_position_check: bool = False):
    """Execute trade based on signal with duplicate prevention
    
    Args:
        symbol: Trading symbol
        signal_data: Signal data dict
        skip_position_check: If True, skip checking for existing positions (used after closing opposite position)
    """
    global _bot, _bot_status, _last_traded_signal, _pullback_config
    
    # 🔒 ACQUIRE TRADE LOCK FIRST - ป้องกัน race condition!
    if not _acquire_trade_lock(symbol):
        logger.warning(f"⛔ Trade blocked for {symbol}: Another trade in progress (LOCKED)")
        return
    
    try:
        # Double check - only execute in AUTO mode
        if _bot_status["mode"] != BotMode.AUTO.value:
            logger.warning(f"⛔ Trade blocked - not in AUTO mode")
            return
        
        
        # 🎯 PULLBACK CHECK - เฉพาะเมื่อเปิดใช้งาน
        current_price = signal_data.get("current_price", 0)
        if _pullback_config.get("enabled", False) and current_price > 0:
            can_enter_pullback, pullback_reason = _check_pullback_entry(symbol, signal_data, current_price)
            if not can_enter_pullback:
                logger.info(f"⏳ PULLBACK WAIT: {symbol} - {pullback_reason}")
                return  # ❌ ไม่เข้าเทรด - รอ pullback
        
        # 🔥 DUPLICATE PREVENTION CHECK (can skip if coming from reverse signal close)
        if not skip_position_check:
            can_trade, reason = await _can_trade_signal(symbol, signal_data)
            if not can_trade:
                logger.info(f"⛔ Trade blocked for {symbol}: {reason}")
                return
        
        # 🔒 DOUBLE CHECK: ตรวจสอบ position อีกครั้งก่อนเปิด (ป้องกัน race condition)
        has_position = await _check_open_positions(symbol)
        if has_position and not skip_position_check:
            logger.warning(f"⛔ Trade blocked for {symbol}: Position already exists (DOUBLE CHECK)")
            return
        
        if _bot and _bot.trading_engine:
            original_signal = signal_data["signal"]
            
            # 🔀 CONTRARIAN MODE - กลับสัญญาณ!
            final_signal = _apply_contrarian_mode(original_signal)
            
            # Determine side from FINAL signal (after contrarian)
            side = "BUY" if "BUY" in final_signal else "SELL"
            signal_id = _generate_signal_id(symbol, final_signal, signal_data.get("confidence", 0))
            
            if original_signal != final_signal:
                logger.info(f"🔀 CONTRARIAN MODE: Original={original_signal} → Final={final_signal}")
            
            logger.info(f"🎯 Attempting trade: {symbol} {side} (Signal ID: {signal_id})")
            
            # 🔧 Modify analysis to use reversed signal
            analysis = _bot_status["last_analysis"].get(symbol)
            if not analysis:
                logger.warning(f"⚠️ No analysis found for {symbol}")
                return
            
            # Create modified analysis with reversed signal
            modified_analysis = analysis.copy()
            modified_analysis["signal"] = final_signal
            modified_analysis["original_signal"] = original_signal
            modified_analysis["contrarian_applied"] = (original_signal != final_signal)
            
            # 🆕 UNIFIED LOT SIZING: Calculate safe lot and pass to ai_trading_bot
            # This ensures unified_bot's config is used!
            try:
                balance = await _bot.trading_engine.broker.get_balance() or 300
                current_price = signal_data.get("current_price", 0)
                sl_price = signal_data.get("stop_loss", 0)
                
                if current_price > 0 and sl_price > 0:
                    sl_distance = abs(current_price - sl_price)
                else:
                    sl_distance = 0
                
                safe_lot = _calculate_safe_lot_size(balance, symbol, sl_distance, current_price)
                modified_analysis["override_lot_size"] = safe_lot
                logger.info(f"🛡️ UNIFIED LOT: Passing safe_lot={safe_lot} to ai_trading_bot")
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate safe lot: {e}")
            
            result = await _bot.execute_trade(modified_analysis)
            
            # 🔥 FIX: Check result properly - ai_trading_bot returns {"success": True/False, "action": "EXECUTED/FAILED", ...}
            trade_success = False
            if result:
                # Check both "success" key and "action" key for backwards compatibility
                trade_success = result.get("success", False) or result.get("action") == "EXECUTED"
            
            if trade_success:
                # ✅ Record successful trade to prevent duplicates
                _last_traded_signal[symbol] = {
                    "signal": final_signal,
                    "original_signal": original_signal,
                    "signal_id": signal_id,
                    "timestamp": datetime.now(),
                    "confidence": signal_data.get("confidence", 0),
                    "side": side,
                    "contrarian": (original_signal != final_signal)
                }
                
                # 🆕 Also track the position for sync
                ticket = result.get("ticket") or result.get("order", {}).get("id")
                if ticket:
                    _known_positions[str(ticket)] = {
                        "symbol": symbol,
                        "side": side,
                        "opened_at": datetime.now().isoformat()
                    }
                    logger.info(f"📍 Position tracked: #{ticket} {symbol} {side}")
                
                contrarian_tag = " [CONTRARIAN]" if original_signal != final_signal else ""
                logger.info(f"✅ Trade executed: {symbol} {side}{contrarian_tag} (ID: {signal_id}) - Cooldown {_trade_cooldown_seconds}s started")
                _bot_status["daily_stats"]["trades"] += 1
                
                # 🔄 Force sync with MT5 after trade
                await _sync_positions_with_mt5()
            else:
                reason = result.get("reason", "Unknown") if result else "No result"
                action = result.get("action", "UNKNOWN") if result else "NO_RESULT"
                logger.warning(f"⚠️ Trade not executed: {action} - {reason}")
                
    except Exception as e:
        logger.error(f"❌ Trade execution error: {e}")
    finally:
        # 🔓 ALWAYS release lock after trade attempt
        _release_trade_lock(symbol)


async def _stop_bot_internal():
    """Internal function to stop bot"""
    global _bot, _bot_task, _bot_status
    
    _bot_status["running"] = False
    
    if _bot_task:
        _bot_task.cancel()
        try:
            await _bot_task
        except asyncio.CancelledError:
            pass
        _bot_task = None
    
    
    if _bot:
        try:
            await _bot.stop()
        except:
            pass
    
    _bot_status["mode"] = BotMode.STOPPED.value
    logger.info("?? Bot stopped internally")


# =====================
# API ENDPOINTS
# =====================

@router.get("/status")
async def get_unified_status():
    """
    📊 Get complete unified status
    
    Returns bot status, current signals, account info all in one call
    🆕 With timeout protection to prevent API freeze
    """
    global _bot, _bot_status
    
    try:
        # Get account info with timeout protection
        account = {"balance": 0, "equity": 0, "profit": 0, "free_margin": 0, "margin_level": 0}
        try:
            if _bot and _bot.trading_engine:
                # 🆕 Use asyncio.wait_for with 5 second timeout
                try:
                    balance = await asyncio.wait_for(
                        _bot.trading_engine.broker.get_balance(), 
                        timeout=5.0
                    )
                    account_info = await asyncio.wait_for(
                        _bot.trading_engine.broker.get_account_info(),
                        timeout=5.0
                    )
                    if account_info:
                        equity = account_info.get("equity", balance)
                        margin = account_info.get("margin", 0)
                        account = {
                            "balance": float(balance) if balance else 0,
                            "equity": float(equity) if equity else 0,
                            "profit": float(account_info.get("profit", 0)),
                            "free_margin": float(account_info.get("free_margin", balance or 0)),
                            "margin_level": float((equity / margin * 100) if margin and margin > 0 else 0)
                        }
                except asyncio.TimeoutError:
                    logger.warning("⏱️ Account info fetch timed out - using cached")
                    # Use last known account info from bot status if available
        except Exception as e:
            logger.warning(f"Failed to get account: {e}")
        
        # 🔧 Convert all numpy types to JSON-serializable
        return _convert_to_json_serializable({
            "bot": {
                "mode": _bot_status.get("mode", "stopped"),
                "running": _bot_status.get("running", False),
                "initialized": _bot_status.get("initialized", False),
                "symbols": _bot_status.get("symbols", []),
                "timeframe": _bot_status.get("timeframe", "H1"),
                "signal_mode": _bot_status.get("signal_mode", "technical"),
                "quality": _bot_status.get("quality", "MEDIUM"),
                "interval": _bot_status.get("interval", 60),
                "auto_trade": _bot_status.get("auto_trade", False),
                "started_at": _bot_status.get("started_at"),
                "error": _bot_status.get("error")
            },
            "signals": _bot_status.get("last_signal", {}),
            "layers": _bot_status.get("layer_status", {}),
            "daily_stats": _bot_status.get("daily_stats", {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}),
            "account": account,
            "trade_protection": _get_trade_protection_info(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting unified status: {e}")
        return {
            "bot": {
                "mode": "stopped",
                "running": False,
                "error": str(e)
            },
            "signals": {},
            "layers": {},
            "daily_stats": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "account": {"balance": 0, "equity": 0, "profit": 0, "free_margin": 0, "margin_level": 0},
            "timestamp": datetime.now().isoformat()
        }


@router.post("/start")
async def start_unified_bot(request: StartBotRequest, background_tasks: BackgroundTasks):
    """
    ?? Start the unified trading bot
    
    Modes:
    - 'auto': Bot analyzes AND auto-trades
    - 'manual': Bot analyzes only, you trade manually
    
    ?? Only ONE mode can run at a time!
    """
    global _bot, _bot_task, _bot_status
    
    # Check if already running
    if _bot_status["running"]:
        current_mode = _bot_status["mode"]
        return {
            "status": "already_running", 
            "message": f"Bot already running in {current_mode.upper()} mode. Stop it first or use /switch-mode",
            "current_mode": current_mode
        }
    
    # Validate mode
    mode = request.mode.lower()
    if mode not in ["auto", "manual"]:
        raise HTTPException(status_code=400, detail="Mode must be 'auto' or 'manual'")
    
    # Parse symbols
    symbols = [s.strip() for s in request.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = ["XAUUSDm"]
    
    # Map quality string to enum
    from ai_trading_bot import SignalQuality
    quality_map = {
        "LOW": SignalQuality.LOW,
        "MEDIUM": SignalQuality.MEDIUM,
        "HIGH": SignalQuality.HIGH,
        "PREMIUM": SignalQuality.PREMIUM
    }
    quality_enum = quality_map.get(request.quality.upper(), SignalQuality.MEDIUM)
    
    # Create/reconfigure bot
    from ai_trading_bot import AITradingBot
    _bot = AITradingBot(
        symbols=symbols,
        timeframe=request.timeframe,
        min_quality=quality_enum,
        broker_type="MT5",
        signal_mode=request.signal_mode
    )
    
    # Initialize
    try:
        await _bot.initialize()
        _bot_status["initialized"] = True
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        return {"status": "error", "message": f"Failed to initialize: {e}"}
    
    # Determine auto_trade based on mode
    auto_trade = (mode == "auto")
    
    # Update status
    _bot_status.update({
        "mode": BotMode.AUTO.value if auto_trade else BotMode.MANUAL.value,
        "running": True,
        "auto_trade": auto_trade,
        "symbols": symbols,
        "timeframe": request.timeframe,
        "signal_mode": request.signal_mode,
        "quality": request.quality,
        "interval": request.interval,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "last_signal": {},
        "last_analysis": {},
        "layer_status": {}
    })
    
    # Start background loop
    _bot_task = asyncio.create_task(
        _run_bot_loop(request.interval, auto_trade)
    )
    
    mode_icon = "??" if auto_trade else "??"
    mode_desc = "AUTO (will trade automatically)" if auto_trade else "MANUAL (analysis only)"
    logger.info(f"{mode_icon} Unified bot started: {symbols} @ {request.timeframe} - {mode_desc}")
    
    return {
        "status": "started",
        "mode": _bot_status["mode"],
        "mode_description": mode_desc,
        "symbols": symbols,
        "timeframe": request.timeframe,
        "signal_mode": request.signal_mode,
        "quality": request.quality,
        "interval": request.interval,
        "auto_trade": auto_trade
    }


@router.post("/stop")
async def stop_unified_bot():
    """
    ?? Stop the unified trading bot
    """
    global _bot, _bot_task, _bot_status
    
    if not _bot_status["running"]:
        return {"status": "not_running", "message": "Bot is not running"}
    
    previous_mode = _bot_status["mode"]
    
    # Stop the bot
    await _stop_bot_internal()
    
    logger.info(f"?? Unified bot stopped (was in {previous_mode} mode)")
    
    return {
        "status": "stopped", 
        "message": f"Bot stopped successfully (was in {previous_mode} mode)",
        "previous_mode": previous_mode
    }


@router.post("/switch-mode")
async def switch_bot_mode(request: SwitchModeRequest):
    """
    ?? Switch bot mode (AUTO <-> MANUAL)
    
    This will restart the bot in the new mode.
    """
    global _bot_status
    
    new_mode = request.mode.lower()
    if new_mode not in ["auto", "manual"]:
        raise HTTPException(status_code=400, detail="Mode must be 'auto' or 'manual'")
    
    current_mode = _bot_status["mode"]
    
    # If not running, just return info
    if not _bot_status["running"]:
        return {
            "status": "not_running",
            "message": "Bot is not running. Use /start to start it with desired mode.",
            "requested_mode": new_mode
        }
    
    # If same mode, do nothing
    if (new_mode == "auto" and current_mode == BotMode.AUTO.value) or \
       (new_mode == "manual" and current_mode == BotMode.MANUAL.value):
        return {
            "status": "no_change",
            "message": f"Bot is already in {new_mode.upper()} mode",
            "current_mode": current_mode
        }
    
    # Stop current bot
    logger.info(f"?? Switching mode: {current_mode} ? {new_mode}")
    
    # Save current settings
    symbols = _bot_status["symbols"]
    timeframe = _bot_status["timeframe"]
    signal_mode = _bot_status["signal_mode"]
    quality = _bot_status["quality"]
    interval = _bot_status["interval"]
    
    # Stop
    await _stop_bot_internal()
    
    # Wait a bit
    await asyncio.sleep(0.5)
    
    # Restart with new mode
    auto_trade = (new_mode == "auto")
    
    # Reinitialize
    from ai_trading_bot import AITradingBot, SignalQuality
    quality_map = {
        "LOW": SignalQuality.LOW,
        "MEDIUM": SignalQuality.MEDIUM,
        "HIGH": SignalQuality.HIGH,
        "PREMIUM": SignalQuality.PREMIUM
    }
    quality_enum = quality_map.get(quality.upper(), SignalQuality.MEDIUM)
    
    global _bot, _bot_task
    _bot = AITradingBot(
        symbols=symbols,
        timeframe=timeframe,
        min_quality=quality_enum,
        broker_type="MT5",
        signal_mode=signal_mode
    )
    await _bot.initialize()
    
    # Update status
    _bot_status.update({
        "mode": BotMode.AUTO.value if auto_trade else BotMode.MANUAL.value,
        "running": True,
        "auto_trade": auto_trade,
        "initialized": True,
        "error": None,
        "started_at": datetime.now().isoformat()
    })
    
    # Start loop
    _bot_task = asyncio.create_task(
        _run_bot_loop(interval, auto_trade)
    )
    
    mode_desc = "AUTO (will trade automatically)" if auto_trade else "MANUAL (analysis only)"
    logger.info(f"? Mode switched to {new_mode.upper()}")
    
    return {
        "status": "switched",
        "previous_mode": current_mode,
        "new_mode": _bot_status["mode"],
        "mode_description": mode_desc,
        "auto_trade": auto_trade,
        "message": f"Successfully switched to {new_mode.upper()} mode"
    }
@router.get("/signal/{symbol}")
async def get_signal_for_symbol(symbol: str):
    """
    📊 Get current signal for a specific symbol
    """
    global _bot_status
    
    try:
        signal = _bot_status["last_signal"].get(symbol)
        
        if not signal:
            return {
                "status": "no_signal",
                "symbol": symbol,
                "signal": "WAIT",
                "confidence": 0,
                "quality": "SKIP",
                "bot_mode": _bot_status["mode"],
                "message": "No analysis available for this symbol. Start the bot first."
            }
        
        # Build response safely - 🐛 FIX: Handle None values and numpy types properly
        # Convert buy_score/sell_score to native Python int (could be numpy.int64)
        buy_score_raw = signal.get("buy_score") or 0
        sell_score_raw = signal.get("sell_score") or 0
        buy_score = int(buy_score_raw) if hasattr(buy_score_raw, '__int__') else 0
        sell_score = int(sell_score_raw) if hasattr(sell_score_raw, '__int__') else 0
        
        response = {
            "status": "ok",
            "bot_mode": _bot_status["mode"],
            "symbol": signal.get("symbol", symbol),
            "signal": signal.get("signal", "WAIT"),
            "confidence": float(signal.get("confidence") or 0),
            "quality": signal.get("quality", "SKIP"),
            "current_price": float(signal.get("current_price") or 0),
            "stop_loss": float(signal.get("stop_loss") or 0),
            "take_profit": float(signal.get("take_profit") or 0),
            "trade_status": signal.get("trade_status", "N/A"),
            "market_regime": signal.get("market_regime", "UNKNOWN"),
            "timestamp": signal.get("timestamp", datetime.now().isoformat()),
            # 🔥 NEW: Add explicit fields for frontend
            "buy_score": buy_score,
            "sell_score": sell_score,
            "session": signal.get("session") or "N/A",
            "trend": signal.get("trend") or signal.get("market_regime") or "UNKNOWN",
        }
        
        # 🔔 ADD HEALTH INFO (Early Warning System)
        if "health" in signal:
            response["health"] = _convert_to_json_serializable(signal["health"])
        else:
            # Generate health on-the-fly if not present
            health = _get_signal_health_summary(symbol)
            response["health"] = _convert_to_json_serializable(health)
        
        # Add optional fields if present
        if "scores" in signal:
            response["scores"] = _convert_to_json_serializable(signal["scores"])
        if "indicators" in signal:
            response["indicators"] = _convert_to_json_serializable(signal["indicators"])
        if "factors" in signal:
            response["factors"] = _convert_to_json_serializable(signal["factors"])
        
        # 🐛 FIX: Ensure entire response is JSON serializable (handles any remaining numpy types)
        return _convert_to_json_serializable(response)
        
    except Exception as e:
        logger.error(f"Error getting signal for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "signal": "WAIT",
            "confidence": 0,
            "quality": "SKIP",
            "bot_mode": _bot_status.get("mode", "stopped"),
            "error": str(e),
            "message": f"Error fetching signal: {str(e)}"
        }


@router.get("/layers/{symbol}")
async def get_layers_for_symbol(symbol: str):
    """
    🏗️ Get 20-layer status for a specific symbol
    """
    global _bot_status
    
    try:
        layers = _bot_status["layer_status"].get(symbol)
        
        if not layers:
            return {
                "status": "no_data",
                "symbol": symbol,
                "layers": [],
                "passed": 0,
                "total": 20,
                "pass_rate": 0,
                "bot_mode": _bot_status["mode"],
                "message": "No layer data available. Start the bot first."
            }
        
        return _convert_to_json_serializable({
            "status": "ok",
            "symbol": symbol,
            "bot_mode": _bot_status["mode"],
            "layers": layers.get("layers", []),
            "passed": layers.get("passed", 0),
            "total": layers.get("total", 20),
            "pass_rate": layers.get("pass_rate", 0)
        })
        
    except Exception as e:
        logger.error(f"Error getting layers for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "layers": [],
            "passed": 0,
            "total": 20,
            "pass_rate": 0,
            "bot_mode": _bot_status.get("mode", "stopped"),
            "error": str(e)
        }


@router.get("/analysis/{symbol}")
async def get_full_analysis(symbol: str):
    """
    ?? Get full analysis data for a symbol
    """
    global _bot_status
    
    analysis = _bot_status["last_analysis"].get(symbol)
    
    if not analysis:
        return {
            "status": "no_analysis",
            "symbol": symbol,
            "bot_mode": _bot_status["mode"],
            "message": "Run bot to get analysis"
        }
    
    return {
        "status": "ok",
        "symbol": symbol,
        "bot_mode": _bot_status["mode"],
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/analyze/{symbol}")
async def analyze_symbol_now(symbol: str):
    """
    ?? Trigger immediate analysis for a symbol (one-shot)
    
    This works even if bot is not running.
    """
    global _bot, _bot_status
    
    if not _bot:
        _bot = _get_bot_instance()
        await _bot.initialize()
    
    try:
        analysis = await _bot.analyze_symbol(symbol)
        
        if analysis:
            # Update global status
            _bot_status["last_analysis"][symbol] = analysis
            _bot_status["last_signal"][symbol] = {
                "symbol": symbol,
                "signal": analysis.get("signal", "WAIT"),
                "confidence": analysis.get("enhanced_confidence", 0),
                "quality": analysis.get("quality", "SKIP"),
                "current_price": analysis.get("current_price", 0),
                "stop_loss": analysis.get("risk_management", {}).get("stop_loss", 0),
                "take_profit": analysis.get("risk_management", {}).get("take_profit", 0),
                "timestamp": datetime.now().isoformat()
            }
            _bot_status["layer_status"][symbol] = _extract_layer_status(symbol)
            
            return {
                "status": "ok",
                "symbol": symbol,
                "signal": analysis.get("signal", "WAIT"),
                "confidence": analysis.get("enhanced_confidence", 0),
                "quality": analysis.get("quality", "SKIP"),
                "bot_mode": _bot_status["mode"],
                "analysis": analysis
            }
        else:
            return {
                "status": "no_signal",
                "symbol": symbol,
                "message": "Analysis returned no signal"
            }
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "error": str(e)
        }


@router.post("/execute")
async def execute_manual_trade(request: ManualTradeRequest):
    """
    ?? Execute a trade manually
    
    Works in any mode (even MANUAL mode for manual trading)
    """
    global _bot, _bot_status
    
    if not _bot or not _bot.trading_engine:
        raise HTTPException(status_code=400, detail="Bot not initialized. Start bot first.")
    
    try:
        side = request.side.upper()
        if side not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Side must be 'BUY' or 'SELL'")
        
        result = await _bot.execute_trade(
            symbol=request.symbol,
            side=side,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            confidence=100  # Manual trade = 100% confidence
        )
        
        if result and result.get("success"):
            _bot_status["daily_stats"]["trades"] += 1
            logger.info(f"? Manual trade executed: {request.symbol} {side}")
            return {
                "status": "success",
                "message": f"Trade executed: {side} {request.symbol}",
                "result": result
            }
        else:
            return {
                "status": "failed",
                "message": f"Trade not executed: {result.get('reason', 'Unknown')}",
                "result": result
            }
            
    except Exception as e:
        logger.error(f"Manual trade error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# MODE INFO ENDPOINT
# =====================

@router.get("/modes")
async def get_available_modes():
    """
    ?? Get information about available bot modes
    """
    return {
        "modes": {
            "auto": {
                "name": "AUTO",
                "description": "Bot analyzes market AND executes trades automatically",
                "auto_trade": True,
                "icon": "??"
            },
            "manual": {
                "name": "MANUAL", 
                "description": "Bot analyzes market only. You execute trades manually",
                "auto_trade": False,
                "icon": "??"
            }
        },
        "current_mode": _bot_status["mode"],
        "running": _bot_status["running"],
        "note": "Only ONE mode can be active at a time. Use /switch-mode to change."
    }


# =====================
# TRADE PROTECTION STATUS
# =====================

@router.get("/protection")
async def get_trade_protection_status():
    """
    ??? Get trade protection status (cooldowns, last trades, etc.)
    """
    global _last_traded_signal, _trade_cooldown_seconds
    
    protection_status = {}
    
    for symbol, last_trade in _last_traded_signal.items():
        if last_trade:
            last_time = last_trade.get("timestamp")
            if last_time:
                elapsed = (datetime.now() - last_time).total_seconds()
                cooldown_remaining = max(0, _trade_cooldown_seconds - elapsed)
                can_trade = cooldown_remaining == 0
            else:
                elapsed = 0
                cooldown_remaining = 0
                can_trade = True
            
            protection_status[symbol] = {
                "last_signal": last_trade.get("signal"),
                "last_signal_id": last_trade.get("signal_id"),
                "last_trade_time": last_time.isoformat() if last_time else None,
                "elapsed_seconds": int(elapsed),
                "cooldown_remaining": int(cooldown_remaining),
                "can_trade_now": can_trade,
                "last_side": last_trade.get("side"),
                "last_confidence": last_trade.get("confidence", 0)
            }
    
    return {
        "cooldown_seconds": _trade_cooldown_seconds,
        "symbols": protection_status,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/protection/reset")
async def reset_trade_protection(symbol: str = None):
    """
    ?? Reset trade protection (cooldown) for a symbol or all symbols
    """
    global _last_traded_signal
    
    if symbol:
        if symbol in _last_traded_signal:
            del _last_traded_signal[symbol]
            return {"status": "reset", "symbol": symbol}
        else:
            return {"status": "not_found", "symbol": symbol}
    else:
        _last_traded_signal.clear()
        return {"status": "reset_all"}


# =====================
# 🔀 CONTRARIAN MODE
# =====================

@router.get("/contrarian")
async def get_contrarian_status():
    """
    🔀 Get Contrarian Mode status
    
    Contrarian = กลับสัญญาณ (BUY→SELL, SELL→BUY)
    ใช้เมื่อสัญญาณเดิมผิดบ่อย
    """
    global _contrarian_mode
    
    return {
        "config": _contrarian_mode,
        "description": "กลับสัญญาณอัตโนมัติ - ถ้าระบบบอก BUY จะเทรด SELL แทน",
        "mapping": {
            "BUY": "SELL",
            "SELL": "BUY",
            "STRONG_BUY": "STRONG_SELL",
            "STRONG_SELL": "STRONG_BUY"
        }
    }


@router.post("/contrarian/toggle")
async def toggle_contrarian_mode(enabled: bool = True):
    """
    🔀 Enable/Disable Contrarian Mode
    
    - enabled=true: กลับสัญญาณ (BUY→SELL, SELL→BUY)
    - enabled=false: ใช้สัญญาณปกติ
    """
    global _contrarian_mode
    
    _contrarian_mode["enabled"] = enabled
    
    status = "ENABLED 🔀" if enabled else "DISABLED"
    logger.info(f"🔀 Contrarian Mode: {status}")
    
    return {
        "status": "success",
        "contrarian_enabled": enabled,
        "message": f"Contrarian Mode {status}",
        "note": "BUY→SELL, SELL→BUY" if enabled else "Using original signals"
    }


@router.post("/contrarian/configure")
async def configure_contrarian_mode(
    enabled: bool = None,
    reverse_signal: bool = None,
    reverse_strong_signal: bool = None
):
    """
    🔀 Configure Contrarian Mode
    
    - enabled: เปิด/ปิด Contrarian Mode
    - reverse_signal: กลับ BUY/SELL
    - reverse_strong_signal: กลับ STRONG_BUY/STRONG_SELL
    """
    global _contrarian_mode
    
    changes = []
    
    if enabled is not None:
        _contrarian_mode["enabled"] = enabled
        changes.append(f"enabled: {enabled}")
    
    if reverse_signal is not None:
        _contrarian_mode["reverse_signal"] = reverse_signal
        changes.append(f"reverse_signal: {reverse_signal}")
    
    if reverse_strong_signal is not None:
        _contrarian_mode["reverse_strong_signal"] = reverse_strong_signal
        changes.append(f"reverse_strong_signal: {reverse_strong_signal}")
    
    logger.info(f"🔀 Contrarian config updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _contrarian_mode
    }


# =====================
# REVERSE SIGNAL CLOSE
# =====================

@router.get("/reverse-signal")
async def get_reverse_signal_status():
    """
    🔄 Get reverse signal close status
    """
    global _enable_reverse_signal_close
    
    return {
        "enabled": _enable_reverse_signal_close,
        "description": "ปิด position อัตโนมัติเมื่อสัญญาณมาตรงข้าม",
        "example": "มี SELL อยู่ + สัญญาณ BUY มา → ปิด SELL ทันที"
    }


@router.post("/reverse-signal/toggle")
async def toggle_reverse_signal_close(enabled: bool = True):
    """
    🔄 Enable/Disable reverse signal close feature
    
    - enabled=true: ปิด position เมื่อสัญญาณตรงข้าม (แนะนำ)
    - enabled=false: ไม่ปิดอัตโนมัติ
    """
    global _enable_reverse_signal_close
    
    
    _enable_reverse_signal_close = enabled
    
    status = "enabled" if enabled else "disabled"
    logger.info(f"🔄 Reverse Signal Close: {status}")
    
    return {
        "status": "success",
        "reverse_signal_close": enabled,
        "message": f"Reverse signal close {status}"
    }


# =====================
# 🎯 PULLBACK ENTRY STRATEGY
# =====================

@router.get("/pullback")
async def get_pullback_status():
    """
    🎯 Get Pullback Entry configuration and pending signals
    
    Pullback = รอให้ราคาย่อตัวก่อนเข้าเทรด
    - BUY signal มา → รอราคาลงก่อน → เด้งกลับ → เข้า BUY
    - SELL signal มา → รอราคาขึ้นก่อน → กลับลง → เข้า SELL
    """
    global _pullback_config, _pending_signals
    
    # Format pending signals for display
    pending_display = {}
    for symbol, data in _pending_signals.items():
        pending_display[symbol] = {
            "signal": data.get("signal"),
            "price_at_signal": data.get("price_at_signal"),
            "lowest_price": data.get("lowest_price"),
            "highest_price": data.get("highest_price"),
            "pullback_detected": data.get("pullback_detected", False),
            "stable_count": data.get("stable_count", 0),
            "timestamp": data.get("timestamp").isoformat() if data.get("timestamp") else None,
            "age_seconds": int((datetime.now() - data.get("timestamp")).total_seconds()) if data.get("timestamp") else 0,
        }
    
    return {
        "config": _pullback_config,
        "pending_signals": pending_display,
        "description": {
            "enabled": "เปิด/ปิด pullback entry",
            "min_pullback_percent": "ราคาต้องย่อขั้นต่ำ X% ก่อนเข้า",
            "max_pullback_percent": "ถ้าย่อเกิน X% = signal ผิด ยกเลิก",
            "wait_for_stabilization": "รอให้ราคานิ่งก่อนเข้า",
            "stabilization_candles": "จำนวนแท่งที่ต้องเด้งกลับ",
            "max_wait_minutes": "รอสูงสุดกี่นาที",
        },
        "gold_example": {
            "price": 3300,
            "min_pullback": f"${3300 * 0.03 / 100:.2f} (0.03%)",
            "max_pullback": f"${3300 * 0.20 / 100:.2f} (0.20%)",
        }
    }


@router.post("/pullback/toggle")
async def toggle_pullback_entry(enabled: bool = True):
    """
    🎯 Enable/Disable Pullback Entry Strategy
    
    - enabled=true: รอ pullback ก่อนเข้าเทรด (Entry ดีกว่า!)
    - enabled=false: เข้าเทรดทันทีเมื่อมีสัญญาณ
    """
    global _pullback_config, _pending_signals
    
    _pullback_config["enabled"] = enabled
    
    # Clear pending signals when disabled
    if not enabled:
        cleared = len(_pending_signals)
        _pending_signals.clear()
        logger.info(f"🎯 Pullback disabled - cleared {cleared} pending signals")
    
    status = "ENABLED ✅" if enabled else "DISABLED ❌"
    logger.info(f"🎯 Pullback Entry: {status}")
    
    return {
        "status": "success",
        "pullback_enabled": enabled,
        "message": f"Pullback Entry {status}",
        "note": "รอ pullback = Entry ที่ราคาดีกว่า!" if enabled else "เข้าเทรดทันทีเมื่อมีสัญญาณ"
    }


@router.post("/pullback/configure")
async def configure_pullback(
    min_pullback_percent: float = None,
    max_pullback_percent: float = None,
    wait_for_stabilization: bool = None,
    stabilization_candles: int = None,
    max_wait_minutes: int = None
):
    """
    🎯 Configure Pullback Entry Strategy
    
    - min_pullback_percent: ราคาต้องย่อขั้นต่ำ X% (default: 0.03 = ~$1 Gold)
    - max_pullback_percent: ย่อเกิน X% = ยกเลิก signal (default: 0.20 = ~$7 Gold)
    - wait_for_stabilization: รอให้ราคานิ่งก่อนเข้า
    - stabilization_candles: จำนวนแท่งที่ต้องเด้งกลับ (1-3)
    - max_wait_minutes: รอสูงสุดกี่นาที (5-30)
    """
    global _pullback_config
    
    changes = []
    
    if min_pullback_percent is not None:
        _pullback_config["min_pullback_percent"] = max(0.01, min(0.5, min_pullback_percent))
        changes.append(f"min_pullback: {_pullback_config['min_pullback_percent']}%")
    
    if max_pullback_percent is not None:
        _pullback_config["max_pullback_percent"] = max(0.1, min(2.0, max_pullback_percent))
        changes.append(f"max_pullback: {_pullback_config['max_pullback_percent']}%")
    
    if wait_for_stabilization is not None:
        _pullback_config["wait_for_stabilization"] = wait_for_stabilization
        changes.append(f"wait_for_stabilization: {wait_for_stabilization}")
    
    if stabilization_candles is not None:
        _pullback_config["stabilization_candles"] = max(1, min(5, stabilization_candles))
        changes.append(f"stabilization_candles: {_pullback_config['stabilization_candles']}")
    
    if max_wait_minutes is not None:
        _pullback_config["max_wait_minutes"] = max(3, min(60, max_wait_minutes))
        changes.append(f"max_wait_minutes: {_pullback_config['max_wait_minutes']}")
    
    logger.info(f"🎯 Pullback config updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _pullback_config
    }


@router.post("/pullback/clear")
async def clear_pending_signals(symbol: str = None):
    """
    🎯 Clear pending pullback signals
    
    - symbol: Clear เฉพาะ symbol นี้
    - ไม่ระบุ: Clear ทั้งหมด
    """
    global _pending_signals
    
    if symbol:
        if symbol in _pending_signals:
            del _pending_signals[symbol]
            return {"status": "cleared", "symbol": symbol}
        else:
            return {"status": "not_found", "symbol": symbol}
    else:
        count = len(_pending_signals)
        _pending_signals.clear()
        return {"status": "cleared_all", "count": count}





# =====================
# 📊 SIGNAL STRENGTH ENDPOINTS
# =====================

@router.get("/signal-strength/{symbol}")
async def get_signal_strength(symbol: str):
    """
    📊 Get Signal Strength for a symbol
    
    Shows:
    - Strength score (0-100)
    - Recommendation (STRONG/OK/WEAK/DANGER)
    - Trend analysis (rising/falling/stable)
    - Direction changes
    
    Use this to determine if it's safe to trade/DCA
    """
    global _bot_status, _signal_strength_tracker
    
    signal = _bot_status.get("last_signal", {}).get(symbol)
    
    if not signal:
        return {
            "status": "no_signal",
            "symbol": symbol,
            "message": "No signal data available. Start bot first."
        }
    
    # Get strength score
    strength = _get_signal_strength_score(symbol, signal)
    
    # Get tracker data
    tracker = _signal_strength_tracker.get(symbol, {})
    
    return {
        "status": "ok",
        "symbol": symbol,
        "strength": strength,
        "history": {
            "confidence": tracker.get("confidence_history", [])[-10:],
            "quality": tracker.get("quality_history", [])[-10:],
            "signals": tracker.get("signal_history", [])[-10:],
            "direction_changes": tracker.get("direction_changes", 0),
        },
        "is_weakening": _check_signal_weakening_for_dca(symbol, signal),
        "safe_to_dca": strength["score"] >= 70 and not _check_signal_weakening_for_dca(symbol, signal),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/signal-strength")
async def get_all_signal_strengths():
    """
    📊 Get Signal Strength for all tracked symbols
    """
    global _bot_status, _signal_strength_tracker
    
    results = {}
    
    for symbol in _bot_status.get("symbols", []):
        signal = _bot_status.get("last_signal", {}).get(symbol)
        if signal:
            strength = _get_signal_strength_score(symbol, signal)
            is_weakening = _check_signal_weakening_for_dca(symbol, signal)
            results[symbol] = {
                "score": strength["score"],
                "recommendation": strength["recommendation"],
                "signal": signal.get("signal", "WAIT"),
                "confidence": signal.get("confidence", 0),
                "quality": signal.get("quality", "SKIP"),
                "is_weakening": is_weakening,
                "safe_to_dca": strength["score"] >= 70 and not is_weakening,
            }
    
    return {
        "status": "ok",
        "symbols": results,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/signal-strength/reset/{symbol}")
async def reset_signal_strength_tracking(symbol: str = None):
    """
    📊 Reset signal strength tracking
    """
    global _signal_strength_tracker
    
    if symbol:
        if symbol in _signal_strength_tracker:
            del _signal_strength_tracker[symbol]
            return {"status": "reset", "symbol": symbol}
        else:
            return {"status": "not_found", "symbol": symbol}
    else:
        count = len(_signal_strength_tracker)
        _signal_strength_tracker.clear()
        return {"status": "reset_all", "cleared_count": count}


# =====================
# 🔔 SIGNAL HEALTH ENDPOINTS - Early Warning System
# =====================

@router.get("/signal-health/{symbol}")
async def get_signal_health(symbol: str):
    """
    📊 Get Signal Health for a symbol
    
    Shows:
    - Current vs Peak confidence
    - Momentum (RISING/STABLE/FALLING/FADING)
    - Trend direction
    - Alert level (OK/WARNING/DANGER)
    - Early warning if signal is weakening
    
    Use this to monitor signal strength and get early warnings!
    """
    global _bot_status, _signal_health, _signal_strength_tracker
    
    signal = _bot_status.get("last_signal", {}).get(symbol)
    
    if not signal:
        return {
            "status": "no_signal",
            "symbol": symbol,
            "message": "No signal data. Start bot first."
        }
    
    # Get health summary
    health = _get_signal_health_summary(symbol)
    
    # Get history for chart
    tracker = _signal_strength_tracker.get(symbol, {})
    
    return _convert_to_json_serializable({
        "status": "ok",
        "symbol": symbol,
        **health,
        "history": {
            "confidence": tracker.get("confidence_history", [])[-20:],
            "quality": tracker.get("quality_history", [])[-20:],
            "direction_changes": tracker.get("direction_changes", 0),
        },
        "recommendations": _get_health_recommendations(health),
        "timestamp": datetime.now().isoformat()
    })


@router.get("/signal-health")
async def get_all_signal_health():
    """
    📊 Get Signal Health for all tracked symbols
    """
    global _bot_status, _signal_health
    
    results = {}
    
    for symbol in _bot_status.get("symbols", []):
        health = _get_signal_health_summary(symbol)
        results[symbol] = health
    
    # Count alerts
    danger_count = sum(1 for h in results.values() if h.get("alert_level") == "DANGER")
    warning_count = sum(1 for h in results.values() if h.get("alert_level") == "WARNING")
    
    return _convert_to_json_serializable({
        "status": "ok",
        "symbols": results,
        "summary": {
            "total_symbols": len(results),
            "danger_alerts": danger_count,
            "warning_alerts": warning_count,
            "ok_count": len(results) - danger_count - warning_count,
        },
        "timestamp": datetime.now().isoformat()
    })


@router.post("/signal-health/reset/{symbol}")
async def reset_signal_health(symbol: str = None):
    """
    🔄 Reset signal health tracking (clears peak values)
    """
    global _signal_health
    
    if symbol:
        if symbol in _signal_health:
            del _signal_health[symbol]
            return {"status": "reset", "symbol": symbol}
        else:
            return {"status": "not_found", "symbol": symbol}
    else:
        count = len(_signal_health)
        _signal_health.clear()
        return {"status": "reset_all", "cleared_count": count}


@router.post("/signal-fade/configure")
async def configure_signal_fade_alerts(
    enabled: bool = None,
    confidence_drop_threshold: float = None,
    alert_on_quality_drop: bool = None,
    momentum_window_size: int = None,
):
    """
    🔔 Configure Signal Fade Alert settings
    
    - enabled: Enable/disable fade detection
    - confidence_drop_threshold: Alert when dropped X% from peak (default: 10)
    - alert_on_quality_drop: Alert when quality drops
    - momentum_window_size: Number of readings for momentum calc (default: 5)
    """
    global _signal_fade_config
    
    changes = []
    
    if enabled is not None:
        _signal_fade_config["enabled"] = enabled
        changes.append(f"enabled: {enabled}")
    
    if confidence_drop_threshold is not None:
        _signal_fade_config["confidence_drop_threshold_percent"] = max(5, min(30, confidence_drop_threshold))
        changes.append(f"threshold: {_signal_fade_config['confidence_drop_threshold_percent']}%")
    
    if alert_on_quality_drop is not None:
        _signal_fade_config["alert_on_quality_drop"] = alert_on_quality_drop
        changes.append(f"quality_alert: {alert_on_quality_drop}")
    
    if momentum_window_size is not None:
        _signal_fade_config["momentum_window_size"] = max(3, min(10, momentum_window_size))
        changes.append(f"window: {_signal_fade_config['momentum_window_size']}")
    
    logger.info(f"🔔 Signal fade config updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _signal_fade_config
    }


@router.get("/signal-fade/auto-action")
async def get_signal_fade_auto_action_config():
    """
    🤖 Get Signal Fade Auto-Action configuration
    
    Shows automatic actions when signal fades:
    - WARNING → Block new trades + Move SL to break-even
    - DANGER → Block new trades + Close profitable positions
    """
    global _signal_fade_config
    
    return {
        "config": {
            "auto_action_enabled": _signal_fade_config.get("auto_action_enabled", True),
            "block_new_trades_on_warning": _signal_fade_config.get("block_new_trades_on_warning", True),
            "block_new_trades_on_danger": _signal_fade_config.get("block_new_trades_on_danger", True),
            "move_sl_to_breakeven_on_warning": _signal_fade_config.get("move_sl_to_breakeven_on_warning", True),
            "close_profitable_on_danger": _signal_fade_config.get("close_profitable_on_danger", True),
            "min_profit_percent_to_close_on_danger": _signal_fade_config.get("min_profit_percent_to_close_on_danger", 5.0),
            "min_profit_to_move_sl": _signal_fade_config.get("min_profit_to_move_sl", 0.5),
        },
        "description": {
            "auto_action_enabled": "เปิด/ปิด auto-action ทั้งหมด",
            "block_new_trades_on_warning": "Block การเปิด position ใหม่เมื่อ WARNING",
            "block_new_trades_on_danger": "Block การเปิด position ใหม่เมื่อ DANGER",
            "move_sl_to_breakeven_on_warning": "ย้าย SL มา break-even เมื่อ WARNING (ต้องมีกำไร)",
            "close_profitable_on_danger": "ปิด position ที่มีกำไรเมื่อ DANGER",
            "min_profit_percent_to_close_on_danger": "% กำไรขั้นต่ำ (ของ balance) ก่อนปิดเมื่อ DANGER",
            "min_profit_to_move_sl": "% กำไรขั้นต่ำ (ของ balance) ก่อนย้าย SL",
        },
        "actions_summary": {
            "OK": "✅ เทรดตามปกติ",
            "WARNING": "🟡 Block เปิดใหม่ + ย้าย SL มา break-even (ถ้ามีกำไร)",
            "DANGER": "🔴 Block เปิดใหม่ + ปิด position ที่มีกำไร >= X%",
        }
    }


@router.post("/signal-fade/auto-action/configure")
async def configure_signal_fade_auto_action(
    auto_action_enabled: bool = None,
    block_on_warning: bool = None,
    block_on_danger: bool = None,
    move_sl_on_warning: bool = None,
    close_on_danger: bool = None,
    min_profit_percent_to_close: float = None,
    min_profit_to_move_sl: float = None,
):
    """
    🤖 Configure Signal Fade Auto-Action
    
    - auto_action_enabled: เปิด/ปิด auto-action ทั้งหมด
    - block_on_warning: Block เปิด position ใหม่เมื่อ WARNING
    - block_on_danger: Block เปิด position ใหม่เมื่อ DANGER
    - move_sl_on_warning: ย้าย SL มา break-even เมื่อ WARNING
    - close_on_danger: ปิด position ที่มีกำไรเมื่อ DANGER
    - min_profit_percent_to_close: % กำไรขั้นต่ำก่อนปิด (1-20%)
    - min_profit_to_move_sl: % กำไรขั้นต่ำก่อนย้าย SL (0.1-5%)
    """
    global _signal_fade_config
    
    changes = []
    
    if auto_action_enabled is not None:
        _signal_fade_config["auto_action_enabled"] = auto_action_enabled
        changes.append(f"auto_action: {'ON' if auto_action_enabled else 'OFF'}")
    
    if block_on_warning is not None:
        _signal_fade_config["block_new_trades_on_warning"] = block_on_warning
        changes.append(f"block_on_warning: {block_on_warning}")
    
    if block_on_danger is not None:
        _signal_fade_config["block_new_trades_on_danger"] = block_on_danger
        changes.append(f"block_on_danger: {block_on_danger}")
    
    if move_sl_on_warning is not None:
        _signal_fade_config["move_sl_to_breakeven_on_warning"] = move_sl_on_warning
        changes.append(f"move_sl_on_warning: {move_sl_on_warning}")
    
    if close_on_danger is not None:
        _signal_fade_config["close_profitable_on_danger"] = close_on_danger
        changes.append(f"close_on_danger: {close_on_danger}")
    
    if min_profit_percent_to_close is not None:
        _signal_fade_config["min_profit_percent_to_close_on_danger"] = max(1, min(20, min_profit_percent_to_close))
        changes.append(f"min_profit_to_close: {_signal_fade_config['min_profit_percent_to_close_on_danger']}%")
    
    if min_profit_to_move_sl is not None:
        _signal_fade_config["min_profit_to_move_sl"] = max(0.1, min(5, min_profit_to_move_sl))
        changes.append(f"min_profit_to_move_sl: {_signal_fade_config['min_profit_to_move_sl']}%")
    
    logger.info(f"🤖 Signal fade auto-action configured: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": {
            "auto_action_enabled": _signal_fade_config.get("auto_action_enabled"),
            "block_new_trades_on_warning": _signal_fade_config.get("block_new_trades_on_warning"),
            "block_new_trades_on_danger": _signal_fade_config.get("block_new_trades_on_danger"),
            "move_sl_to_breakeven_on_warning": _signal_fade_config.get("move_sl_to_breakeven_on_warning"),
            "close_profitable_on_danger": _signal_fade_config.get("close_profitable_on_danger"),
            "min_profit_percent_to_close_on_danger": _signal_fade_config.get("min_profit_percent_to_close_on_danger"),
            "min_profit_to_move_sl": _signal_fade_config.get("min_profit_to_move_sl"),
        }
    }


@router.post("/signal-fade/auto-action/toggle")
async def toggle_signal_fade_auto_action(enabled: bool = True):
    """
    🤖 Enable/Disable Signal Fade Auto-Action
    
    - enabled=true: เปิด auto-action (ปิด position อัตโนมัติเมื่อ signal fading)
    - enabled=false: ปิด auto-action (แจ้งเตือนอย่างเดียว ไม่จัดการ position)
    """
    global _signal_fade_config
    
    _signal_fade_config["auto_action_enabled"] = enabled
    
    status = "🤖 ENABLED - จัดการ position อัตโนมัติ" if enabled else "📢 DISABLED - แจ้งเตือนอย่างเดียว"
    logger.info(f"🤖 Signal Fade Auto-Action: {status}")
    
    return {
        "status": "success",
        "auto_action_enabled": enabled,
        "message": status,
        "actions_when_enabled": {
            "WARNING": "Block เปิดใหม่ + ย้าย SL มา break-even",
            "DANGER": "Block เปิดใหม่ + ปิด position ที่มีกำไร >= 5%",
        }
    }


# =====================
# 🎯 AGGRESSIVE TRADING MODE
# =====================

# =====================
# 🎯 SCORE GAP FILTER - ป้องกันสัญญาณไม่ชัดเจน
# =====================

@router.get("/score-gap")
async def get_score_gap_config():
    """
    🎯 Get Score Gap Filter configuration
    
    Score Gap Filter ป้องกันสัญญาณที่ไม่ชัดเจน:
    - ถ้า Buy=5 vs Sell=6 (gap=1) → ไม่เทรด! ไม่รู้จะไปทางไหน
    - ถ้า Buy=8 vs Sell=3 (gap=5) → เทรด BUY! ชัดเจนมาก
    """
    global _score_gap_config
    
    return {
        "config": _score_gap_config,
        "description": {
            "min_score_gap_gold": "🥇 Gold: Gap ขั้นต่ำระหว่าง Buy/Sell Score",
            "min_score_gap_forex": "💱 Forex: Gap ขั้นต่ำระหว่าง Buy/Sell Score",
            "min_dominant_score_gold": "🥇 Gold: Score ที่ชนะต้อง >= X",
            "min_dominant_score_forex": "💱 Forex: Score ที่ชนะต้อง >= X",
            "confidence_bonus_gap_5": "📈 Gap >= 5 = +X% confidence",
            "confidence_penalty_gap_2": "📉 Gap 2 = X% confidence penalty",
        },
        "examples": {
            "blocked": "Buy=5, Sell=6, Gap=1 → ❌ BLOCKED (gap < 2)",
            "weak": "Buy=7, Sell=5, Gap=2 → ⚠️ LOW confidence (-5%)",
            "good": "Buy=8, Sell=4, Gap=4 → ✅ HIGH confidence (+5%)",
            "strong": "Buy=10, Sell=3, Gap=7 → 🔥 PREMIUM confidence (+10%)",
        }
    }


@router.post("/score-gap/toggle")
async def toggle_score_gap_filter(enabled: bool = True):
    """
    🎯 Enable/Disable Score Gap Filter
    
    - enabled=true: Block สัญญาณที่ไม่ชัดเจน (แนะนำ!)
    - enabled=false: อนุญาตสัญญาณทั้งหมด (เสี่ยงสูง)
    """
    global _score_gap_config
    
    _score_gap_config["enabled"] = enabled
    
    status = "ENABLED ✅" if enabled else "DISABLED ⚠️"
    logger.info(f"🎯 Score Gap Filter: {status}")
    
    return {
        "status": "success",
        "score_gap_filter_enabled": enabled,
        "message": f"Score Gap Filter {status}",
        "warning": "⚠️ Disabling may increase false signals!" if not enabled else None
    }


@router.post("/score-gap/configure")
async def configure_score_gap_filter(
    min_score_gap_gold: int = None,
    min_score_gap_forex: int = None,
    min_dominant_score_gold: int = None,
    min_dominant_score_forex: int = None,
):
    """
    🎯 Configure Score Gap Filter
    
    - min_score_gap_gold: Gap ขั้นต่ำสำหรับ Gold (1-5, default: 2)
    - min_score_gap_forex: Gap ขั้นต่ำสำหรับ Forex (2-6, default: 3)
    - min_dominant_score_gold: Score ที่ชนะต้อง >= X (5-10, default: 7)
    - min_dominant_score_forex: Score ที่ชนะต้อง >= X (5-10, default: 7)
    """
    global _score_gap_config
    
    changes = []
    
    if min_score_gap_gold is not None:
        _score_gap_config["min_score_gap_gold"] = max(1, min(5, min_score_gap_gold))
        changes.append(f"gold_gap: {_score_gap_config['min_score_gap_gold']}")
    
    if min_score_gap_forex is not None:
        _score_gap_config["min_score_gap_forex"] = max(2, min(6, min_score_gap_forex))
        changes.append(f"forex_gap: {_score_gap_config['min_score_gap_forex']}")
    
    if min_dominant_score_gold is not None:
        _score_gap_config["min_dominant_score_gold"] = max(5, min(10, min_dominant_score_gold))
        changes.append(f"gold_min_score: {_score_gap_config['min_dominant_score_gold']}")
    
    if min_dominant_score_forex is not None:
        _score_gap_config["min_dominant_score_forex"] = max(5, min(10, min_dominant_score_forex))
        changes.append(f"forex_min_score: {_score_gap_config['min_dominant_score_forex']}")
    
    logger.info(f"🎯 Score Gap config updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _score_gap_config
    }


@router.post("/score-gap/preset/{preset}")
async def set_score_gap_preset(preset: str):
    """
    🎯 Set Score Gap Filter Preset
    
    Presets:
    - strict: เทรดน้อย แต่ win rate สูง (Gap >= 3, Score >= 8)
    - balanced: สมดุล (Gap >= 2, Score >= 7) - แนะนำ!
    - relaxed: เทรดเยอะ win rate ปานกลาง (Gap >= 1, Score >= 6)
    """
    global _score_gap_config
    
    presets = {
        "strict": {
            "min_score_gap_gold": 3,
            "min_score_gap_forex": 4,
            "min_dominant_score_gold": 8,
            "min_dominant_score_forex": 8,
            "description": "🎯 Strict: เทรดน้อย Win Rate สูง ~85%+"
        },
        "balanced": {
            "min_score_gap_gold": 2,
            "min_score_gap_forex": 3,
            "min_dominant_score_gold": 7,
            "min_dominant_score_forex": 7,
            "description": "⚖️ Balanced: สมดุล Win Rate ~80% (แนะนำ!)"
        },
        "relaxed": {
            "min_score_gap_gold": 1,
            "min_score_gap_forex": 2,
            "min_dominant_score_gold": 6,
            "min_dominant_score_forex": 6,
            "description": "📈 Relaxed: เทรดเยอะ Win Rate ~70%"
        }
    }
    
    if preset not in presets:
        return {"status": "error", "message": f"Unknown preset: {preset}. Available: {list(presets.keys())}"}
    
    config = presets[preset]
    
    _score_gap_config["min_score_gap_gold"] = config["min_score_gap_gold"]
    _score_gap_config["min_score_gap_forex"] = config["min_score_gap_forex"]
    _score_gap_config["min_dominant_score_gold"] = config["min_dominant_score_gold"]
    _score_gap_config["min_dominant_score_forex"] = config["min_dominant_score_forex"]
    
    logger.info(f"🎯 Preset '{preset}' activated: {config['description']}")
    
    return {
        "status": "success",
        "preset": preset,
        "description": config["description"],
        "config": _score_gap_config
    }


# =====================
# 🎯 AGGRESSIVE TRADING MODE
# =====================

@router.get("/aggressive")
async def get_aggressive_config():
    """
    🎯 Get Aggressive Trading configuration
    
    Aggressive mode = เทรดเยอะ กำไรเยอะ
    """
    global _aggressive_config, _trade_cooldown_seconds
    
    return {
        "config": _aggressive_config,
        "cooldown_seconds": _trade_cooldown_seconds,
        "description": {
            "min_confidence_to_trade": "Minimum confidence % เพื่อเทรด",
            "signal_window_minutes": "Signal ID window (นาที) - ยิ่งน้อย ยิ่งเทรดเยอะ",
            "allow_same_direction_reentry": "อนุญาต re-entry ทิศทางเดียวกัน",
            "min_profit_for_wait_close": "กำไรขั้นต่ำที่จะปิดเมื่อ WAIT signal",
            "quick_scalp_mode": "Scalping mode (เทรดถี่มาก)"
        }
    }




@router.post("/aggressive/configure")
async def configure_aggressive_mode(
    min_confidence: float = None,
    signal_window_minutes: int = None,
    allow_reentry: bool = None,
    min_profit_for_wait: float = None,
    cooldown_seconds: int = None,
    scalp_mode: bool = None
):
    """
    🎯 Configure Aggressive Trading Mode
    
    - min_confidence: 60-80 (default: 60)
    - signal_window_minutes: 1-15 (default: 5)
    - allow_reentry: true/false
    - min_profit_for_wait: $100-$1000 (default: $500)
    - cooldown_seconds: 5-60 (default: 10)
    - scalp_mode: true/false (experimental)
    """
    global _aggressive_config, _trade_cooldown_seconds
    
    changes = []
    
    if min_confidence is not None:
        _aggressive_config["min_confidence_to_trade"] = max(50, min(85, min_confidence))
        changes.append(f"min_confidence: {_aggressive_config['min_confidence_to_trade']}%")
    
    if signal_window_minutes is not None:
        _aggressive_config["signal_window_minutes"] = max(1, min(15, signal_window_minutes))
        changes.append(f"signal_window: {_aggressive_config['signal_window_minutes']} mins")
    
    if allow_reentry is not None:
        _aggressive_config["allow_same_direction_reentry"] = allow_reentry
        changes.append(f"allow_reentry: {allow_reentry}")
    
    if min_profit_for_wait is not None:
        _aggressive_config["min_profit_for_wait_close"] = max(50, min(5000, min_profit_for_wait))
        changes.append(f"min_profit_for_wait: ${_aggressive_config['min_profit_for_wait_close']}")
    
    if cooldown_seconds is not None:
        _trade_cooldown_seconds = max(5, min(60, cooldown_seconds))
        changes.append(f"cooldown: {_trade_cooldown_seconds}s")
    
    if scalp_mode is not None:
        _aggressive_config["quick_scalp_mode"] = scalp_mode
        if scalp_mode:
            # Ultra aggressive settings for scalping
            _trade_cooldown_seconds = 5
            _aggressive_config["signal_window_minutes"] = 1
            _aggressive_config["min_confidence_to_trade"] = 55
            changes.append("SCALP MODE ACTIVATED!")
    
    logger.info(f"🎯 Aggressive config updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _aggressive_config,
        "cooldown_seconds": _trade_cooldown_seconds
    }


@router.post("/aggressive/preset/{preset}")
async def set_aggressive_preset(preset: str):
    """
    🎯 Set Aggressive Trading Preset
    
    Presets:
    - conservative: Winrate สูง แต่เทรดน้อย
    - balanced: สมดุล (default)
    - aggressive: เทรดเยอะ กำไรเยอะ
    - ultra: Ultra aggressive (scalping)
    """
    global _aggressive_config, _trade_cooldown_seconds
    
    presets = {
        "conservative": {
            "min_confidence_to_trade": 75,
            "signal_window_minutes": 15,
            "min_profit_for_wait_close": 200,
            "cooldown": 30,
            "description": "Winrate สูง ~90% แต่เทรดน้อย"
        },
        "balanced": {
            "min_confidence_to_trade": 65,
            "signal_window_minutes": 10,
            "min_profit_for_wait_close": 300,
            "cooldown": 20,
            "description": "สมดุล Winrate ~80% เทรดปานกลาง"
        },
        "aggressive": {
            "min_confidence_to_trade": 60,
            "signal_window_minutes": 5,
            "min_profit_for_wait_close": 500,
            "cooldown": 10,
            "description": "เทรดเยอะ Winrate ~75% กำไรเยอะ"
        },
        "ultra": {
            "min_confidence_to_trade": 55,
            "signal_window_minutes": 2,
            "min_profit_for_wait_close": 1000,
            "cooldown": 5,
            "description": "Ultra aggressive Winrate ~70% เทรดมากที่สุด"
        }
    }
    
    if preset not in presets:
        return {"status": "error", "message": f"Unknown preset: {preset}. Available: {list(presets.keys())}"}
    
    config = presets[preset]
    _aggressive_config["min_confidence_to_trade"] = config["min_confidence_to_trade"]
    _aggressive_config["signal_window_minutes"] = config["signal_window_minutes"]
    _aggressive_config["min_profit_for_wait_close"] = config["min_profit_for_wait_close"]
    _trade_cooldown_seconds = config["cooldown"]
    
    logger.info(f"🎯 Preset '{preset}' activated: {config['description']}")
    
    return {
        "status": "success",
        "preset": preset,
        "description": config["description"],
        "config": _aggressive_config,
        "cooldown_seconds": _trade_cooldown_seconds
    }


# =====================
# 📈 SMART DCA (Dollar Cost Averaging)
# =====================

@router.get("/dca")
async def get_dca_status():
    """
    📈 Get Smart DCA configuration and tracking status
    
    Shows:
    - DCA configuration
    - Current tracking for each symbol
    - DCA opportunities
    """
    global _dca_config, _dca_tracking, _bot
    
    # Get current positions with DCA info
    positions_info = []
    try:
        if _bot and _bot.trading_engine:
            positions = await _bot.trading_engine.broker.get_positions()
            if positions:
                for pos in positions:
                    if isinstance(pos, dict):
                        pos_symbol = pos.get("symbol", "")
                        pos_side = pos.get("side", "")
                        pos_price = float(pos.get("open_price", 0) or pos.get("price_open", 0) or 0)
                        pos_pnl = float(pos.get("profit", 0) or 0)
                    else:
                        pos_symbol = getattr(pos, "symbol", "")
                        pos_side = getattr(pos, "side", "")
                        if hasattr(pos_side, "value"):
                            pos_side = pos_side.value
                        pos_price = float(getattr(pos, "open_price", 0) or getattr(pos, "price_open", 0) or 0)
                        pos_pnl = float(getattr(pos, "profit", 0) or 0)
                    
                    tracking = _dca_tracking.get(pos_symbol, {})
                    max_entries = _dca_config.get("max_dca_entries", 2)
                    current_entries = tracking.get("entries", 1)
                    
                    positions_info.append({
                        "symbol": pos_symbol,
                        "side": str(pos_side).upper(),
                        "entry_price": pos_price,
                        "current_pnl": pos_pnl,
                        "dca_entries": current_entries,
                        "max_entries": max_entries + 1,
                        "can_dca": current_entries <= max_entries,
                        "peak_adverse": tracking.get("peak_adverse"),
                        "last_dca_time": tracking.get("last_dca_time").isoformat() if tracking.get("last_dca_time") else None,
                    })
    except Exception as e:
        logger.warning(f"Error getting positions for DCA: {e}")
    
    return {
        "config": _dca_config,
        "tracking": {k: {
            "entries": v.get("entries", 1),
            "first_entry_price": v.get("first_entry_price"),
            "peak_adverse": v.get("peak_adverse"),
            "side": v.get("side"),
            "last_dca_time": v.get("last_dca_time").isoformat() if v.get("last_dca_time") else None,
        } for k, v in _dca_tracking.items()},
        "positions": positions_info,
        "description": {
            "max_dca_entries": "จำนวนครั้งเข้าซ้ำสูงสุด",
            "min_retracement_percent": "ราคาต้องย่อกี่ % ก่อนเข้าซ้ำ",
            "wait_for_reversal": "รอให้ราคากลับตัวก่อนเข้าซ้ำ",
            "signal_must_persist": "สัญญาณต้องยังคงเป็นทิศทางเดิม",
            "min_time_between_dca": "ห่างกันอย่างน้อยกี่วินาที",
        }
    }


@router.post("/dca/toggle")
async def toggle_dca(enabled: bool = True):
    """
    📈 Enable/Disable Smart DCA
    
    - enabled=true: เปิดใช้งาน DCA
    - enabled=false: ปิดใช้งาน DCA
    """
    global _dca_config
    
    _dca_config["enabled"] = enabled
    
    status = "ENABLED" if enabled else "DISABLED"
    logger.info(f"📈 Smart DCA: {status}")
    
    return {
        "status": "success",
        "dca_enabled": enabled,
        "message": f"Smart DCA {status}"
    }


@router.post("/dca/configure")
async def configure_dca(
    max_dca_entries: int = None,
    min_retracement_percent: float = None,
    wait_for_reversal: bool = None,
    reversal_candles: int = None,
    signal_must_persist: bool = None,
    min_time_between_dca: int = None,
    lot_multiplier: float = None,
    max_loss_percent_before_dca: float = None,
    min_balance_for_dca: float = None
):
    """
    📈 Configure Smart DCA settings (PERCENT BASED)
    
    - max_dca_entries: จำนวนครั้งเข้าซ้ำสูงสุด (1-5)
    - min_retracement_percent: ราคาต้องย่อกี่ % (0.1-2.0)
    - wait_for_reversal: รอให้ราคากลับตัวก่อน
    - reversal_candles: รอกี่ candle ที่กลับตัว
    - signal_must_persist: สัญญาณต้องยังคงเดิม
    - min_time_between_dca: ห่างกันกี่วินาที
    - lot_multiplier: Lot size เท่าไหร่ (1.0 = เท่าเดิม)
    - max_loss_percent_before_dca: % ขาดทุนสูงสุดก่อน DCA (default: 5%)
    - min_balance_for_dca: Balance ขั้นต่ำสำหรับ DCA (default: $500)
    """
    global _dca_config
    
    changes = []
    
    if max_dca_entries is not None:
        _dca_config["max_dca_entries"] = max(1, min(5, max_dca_entries))
        changes.append(f"max_dca_entries: {_dca_config['max_dca_entries']}")
    
    if min_retracement_percent is not None:
        _dca_config["min_retracement_percent"] = max(0.05, min(2.0, min_retracement_percent))
        changes.append(f"min_retracement: {_dca_config['min_retracement_percent']}%")
    
    if wait_for_reversal is not None:
        _dca_config["wait_for_reversal"] = wait_for_reversal
        changes.append(f"wait_for_reversal: {wait_for_reversal}")
    
    if reversal_candles is not None:
        _dca_config["reversal_candles"] = max(1, min(5, reversal_candles))
        changes.append(f"reversal_candles: {_dca_config['reversal_candles']}")
    
    if signal_must_persist is not None:
        _dca_config["signal_must_persist"] = signal_must_persist
        changes.append(f"signal_must_persist: {signal_must_persist}")
    
    if min_time_between_dca is not None:
        _dca_config["min_time_between_dca"] = max(60, min(3600, min_time_between_dca))
        changes.append(f"min_time_between_dca: {_dca_config['min_time_between_dca']}s")
    
    if lot_multiplier is not None:
        _dca_config["lot_multiplier"] = max(0.5, min(3.0, lot_multiplier))
        changes.append(f"lot_multiplier: {_dca_config['lot_multiplier']}x")
    
    if max_loss_percent_before_dca is not None:
        _dca_config["max_loss_percent_before_dca"] = max(1, min(20, max_loss_percent_before_dca))
        changes.append(f"max_loss_percent: {_dca_config['max_loss_percent_before_dca']}%")
    
    if min_balance_for_dca is not None:
        _dca_config["min_balance_for_dca"] = max(100, min_balance_for_dca)
        changes.append(f"min_balance: ${_dca_config['min_balance_for_dca']}")
    
    logger.info(f"📈 DCA config updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _dca_config
    }


@router.post("/dca/reset")
async def reset_dca_tracking(symbol: str = None):
    """
    📈 Reset DCA tracking
    
    - symbol: Reset tracking สำหรับ symbol นี้
    - ถ้าไม่ระบุ: Reset ทั้งหมด
    """
    global _dca_tracking
    
    if symbol:
        if symbol in _dca_tracking:
            del _dca_tracking[symbol]
            return {"status": "reset", "symbol": symbol}
        else:
            return {"status": "not_found", "symbol": symbol}
    else:
        count = len(_dca_tracking)
        _dca_tracking.clear()
        return {"status": "reset_all", "cleared_count": count}


# =====================
# 💰 SMART PROFIT PROTECTION
# =====================

@router.get("/profit-protection")
async def get_profit_protection_status():
    """
    💰 Get Smart Profit Protection status and configuration
    """
    global _profit_protection_config, _peak_profit_by_position, _bot
    
    # Get current positions with peaks
    positions_info = []
    try:
        if _bot and _bot.trading_engine:
            positions = await _bot.trading_engine.broker.get_positions()
            if positions:
                for pos in positions:
                    if isinstance(pos, dict):
                        pos_id = pos.get("ticket") or pos.get("id")
                        pos_symbol = pos.get("symbol", "")
                        pos_pnl = float(pos.get("profit", 0) or 0)
                        pos_side = pos.get("side", "")
                    else:
                        pos_id = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                        pos_symbol = getattr(pos, "symbol", "")
                        pos_pnl = float(getattr(pos, "profit", 0) or 0)
                        pos_side = getattr(pos, "side", "")
                    
                    peak = _peak_profit_by_position.get(pos_id, pos_pnl)
                    drawdown_pct = ((peak - pos_pnl) / peak * 100) if peak > 0 else 0
                    trigger_pct = _profit_protection_config.get("profit_drawdown_percent", 30)
                    
                    positions_info.append({
                        "ticket": pos_id,
                        "symbol": pos_symbol,
                        "side": pos_side,
                        "current_profit": pos_pnl,
                        "peak_profit": peak,
                        "drawdown_percent": round(drawdown_pct, 1),
                        "trigger_at_percent": trigger_pct,
                        "will_close_at": round(peak * (1 - trigger_pct/100), 2) if peak > 0 else 0,
                        "protected": pos_pnl >= _profit_protection_config.get("min_profit_to_protect", 100)
                    })
    except Exception as e:
        logger.warning(f"Error getting positions for profit protection: {e}")
    
    return {
        "config": _profit_protection_config,
        "positions": positions_info,
        "description": {
            "profit_drawdown_percent": "ปิดเมื่อกำไรลดลง X% จาก peak",
            "min_profit_to_protect": "เริ่ม protect เมื่อกำไร >= $X",
            "trailing_stop_trigger": "เริ่ม trailing เมื่อกำไร >= $X",
            "trailing_stop_distance": "trailing stop ห่าง $X จาก current profit"
        }
    }


@router.post("/profit-protection/toggle")
async def toggle_profit_protection(enabled: bool = True):
    """
    💰 Enable/Disable Smart Profit Protection
    """
    global _profit_protection_config
    
    _profit_protection_config["enabled"] = enabled
    
    status = "ENABLED" if enabled else "DISABLED"
    logger.info(f"💰 Smart Profit Protection: {status}")
    
    return {
        "status": "success",
        "profit_protection_enabled": enabled,
        "message": f"Smart Profit Protection {status}"
    }


@router.post("/profit-protection/configure")
async def configure_profit_protection(
    profit_drawdown_percent: float = None,
    min_profit_to_protect: float = None,
    trailing_stop_trigger: float = None,
    trailing_stop_distance: float = None
):
    """
    💰 Configure Smart Profit Protection settings
    
    - profit_drawdown_percent: ปิดเมื่อกำไรลดลง X% จาก peak (default: 30)
    - min_profit_to_protect: เริ่ม protect เมื่อกำไร >= $X (default: 100)
    - trailing_stop_trigger: เริ่ม trailing เมื่อกำไร >= $X (default: 500)
    - trailing_stop_distance: trailing stop ห่าง $X (default: 200)
    """
    global _profit_protection_config
    
    if profit_drawdown_percent is not None:
        _profit_protection_config["profit_drawdown_percent"] = max(5, min(80, profit_drawdown_percent))
    
    if min_profit_to_protect is not None:
        _profit_protection_config["min_profit_to_protect"] = max(10, min_profit_to_protect)
    
    if trailing_stop_trigger is not None:
        _profit_protection_config["trailing_stop_trigger"] = max(50, trailing_stop_trigger)
    
    if trailing_stop_distance is not None:
        _profit_protection_config["trailing_stop_distance"] = max(20, trailing_stop_distance)
    
    logger.info(f"💰 Profit Protection configured: {_profit_protection_config}")
    
    return {
        "status": "success",
        "config": _profit_protection_config,
        "message": "Configuration updated"
    }


@router.post("/profit-protection/reset-peaks")
async def reset_peak_profits():
    """
    💰 Reset all peak profit tracking
    
    Use this when you want to start fresh tracking
    """
    global _peak_profit_by_position
    
    count = len(_peak_profit_by_position)
    _peak_profit_by_position.clear()
    
    logger.info(f"💰 Reset {count} peak profit records")
    
    return {
        "status": "success",
        "cleared_count": count,
        "message": f"Cleared {count} peak profit records"
    }


# =====================
# 🔥 STABILITY ENDPOINTS - 10 Year Runtime
# =====================

@router.get("/stability")
async def get_stability_status():
    """
    🔥 Get system stability status and runtime statistics
    
    Shows:
    - Uptime
    - Restart count
    - Memory usage
    - Watchdog status
    - Error count
    """
    global _stability_config, _runtime_stats, _watchdog_task, _last_successful_cycle
    
    memory_mb = _get_memory_usage_mb()
    
    # Calculate uptime
    uptime_seconds = _runtime_stats.get("total_uptime_seconds", 0)
    uptime_days = uptime_seconds // 86400
    uptime_hours = (uptime_seconds % 86400) // 3600
    uptime_minutes = (uptime_seconds % 3600) // 60
    
    # Check watchdog
    watchdog_alive = _watchdog_task is not None and not _watchdog_task.done()
    
    # Last successful cycle age
    last_cycle_age = None
    if _last_successful_cycle:
        last_cycle_age = int((datetime.now() - _last_successful_cycle).total_seconds())
    
    return {
        "stability": {
            "config": _stability_config,
            "uptime": {
                "total_seconds": uptime_seconds,
                "formatted": f"{uptime_days}d {uptime_hours}h {uptime_minutes}m",
                "days": uptime_days,
                "hours": uptime_hours,
                "minutes": uptime_minutes,
            },
            "runtime_stats": _runtime_stats,
            "memory": {
                "current_mb": round(memory_mb, 1),
                "max_mb": _stability_config.get("max_memory_mb", 2048),
                "usage_percent": round(memory_mb / _stability_config.get("max_memory_mb", 2048) * 100, 1),
            },
            "watchdog": {
                "enabled": True,
                "alive": watchdog_alive,
                "interval_seconds": _stability_config.get("watchdog_interval_seconds", 60),
            },
            "last_successful_cycle": {
                "timestamp": _last_successful_cycle.isoformat() if _last_successful_cycle else None,
                "age_seconds": last_cycle_age,
            },
            "health": "HEALTHY" if watchdog_alive and (last_cycle_age is None or last_cycle_age < 120) else "WARNING",
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/stability/restart")
async def manual_restart_bot():
    """
    🔄 Manually trigger bot restart
    
    Use this to force restart the bot without stopping the API
    """
    global _runtime_stats
    
    logger.info("🔄 Manual restart requested")
    
    success = await _auto_restart_bot()
    
    return {
        "status": "success" if success else "failed",
        "message": "Bot restart triggered" if success else "Restart failed",
        "restart_count": _runtime_stats.get("restart_count", 0),
    }


@router.post("/stability/cleanup")
async def manual_memory_cleanup():
    """
    🧹 Manually trigger memory cleanup
    
    Forces garbage collection and clears old data
    """
    global _runtime_stats
    
    before_mb = _get_memory_usage_mb()
    _cleanup_memory()
    after_mb = _get_memory_usage_mb()
    
    freed_mb = before_mb - after_mb
    
    return {
        "status": "success",
        "before_mb": round(before_mb, 1),
        "after_mb": round(after_mb, 1),
        "freed_mb": round(max(0, freed_mb), 1),
        "total_cleanups": _runtime_stats.get("memory_cleanups", 0),
    }


@router.post("/stability/save-state")
async def manual_save_state():
    """
    💾 Manually save bot state to file
    
    State will be automatically restored on restart
    """
    _save_state()
    
    return {
        "status": "success",
        "message": "State saved successfully",
        "file": _stability_config.get("state_file_path", "bot_state.json"),
    }


@router.get("/stability/load-state")
async def get_saved_state():
    """
    📂 Get saved bot state from file
    """
    state = _load_state()
    
    if state:
        return {
            "status": "ok",
            "state": state,
        }
    else:
        return {
            "status": "no_state",
            "message": "No saved state found",
        }


@router.post("/stability/configure")
async def configure_stability(
    auto_restart_enabled: bool = None,
    max_restart_attempts: int = None,
    restart_cooldown_seconds: int = None,
    watchdog_interval_seconds: int = None,
    memory_cleanup_interval: int = None,
    max_memory_mb: int = None,
    heartbeat_timeout_seconds: int = None,
):
    """
    🔧 Configure stability settings
    
    - auto_restart_enabled: Enable/disable auto-restart on crash
    - max_restart_attempts: Max restart attempts before giving up
    - restart_cooldown_seconds: Wait time between restarts
    - watchdog_interval_seconds: Health check interval
    - memory_cleanup_interval: Memory cleanup interval
    - max_memory_mb: Max memory before forced cleanup
    - heartbeat_timeout_seconds: Max time without heartbeat
    """
    global _stability_config
    
    changes = []
    
    if auto_restart_enabled is not None:
        _stability_config["auto_restart_enabled"] = auto_restart_enabled
        changes.append(f"auto_restart: {auto_restart_enabled}")
    
    if max_restart_attempts is not None:
        _stability_config["max_restart_attempts"] = max(1, max_restart_attempts)
        changes.append(f"max_restarts: {max_restart_attempts}")
    
    if restart_cooldown_seconds is not None:
        _stability_config["restart_cooldown_seconds"] = max(5, restart_cooldown_seconds)
        changes.append(f"restart_cooldown: {restart_cooldown_seconds}s")
    
    if watchdog_interval_seconds is not None:
        _stability_config["watchdog_interval_seconds"] = max(10, watchdog_interval_seconds)
        changes.append(f"watchdog_interval: {watchdog_interval_seconds}s")
    
    if memory_cleanup_interval is not None:
        _stability_config["memory_cleanup_interval"] = max(60, memory_cleanup_interval)
        changes.append(f"memory_cleanup: {memory_cleanup_interval}s")
    
    if max_memory_mb is not None:
        _stability_config["max_memory_mb"] = max(256, max_memory_mb)
        changes.append(f"max_memory: {max_memory_mb}MB")
    
    if heartbeat_timeout_seconds is not None:
        _stability_config["heartbeat_timeout_seconds"] = max(30, heartbeat_timeout_seconds)
        changes.append(f"heartbeat_timeout: {heartbeat_timeout_seconds}s")
    
    logger.info(f"🔧 Stability config updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _stability_config,
    }


@router.post("/stability/reset-stats")
async def reset_runtime_stats():
    """
    📊 Reset runtime statistics
    
    Use this to start fresh statistics counting
    """
    global _runtime_stats
    
    _runtime_stats = {
        "total_uptime_seconds": 0,
        "restart_count": 0,
        "last_restart_time": None,
        "last_heartbeat": None,
        "errors_count": 0,
        "recoveries_count": 0,
        "memory_cleanups": 0,
        "started_at": datetime.now().isoformat(),
    }
    
    logger.info("📊 Runtime stats reset")
    
    return {
        "status": "success",
        "message": "Runtime statistics reset",
        "stats": _runtime_stats,
    }


# =====================
# 🔄 POSITION SYNC ENDPOINTS
# =====================

@router.post("/sync-positions")
async def force_sync_positions():
    """
    🔄 Force sync positions with MT5
    
    Use this when bot thinks there's a position but MT5 doesn't have it
    (e.g., after SL/TP hit externally)
    """
    global _bot, _known_positions, _last_traded_signal, _peak_profit_by_position
    
    if not _bot or not _bot.trading_engine:
        return {
            "status": "error",
            "message": "Bot not initialized"
        }
    
    try:
        # Get fresh positions from MT5
        positions = await _bot.trading_engine.broker.get_positions()
        
        # Build current state
        mt5_tickets = set()
        mt5_symbols = set()
        mt5_positions = []
        
        for pos in (positions or []):
            if isinstance(pos, dict):
                ticket = pos.get("ticket") or pos.get("id")
                symbol = pos.get("symbol", "")
                side = pos.get("side", "")
                profit = pos.get("profit", 0)
            else:
                ticket = getattr(pos, "ticket", None) or getattr(pos, "id", None)
                symbol = getattr(pos, "symbol", "")
                side = getattr(pos, "side", "")
                if hasattr(side, "value"):
                    side = side.value
                profit = getattr(pos, "profit", 0)
            
            if ticket:
                mt5_tickets.add(str(ticket))
                mt5_positions.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "side": str(side).upper(),
                    "profit": profit
                })
            if symbol:
                mt5_symbols.add(symbol.upper())
        
        # Find and clear orphan entries
        orphans_cleared = []
        known_before = dict(_known_positions)
        
        for ticket in list(_known_positions.keys()):
            if str(ticket) not in mt5_tickets:
                info = _known_positions[ticket]
                orphans_cleared.append({
                    "ticket": ticket,
                    "symbol": info.get("symbol", ""),
                    "side": info.get("side", "")
                })
                del _known_positions[ticket]
                
                # Clear cooldown
                symbol = info.get("symbol", "")
                if symbol and symbol in _last_traded_signal:
                    del _last_traded_signal[symbol]
                if symbol and symbol.upper() in _last_traded_signal:
                    del _last_traded_signal[symbol.upper()]
                
                # Clear peak profit
                if ticket in _peak_profit_by_position:
                    del _peak_profit_by_position[ticket]
        
        # Add new positions not in known
        new_tracked = []
        for pos in mt5_positions:
            ticket = str(pos["ticket"])
            if ticket not in _known_positions:
                _known_positions[ticket] = {
                    "symbol": pos["symbol"],
                    "side": pos["side"]
                }
                new_tracked.append(pos)
        
        logger.info(f"🔄 FORCE SYNC: MT5={len(mt5_positions)}, Known before={len(known_before)}, Orphans cleared={len(orphans_cleared)}, New tracked={len(new_tracked)}")
        
        return {
            "status": "success",
            "mt5_positions": mt5_positions,
            "mt5_count": len(mt5_positions),
            "orphans_cleared": orphans_cleared,
            "new_tracked": new_tracked,
            "known_positions_now": dict(_known_positions),
            "cooldowns_active": list(_last_traded_signal.keys()),
            "message": f"Synced! Cleared {len(orphans_cleared)} orphans, tracking {len(_known_positions)} positions"
        }
        
    except Exception as e:
        logger.error(f"Force sync error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/positions-debug")
async def get_positions_debug():
    """
    🔍 Debug endpoint to see position tracking state
    """
    global _bot, _known_positions, _last_traded_signal, _peak_profit_by_position
    
    mt5_positions = []
    try:
        if _bot and _bot.trading_engine:
            positions = await _bot.trading_engine.broker.get_positions()
            for pos in (positions or []):
                if isinstance(pos, dict):
                    mt5_positions.append({
                        "ticket": pos.get("ticket") or pos.get("id"),
                        "symbol": pos.get("symbol", ""),
                        "side": pos.get("side", ""),
                        "profit": pos.get("profit", 0),
                    })
                else:
                    side = getattr(pos, "side", "")
                    if hasattr(side, "value"):
                        side = side.value
                    mt5_positions.append({
                        "ticket": getattr(pos, "ticket", None) or getattr(pos, "id", None),
                        "symbol": getattr(pos, "symbol", ""),
                        "side": str(side),
                        "profit": getattr(pos, "profit", 0),
                    })
    except Exception as e:
        logger.warning(f"Error getting MT5 positions: {e}")
    
    return {
        "mt5_positions": mt5_positions,
        "mt5_count": len(mt5_positions),
        "known_positions": dict(_known_positions),
        "known_count": len(_known_positions),
        "cooldowns": {k: {
            "signal": v.get("signal"),
            "timestamp": v.get("timestamp").isoformat() if v.get("timestamp") else None,
        } for k, v in _last_traded_signal.items()},
        "peak_profits": dict(_peak_profit_by_position),
        "sync_status": "OK" if len(mt5_positions) == len(_known_positions) else "MISMATCH",
    }


@router.post("/reset-all-tracking")
async def reset_all_tracking_data():
    """
    🧹 EMERGENCY RESET - ล้างข้อมูล tracking ทั้งหมด
    
    ใช้เมื่อ:
    - Bot คิดว่ามี position แต่ MT5 ไม่มี
    - Cooldown ติด แต่อยากเทรดใหม่
    - daily_stats ผิดปกติ
    
    WARNING: ข้อมูลทั้งหมดจะถูกล้าง!
    """
    global _known_positions, _last_traded_signal, _peak_profit_by_position, _bot_status, _pending_signals
    
    # Count before clearing
    counts = {
        "known_positions": len(_known_positions),
        "cooldowns": len(_last_traded_signal),
        "peak_profits": len(_peak_profit_by_position),
        "pending_signals": len(_pending_signals),
    }
    
    # Clear all tracking
    _known_positions.clear()
    _last_traded_signal.clear()
    _peak_profit_by_position.clear()
    _pending_signals.clear()
    
    # Reset daily stats
    today = datetime.now().date().isoformat()
    old_stats = dict(_bot_status["daily_stats"])
    _bot_status["daily_stats"] = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "last_reset_date": today
    }
    
    logger.warning(f"🧹 EMERGENCY RESET executed!")
    logger.warning(f"   Cleared: positions={counts['known_positions']}, cooldowns={counts['cooldowns']}, peaks={counts['peak_profits']}, pending={counts['pending_signals']}")
    logger.warning(f"   Old stats: trades={old_stats.get('trades', 0)}, pnl=${old_stats.get('pnl', 0):.2f}")
    
    
    return {
        "status": "success",
        "message": "All tracking data cleared!",
        "cleared": counts,
        "old_daily_stats": old_stats,
        "new_daily_stats": _bot_status["daily_stats"],
        "note": "Bot can now trade fresh"
    }


@router.post("/reset-daily-stats")
async def reset_daily_stats_only():
    """
    📊 Reset daily_stats only
    
    ใช้เมื่อ P&L แสดงผิด หรืออยากเริ่มนับใหม่
    """
    global _bot_status
    
    today = datetime.now().date().isoformat()
    old_stats = dict(_bot_status["daily_stats"])
    
    _bot_status["daily_stats"] = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "last_reset_date": today
    }
    
    logger.info(f"📊 Daily stats manually reset")
    logger.info(f"   Old: trades={old_stats.get('trades', 0)}, W:{old_stats.get('wins', 0)} L:{old_stats.get('losses', 0)}, PnL:${old_stats.get('pnl', 0):.2f}")
    
    return {
        "status": "success",
        "message": "Daily stats reset!",
        "old_stats": old_stats,
        "new_stats": _bot_status["daily_stats"]
    }


# =====================
# 📊 TRADE HISTORY - ประวัติเทรดที่แม่นยำ
# =====================
# 📈 AUTO TRAILING STOP - ยก SL อัตโนมัติ
# =====================

@router.get("/trailing-stop")
async def get_trailing_stop_config():
    """
    📈 Get Auto Trailing Stop configuration (PERCENT BASED)
    """
    global _trailing_stop_config, _last_trailing_sl, _bot
    
    # Get current balance for example values
    balance = 1000
    try:
        if _bot and _bot.trading_engine:
            balance = await _bot.trading_engine.broker.get_balance() or 1000
    except:
        pass
    
    return {
        "config": _trailing_stop_config,
        "active_trails": _last_trailing_sl,
        "current_balance": balance,
        "current_values": {
            "trigger_profit": f"${_get_trailing_trigger(balance):.2f}",
            "trail_distance": f"${_get_trailing_distance(balance):.2f}",
            "step_size": f"${_get_trailing_step(balance):.2f}",
            "lock_profit": f"${_get_trailing_lock(balance):.2f}",
        },
        "description": {
            "trigger_profit_percent": f"เริ่ม trail เมื่อกำไร >= X% ของ balance",
            "trail_distance_percent": f"SL ห่างจากราคา X% ของ balance",
            "step_size_percent": f"ยก SL ทีละ X% ของ balance",
            "lock_profit_percent": f"เมื่อกำไร >= X% ยก SL เข้ามาล็อกกำไร",
        },
        "examples": {
            "$100 port": f"trigger: ${_get_trailing_trigger(100):.0f}, lock: ${_get_trailing_lock(100):.0f}",
            "$1,000 port": f"trigger: ${_get_trailing_trigger(1000):.0f}, lock: ${_get_trailing_lock(1000):.0f}",
            "$10,000 port": f"trigger: ${_get_trailing_trigger(10000):.0f}, lock: ${_get_trailing_lock(10000):.0f}",
            "$100,000 port": f"trigger: ${_get_trailing_trigger(100000):.0f}, lock: ${_get_trailing_lock(100000):.0f}",
        }
    }


@router.post("/trailing-stop/toggle")
async def toggle_trailing_stop(enabled: bool = True):
    """
    📈 Enable/Disable Auto Trailing Stop
    """
    global _trailing_stop_config
    
    _trailing_stop_config["enabled"] = enabled
    
    status = "ENABLED ✅" if enabled else "DISABLED ❌"
    logger.info(f"📈 Auto Trailing Stop: {status}")
    
    return {
        "status": "success",
        "trailing_stop_enabled": enabled,
        "message": f"Auto Trailing Stop {status}"
    }


@router.post("/trailing-stop/configure")
async def configure_trailing_stop(
    trigger_profit_percent: float = None,
    trail_distance_percent: float = None,
    step_size_percent: float = None,
    lock_profit_percent: float = None
):
    """
    📈 Configure Auto Trailing Stop (PERCENT BASED)
    
    ค่าทั้งหมดเป็น % ของ balance - ทำงานกับทุกขนาด port!
    
    - trigger_profit_percent: เริ่ม trail เมื่อกำไร >= X% (default: 5%)
    - trail_distance_percent: SL ห่าง X% ของ balance (default: 2.5%)
    - step_size_percent: ยก SL ทีละ X% (default: 0.5%)
    - lock_profit_percent: ล็อกกำไรเมื่อ >= X% (default: 10%)
    
    Examples ($1000 balance):
    - trigger 5% = $50
    - lock 10% = $100
    """
    global _trailing_stop_config
    
    changes = []
    
    if trigger_profit_percent is not None:
        _trailing_stop_config["trigger_profit_percent"] = max(1, min(20, trigger_profit_percent))
        changes.append(f"trigger: {_trailing_stop_config['trigger_profit_percent']}%")
    
    if trail_distance_percent is not None:
        _trailing_stop_config["trail_distance_percent"] = max(0.5, min(10, trail_distance_percent))
        changes.append(f"distance: {_trailing_stop_config['trail_distance_percent']}%")
    
    if step_size_percent is not None:
        _trailing_stop_config["step_size_percent"] = max(0.1, min(5, step_size_percent))
        changes.append(f"step: {_trailing_stop_config['step_size_percent']}%")
    
    if lock_profit_percent is not None:
        _trailing_stop_config["lock_profit_percent"] = max(2, min(50, lock_profit_percent))
        changes.append(f"lock: {_trailing_stop_config['lock_profit_percent']}%")
    
    logger.info(f"📈 Trailing stop configured: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _trailing_stop_config,
        "example_values": {
            "$200 port": f"trigger: ${_get_trailing_trigger(200):.0f}, lock: ${_get_trailing_lock(200):.0f}",
            "$1,000 port": f"trigger: ${_get_trailing_trigger(1000):.0f}, lock: ${_get_trailing_lock(1000):.0f}",
            "$10,000 port": f"trigger: ${_get_trailing_trigger(10000):.0f}, lock: ${_get_trailing_lock(10000):.0f}",
        }
    }


# =====================
# 📊 TRADE HISTORY - ประวัติเทรดที่แม่นยำ
# =====================

@router.get("/trade-history")
async def get_trade_history(limit: int = 50):
    """
    📊 Get Trade History with accurate Win/Loss tracking
    
    Features:
    - PnL-based win/loss (not SL/TP based!)
    - SL hit but profit → WIN ✅ (trailing stop)
    - TP hit → WIN ✅
    - Shows close reason (SL/TP/MANUAL)
    """
    global _trade_history, _bot_status
    
    # Calculate stats from history
    trades = _trade_history[-limit:] if limit > 0 else _trade_history
    
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades if t.get("pnl", 0) < 0)
    breakeven = sum(1 for t in trades if t.get("pnl", 0) == 0)
    
    win_rate = (wins / len(trades) * 100) if trades else 0
    
    # Group by symbol
    by_symbol = {}
    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            by_symbol[sym]["wins"] += 1
        elif t.get("pnl", 0) < 0:
            by_symbol[sym]["losses"] += 1
    
    # Calculate per-symbol win rate
    for sym in by_symbol:
        total = by_symbol[sym]["trades"]
        by_symbol[sym]["win_rate"] = (by_symbol[sym]["wins"] / total * 100) if total > 0 else 0
    
    return {
        "trades": list(reversed(trades)),  # Newest first
        "count": len(trades),
        "stats": {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "average_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        },
        "by_symbol": by_symbol,
        "daily_stats": _bot_status.get("daily_stats", {}),
        "note": "Win/Loss based on PnL (SL hit with profit = WIN!)",
    }


@router.get("/trade-history/today")
async def get_today_trade_history():
    """
    📊 Get Today's Trade History only
    """
    global _trade_history
    
    today = datetime.now().date().isoformat()
    
    today_trades = []
    for t in _trade_history:
        close_time = t.get("close_time", "")
        if isinstance(close_time, str) and close_time.startswith(today):
            today_trades.append(t)
    
    total_pnl = sum(t.get("pnl", 0) for t in today_trades)
    wins = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
    losses = sum(1 for t in today_trades if t.get("pnl", 0) < 0)
    
    return {
        "date": today,
        "trades": list(reversed(today_trades)),
        "count": len(today_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(today_trades) * 100, 1) if today_trades else 0,
        "total_pnl": round(total_pnl, 2),
    }


# =====================
# 🥇 SYMBOL WHITELIST - Gold Only Mode
# =====================

@router.get("/whitelist")
async def get_symbol_whitelist():
    """
    🥇 Get Symbol Whitelist configuration
    
    Shows which symbols are allowed to trade
    """
    global _symbol_whitelist
    
    return {
        "config": _symbol_whitelist,
        "description": {
            "enabled": "เปิด/ปิด whitelist filter",
            "allowed_symbols": "รายชื่อ symbols ที่อนุญาตเทรด",
            "block_forex": "Block Forex pairs ทั้งหมด (เทรดเฉพาะ Gold)",
        },
        "status": "GOLD ONLY MODE" if _symbol_whitelist.get("block_forex", True) else "ALL SYMBOLS"
    }


@router.post("/whitelist/gold-only")
async def set_gold_only_mode(enabled: bool = True):
    """
    🥇 Enable Gold-Only Trading Mode
    
    - enabled=true: เทรดเฉพาะ Gold (XAUUSDm)
    - enabled=false: เทรดทุก symbol
    """
    global _symbol_whitelist
    
    _symbol_whitelist["enabled"] = enabled
    _symbol_whitelist["block_forex"] = enabled
    
    if enabled:
        _symbol_whitelist["allowed_symbols"] = ["XAUUSDm", "XAUUSD", "GOLD"]
        status = "🥇 GOLD ONLY MODE ENABLED!"
        logger.info(f"🥇 Gold-Only Mode: ENABLED - Forex blocked!")
    else:
        _symbol_whitelist["allowed_symbols"] = []
        status = "ALL SYMBOLS MODE"
        logger.info(f"🌐 All Symbols Mode: ENABLED - Forex allowed")
    
    return {
        "status": "success",
        "gold_only_mode": enabled,
        "message": status,
        "config": _symbol_whitelist
    }


@router.post("/whitelist/configure")
async def configure_symbol_whitelist(
    enabled: bool = None,
    allowed_symbols: List[str] = None,
    block_forex: bool = None
):
    """
    🔧 Configure Symbol Whitelist
    
    - enabled: เปิด/ปิด whitelist
    - allowed_symbols: รายชื่อ symbols ที่อนุญาต (e.g., ["XAUUSDm", "EURUSDm"])
    - block_forex: Block Forex pairs ทั้งหมด
    """
    global _symbol_whitelist
    
    changes = []
    
    if enabled is not None:
        _symbol_whitelist["enabled"] = enabled
        changes.append(f"enabled: {enabled}")
    
    if allowed_symbols is not None:
        _symbol_whitelist["allowed_symbols"] = allowed_symbols
        changes.append(f"allowed_symbols: {allowed_symbols}")
    
    if block_forex is not None:
        _symbol_whitelist["block_forex"] = block_forex
        changes.append(f"block_forex: {block_forex}")
    
    logger.info(f"🔧 Symbol whitelist updated: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _symbol_whitelist
    }


# =====================
# 🛡️ ANTI-WIPEOUT PROTECTION - ป้องกันล้างพอร์ต!
# =====================

@router.get("/anti-wipeout")
async def get_anti_wipeout_status():
    """
    🛡️ Get Anti-Wipeout Protection configuration
    
    🔥 UNIVERSAL SCALING - รองรับ $100 ถึง $2,000,000,000!
    
    Features:
    - Max lot size limiter (based on balance %)
    - Minimum SL distance (% of price, not fixed points!)
    - Higher timeframe trend alignment
    - Counter-trend blocking
    
    Example for $300 balance:
    - Max lot for Gold: 0.01 (=$300/1000 × 0.02)
    - Risk per trade: $3 (=1% of $300)
    - Gold SL: 0.3% of price (~$15 at $5000)
    """
    global _anti_wipeout_config, _bot
    
    # Get current balance for examples
    balance = 300
    try:
        if _bot and _bot.trading_engine:
            balance = await _bot.trading_engine.broker.get_balance() or 300
    except:
        pass
    
    # Calculate current limits
    gold_max_lot = round((balance / 1000) * _anti_wipeout_config.get("gold_max_lot_per_1000", 0.02), 2)
    gold_max_lot = max(0.01, gold_max_lot)  # Minimum 0.01 lot
    forex_max_lot = round((balance / 1000) * _anti_wipeout_config.get("forex_max_lot_per_1000", 0.05), 2)
    forex_max_lot = max(0.01, forex_max_lot)
    max_risk_percent = _anti_wipeout_config.get("max_risk_per_trade_percent", 1.0)
    max_risk_amount = balance * (max_risk_percent / 100)
    
    # Get SL % config
    gold_sl_min = _anti_wipeout_config.get("gold_sl_percent_min", 0.3)
    gold_sl_max = _anti_wipeout_config.get("gold_sl_percent_max", 1.0)
    
    # Calculate SL in $ at current gold price (~$5000)
    gold_price_estimate = 5000
    gold_min_sl_usd = gold_price_estimate * (gold_sl_min / 100)
    gold_max_sl_usd = gold_price_estimate * (gold_sl_max / 100)
    
    return {
        "config": _anti_wipeout_config,
        "current_balance": balance,
        "current_limits": {
            "gold_max_lot": gold_max_lot,
            "forex_max_lot": forex_max_lot,
            "max_risk_percent": f"{max_risk_percent}%",
            "max_risk_amount": f"${max_risk_amount:.2f}",
            "gold_sl_range": f"{gold_sl_min}% - {gold_sl_max}% of price",
            "gold_sl_usd_estimate": f"${gold_min_sl_usd:.0f} - ${gold_max_sl_usd:.0f} (at $5000 gold)",
        },
        "scaling_examples": {
            "$100 port": {
                "gold_max_lot": 0.01,
                "max_risk": f"${100 * max_risk_percent / 100:.2f}",
            },
            "$300 port": {
                "gold_max_lot": max(0.01, round(0.3 * 0.02, 2)),
                "max_risk": f"${300 * max_risk_percent / 100:.2f}",
            },
            "$1,000 port": {
                "gold_max_lot": round(1 * 0.02, 2),
                "max_risk": f"${1000 * max_risk_percent / 100:.2f}",
            },
            "$10,000 port": {
                "gold_max_lot": round(10 * 0.02, 2),
                "max_risk": f"${10000 * max_risk_percent / 100:.2f}",
            },
            "$100,000 port": {
                "gold_max_lot": round(100 * 0.02, 2),
                "max_risk": f"${100000 * max_risk_percent / 100:.2f}",
            },
            "$1,000,000 port": {
                "gold_max_lot": round(1000 * 0.02, 2),
                "max_risk": f"${1000000 * max_risk_percent / 100:.2f}",
            },
        },
        "why_important": "ล้างพอร์ตเพราะ: Lot size ใหญ่เกินไป + SL แคบเกินไป + เข้าสวนเทรนด์",
        "note": "Config ใช้ % ทั้งหมด - รองรับ $100 ถึง $2,000,000,000!"
    }


@router.post("/anti-wipeout/toggle")
async def toggle_anti_wipeout(enabled: bool = True):
    """
    🛡️ Enable/Disable Anti-Wipeout Protection
    
    ⚠️ WARNING: Disabling this increases risk of account wipeout!
    """
    global _anti_wipeout_config
    
    _anti_wipeout_config["enabled"] = enabled
    
    status = "ENABLED ✅ (SAFE)" if enabled else "DISABLED ⚠️ (RISKY!)"
    logger.info(f"🛡️ Anti-Wipeout Protection: {status}")
    
    return {
        "status": "success",
        "anti_wipeout_enabled": enabled,
        "message": f"Anti-Wipeout Protection {status}",
        "warning": "RISKY! Account may be wiped out easily!" if not enabled else None
    }


@router.post("/anti-wipeout/configure")
async def configure_anti_wipeout(
    max_risk_per_trade_percent: float = None,
    gold_max_lot_per_1000: float = None,
    forex_max_lot_per_1000: float = None,
    gold_sl_percent_min: float = None,
    gold_sl_percent_max: float = None,
    block_counter_trend: bool = None,
    min_trend_strength: int = None,
):
    """
    🛡️ Configure Anti-Wipeout Protection (% BASED!)
    
    🔥 UNIVERSAL SCALING - รองรับ $100 ถึง $2,000,000,000!
    
    - max_risk_per_trade_percent: Risk % ต่อเทรด (default: 1.0, range: 0.5-3.0)
    - gold_max_lot_per_1000: Max lot Gold ต่อทุก $1000 (default: 0.02, range: 0.01-0.10)
    - forex_max_lot_per_1000: Max lot Forex ต่อทุก $1000 (default: 0.05)
    - gold_sl_percent_min: Gold SL ขั้นต่ำ (% of price, default: 0.3)
    - gold_sl_percent_max: Gold SL สูงสุด (% of price, default: 1.0)
    - block_counter_trend: ห้ามเทรดสวนเทรนด์ (default: True)
    - min_trend_strength: Trend strength ขั้นต่ำที่จะ block (default: 60%)
    """
    global _anti_wipeout_config
    
    changes = []
    
    if max_risk_per_trade_percent is not None:
        _anti_wipeout_config["max_risk_per_trade_percent"] = max(0.5, min(3.0, max_risk_per_trade_percent))
        changes.append(f"max_risk: {_anti_wipeout_config['max_risk_per_trade_percent']}%")
    
    if gold_max_lot_per_1000 is not None:
        _anti_wipeout_config["gold_max_lot_per_1000"] = max(0.01, min(0.10, gold_max_lot_per_1000))
        changes.append(f"gold_max_lot: {_anti_wipeout_config['gold_max_lot_per_1000']} per $1000")
    
    if forex_max_lot_per_1000 is not None:
        _anti_wipeout_config["forex_max_lot_per_1000"] = max(0.01, min(0.20, forex_max_lot_per_1000))
        changes.append(f"forex_max_lot: {_anti_wipeout_config['forex_max_lot_per_1000']} per $1000")
    
    if gold_sl_percent_min is not None:
        _anti_wipeout_config["gold_sl_percent_min"] = max(0.1, min(1.0, gold_sl_percent_min))
        changes.append(f"gold_sl_min: {_anti_wipeout_config['gold_sl_percent_min']}% of price")
    
    if gold_sl_percent_max is not None:
        _anti_wipeout_config["gold_sl_percent_max"] = max(0.3, min(3.0, gold_sl_percent_max))
        changes.append(f"gold_sl_max: {_anti_wipeout_config['gold_sl_percent_max']}% of price")
    
    if block_counter_trend is not None:
        _anti_wipeout_config["block_counter_trend"] = block_counter_trend
        changes.append(f"block_counter_trend: {block_counter_trend}")
    
    if min_trend_strength is not None:
        _anti_wipeout_config["min_trend_strength"] = max(30, min(90, min_trend_strength))
        changes.append(f"min_trend_strength: {_anti_wipeout_config['min_trend_strength']}%")
    
    logger.info(f"🛡️ Anti-Wipeout configured: {changes}")
    
    return {
        "status": "success",
        "changes": changes,
        "config": _anti_wipeout_config
    }


@router.post("/anti-wipeout/preset/{preset}")
async def set_anti_wipeout_preset(preset: str):
    """
    🛡️ Set Anti-Wipeout Preset (% BASED - Universal Scaling!)
    
    Presets:
    - ultra_safe: Very conservative (Risk 0.5%, สำหรับ port เล็ก $100-$500)
    - safe: Recommended for most (Risk 1%, default)
    - moderate: More trades (Risk 1.5%)
    - aggressive: Higher risk (Risk 2% - NOT recommended for small accounts!)
    """
    global _anti_wipeout_config
    
    presets = {
        "ultra_safe": {
            "max_risk_per_trade_percent": 0.5,   # 🔥 ลดเหลือ 0.5%!
            "gold_max_lot_per_1000": 0.01,       # 🔥 ลดเหลือ 0.01 lot per $1000
            "gold_sl_percent_min": 0.4,          # 🔥 SL >= 0.4% of price
            "gold_sl_percent_max": 1.2,          # 🔥 SL <= 1.2% of price
            "block_counter_trend": True,
            "min_trend_strength": 50,
            "description": "Ultra safe สำหรับ $100-$500 - Risk 0.5% ต่อเทรด"
        },
        "safe": {
            "max_risk_per_trade_percent": 1.0,   # 🔥 ลดเหลือ 1%!
            "gold_max_lot_per_1000": 0.02,       # 🔥 ลดเหลือ 0.02 lot per $1000
            "gold_sl_percent_min": 0.3,          # 🔥 SL >= 0.3% of price
            "gold_sl_percent_max": 1.0,          # 🔥 SL <= 1.0% of price
            "block_counter_trend": True,
            "min_trend_strength": 60,
            "description": "แนะนำสำหรับส่วนใหญ่ - Risk 1% ต่อเทรด"
        },
        "moderate": {
            "max_risk_per_trade_percent": 1.5,   # 🔥 1.5%
            "gold_max_lot_per_1000": 0.03,       # 🔥 0.03 lot per $1000
            "gold_sl_percent_min": 0.25,         # 🔥 SL >= 0.25% of price
            "gold_sl_percent_max": 0.8,          # 🔥 SL <= 0.8% of price
            "block_counter_trend": True,
            "min_trend_strength": 65,
            "description": "เทรดเยอะขึ้น - Risk 1.5% ต่อเทรด"
        },
        "aggressive": {
            "max_risk_per_trade_percent": 2.0,   # 🔥 2%
            "gold_max_lot_per_1000": 0.05,       # 🔥 0.05 lot per $1000
            "gold_sl_percent_min": 0.2,          # 🔥 SL >= 0.2% of price
            "gold_sl_percent_max": 0.6,          # 🔥 SL <= 0.6% of price
            "block_counter_trend": True,         # 🔥 ยังบล็อกสวนเทรนด์!
            "min_trend_strength": 70,
            "description": "⚠️ สำหรับ port $5000+ เท่านั้น - Risk 2% ต่อเทรด"
        }
    }
    
    if preset not in presets:
        return {"status": "error", "message": f"Unknown preset: {preset}. Available: {list(presets.keys())}"}
    
    config = presets[preset]
    
    _anti_wipeout_config["max_risk_per_trade_percent"] = config["max_risk_per_trade_percent"]
    _anti_wipeout_config["gold_max_lot_per_1000"] = config["gold_max_lot_per_1000"]
    _anti_wipeout_config["gold_sl_percent_min"] = config["gold_sl_percent_min"]
    _anti_wipeout_config["gold_sl_percent_max"] = config["gold_sl_percent_max"]
    _anti_wipeout_config["block_counter_trend"] = config["block_counter_trend"]
    _anti_wipeout_config["min_trend_strength"] = config["min_trend_strength"]
    
    logger.info(f"🛡️ Preset '{preset}' activated: {config['description']}")
    
    return {
        "status": "success",
        "preset": preset,
        "description": config["description"],
        "config": _anti_wipeout_config,
        "warning": "⚠️ สำหรับ port $5000+ เท่านั้น!" if preset == "aggressive" else None,
        "scaling_examples": {
            "$300 port": f"max_lot={max(0.01, 0.3 * config['gold_max_lot_per_1000']):.2f}, max_risk=${300 * config['max_risk_per_trade_percent'] / 100:.2f}",
            "$1000 port": f"max_lot={config['gold_max_lot_per_1000']:.2f}, max_risk=${1000 * config['max_risk_per_trade_percent'] / 100:.2f}",
            "$10000 port": f"max_lot={10 * config['gold_max_lot_per_1000']:.2f}, max_risk=${10000 * config['max_risk_per_trade_percent'] / 100:.2f}",
        }
    }
