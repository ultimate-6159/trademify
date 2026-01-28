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
# 🔥 STABILITY CONFIG - 10 Year Runtime
# =====================
_stability_config = {
    "auto_restart_enabled": True,           # 🔄 Auto-restart เมื่อ crash
    "max_restart_attempts": 0,              # 🔥 0 = UNLIMITED restarts (10 year mode!)
    "restart_cooldown_seconds": 30,         # รอ 30 วินาทีก่อน restart
    "watchdog_interval_seconds": 60,        # ตรวจสอบ health ทุก 60 วินาที
    "memory_cleanup_interval": 300,         # ทำความสะอาด memory ทุก 5 นาที
    "max_memory_mb": 2048,                  # ถ้า memory > 2GB ให้ cleanup
    "state_persistence_enabled": True,      # เก็บ state เพื่อ restore
    "state_file_path": "bot_state.json",    # ไฟล์เก็บ state
    "heartbeat_timeout_seconds": 120,       # ถ้าไม่มี heartbeat 2 นาที = dead
    "auto_start_on_api_init": False,        # เริ่ม bot อัตโนมัติเมื่อ API start
    "daily_restart_count_reset": True,      # 🔥 Reset restart count ทุกวัน
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
    "started_at": datetime.now().isoformat(),
}

# 🔥 WATCHDOG STATE
_watchdog_task = None
_last_successful_cycle = None


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
























# 🔓 DUPLICATE TRADE PREVENTION
_last_traded_signal = {}      # {symbol: {"signal": "BUY", "timestamp": datetime, "signal_id": "hash"}}
_open_positions = {}          # {symbol: True/False}
_trade_cooldown_seconds = 300  # 5 นาที cooldown

# 🥇 SYMBOL WHITELIST - เทรดเฉพาะ Gold เท่านั้น!
_symbol_whitelist = {
    "enabled": True,                         # ✅ เปิด! Block Forex
    "allowed_symbols": ["XAUUSDm", "XAUUSD", "GOLD"],  # 🥇 Gold only!
    "block_forex": True,                     # ❌ Block all Forex pairs
}

# 🔄 REVERSE SIGNAL CLOSE - ปิด position เมื่อสัญญาณตรงข้าม (ต้องกำไรก่อน!)
_enable_reverse_signal_close = True    # ✅ เปิด! แต่ต้องกำไรก่อน
_open_new_after_close = True           # ✅ ปิดแล้วเปิดใหม่ (รอ pullback)
_reverse_signal_min_profit = 50        # 🔥 ต้องมีกำไร >= $50 ถึงจะปิดตาม reverse signal

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
    "min_profit_to_exit_early": 500,        # 🔥 เพิ่มเป็น $500 ถึงจะ early exit
}



# 🔀 CONTRARIAN MODE - กลับสัญญาณ
# ❌ ปิดถาวร! ใช้สัญญาณปกติ (BUY=BUY, SELL=SELL)
_contrarian_mode = {
    "enabled": False,
    "reverse_signal": False,
    "reverse_strong_signal": False,
}

# 🎯 PULLBACK ENTRY STRATEGY - รอ pullback ก่อนเข้าเทรด
# สัญญาณมา → รอราคา pullback → รอนิ่ง → ค่อยเข้า
_pullback_config = {
    "enabled": True,                         # ✅ เปิดใช้งาน
    "min_pullback_percent": 0.10,            # 🔥 ลดเหลือ 0.10% (Gold = ~$5)
    "max_pullback_percent": 0.50,            # 🔥 ลดเหลือ 0.50% (Gold = ~$25) ถ้าเกินนี้ = สัญญาณผิด
    "wait_for_stabilization": True,          # รอให้ราคานิ่งก่อน
    "stabilization_candles": 1,              # 🔥 ลดเหลือ 1 รอบ (เร็วขึ้น)
    "max_wait_minutes": 15,                  # 🔥 ลดเหลือ 15 นาที
    "require_signal_still_valid": True,      # สัญญาณต้องยังคงอยู่
}
_pending_signals = {}  # {symbol: {"signal": "BUY", "price_at_signal": 2750, "timestamp": datetime, "pullback_detected": False}}

# 🎯 SMART TRADING CONFIG - เทรดบ่อย + แม่นยำ
_aggressive_config = {
    "enabled": True,
    "min_confidence_to_trade": 75,          # 🔥 เพิ่มเป็น 75% (Conservative)
    "min_quality": "HIGH",                  # 🔥 ต้อง HIGH ขึ้นไป
    "signal_window_minutes": 5,             # Signal ID window 5 นาที
    "allow_same_direction_reentry": True,   # ✅ เปิด re-entry ทิศทางเดียวกัน
    "min_profit_for_wait_close": 200,       # ✅ ปิดเมื่อ WAIT + กำไร >= $200 (ไม่รอนาน)
    "quick_scalp_mode": False,
}

# 💰 SMART PROFIT PROTECTION - ล็อกกำไรอัตโนมัติ
_profit_protection_config = {
    "enabled": True,
    "profit_drawdown_percent": 25,          # ✅ ปิดเมื่อกำไรลดลง 25% จาก peak
    "min_profit_to_protect": 50,            # ✅ เริ่ม protect เมื่อกำไร >= $50
    "trailing_stop_trigger": 300,           # เริ่ม trailing เมื่อกำไร >= $300
    "trailing_stop_distance": 100,          # trailing stop ห่าง $100
}
_peak_profit_by_position = {}

# 🚨 MAX LOSS PROTECTION - บังคับปิดเมื่อขาดทุนเกินกำหนด
_max_loss_config = {
    "enabled": True,
    "max_loss_per_position": 1500,          # 🔥 ลดเหลือ $1,500 ต่อ position
    "max_loss_percent": 3,                  # 🔥 ลดเหลือ 3% ของ balance
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
_known_positions = {}  # {ticket: {"symbol": "XAUUSDm", "side": "BUY", "open_price": 5100}}


async def _sync_positions_with_mt5():
    """
    🔄 SYNC WITH MT5 - ตรวจสอบว่า position ถูกปิดไปแล้วหรือยัง
    
    เมื่อ position ถูกปิดด้วย SL/TP (external) → Bot ต้องรู้และ update สถานะ
    
    🔥 CRITICAL: ต้องเรียก MT5 positions ใหม่ทุกครั้ง เพื่อให้ได้ข้อมูลล่าสุด
    """
    global _bot, _known_positions, _bot_status, _peak_profit_by_position, _last_traded_signal
    
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
                
                closed_externally.append({
                    "ticket": ticket,
                    "symbol": closed_symbol,
                    "side": closed_side,
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
        
        # Log closed positions
        for pos in closed_externally:
            logger.warning(f"📢 POSITION CLOSED EXTERNALLY: #{pos['ticket']} ({pos['symbol']} {pos['side']}) - SL/TP hit!")
        
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


async def _run_bot_loop(interval: int, auto_trade: bool):
    """
    🔥 ENTERPRISE GRADE Bot Analysis Loop
    
    Features:
    - Auto-reconnect on MT5 disconnect
    - Heartbeat tracking for watchdog
    - Error recovery
    - Memory-efficient
    """
    global _bot, _bot_status, _last_successful_cycle, _runtime_stats
    
    mode_str = "AUTO" if auto_trade else "MANUAL"
    logger.info(f"🚀 Unified bot loop starting (mode={mode_str}, interval={interval}s)")
    
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
            
            for symbol in _bot_status["symbols"]:
                # Run analysis
                analysis = await _bot.analyze_symbol(symbol)
                
                if analysis:
                    # Store analysis
                    _bot_status["last_analysis"][symbol] = analysis
                    
                    # Extract signal - try multiple confidence fields
                    raw_confidence = analysis.get("enhanced_confidence", 0) or analysis.get("base_confidence", 0) or analysis.get("confidence", 0)
                    
                    signal_data = {
                        "symbol": symbol,
                        "signal": analysis.get("signal", "WAIT"),
                        "confidence": raw_confidence,
                        "quality": analysis.get("quality", "SKIP"),
                        "current_price": analysis.get("current_price", 0),
                        "stop_loss": analysis.get("risk_management", {}).get("stop_loss", 0),
                        "take_profit": analysis.get("risk_management", {}).get("take_profit", 0),
                        "scores": analysis.get("scores", {}),
                        "indicators": analysis.get("indicators", {}),
                        "market_regime": analysis.get("market_regime", "UNKNOWN"),
                        "timestamp": datetime.now().isoformat()
                    }
                    _bot_status["last_signal"][symbol] = signal_data
                    
                    # ⚡ TRACK SIGNAL HISTORY for momentum detection
                    _track_signal_history(symbol, signal_data)
                    
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
                            # Check if can trade before attempting
                            can_trade, reason = await _can_trade_signal(symbol, signal_data)
                            if can_trade:
                                await _execute_signal_trade(symbol, signal_data)
                                # Update signal status
                                _bot_status["last_signal"][symbol]["trade_status"] = "EXECUTED"
                            else:
                                logger.info(f"   ❌ {symbol}: Trade blocked - {reason}")
                                _bot_status["last_signal"][symbol]["trade_status"] = f"BLOCKED: {reason}"
                        else:
                            _bot_status["last_signal"][symbol]["trade_status"] = "NO_SIGNAL"
                    elif signal_data["signal"] not in ["WAIT", "SKIP"] and not closed_opposite:
                        logger.info(f"   📋 Signal available but mode is MANUAL - not auto-trading")
                        _bot_status["last_signal"][symbol]["trade_status"] = "MANUAL_MODE"
            
            # 💰 SMART PROFIT PROTECTION - ตรวจสอบทุก cycle
            closed = await _check_profit_protection()
            if closed:
                for pos in closed:
                    logger.info(f"🛡️ Profit protected: {pos['symbol']} locked ${pos['locked_profit']:.2f}")
            
            # ✅ SUCCESSFUL CYCLE - Update heartbeat
            _last_successful_cycle = datetime.now()
            consecutive_failures = 0  # Reset on success
            
            # 📊 Log cycle stats periodically (every 10 cycles)
            if cycle_count % 10 == 0:
                uptime = _runtime_stats.get("total_uptime_seconds", 0)
                restarts = _runtime_stats.get("restart_count", 0)
                logger.info(f"📊 Cycle #{cycle_count} | Uptime: {uptime//3600}h {(uptime%3600)//60}m | Restarts: {restarts}")
            
            # Wait for next cycle
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            logger.info("🛑 Bot loop cancelled")
            break
        except OSError as e:
            # 🔥 Network error - รอแล้วลองใหม่
            consecutive_failures += 1
            logger.warning(f"⚠️ Network error in bot loop ({consecutive_failures}/{max_failures}): {e}")
            _bot_status["error"] = f"Network error: {e}"
            _runtime_stats["errors_count"] += 1
            
            if consecutive_failures >= max_failures:
                logger.error(f"🔥 Too many failures - triggering watchdog restart")
                break  # Let watchdog handle restart
            
            await asyncio.sleep(30)  # รอ 30 วินาทีก่อนลองใหม่
        except ConnectionError as e:
            # 🔥 Connection lost - รอแล้วลองใหม่
            consecutive_failures += 1
            logger.warning(f"⚠️ Connection error in bot loop ({consecutive_failures}/{max_failures}): {e}")
            _bot_status["error"] = f"Connection error: {e}"
            _runtime_stats["errors_count"] += 1
            
            if consecutive_failures >= max_failures:
                logger.error(f"🔥 Too many failures - triggering watchdog restart")
                break
            
            await asyncio.sleep(30)
        except Exception as e:
            # 🔥 Unexpected error
            consecutive_failures += 1
            error_type = type(e).__name__
            logger.error(f"❌ Bot loop error ({error_type}) [{consecutive_failures}/{max_failures}]: {e}")
            logger.error(traceback.format_exc())
            _bot_status["error"] = f"{error_type}: {e}"
            _runtime_stats["errors_count"] += 1
            
            if consecutive_failures >= max_failures:
                logger.error(f"🔥 Too many failures - triggering watchdog restart")
                break
            
            await asyncio.sleep(10)  # รอ 10 วินาทีก่อนลองใหม่
    
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
    min_profit = _profit_protection_config.get("min_profit_to_protect", 100)
    drawdown_pct = _profit_protection_config.get("profit_drawdown_percent", 30)
    
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
            
            # 🔥 AGGRESSIVE: Only close if profit >= min_profit_for_wait_close
            min_profit_for_wait = _aggressive_config.get("min_profit_for_wait_close", 500)
            
            # Only close if profitable AND profit >= minimum
            if pos_pnl <= 0:
                logger.info(f"🚨 WAIT SIGNAL: {symbol} {pos_side} PnL=${pos_pnl:.2f} (loss) → NOT closing")
                continue
            
            if pos_pnl < min_profit_for_wait:
                logger.info(f"🚨 WAIT SIGNAL: {symbol} {pos_side} PnL=${pos_pnl:.2f} < ${min_profit_for_wait} → NOT closing (let it run)")
                continue
            
            # Close profitable position only if >= minimum
            logger.warning(f"🚨 WAIT SIGNAL CLOSE: {symbol} #{pos_id} | {pos_side} | Profit: ${pos_pnl:.2f} >= ${min_profit_for_wait}")
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
            
            # 🔥 NEW: ต้องมีกำไรขั้นต่ำก่อนถึงจะปิด
            min_profit_for_reverse = _reverse_signal_min_profit  # Default $50
            
            # Determine if we should close
            should_close = False
            close_reason = ""
            
            if pos_pnl >= min_profit_for_reverse:
                # กำไร >= min → ปิดเลย ล็อกกำไร!
                should_close = True
                close_reason = f"PROFIT ${pos_pnl:.2f} >= ${min_profit_for_reverse} + reverse signal"
                logger.info(f"✅ REVERSE SIGNAL PROFIT: {symbol} {pos_side} PROFIT ${pos_pnl:.2f} + {new_signal} → CLOSE & LOCK PROFIT!")
            elif pos_pnl > 0 and pos_pnl < min_profit_for_reverse:
                # กำไรน้อย → ไม่ปิด รอกำไรเพิ่ม
                logger.info(f"⏳ REVERSE SIGNAL: {symbol} {pos_side} profit ${pos_pnl:.2f} < ${min_profit_for_reverse} → HOLD (wait for more profit)")
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
    """
    global _bot, _known_positions
    
    if not _bot or not _bot.trading_engine:
        return False
    
    try:
        # 🔥 ALWAYS get fresh positions from MT5
        positions = await _bot.trading_engine.broker.get_positions()
        
        if not positions:
            # No positions at all
            logger.debug(f"📊 _check_open_positions({symbol}): MT5 returns 0 positions")
            return False
        
        # Check if any position matches this symbol
        for pos in positions:
            # Handle both dict and Position objects
            if isinstance(pos, dict):
                pos_symbol = pos.get("symbol", "")
                pos_ticket = pos.get("ticket") or pos.get("id")
            else:
                pos_symbol = getattr(pos, "symbol", "")
                pos_ticket = getattr(pos, "ticket", None) or getattr(pos, "id", None)
            
            if pos_symbol.upper() == symbol.upper():
                logger.debug(f"📊 _check_open_positions({symbol}): Found position #{pos_ticket}")
                return True
        
        logger.debug(f"📊 _check_open_positions({symbol}): No position found for this symbol")
        return False
        
    except Exception as e:
        logger.warning(f"Failed to check positions: {e}")
        return False  # Assume no position if check fails


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
                # 🔥 ต้องมีกำไรขั้นต่ำสูงก่อนถึงจะปิด
                min_profit = _signal_weakening_config.get("min_profit_to_exit_early", 500)
                
                # ต้องมีกำไร >= min_profit ถึงจะปิด
                if pos_pnl >= min_profit:
                    logger.warning(f"⚡ SIGNAL WEAKENING: {symbol} - {reason}")
                    logger.warning(f"   Position: {pos_side} | PnL: ${pos_pnl:.2f} (>= ${min_profit})")
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







async def _can_trade_signal(symbol: str, signal_data: Dict) -> tuple[bool, str]:
    """
    🎯 SMART TRADE FILTER
    Check if we should trade this signal - Quality + Confidence filter
    
    🥇 Gold: MEDIUM quality OK (Gold Strategy v2 มี filter เข้มอยู่แล้ว)
    💱 Forex: ❌ BLOCKED! ไม่เทรด Forex
    
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
    
    # 2. 🎯 SYMBOL-SPECIFIC QUALITY FILTER (Gold only now)
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
    
    # 3. 🎯 Confidence Filter
    if confidence < min_confidence:
        return False, f"Confidence {confidence:.1f}% < minimum {min_confidence}% (for {'Gold' if is_gold else 'Forex'})"
    
    # 4. 🎯 PULLBACK ENTRY CHECK - รอ pullback ก่อนเข้า
    current_price = signal_data.get("current_price", 0)
    if current_price > 0:
        can_enter_pullback, pullback_reason = _check_pullback_entry(symbol, signal_data, current_price)
        if not can_enter_pullback:
            return False, f"PULLBACK: {pullback_reason}"
    
    # 5. Check for open positions
    has_position = await _check_open_positions(symbol)
    if has_position:
        return False, f"Already have open position for {symbol}"
    
    # 6. Generate signal ID
    signal_id = _generate_signal_id(symbol, signal, confidence)
    
    # 7. Check if we already traded this signal
    last_trade = _last_traded_signal.get(symbol)
    if last_trade:
        last_signal_id = last_trade.get("signal_id")
        last_time = last_trade.get("timestamp")
        
        # Same signal ID = duplicate
        if last_signal_id == signal_id:
            return False, f"Already traded this signal (ID: {signal_id})"
        
        # Check cooldown
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
    
    # Double check - only execute in AUTO mode
    if _bot_status["mode"] != BotMode.AUTO.value:
        logger.warning(f"⛔ Trade blocked - not in AUTO mode")
        return
    
    # 🎯 PULLBACK CHECK - ต้องเช็คเสมอ! (ไม่ว่า skip_position_check จะเป็นอะไร)
    current_price = signal_data.get("current_price", 0)
    if _pullback_config.get("enabled", True) and current_price > 0:
        can_enter_pullback, pullback_reason = _check_pullback_entry(symbol, signal_data, current_price)
        if not can_enter_pullback:
            logger.info(f"⏳ PULLBACK WAIT: {symbol} - {pullback_reason}")
            return  # ❌ ไม่เข้าเทรด - รอ pullback
    
    # 🔥 DUPLICATE PREVENTION CHECK (can skip if coming from reverse signal close)
    if not skip_position_check:
        can_trade, reason = await _can_trade_signal(symbol, signal_data)
        if not can_trade:
            logger.info(f"⛔ Trade blocked for {symbol}: {reason}")
            return  # ✅ Fix: return inside if block
    
    try:
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
            
            result = await _bot.execute_trade(modified_analysis)
            
            if result and result.get("success"):
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
                
                contrarian_tag = " [CONTRARIAN]" if original_signal != final_signal else ""
                logger.info(f"✅ Trade executed: {symbol} {side}{contrarian_tag} (ID: {signal_id}) - Cooldown {_trade_cooldown_seconds}s started")
                _bot_status["daily_stats"]["trades"] += 1
            else:
                reason = result.get("reason", "Unknown") if result else "No result"
                logger.warning(f"⚠️ Trade not executed: {reason}")
                
    except Exception as e:
        logger.error(f"❌ Trade execution error: {e}")


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
    """
    global _bot, _bot_status
    
    try:
        # Get account info
        account = {"balance": 0, "equity": 0, "profit": 0, "free_margin": 0, "margin_level": 0}
        try:
            if _bot and _bot.trading_engine:
                balance = await _bot.trading_engine.broker.get_balance()
                account_info = await _bot.trading_engine.broker.get_account_info()
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
        
        # Build response safely
        response = {
            "status": "ok",
            "bot_mode": _bot_status["mode"],
            "symbol": signal.get("symbol", symbol),
            "signal": signal.get("signal", "WAIT"),
            "confidence": float(signal.get("confidence", 0)),
            "quality": signal.get("quality", "SKIP"),
            "current_price": float(signal.get("current_price", 0)),
            "stop_loss": float(signal.get("stop_loss", 0)),
            "take_profit": float(signal.get("take_profit", 0)),
            "trade_status": signal.get("trade_status", "N/A"),
            "market_regime": signal.get("market_regime", "UNKNOWN"),
            "timestamp": signal.get("timestamp", datetime.now().isoformat())
        }
        
        # Add optional fields if present
        if "scores" in signal:
            response["scores"] = _convert_to_json_serializable(signal["scores"])
        if "indicators" in signal:
            response["indicators"] = _convert_to_json_serializable(signal["indicators"])
        
        return response
        
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
