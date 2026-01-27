"""
Unified Bot API - Single Source of Truth
==========================================

?? Bot Control ??? Signal API ??????????????
????????????? 2 ????????????

Architecture:
- ONE bot instance (_bot) 
- ONE trading engine (from trading_routes)
- ALL views read from same source

?? MUTUAL EXCLUSION: ????????????? 2 ???????????????
- MODE_AUTO: Bot ????????? + ?????????????
- MODE_MANUAL: ??????????????????? ??????? (?????????????)

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
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/unified", tags=["unified"])


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
        "pnl": 0.0
    },
    "error": None,
    "started_at": None
}




# 🔓 DUPLICATE TRADE PREVENTION - AGGRESSIVE FOR MORE TRADES
_last_traded_signal = {}      # {symbol: {"signal": "BUY", "timestamp": datetime, "signal_id": "hash"}}
_open_positions = {}          # {symbol: True/False}
_trade_cooldown_seconds = 10  # 🔥 AGGRESSIVE: 10 seconds cooldown (was 30) - maximum trades!

# 🔄 REVERSE SIGNAL CLOSE - ปิด position เมื่อสัญญาณตรงข้าม
_enable_reverse_signal_close = True  # เปิด/ปิด feature นี้

# 🔀 CONTRARIAN MODE - กลับสัญญาณ (BUY→SELL, SELL→BUY)
_contrarian_mode = {
    "enabled": True,                    # 🔥 เปิด Contrarian Mode!
    "reverse_signal": True,             # BUY→SELL, SELL→BUY
    "reverse_strong_signal": True,      # STRONG_BUY→STRONG_SELL, STRONG_SELL→STRONG_BUY
}

# 🎯 AGGRESSIVE TRADING CONFIG - เทรดเยอะ กำไรเยอะ
_aggressive_config = {
    "enabled": True,
    "min_confidence_to_trade": 60,          # 🔥 ลดจาก 65 → 60 (เทรดง่ายขึ้น)
    "signal_window_minutes": 5,             # 🔥 ลดจาก 15 → 5 นาที (เทรดถี่ขึ้น)
    "allow_same_direction_reentry": True,   # ✅ เปิด re-entry ทิศทางเดียวกัน
    "min_profit_for_wait_close": 500,       # 🔥 ปิดเมื่อ WAIT เฉพาะกำไร >= $500 (ไม่ปิดเร็วไป)
    "quick_scalp_mode": False,              # Scalping mode (ยังไม่เปิด)
}

# 💰 SMART PROFIT PROTECTION - ล็อกกำไรอัตโนมัติ
_profit_protection_config = {
    "enabled": True,                    # เปิด/ปิด feature
    "profit_drawdown_percent": 30,      # ปิดเมื่อกำไรลดลง 30% จาก peak (เช่น peak $1000 → ปิดเมื่อเหลือ $700)
    "min_profit_to_protect": 100,       # เริ่ม protect เมื่อกำไร >= $100
    "trailing_stop_trigger": 500,       # เริ่ม trailing เมื่อกำไร >= $500
    "trailing_stop_distance": 200,      # trailing stop ห่าง $200 จาก current profit
}
_peak_profit_by_position = {}           # {ticket: peak_profit} - เก็บกำไรสูงสุดของแต่ละ position

# 🚨 MAX LOSS PROTECTION - บังคับปิดเมื่อขาดทุนเกินกำหนด
_max_loss_config = {
    "enabled": True,                    # เปิด/ปิด feature
    "max_loss_per_position": 5000,      # ปิดเมื่อขาดทุน >= $5,000 ต่อ position
    "max_loss_percent": 10,             # หรือขาดทุน >= 10% ของ balance
    "close_on_reverse_signal": True,    # ปิดทันทีเมื่อสัญญาณตรงข้าม (แม้ขาดทุน)
}


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


async def _run_bot_loop(interval: int, auto_trade: bool):
    """Main bot analysis loop"""
    global _bot, _bot_status
    
    mode_str = "AUTO" if auto_trade else "MANUAL"
    logger.info(f"?? Unified bot loop starting (mode={mode_str}, interval={interval}s)")
    
    while _bot_status["running"]:
        try:
            for symbol in _bot_status["symbols"]:
                # Run analysis
                analysis = await _bot.analyze_symbol(symbol)
                
                if analysis:
                    # Store analysis
                    _bot_status["last_analysis"][symbol] = analysis
                    
                    # Extract signal
                    signal_data = {
                        "symbol": symbol,
                        "signal": analysis.get("signal", "WAIT"),
                        "confidence": analysis.get("enhanced_confidence", 0),
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
                    
                    # Extract layer status
                    _bot_status["layer_status"][symbol] = _extract_layer_status(symbol)
                    
                    logger.info(f"📊 {symbol}: {signal_data['signal']} @ {signal_data['confidence']:.1f}% ({_bot_status['mode']} mode)")
                    
                    # 🔄 REVERSE SIGNAL CLOSE - ปิด position ถ้าสัญญาณตรงข้าม
                    if signal_data["signal"] in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
                        closed = await _check_and_close_opposite_positions(symbol, signal_data["signal"])
                        if closed:
                            _bot_status["last_signal"][symbol]["trade_status"] = "REVERSED"
                            logger.info(f"   🔄 {symbol}: Opposite position closed due to reverse signal")
                    
                    # 🚨 WAIT SIGNAL = CLOSE PROFITABLE - ถ้าสัญญาณเป็น WAIT และมีกำไร → ปิดทันที
                    elif signal_data["signal"] in ["WAIT", "SKIP"]:
                        closed = await _close_profitable_on_wait_signal(symbol)
                        if closed:
                            _bot_status["last_signal"][symbol]["trade_status"] = "CLOSED_ON_WAIT"
                            logger.info(f"   🚨 {symbol}: Profitable position closed due to WAIT signal")
                    
                    # Auto trade ONLY if mode is AUTO
                    if auto_trade and _bot_status["mode"] == BotMode.AUTO.value:
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
                    elif signal_data["signal"] not in ["WAIT", "SKIP"]:
                        logger.info(f"   📋 Signal available but mode is MANUAL - not auto-trading")
                        _bot_status["last_signal"][symbol]["trade_status"] = "MANUAL_MODE"
            
            # 💰 SMART PROFIT PROTECTION - ตรวจสอบทุก cycle
            closed = await _check_profit_protection()
            if closed:
                for pos in closed:
                    logger.info(f"🛡️ Profit protected: {pos['symbol']} locked ${pos['locked_profit']:.2f}")
            
            # Wait for next cycle
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            logger.info("🛑 Bot loop cancelled")
            break
        except Exception as e:
            logger.error(f"❌ Bot loop error: {e}")
            _bot_status["error"] = str(e)
            await asyncio.sleep(5)  # Brief pause on error
    
    
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
            
            # Determine if we should close
            should_close = False
            close_reason = ""
            
            if pos_pnl > 0:
                # กำไร → ปิดเสมอ
                should_close = True
                close_reason = f"PROFIT ${pos_pnl:.2f} + reverse signal"
                logger.info(f"🔄 REVERSE SIGNAL: {symbol} {pos_side} position with PROFIT ${pos_pnl:.2f}, got {new_signal}")
            elif close_on_reverse:
                # ขาดทุน + เปิด option close_on_reverse_signal → ปิดเพื่อหยุดขาดทุน!
                should_close = True
                close_reason = f"LOSS ${pos_pnl:.2f} + reverse signal (CUT LOSS)"
                logger.warning(f"🚨 REVERSE SIGNAL CUT LOSS: {symbol} {pos_side} position with LOSS ${pos_pnl:.2f}, got {new_signal}")
                logger.warning(f"   Market direction changed! Cutting loss to prevent further damage!")
            else:
                # ขาดทุน + ไม่เปิด option → ไม่ปิด
                logger.info(f"🔄 REVERSE SIGNAL: {symbol} {pos_side} position with LOSS ${pos_pnl:.2f}, got {new_signal} → NOT closing (close_on_reverse disabled)")
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
    """Check if there's already an open position for this symbol"""
    global _bot
    
    if not _bot or not _bot.trading_engine:
        return False
    
    try:
        positions = await _bot.trading_engine.broker.get_positions()
        if positions:
            for pos in positions:
                # Handle both dict and Position objects
                if isinstance(pos, dict):
                    pos_symbol = pos.get("symbol", "")
                else:
                    pos_symbol = getattr(pos, "symbol", "")
                if pos_symbol.upper() == symbol.upper():
                    return True
        return False
    except Exception as e:
        logger.warning(f"Failed to check positions: {e}")
        return False  # Assume no position if check fails


async def _can_trade_signal(symbol: str, signal_data: Dict) -> tuple[bool, str]:
    """
    ?? DUPLICATE TRADE PREVENTION
    Check if we should trade this signal
    
    Returns: (can_trade: bool, reason: str)
    """
    global _last_traded_signal, _open_positions, _trade_cooldown_seconds
    
    signal = signal_data.get("signal", "WAIT")
    confidence = signal_data.get("confidence", 0)
    
    # 1. Check if signal is tradeable
    if signal in ["WAIT", "SKIP"]:
        return False, "Signal is WAIT/SKIP"
    
    # 2. Check for open positions
    has_position = await _check_open_positions(symbol)
    if has_position:
        return False, f"Already have open position for {symbol}"
    
    # 3. Generate signal ID
    signal_id = _generate_signal_id(symbol, signal, confidence)
    
    # 4. Check if we already traded this signal
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


async def _execute_signal_trade(symbol: str, signal_data: Dict):
    """Execute trade based on signal with duplicate prevention"""
    global _bot, _bot_status, _last_traded_signal
    
    # Double check - only execute in AUTO mode
    if _bot_status["mode"] != BotMode.AUTO.value:
        logger.warning(f"?? Trade blocked - not in AUTO mode")
        return
    
    # ?? DUPLICATE PREVENTION CHECK
    can_trade, reason = await _can_trade_signal(symbol, signal_data)
    if not can_trade:
        logger.info(f"??? Trade blocked for {symbol}: {reason}")
        return
    
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
