"""
Trading API Routes
API endpoints สำหรับระบบเทรดอัตโนมัติ

🚨 PRODUCTION: Exness MT5 on Windows VPS only

Security:
- Sensitive endpoints require API Key authentication
- Use X-API-Key header or api_key query parameter
"""
import logging
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from api.security import verify_api_key, check_rate_limit, ValidatedTradeRequest
from trading.engine import TradingEngine, RiskManager, OrderSide, OrderType, Order
from trading.binance_connector import BinanceBroker, BinanceConfig
from trading.mt5_connector import MT5Broker, MT5Config
from trading.position_manager import PositionManager, PositionConfig, TrailingStopConfig, TrailingStopMode
from trading.settings import TradingConfig, BrokerType
from analysis import VotingSystem, VoteResult, Signal
from services.shared_state_service import get_shared_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/trading", tags=["trading"])

# Global instances (จะถูก initialize ตอน startup)
trading_engine: Optional[TradingEngine] = None
position_manager: Optional[PositionManager] = None
# โหลด config จากไฟล์ (ถ้ามี) - จะจำค่าแม้ปิดเครื่อง
trading_config: TradingConfig = TradingConfig.load_from_file()


# =====================
# Request/Response Models
# =====================

class RiskSettings(BaseModel):
    """Risk management settings"""
    max_risk_per_trade: Optional[float] = Field(None, ge=0.1, le=10.0)
    max_daily_loss: Optional[float] = Field(None, ge=1.0, le=20.0)
    max_positions: Optional[int] = Field(None, ge=1, le=20)


class SignalSettings(BaseModel):
    """Signal settings"""
    min_confidence: Optional[float] = Field(None, ge=50.0, le=100.0)
    min_quality: Optional[str] = Field(None, description="PREMIUM, HIGH, MEDIUM, LOW")
    allowed_signals: Optional[List[str]] = None


class PatternSettings(BaseModel):
    """Pattern matching settings"""
    top_k: Optional[int] = Field(None, ge=5, le=100)
    min_correlation: Optional[float] = Field(None, ge=0.5, le=1.0)
    window_size: Optional[int] = Field(None, ge=20, le=200)


class TradingSettingsRequest(BaseModel):
    """Request สำหรับอัพเดทการตั้งค่า"""
    enabled: Optional[bool] = None
    broker_type: Optional[str] = None
    # Nested settings from frontend
    risk: Optional[RiskSettings] = None
    signals: Optional[SignalSettings] = None
    pattern: Optional[PatternSettings] = None
    symbols: Optional[List[str]] = None
    timeframe: Optional[str] = None
    # Legacy flat fields (backwards compatibility)
    max_risk_per_trade: Optional[float] = Field(None, ge=0.1, le=10.0)
    max_daily_loss: Optional[float] = Field(None, ge=1.0, le=20.0)
    max_positions: Optional[int] = Field(None, ge=1, le=20)
    min_confidence: Optional[float] = Field(None, ge=50.0, le=100.0)
    allowed_signals: Optional[List[str]] = None


class ManualTradeRequest(BaseModel):
    """Request สำหรับเปิด Position แบบ manual"""
    symbol: str
    side: str  # BUY or SELL
    quantity: float = Field(gt=0)
    order_type: str = "MARKET"
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class ModifyPositionRequest(BaseModel):
    """Request สำหรับแก้ไข Position"""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class SignalTradeRequest(BaseModel):
    """Request สำหรับเทรดตามสัญญาณ"""
    symbol: str
    signal: str
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


# =====================
# Initialization
# =====================

async def init_trading_system(config: TradingConfig = None):
    """Initialize trading system"""
    global trading_engine, position_manager, trading_config
    
    if config:
        trading_config = config
    
    # Create broker based on config
    if trading_config.broker_type == BrokerType.BINANCE:
        broker = BinanceBroker(BinanceConfig(
            api_key=trading_config.binance_api_key,
            api_secret=trading_config.binance_api_secret,
            testnet=trading_config.binance_testnet,
        ))
    elif trading_config.broker_type == BrokerType.MT5:
        broker = MT5Broker(MT5Config(
            login=trading_config.mt5_login,
            password=trading_config.mt5_password,
            server=trading_config.mt5_server,
        ))
    else:
        # Default to MT5 broker
        logger.warning(f"Unknown broker type: {trading_config.broker_type}, defaulting to MT5")
        broker = MT5Broker(MT5Config(
            login=trading_config.mt5_login,
            password=trading_config.mt5_password,
            server=trading_config.mt5_server,
        ))
    
    # Create risk manager with confidence based on quality setting
    effective_min_confidence = trading_config.get_min_confidence_from_quality()
    risk_manager = RiskManager(
        max_risk_per_trade=trading_config.max_risk_per_trade,
        max_daily_loss=trading_config.max_daily_loss,
        max_positions=trading_config.max_positions,
        min_confidence=effective_min_confidence,  # Use quality-based threshold
        max_drawdown=trading_config.max_drawdown,
    )
    logger.info(f"Risk manager initialized with min_confidence={effective_min_confidence}% (from {trading_config.min_quality} quality)")
    
    # Create trading engine
    trading_engine = TradingEngine(
        broker=broker,
        risk_manager=risk_manager,
        max_positions=trading_config.max_positions,
        enabled=trading_config.enabled,
    )
    
    # Create position manager
    position_config = PositionConfig(
        max_holding_time=trading_config.max_holding_hours,
    )
    
    if trading_config.trailing_stop_enabled:
        position_config.trailing_stop = TrailingStopConfig(
            mode=TrailingStopMode.PERCENT,
            value=trading_config.trailing_stop_percent,
            activation_profit=trading_config.trailing_stop_activation,
        )
    
    if trading_config.break_even_enabled:
        position_config.break_even_at = trading_config.break_even_at_percent
    
    # Get shared state for multi-VPS sync
    shared_state = get_shared_state()
    
    position_manager = PositionManager(
        broker=broker,
        config=position_config,
        shared_state=shared_state,
        enable_sync=True  # Enable multi-VPS sync
    )
    
    # Start systems if enabled
    if trading_config.enabled:
        await trading_engine.start()
        await position_manager.start(symbols=trading_config.symbols)
    
    return True


# =====================
# Status & Settings Endpoints
# =====================

@router.get("/status")
async def get_trading_status():
    """ดึงสถานะของระบบเทรด"""
    if not trading_engine:
        return {
            "initialized": False,
            "message": "Trading system not initialized"
        }
    
    return {
        "initialized": True,
        "enabled": trading_engine.enabled,
        "running": trading_engine._running,
        "broker_connected": trading_engine.broker._connected,
        "open_positions": len(trading_engine.positions),
        "stats": trading_engine.get_stats(),
    }


@router.get("/settings")
async def get_trading_settings():
    """ดึงการตั้งค่าปัจจุบัน"""
    return trading_config.to_dict()


@router.put("/settings")
async def update_trading_settings(
    request: TradingSettingsRequest,
    api_key: str = Depends(verify_api_key)
):
    """อัพเดทการตั้งค่า (🔒 Requires API Key)"""
    global trading_config
    
    if request.enabled is not None:
        trading_config.enabled = request.enabled
    if request.broker_type is not None:
        trading_config.broker_type = BrokerType(request.broker_type)
    
    # Handle nested risk settings from frontend
    if request.risk:
        if request.risk.max_risk_per_trade is not None:
            trading_config.max_risk_per_trade = request.risk.max_risk_per_trade
        if request.risk.max_daily_loss is not None:
            trading_config.max_daily_loss = request.risk.max_daily_loss
        if request.risk.max_positions is not None:
            trading_config.max_positions = request.risk.max_positions
    
    # Handle nested signal settings from frontend
    if request.signals:
        if request.signals.min_quality is not None:
            trading_config.min_quality = request.signals.min_quality
            # Auto-update min_confidence based on quality
            trading_config.min_confidence = trading_config.get_min_confidence_from_quality()
        if request.signals.min_confidence is not None:
            trading_config.min_confidence = request.signals.min_confidence
        if request.signals.allowed_signals is not None:
            trading_config.allowed_signals = request.signals.allowed_signals
    
    # Handle nested pattern settings (stored in config for reference)
    if request.pattern:
        if request.pattern.top_k is not None:
            trading_config.top_k_patterns = request.pattern.top_k
        if request.pattern.min_correlation is not None:
            trading_config.min_correlation = request.pattern.min_correlation
        if request.pattern.window_size is not None:
            trading_config.window_size = request.pattern.window_size
    
    # Handle flat settings from frontend
    if request.symbols is not None:
        trading_config.symbols = request.symbols
    if request.timeframe is not None:
        trading_config.timeframe = request.timeframe
    
    # Legacy flat fields (backwards compatibility)
    if request.max_risk_per_trade is not None:
        trading_config.max_risk_per_trade = request.max_risk_per_trade
    if request.max_daily_loss is not None:
        trading_config.max_daily_loss = request.max_daily_loss
    if request.max_positions is not None:
        trading_config.max_positions = request.max_positions
    if request.min_confidence is not None:
        trading_config.min_confidence = request.min_confidence
    if request.allowed_signals is not None:
        trading_config.allowed_signals = request.allowed_signals
    
    # บันทึก config ลงไฟล์ (จะจำค่าแม้ปิดเครื่อง)
    trading_config.save_to_file()
    
    # Reinitialize with new settings
    await init_trading_system(trading_config)
    
    return {"message": "Settings updated and saved to file", "settings": trading_config.to_dict()}


# =====================
# Control Endpoints
# =====================

@router.post("/start")
async def start_trading(api_key: str = Depends(verify_api_key)):
    """เริ่มระบบเทรดอัตโนมัติ (🔒 Requires API Key)"""
    if not trading_engine:
        await init_trading_system()
    
    success = await trading_engine.start()
    await position_manager.start()
    
    if success:
        return {"message": "Trading system started", "status": "running"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start trading system")


@router.post("/stop")
async def stop_trading(api_key: str = Depends(verify_api_key)):
    """หยุดระบบเทรดอัตโนมัติ (🔒 Requires API Key)"""
    if not trading_engine:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    await trading_engine.stop()
    await position_manager.stop()
    
    return {"message": "Trading system stopped", "status": "stopped"}


@router.post("/pause")
async def pause_trading(api_key: str = Depends(verify_api_key)):
    """พักระบบเทรด (ไม่เปิด Position ใหม่) (🔒 Requires API Key)"""
    if not trading_engine:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    trading_engine.enabled = False
    
    return {"message": "Trading paused", "status": "paused"}


@router.post("/resume")
async def resume_trading(api_key: str = Depends(verify_api_key)):
    """เริ่มระบบเทรดต่อ (🔒 Requires API Key)"""
    if not trading_engine:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    trading_engine.enabled = True
    
    return {"message": "Trading resumed", "status": "running"}


# =====================
# Position Endpoints
# =====================

@router.get("/positions")
async def get_positions():
    """ดึง Position ที่เปิดอยู่ทั้งหมด"""
    if not position_manager:
        return {"positions": []}
    
    positions = position_manager.get_open_positions()
    return {
        "positions": [p.to_dict() for p in positions],
        "count": len(positions),
        "total_pnl": sum(p.pnl for p in positions),
    }


@router.get("/positions/{position_id}")
async def get_position(position_id: str):
    """ดึง Position ตาม ID"""
    if not position_manager:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    position = position_manager.get_position(position_id)
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    
    return position.to_dict()


@router.post("/positions")
async def open_position(
    request: ManualTradeRequest,
    api_key: str = Depends(verify_api_key)
):
    """เปิด Position ใหม่ (manual) (🔒 Requires API Key)"""
    if not trading_engine:
        await init_trading_system()
    
    # Validate side
    try:
        side = OrderSide(request.side.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid side. Use BUY or SELL")
    
    # Create order
    order = Order(
        id=f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        symbol=request.symbol,
        side=side,
        order_type=OrderType(request.order_type.upper()),
        quantity=request.quantity,
        price=request.price,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
    )
    
    # Execute
    result = await trading_engine.broker.place_order(order)
    
    if result.success:
        return {
            "message": "Position opened",
            "order": result.order.to_dict() if result.order else None,
            "position": result.position.to_dict() if result.position else None,
        }
    else:
        raise HTTPException(status_code=400, detail=result.error or "Failed to open position")


@router.put("/positions/{position_id}")
async def modify_position(
    position_id: str,
    request: ModifyPositionRequest,
    api_key: str = Depends(verify_api_key)
):
    """แก้ไข Position (SL/TP) (🔒 Requires API Key)"""
    if not position_manager:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    result = await position_manager.modify_position(
        position_id=position_id,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Position not found")
    
    if result.success:
        return {"message": "Position modified", "position": result.position.to_dict() if result.position else None}
    else:
        raise HTTPException(status_code=400, detail=result.error)


@router.delete("/positions/{position_id}")
async def close_position(
    position_id: str,
    api_key: str = Depends(verify_api_key)
):
    """ปิด Position (🔒 Requires API Key)"""
    if not position_manager:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    result = await position_manager.close_position(position_id, "Manual close via API")
    
    if not result:
        raise HTTPException(status_code=404, detail="Position not found")
    
    if result.success:
        return {
            "message": "Position closed",
            "position": result.position.to_dict() if result.position else None,
        }
    else:
        raise HTTPException(status_code=400, detail=result.error)


@router.delete("/positions")
async def close_all_positions(api_key: str = Depends(verify_api_key)):
    """ปิดทุก Position (🔒 Requires API Key)"""
    if not position_manager:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    results = await position_manager.close_all_positions("Close all via API")
    
    return {
        "message": f"Closed {len(results)} positions",
        "results": [r.to_dict() for r in results],
    }


# =====================
# Signal Integration
# =====================

@router.post("/signal")
async def process_signal(
    request: SignalTradeRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """ประมวลผลสัญญาณและเทรด (ใช้โดย Voting System) (🔒 Requires API Key)"""
    if not trading_engine:
        await init_trading_system()
    
    if not trading_engine.enabled:
        return {
            "message": "Trading disabled",
            "action": "none",
        }
    
    # Check if signal is allowed
    if request.signal not in trading_config.allowed_signals:
        return {
            "message": f"Signal {request.signal} not in allowed list",
            "action": "skipped",
        }
    
    # Create VoteResult
    try:
        signal = Signal[request.signal]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid signal: {request.signal}")
    
    vote_result = VoteResult(
        signal=signal,
        confidence=request.confidence,
        bullish_votes=0,
        bearish_votes=0,
        total_votes=0,
        average_movement=np.array([]),
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
    )
    
    # Process in background
    result = await trading_engine.process_signal(vote_result, request.symbol)
    
    if result:
        return {
            "message": "Signal processed",
            "action": "trade_executed" if result.success else "trade_failed",
            "result": result.to_dict(),
        }
    
    return {
        "message": "Signal processed",
        "action": "no_trade",
    }


# =====================
# Statistics
# =====================

@router.get("/stats")
async def get_trading_stats():
    """ดึงสถิติการเทรด"""
    stats = {
        "engine": trading_engine.get_stats() if trading_engine else {},
        "positions": position_manager.get_stats() if position_manager else {},
        "risk": trading_engine.risk_manager.to_dict() if trading_engine else {},
    }
    
    return stats


@router.get("/history")
async def get_trade_history(limit: int = 50):
    """ดึงประวัติการเทรด"""
    if not position_manager:
        return {"trades": []}
    
    trades = position_manager.closed_positions[-limit:]
    
    return {
        "trades": [t.to_dict() for t in trades],
        "count": len(trades),
    }


# =====================
# Account
# =====================

@router.get("/account")
async def get_account_info():
    """ดึงข้อมูลบัญชี"""
    if not trading_engine:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    account = await trading_engine.broker.get_account_info()
    balance = await trading_engine.broker.get_balance()
    
    return {
        "account": account,
        "balance": balance,
        "broker_type": trading_config.broker_type.value,
    }


@router.get("/price/{symbol}")
async def get_current_price(symbol: str):
    """ดึงราคาปัจจุบัน"""
    if not trading_engine:
        await init_trading_system()
    
    price = await trading_engine.broker.get_current_price(symbol)
    
    return {
        "symbol": symbol,
        "price": price,
        "timestamp": datetime.now().isoformat(),
    }


# =====================
# Cluster Management (Multi-VPS Sync)
# =====================

@router.get("/cluster/status")
async def get_cluster_status():
    """ดึงสถานะ cluster (multi-VPS)"""
    if not position_manager:
        return {
            "cluster_enabled": False,
            "message": "Position manager not initialized"
        }
    
    if not position_manager.shared_state:
        return {
            "cluster_enabled": False,
            "message": "Shared state not available - running in local mode"
        }
    
    summary = position_manager.shared_state.get_cluster_summary()
    
    return {
        "cluster_enabled": True,
        "node_id": position_manager.shared_state.node_id,
        "hostname": position_manager.shared_state.hostname,
        "total_nodes": summary.get("total_nodes", 0),
        "nodes": summary.get("nodes", []),
        "total_positions": summary.get("total_positions", 0),
        "total_pnl": summary.get("total_pnl", 0),
        "positions_by_symbol": summary.get("positions_by_symbol", {}),
        "timestamp": summary.get("timestamp"),
    }


@router.get("/cluster/positions")
async def get_cluster_positions():
    """ดึง positions ทั้งหมดจาก cluster (ทุก VPS nodes)"""
    if not position_manager:
        return {"positions": [], "message": "Position manager not initialized"}
    
    positions = position_manager.get_all_cluster_positions()
    
    return {
        "positions": [p.to_dict() for p in positions],
        "count": len(positions),
        "node_id": position_manager.shared_state.node_id if position_manager.shared_state else "local",
    }


@router.get("/cluster/history")
async def get_cluster_trade_history(limit: int = 100):
    """ดึงประวัติเทรดจาก cluster ทั้งหมด"""
    if not position_manager:
        return {"trades": [], "message": "Position manager not initialized"}
    
    trades = position_manager.get_trade_history(limit=limit)
    
    return {
        "trades": trades,
        "count": len(trades),
    }


@router.get("/cluster/nodes")
async def get_cluster_nodes():
    """ดึงรายการ nodes ที่ active"""
    if not position_manager or not position_manager.shared_state:
        return {"nodes": [], "message": "Shared state not available"}
    
    nodes = position_manager.shared_state.get_active_nodes()
    
    return {
        "nodes": [n.to_dict() for n in nodes],
        "count": len(nodes),
        "this_node": position_manager.shared_state.node_id,
    }

# =====================
# Smart Brain Endpoints
# =====================

@router.get("/smart-brain/insights")
async def get_smart_brain_insights():
    """
    🧠 Smart Brain Insights - ข้อมูลที่ Bot เรียนรู้
    
    Returns:
    - performance_30d: สถิติ 30 วัน (win_rate, pnl, etc.)
    - best_hours: ชั่วโมงที่เทรดดีที่สุด
    - best_symbols: คู่เงินที่เทรดดีที่สุด
    - adaptive_risk_mult: ตัวคูณ risk ปัจจุบัน
    - patterns_in_memory: จำนวน patterns ที่จำได้
    """
    try:
        from trading.smart_brain import get_smart_brain
        brain = get_smart_brain()
        insights = brain.get_insights()
        return {
            "success": True,
            "insights": insights,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting Smart Brain insights: {e}")
        return {
            "success": False,
            "error": str(e),
            "insights": None,
        }


@router.get("/smart-brain/journal")
async def get_trade_journal(days: int = 30):
    """
    📔 Trade Journal - ประวัติเทรดทั้งหมด
    """
    try:
        from trading.smart_brain import get_smart_brain
        brain = get_smart_brain()
        
        # Get trades
        trades = []
        for t in brain.journal.trades[-100:]:  # Last 100
            trades.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl_percent": t.pnl_percent,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "session": t.session,
                "exit_reason": t.exit_reason,
                "is_win": t.is_win() if t.exit_price else None,
            })
        
        stats = brain.journal.get_stats(days)
        
        return {
            "success": True,
            "trades": trades,
            "stats": stats,
            "total_trades": len(brain.journal.trades),
        }
    except Exception as e:
        logger.error(f"Error getting trade journal: {e}")
        return {"success": False, "error": str(e), "trades": []}


@router.get("/smart-brain/patterns")
async def get_pattern_memory():
    """
    🧩 Pattern Memory - patterns ที่จำได้และสถิติ
    """
    try:
        from trading.smart_brain import get_smart_brain
        brain = get_smart_brain()
        
        patterns = []
        for pattern_id, pattern in brain.pattern_memory.patterns.items():
            patterns.append({
                "pattern_id": pattern_id,
                "win_rate": pattern.win_rate,
                "avg_pnl": pattern.avg_pnl,
                "trade_count": pattern.trade_count,
                "last_seen": pattern.last_seen,
            })
        
        # Sort by trade count
        patterns.sort(key=lambda x: x["trade_count"], reverse=True)
        
        return {
            "success": True,
            "patterns": patterns[:50],  # Top 50
            "total_patterns": len(brain.pattern_memory.patterns),
        }
    except Exception as e:
        logger.error(f"Error getting pattern memory: {e}")
        return {"success": False, "error": str(e), "patterns": []}


@router.get("/smart-brain/time-analysis")
async def get_time_analysis():
    """
    ⏰ Time Analysis - สถิติตามชั่วโมง
    """
    try:
        from trading.smart_brain import get_smart_brain
        brain = get_smart_brain()
        
        best_hours = brain.time_analyzer.get_best_hours()
        
        # Get all hours stats
        hours_stats = {}
        for hour, stats in brain.time_analyzer.hour_stats.items():
            if stats["trades"] > 0:
                hours_stats[str(hour)] = {
                    "trades": stats["trades"],
                    "wins": stats["wins"],
                    "win_rate": round((stats["wins"] / stats["trades"]) * 100, 1),
                    "total_pnl": round(stats["total_pnl"], 2),
                }
        
        return {
            "success": True,
            "best_hours": best_hours,
            "all_hours": hours_stats,
        }
    except Exception as e:
        logger.error(f"Error getting time analysis: {e}")
        return {"success": False, "error": str(e)}


@router.get("/smart-brain/symbol-analysis")
async def get_symbol_analysis():
    """
    📊 Symbol Analysis - สถิติตามคู่เงิน
    """
    try:
        from trading.smart_brain import get_smart_brain
        brain = get_smart_brain()
        
        best_symbols = brain.symbol_analyzer.get_best_symbols()
        
        # Get all symbol stats
        all_stats = {}
        for symbol, stats in brain.symbol_analyzer.symbol_stats.items():
            if stats["trades"] > 0:
                all_stats[symbol] = {
                    "trades": stats["trades"],
                    "wins": stats["wins"],
                    "win_rate": round((stats["wins"] / stats["trades"]) * 100, 1),
                    "total_pnl": round(stats["total_pnl"], 2),
                    "avg_holding_hours": round(stats["total_holding"] / stats["trades"], 1) if stats["trades"] > 0 else 0,
                }
        
        return {
            "success": True,
            "best_symbols": best_symbols,
            "all_symbols": all_stats,
        }
    except Exception as e:
        logger.error(f"Error getting symbol analysis: {e}")
        return {"success": False, "error": str(e)}


# =====================
# Advanced Intelligence Endpoints
# =====================

@router.post("/intelligence/analyze")
async def analyze_with_intelligence(
    symbol: str,
    side: str = "BUY",
    pattern_confidence: float = 70.0,
):
    """
    🧠 Advanced Intelligence Analysis
    
    วิเคราะห์ด้วย:
    - Market Regime (Trend/Range/Volatile)
    - Momentum (RSI/MACD/Stoch)
    - Support/Resistance
    - Kelly Criterion
    - Confluence Score
    """
    try:
        from trading.advanced_intelligence import get_intelligence
        import numpy as np
        
        intel = get_intelligence()
        
        # Get data from MT5
        if trading_engine and trading_engine.broker:
            # This is simplified - in real use, get actual OHLC data
            import MetaTrader5 as mt5
            
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
            if rates is not None and len(rates) > 50:
                h1_data = {
                    "open": np.array([r['open'] for r in rates], dtype=np.float32),
                    "high": np.array([r['high'] for r in rates], dtype=np.float32),
                    "low": np.array([r['low'] for r in rates], dtype=np.float32),
                    "close": np.array([r['close'] for r in rates], dtype=np.float32),
                }
                
                result = intel.analyze(
                    signal_side=side,
                    pattern_confidence=pattern_confidence,
                    h1_data=h1_data,
                )
                
                return {
                    "success": True,
                    "analysis": result.to_dict(),
                    "timestamp": datetime.now().isoformat(),
                }
        
        return {
            "success": False,
            "error": "Cannot get market data",
        }
        
    except Exception as e:
        logger.error(f"Error in intelligence analysis: {e}")
        return {"success": False, "error": str(e)}


@router.get("/intelligence/regime/{symbol}")
async def get_market_regime(symbol: str):
    """
    🌡️ Get current market regime for a symbol
    """
    try:
        from trading.advanced_intelligence import MarketRegimeDetector
        import numpy as np
        import MetaTrader5 as mt5
        
        detector = MarketRegimeDetector()
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        if rates is None or len(rates) < 100:
            return {"success": False, "error": "Cannot get data"}
        
        highs = np.array([r['high'] for r in rates], dtype=np.float32)
        lows = np.array([r['low'] for r in rates], dtype=np.float32)
        closes = np.array([r['close'] for r in rates], dtype=np.float32)
        
        regime = detector.detect(highs, lows, closes)
        
        return {
            "success": True,
            "symbol": symbol,
            "regime": regime.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting market regime: {e}")
        return {"success": False, "error": str(e)}


@router.get("/intelligence/momentum/{symbol}")
async def get_momentum_scan(symbol: str):
    """
    📈 Get momentum analysis for a symbol
    """
    try:
        from trading.advanced_intelligence import MomentumScanner
        import numpy as np
        import MetaTrader5 as mt5
        
        scanner = MomentumScanner()
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        if rates is None or len(rates) < 50:
            return {"success": False, "error": "Cannot get data"}
        
        highs = np.array([r['high'] for r in rates], dtype=np.float32)
        lows = np.array([r['low'] for r in rates], dtype=np.float32)
        closes = np.array([r['close'] for r in rates], dtype=np.float32)
        
        momentum = scanner.scan(closes, highs, lows)
        
        return {
            "success": True,
            "symbol": symbol,
            "momentum": momentum.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting momentum: {e}")
        return {"success": False, "error": str(e)}


@router.get("/intelligence/sr-levels/{symbol}")
async def get_sr_levels(symbol: str):
    """
    📊 Get Support/Resistance levels for a symbol
    """
    try:
        from trading.advanced_intelligence import SupportResistanceFinder
        import numpy as np
        import MetaTrader5 as mt5
        
        finder = SupportResistanceFinder()
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        if rates is None or len(rates) < 100:
            return {"success": False, "error": "Cannot get data"}
        
        highs = np.array([r['high'] for r in rates], dtype=np.float32)
        lows = np.array([r['low'] for r in rates], dtype=np.float32)
        closes = np.array([r['close'] for r in rates], dtype=np.float32)
        current_price = closes[-1]
        
        levels = finder.find_levels(highs, lows, closes)
        support, resistance = finder.get_nearest_sr(current_price, levels)
        
        return {
            "success": True,
            "symbol": symbol,
            "current_price": float(current_price),
            "nearest_support": support.to_dict() if support else None,
            "nearest_resistance": resistance.to_dict() if resistance else None,
            "all_levels": [l.to_dict() for l in levels[:10]],
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting S/R levels: {e}")
        return {"success": False, "error": str(e)}


# =====================
# 📚 Learning System Endpoints
# =====================

@router.get("/learning/stats")
async def get_learning_statistics():
    """
    📊 Get continuous learning statistics
    Shows what the bot has learned from trading
    """
    try:
        from trading.continuous_learning import get_learning_system
        
        learning = get_learning_system()
        if not learning:
            return {"success": False, "error": "Learning system not initialized"}
        
        stats = learning.get_learning_summary()
        
        return {
            "success": True,
            "learning": stats,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return {"success": False, "error": str(e)}


@router.get("/learning/factor-weights")
async def get_factor_weights():
    """
    📊 Get learned factor weights
    Shows which factors are most predictive
    """
    try:
        from trading.continuous_learning import get_learning_system
        
        learning = get_learning_system()
        if not learning:
            return {"success": False, "error": "Learning system not initialized"}
        
        weights = learning.online_learner.factor_weights
        
        # Sort by importance
        sorted_weights = dict(sorted(
            weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return {
            "success": True,
            "factor_weights": sorted_weights,
            "total_trades_learned": learning.online_learner.sample_count,
            "win_rate": learning.online_learner.ema_win_rate,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting factor weights: {e}")
        return {"success": False, "error": str(e)}


@router.get("/learning/market-cycle")
async def get_market_cycle():
    """
    🌊 Get current market cycle detection
    """
    try:
        from trading.continuous_learning import get_learning_system
        
        learning = get_learning_system()
        if not learning:
            return {"success": False, "error": "Learning system not initialized"}
        
        cycle_info = learning.cycle_detector.detect()
        
        return {
            "success": True,
            "cycle": cycle_info.to_dict(),
            "description": {
                "ACCUMULATION": "Smart money buying - prepare for uptrend",
                "MARKUP": "Uptrend in progress - good for longs",
                "DISTRIBUTION": "Smart money selling - prepare for downtrend",
                "MARKDOWN": "Downtrend in progress - good for shorts",
                "UNKNOWN": "No clear cycle detected",
            }.get(cycle_info.cycle.value, "Unknown"),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting market cycle: {e}")
        return {"success": False, "error": str(e)}


@router.get("/learning/optimized-params")
async def get_optimized_params():
    """
    ⚙️ Get auto-optimized strategy parameters
    """
    try:
        from trading.continuous_learning import get_learning_system
        
        learning = get_learning_system()
        if not learning:
            return {"success": False, "error": "Learning system not initialized"}
        
        optimizer = learning.strategy_optimizer
        
        return {
            "success": True,
            "params": optimizer.best_params.to_dict(),
            "optimization_status": optimizer.get_optimization_status(),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting optimized params: {e}")
        return {"success": False, "error": str(e)}


@router.get("/learning/pattern-evolution")
async def get_pattern_evolution():
    """
    📈 Get pattern evolution tracking
    Shows how patterns perform over time
    """
    try:
        from trading.continuous_learning import get_learning_system
        
        learning = get_learning_system()
        if not learning:
            return {"success": False, "error": "Learning system not initialized"}
        
        tracker = learning.pattern_tracker
        
        return {
            "success": True,
            "total_patterns_tracked": len(tracker.pattern_stats),
            "evolution_summary": tracker.get_evolution_summary(),
            "recent_patterns": tracker.get_recent_patterns(limit=20),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error getting pattern evolution: {e}")
        return {"success": False, "error": str(e)}