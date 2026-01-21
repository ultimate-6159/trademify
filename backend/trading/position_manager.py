"""
Position Manager
จัดการ Position ทั้งหมดในระบบ + Sync กับ SharedStateService
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .engine import (
    BaseBroker,
    Position,
    Order,
    TradeResult,
    OrderSide,
    PositionStatus,
)

# Import SharedStateService
try:
    from services.shared_state_service import (
        SharedStateService,
        SharedPosition,
        SharedTrade,
        get_shared_state
    )
    SHARED_STATE_AVAILABLE = True
except ImportError:
    SHARED_STATE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrailingStopMode(str, Enum):
    """โหมด Trailing Stop"""
    NONE = "NONE"
    FIXED = "FIXED"  # จำนวน pips คงที่
    PERCENT = "PERCENT"  # % ของราคา
    ATR = "ATR"  # ตาม ATR


@dataclass
class TrailingStopConfig:
    """Configuration สำหรับ Trailing Stop"""
    mode: TrailingStopMode = TrailingStopMode.NONE
    value: float = 0.0  # pips หรือ %
    activation_profit: float = 0.0  # เริ่มทำงานเมื่อ profit ถึงค่านี้


@dataclass
class PositionConfig:
    """Configuration สำหรับ Position"""
    max_holding_time: Optional[int] = None  # ชั่วโมง
    trailing_stop: TrailingStopConfig = field(default_factory=TrailingStopConfig)
    break_even_at: Optional[float] = None  # ย้าย SL ไป break-even เมื่อ profit ถึง %


class PositionManager:
    """
    Position Manager
    จัดการ lifecycle ของ Position ทั้งหมด
    รองรับ Shared State สำหรับ multi-VPS sync
    """
    
    def __init__(
        self,
        broker: BaseBroker,
        config: PositionConfig = None,
        shared_state: SharedStateService = None,
        enable_sync: bool = True
    ):
        self.broker = broker
        self.config = config or PositionConfig()
        
        # Shared State for multi-VPS sync
        self.enable_sync = enable_sync and SHARED_STATE_AVAILABLE
        if self.enable_sync:
            self.shared_state = shared_state or get_shared_state()
        else:
            self.shared_state = None
        
        # Position tracking (local cache)
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Statistics
        self.total_pnl: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        
        # Callbacks
        self.on_position_opened: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        self.on_position_updated: Optional[Callable] = None
        self.on_trailing_stop_moved: Optional[Callable] = None
        
        self._running = False
    
    def _position_to_shared(self, position: Position) -> SharedPosition:
        """แปลง Position เป็น SharedPosition"""
        return SharedPosition(
            id=position.id,
            symbol=position.symbol,
            side=position.side.value,
            quantity=position.quantity,
            entry_price=position.entry_price,
            current_price=position.current_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            pnl=position.pnl,
            pnl_percent=position.pnl_percent,
            status=position.status.value,
            node_id=self.shared_state.node_id if self.shared_state else "local",
            created_at=position.opened_at.isoformat() if position.opened_at else datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def _shared_to_position(self, shared: SharedPosition) -> Position:
        """แปลง SharedPosition เป็น Position"""
        return Position(
            id=shared.id,
            symbol=shared.symbol,
            side=OrderSide(shared.side),
            quantity=shared.quantity,
            entry_price=shared.entry_price,
            current_price=shared.current_price,
            stop_loss=shared.stop_loss,
            take_profit=shared.take_profit,
            pnl=shared.pnl,
            pnl_percent=shared.pnl_percent,
            status=PositionStatus(shared.status),
            opened_at=datetime.fromisoformat(shared.created_at) if shared.created_at else datetime.now()
        )
    
    async def start(self, symbols: List[str] = None) -> None:
        """เริ่ม Position Manager"""
        self._running = True
        logger.info("Position Manager started")
        
        # Register node ถ้าใช้ shared state
        if self.shared_state and symbols:
            self.shared_state.register_node(symbols)
            self.shared_state.start_listening()
            
            # Setup callbacks for remote changes
            self.shared_state.on('position_updated', self._on_remote_position_updated)
            self.shared_state.on('position_removed', self._on_remote_position_removed)
        
        # Sync existing positions from broker AND shared state
        await self._sync_positions()
        
        # Start monitoring
        asyncio.create_task(self._monitor_loop())
        
        # Start heartbeat loop if using shared state
        if self.shared_state:
            asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self) -> None:
        """หยุด Position Manager"""
        self._running = False
        
        if self.shared_state:
            self.shared_state.stop_listening()
            self.shared_state.unregister_node()
        
        logger.info("Position Manager stopped")
    
    async def _heartbeat_loop(self) -> None:
        """ส่ง heartbeat ทุก 30 วินาที"""
        while self._running:
            if self.shared_state:
                self.shared_state.heartbeat()
            await asyncio.sleep(30)
    
    def _on_remote_position_updated(self, shared_pos: SharedPosition) -> None:
        """Callback เมื่อ position ถูกอัปเดตจาก node อื่น"""
        # อัปเดต local cache ถ้าเป็น position จาก node อื่น
        if self.shared_state and shared_pos.node_id != self.shared_state.node_id:
            self.positions[shared_pos.id] = self._shared_to_position(shared_pos)
            logger.info(f"Remote position updated: {shared_pos.symbol} from node {shared_pos.node_id}")
    
    def _on_remote_position_removed(self, position_path: str) -> None:
        """Callback เมื่อ position ถูกลบจาก node อื่น"""
        position_id = position_path.strip('/')
        if position_id in self.positions:
            del self.positions[position_id]
            logger.info(f"Remote position removed: {position_id}")
    
    async def _sync_positions(self) -> None:
        """Sync positions จาก broker และ shared state"""
        # Sync จาก broker
        broker_positions = await self.broker.get_positions()
        for pos in broker_positions:
            self.positions[pos.id] = pos
        
        # Sync จาก shared state (positions จาก nodes อื่น)
        if self.shared_state:
            shared_positions = self.shared_state.get_all_positions(status="OPEN")
            for shared_pos in shared_positions:
                if shared_pos.id not in self.positions:
                    self.positions[shared_pos.id] = self._shared_to_position(shared_pos)
        
        logger.info(f"Synced {len(self.positions)} positions (broker + shared state)")
    
    async def _monitor_loop(self) -> None:
        """Loop สำหรับ monitor positions"""
        while self._running:
            try:
                for position_id, position in list(self.positions.items()):
                    # Update current price
                    current_price = await self.broker.get_current_price(position.symbol)
                    position.update_pnl(current_price)
                    
                    # Check trailing stop
                    await self._check_trailing_stop(position, current_price)
                    
                    # Check break-even
                    await self._check_break_even(position, current_price)
                    
                    # Check max holding time
                    await self._check_holding_time(position)
                    
                    # Sync to shared state
                    if self.shared_state:
                        self.shared_state.update_position_price(
                            position.id,
                            current_price,
                            position.pnl,
                            position.pnl_percent
                        )
                    
                    # Notify update
                    if self.on_position_updated:
                        self.on_position_updated(position)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_trailing_stop(self, position: Position, current_price: float) -> None:
        """ตรวจสอบและอัพเดท Trailing Stop"""
        ts_config = self.config.trailing_stop
        
        if ts_config.mode == TrailingStopMode.NONE:
            return
        
        # Check activation threshold
        if position.pnl_percent < ts_config.activation_profit:
            return
        
        new_sl = None
        
        if ts_config.mode == TrailingStopMode.FIXED:
            # Fixed pips
            if position.side == OrderSide.BUY:
                new_sl = current_price - ts_config.value
                if position.stop_loss and new_sl <= position.stop_loss:
                    return
            else:
                new_sl = current_price + ts_config.value
                if position.stop_loss and new_sl >= position.stop_loss:
                    return
                    
        elif ts_config.mode == TrailingStopMode.PERCENT:
            # Percentage
            distance = current_price * (ts_config.value / 100)
            if position.side == OrderSide.BUY:
                new_sl = current_price - distance
                if position.stop_loss and new_sl <= position.stop_loss:
                    return
            else:
                new_sl = current_price + distance
                if position.stop_loss and new_sl >= position.stop_loss:
                    return
        
        if new_sl:
            await self.broker.modify_position(position.id, stop_loss=new_sl)
            position.stop_loss = new_sl
            
            logger.info(f"Trailing stop moved to {new_sl} for {position.symbol}")
            
            if self.on_trailing_stop_moved:
                self.on_trailing_stop_moved(position, new_sl)
    
    async def _check_break_even(self, position: Position, current_price: float) -> None:
        """ตรวจสอบและย้าย SL ไป break-even"""
        if not self.config.break_even_at:
            return
        
        if position.pnl_percent < self.config.break_even_at:
            return
        
        # Already at break-even or better
        if position.side == OrderSide.BUY:
            if position.stop_loss and position.stop_loss >= position.entry_price:
                return
            new_sl = position.entry_price
        else:
            if position.stop_loss and position.stop_loss <= position.entry_price:
                return
            new_sl = position.entry_price
        
        await self.broker.modify_position(position.id, stop_loss=new_sl)
        position.stop_loss = new_sl
        
        logger.info(f"Moved to break-even for {position.symbol}")
    
    async def _check_holding_time(self, position: Position) -> None:
        """ตรวจสอบเวลาถือ Position"""
        if not self.config.max_holding_time:
            return
        
        max_time = timedelta(hours=self.config.max_holding_time)
        
        if datetime.now() - position.opened_at > max_time:
            logger.info(f"Max holding time reached for {position.symbol}, closing...")
            await self.close_position(position.id, "Max holding time reached")
    
    async def open_position(
        self,
        order: Order,
        entry_price: float
    ) -> Optional[Position]:
        """เปิด Position ใหม่"""
        position = Position(
            id=f"POS-{order.id}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
        )
        
        self.positions[position.id] = position
        self.total_trades += 1
        
        # Sync to shared state
        if self.shared_state:
            shared_pos = self._position_to_shared(position)
            self.shared_state.save_position(shared_pos)
        
        if self.on_position_opened:
            self.on_position_opened(position)
        
        logger.info(f"Position opened: {position.symbol} {position.side.value} @ {entry_price}")
        
        return position
    
    async def close_position(
        self,
        position_id: str,
        reason: str = "Manual close"
    ) -> Optional[TradeResult]:
        """ปิด Position"""
        if position_id not in self.positions:
            return None
        
        result = await self.broker.close_position(position_id)
        
        if result.success:
            position = self.positions.pop(position_id)
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.now()
            
            self.closed_positions.append(position)
            self.total_pnl += position.pnl
            
            if position.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Sync to shared state
            if self.shared_state:
                # Save trade history
                trade = SharedTrade(
                    id=f"TRADE-{position_id}",
                    position_id=position_id,
                    symbol=position.symbol,
                    side=position.side.value,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    exit_price=position.current_price,
                    pnl=position.pnl,
                    pnl_percent=position.pnl_percent,
                    node_id=self.shared_state.node_id,
                    opened_at=position.opened_at.isoformat() if position.opened_at else "",
                    closed_at=datetime.now().isoformat(),
                    close_reason=reason
                )
                self.shared_state.save_trade(trade)
                
                # Remove position from shared state
                self.shared_state.delete_position(position_id)
            
            if self.on_position_closed:
                self.on_position_closed(position, reason)
            
            logger.info(f"Position closed: {position.symbol} PnL: {position.pnl:.2f} ({reason})")
        
        return result
    
    async def close_all_positions(self, reason: str = "Close all") -> List[TradeResult]:
        """ปิดทุก Position"""
        results = []
        
        for position_id in list(self.positions.keys()):
            result = await self.close_position(position_id, reason)
            if result:
                results.append(result)
        
        return results
    
    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[TradeResult]:
        """แก้ไข Position"""
        if position_id not in self.positions:
            return None
        
        result = await self.broker.modify_position(position_id, stop_loss, take_profit)
        
        if result.success:
            position = self.positions[position_id]
            if stop_loss:
                position.stop_loss = stop_loss
            if take_profit:
                position.take_profit = take_profit
        
        return result
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """ดึง Position"""
        return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """ดึง Position ตาม symbol"""
        return [p for p in self.positions.values() if p.symbol == symbol]
    
    def get_open_positions(self) -> List[Position]:
        """ดึง Position ที่เปิดอยู่ทั้งหมด"""
        return list(self.positions.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Include cluster info if using shared state
        cluster_info = {}
        if self.shared_state:
            cluster_info = self.shared_state.get_cluster_summary()
        
        return {
            "open_positions": len(self.positions),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(self.total_pnl, 2),
            "current_exposure": sum(p.quantity * p.current_price for p in self.positions.values()),
            "cluster": cluster_info,
            "node_id": self.shared_state.node_id if self.shared_state else "local",
            "sync_enabled": self.enable_sync,
        }
    
    def get_all_cluster_positions(self) -> List[Position]:
        """ดึง positions ทั้งหมดจาก cluster (ทุก nodes)"""
        if self.shared_state:
            shared_positions = self.shared_state.get_all_positions(status="OPEN")
            return [self._shared_to_position(sp) for sp in shared_positions]
        return list(self.positions.values())
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """ดึงประวัติเทรดจาก shared state"""
        if self.shared_state:
            trades = self.shared_state.get_trade_history(limit=limit)
            return [t.to_dict() for t in trades]
        return [p.to_dict() for p in self.closed_positions[-limit:]]
    
    def export_history(self, filepath: str) -> None:
        """Export ประวัติการเทรด"""
        history = [p.to_dict() for p in self.closed_positions]
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Exported {len(history)} trades to {filepath}")
