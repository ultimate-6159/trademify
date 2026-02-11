"""
Trading Engine - Core Auto Trading System
ระบบเทรดอัตโนมัติหลัก

รับสัญญาณจาก Voting System → ตัดสินใจ → ส่งคำสั่งไปยัง Broker
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Import from analysis module - use try/except for flexibility
try:
    from analysis import Signal, VoteResult
except ImportError:
    from analysis.voting_system import Signal, VoteResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """ประเภทคำสั่ง"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """ฝั่งของคำสั่ง"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """สถานะคำสั่ง"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionStatus(str, Enum):
    """สถานะ Position"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


@dataclass
class Order:
    """คำสั่งเทรด"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    commission: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "commission": self.commission,
        }


@dataclass
class Position:
    """Position ที่เปิดอยู่"""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    def update_pnl(self, current_price: float) -> None:
        """อัพเดท P&L"""
        self.current_price = current_price
        
        if self.side == OrderSide.BUY:
            self.pnl = (current_price - self.entry_price) * self.quantity
            self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
            self.pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 2),
            "pnl_percent": round(self.pnl_percent, 2),
        }


@dataclass
class TradeResult:
    """ผลลัพธ์การเทรด"""
    success: bool
    order: Optional[Order] = None
    position: Optional[Position] = None
    message: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "order": self.order.to_dict() if self.order else None,
            "position": self.position.to_dict() if self.position else None,
            "message": self.message,
            "error": self.error,
        }


class BaseBroker(ABC):
    """
    Abstract Base Class สำหรับ Broker Connector
    ทุก Broker ต้อง implement methods เหล่านี้
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """เชื่อมต่อกับ Broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """ตัดการเชื่อมต่อ"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลบัญชี"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> float:
        """ดึงยอดเงินคงเหลือ"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> TradeResult:
        """ส่งคำสั่งเทรด"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """ยกเลิกคำสั่ง"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """ดึง Position ที่เปิดอยู่"""
        pass
    
    @abstractmethod
    async def close_position(self, position_id: str) -> TradeResult:
        """ปิด Position"""
        pass
    
    @abstractmethod
    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeResult:
        """แก้ไข SL/TP ของ Position"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """ดึงราคาปัจจุบัน"""
        pass


class TradingEngine:
    """
    Trading Engine หลัก
    ควบคุมการทำงานทั้งหมดของระบบเทรดอัตโนมัติ
    
    🔧 FIX: Added Trailing Stop and Break-Even protection
    """
    
    def __init__(
        self,
        broker: BaseBroker,
        risk_manager: 'RiskManager',
        max_positions: int = 5,
        enabled: bool = False,
        # 🆕 Trailing Stop Settings
        trailing_stop_enabled: bool = True,
        trailing_stop_trigger_pct: float = 0.5,  # Activate trailing after 0.5% profit
        trailing_stop_distance_pct: float = 0.3,  # Trail 0.3% behind price
        # 🆕 Break-Even Settings  
        break_even_enabled: bool = True,
        break_even_trigger_pct: float = 0.3,  # Move SL to BE after 0.3% profit
        break_even_offset: float = 0.0,  # Offset from entry (0 = exact entry)
    ):
        self.broker = broker
        self.risk_manager = risk_manager
        self.max_positions = max_positions
        self.enabled = enabled
        
        # 🆕 Trailing Stop Config
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_trigger_pct = trailing_stop_trigger_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        
        # 🆕 Break-Even Config
        self.break_even_enabled = break_even_enabled
        self.break_even_trigger_pct = break_even_trigger_pct
        self.break_even_offset = break_even_offset
        
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[TradeResult] = []
        
        # 🆕 Track highest profit per position (for trailing stop)
        self._position_max_profit: Dict[str, float] = {}  # position_id -> max profit %
        self._position_be_applied: Dict[str, bool] = {}  # position_id -> BE applied?
        
        # Callbacks
        self.on_signal_received: Optional[Callable] = None
        self.on_trade_executed: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        
        self._running = False
        
        logger.info(f"🛡️ TradingEngine Protection Settings:")
        logger.info(f"   Trailing Stop: {'ON' if trailing_stop_enabled else 'OFF'} (trigger: {trailing_stop_trigger_pct}%, trail: {trailing_stop_distance_pct}%)")
        logger.info(f"   Break-Even: {'ON' if break_even_enabled else 'OFF'} (trigger: {break_even_trigger_pct}%)")
    
    async def start(self) -> bool:
        """เริ่มระบบเทรดอัตโนมัติ"""
        if self._running:
            logger.warning("Trading engine already running")
            return False
        
        # เชื่อมต่อ Broker
        connected = await self.broker.connect()
        if not connected:
            logger.error("Failed to connect to broker")
            return False
        
        self._running = True
        self.enabled = True
        logger.info("Trading engine started")
        
        # เริ่ม monitoring loop
        asyncio.create_task(self._monitor_positions())
        
        return True
    
    async def stop(self) -> None:
        """หยุดระบบเทรดอัตโนมัติ"""
        self._running = False
        self.enabled = False
        await self.broker.disconnect()
        logger.info("Trading engine stopped")
    
    async def process_signal(self, vote_result: VoteResult, symbol: str) -> Optional[TradeResult]:
        """
        ประมวลผลสัญญาณจาก Voting System
        
        Args:
            vote_result: ผลการ Vote จาก Voting System
            symbol: สัญลักษณ์ที่จะเทรด
            
        Returns:
            TradeResult ถ้าทำการเทรด, None ถ้าไม่เทรด
        """
        if not self.enabled:
            logger.info("Trading disabled, skipping signal")
            return None
        
        if self.on_signal_received:
            self.on_signal_received(vote_result)
        
        # ตรวจสอบว่าควรเทรดหรือไม่
        if vote_result.signal in [Signal.STRONG_BUY, Signal.BUY]:
            side = OrderSide.BUY
        elif vote_result.signal in [Signal.STRONG_SELL, Signal.SELL]:
            side = OrderSide.SELL
        else:
            logger.info(f"Signal {vote_result.signal} - no action")
            return None
        
        # ตรวจสอบ Risk
        can_trade, risk_msg = self.risk_manager.can_trade(
            balance=await self.broker.get_balance(),
            open_positions=len(self.positions),
            confidence=vote_result.confidence
        )
        
        if not can_trade:
            logger.warning(f"Risk check failed: {risk_msg}")
            return None
        
        # สร้างและส่ง Order
        order = Order(
            id=f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=self.risk_manager.calculate_position_size(
                balance=await self.broker.get_balance(),
                risk_percent=2.0  # Default 2% risk
            ),
        )
        
        result = await self.execute_order(order)
        
        return result
    
    async def execute_order(self, order: Order) -> Optional[TradeResult]:
        """
        Execute an order directly
        
        Args:
            order: Order to execute
        
        Returns:
            TradeResult if successful
        """
        if not self.enabled:
            logger.info("Trading disabled, skipping order")
            return None
        
        # ส่งคำสั่งไปยัง Broker
        result = await self.broker.place_order(order)
        
        if result.success and result.position:
            self.positions[result.position.id] = result.position
            self.trade_history.append(result)
            
            if self.on_trade_executed:
                self.on_trade_executed(result)
            
            logger.info(f"Executed {order.side.value} order for {order.symbol}: {order.quantity}")
        
        return result
        
        return result
    
    async def close_all_positions(self) -> List[TradeResult]:
        """ปิดทุก Position"""
        results = []
        
        for position_id in list(self.positions.keys()):
            result = await self.broker.close_position(position_id)
            results.append(result)
            
            if result.success:
                del self.positions[position_id]
                
                if self.on_position_closed:
                    self.on_position_closed(result)
        
        return results
    
    async def _monitor_positions(self) -> None:
        """Monitor และอัพเดท Position"""
        # 🔥 Use instance variable to track closing positions across iterations
        if not hasattr(self, '_closing_positions'):
            self._closing_positions = set()
        if not hasattr(self, '_recently_logged'):
            self._recently_logged = {}  # {position_id: timestamp}
        
        while self._running:
            try:
                # Clean up old logged entries (older than 60 seconds)
                current_time = datetime.now().timestamp()
                self._recently_logged = {
                    pid: ts for pid, ts in self._recently_logged.items() 
                    if current_time - ts < 60
                }
                
                for position_id, position in list(self.positions.items()):
                    # Skip if already trying to close
                    if position_id in self._closing_positions:
                        continue
                    
                    current_price = await self.broker.get_current_price(position.symbol)
                    position.update_pnl(current_price)
                    
                    # 🆕 Apply Break-Even Protection FIRST
                    await self._apply_break_even(position_id, position, current_price)
                    
                    # 🆕 Apply Trailing Stop Protection
                    await self._apply_trailing_stop(position_id, position, current_price)
                    
                    # ตรวจสอบ SL/TP
                    should_close, reason = self._check_exit_conditions(position, current_price)
                    
                    if should_close:
                        self._closing_positions.add(position_id)  # Mark as closing
                        
                        # 🔥 Only log if not recently logged
                        if position_id not in self._recently_logged:
                            logger.info(f"Closing position {position_id}: {reason}")
                            self._recently_logged[position_id] = current_time
                        
                        result = await self.broker.close_position(position_id)
                        
                        if result.success:
                            # 🔥 Verify position is actually closed
                            await asyncio.sleep(0.5)  # Wait for MT5 to process
                            broker_positions = await self.broker.get_positions()
                            still_open = any(
                                str(getattr(p, 'ticket', getattr(p, 'id', ''))) == str(position_id) or
                                str(getattr(p, 'id', '')) == str(position_id)
                                for p in broker_positions
                            )
                            
                            if not still_open:
                                # 🆕 Cleanup tracking dicts
                                self._position_max_profit.pop(position_id, None)
                                self._position_be_applied.pop(position_id, None)
                                
                                del self.positions[position_id]
                                self._closing_positions.discard(position_id)
                                
                                if self.on_position_closed:
                                    self.on_position_closed(result)
                            else:
                                # Position still open - don't remove from closing set
                                logger.warning(f"⚠️ Position {position_id} still open after close attempt")
                        else:
                            # Failed to close - remove from closing set to retry later
                            self._closing_positions.discard(position_id)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(5)
    
    async def _apply_break_even(self, position_id: str, position: Position, current_price: float) -> None:
        """
        🆕 Break-Even Protection: Move SL to entry price when profit reaches threshold
        This locks in NO LOSS even if market reverses
        """
        if not self.break_even_enabled:
            return
        
        # Already applied break-even?
        if self._position_be_applied.get(position_id, False):
            return
        
        # Check if profit reached trigger
        if position.pnl_percent < self.break_even_trigger_pct:
            return
        
        # Calculate break-even SL
        if position.side == OrderSide.BUY:
            new_sl = position.entry_price + self.break_even_offset
            # Only move SL if it's better (higher for BUY)
            if position.stop_loss and new_sl <= position.stop_loss:
                return
        else:  # SELL
            new_sl = position.entry_price - self.break_even_offset
            # Only move SL if it's better (lower for SELL)
            if position.stop_loss and new_sl >= position.stop_loss:
                return
        
        # Apply break-even
        try:
            result = await self.broker.modify_position(position_id, stop_loss=new_sl)
            if result and (result.success if hasattr(result, 'success') else result):
                old_sl = position.stop_loss
                position.stop_loss = new_sl
                self._position_be_applied[position_id] = True
                logger.info(f"🔒 BREAK-EVEN: {position.symbol} SL moved to {new_sl:.5f} (was {old_sl:.5f})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply break-even for {position_id}: {e}")
    
    async def _apply_trailing_stop(self, position_id: str, position: Position, current_price: float) -> None:
        """
        🆕 Trailing Stop: Move SL to lock in profits as price moves in our favor
        This protects profits from sudden reversals
        """
        if not self.trailing_stop_enabled:
            return
        
        # Track max profit
        current_profit_pct = position.pnl_percent
        max_profit = self._position_max_profit.get(position_id, 0.0)
        
        if current_profit_pct > max_profit:
            self._position_max_profit[position_id] = current_profit_pct
            max_profit = current_profit_pct
        
        # Check if trailing stop should be activated
        if max_profit < self.trailing_stop_trigger_pct:
            return
        
        # Calculate trailing stop level
        trail_distance = current_price * (self.trailing_stop_distance_pct / 100)
        
        if position.side == OrderSide.BUY:
            new_sl = current_price - trail_distance
            # Only move SL up (never down for BUY)
            if position.stop_loss and new_sl <= position.stop_loss:
                return
        else:  # SELL
            new_sl = current_price + trail_distance
            # Only move SL down (never up for SELL)
            if position.stop_loss and new_sl >= position.stop_loss:
                return
        
        # Apply new SL
        try:
            result = await self.broker.modify_position(position_id, stop_loss=new_sl)
            if result and (result.success if hasattr(result, 'success') else result):
                old_sl = position.stop_loss
                position.stop_loss = new_sl
                logger.info(f"📈 TRAILING STOP: {position.symbol} SL moved to {new_sl:.5f} (was {old_sl:.5f}, profit: {current_profit_pct:.2f}%)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply trailing stop for {position_id}: {e}")
    
    def _check_exit_conditions(self, position: Position, current_price: float) -> tuple[bool, str]:
        """ตรวจสอบเงื่อนไขการออก"""
        if position.side == OrderSide.BUY:
            # Long position
            if position.stop_loss and current_price <= position.stop_loss:
                return True, "Stop Loss hit"
            if position.take_profit and current_price >= position.take_profit:
                return True, "Take Profit hit"
        else:
            # Short position
            if position.stop_loss and current_price >= position.stop_loss:
                return True, "Stop Loss hit"
            if position.take_profit and current_price <= position.take_profit:
                return True, "Take Profit hit"
        
        return False, ""
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติการเทรด"""
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.position and t.position.pnl > 0)
        losing_trades = sum(1 for t in self.trade_history if t.position and t.position.pnl < 0)
        
        total_pnl = sum(t.position.pnl for t in self.trade_history if t.position)
        
        return {
            "enabled": self.enabled,
            "running": self._running,
            "open_positions": len(self.positions),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "positions": [p.to_dict() for p in self.positions.values()],
        }
    
    async def sync_with_broker(self) -> Dict[str, Any]:
        """
        🔄 Sync internal positions with actual broker positions
        This ensures bot state matches MT5 when positions are closed externally (SL/TP hit)
        
        Returns:
            Dict with sync results: added, removed, unchanged positions
        """
        sync_result = {"added": [], "removed": [], "unchanged": []}
        
        try:
            broker_positions = await self.broker.get_positions()
            broker_position_ids = {
                str(getattr(p, 'ticket', getattr(p, 'id', ''))) 
                for p in broker_positions
            }
            internal_position_ids = set(self.positions.keys())
            
            # 1. Find positions closed externally - remove from internal
            removed_ids = internal_position_ids - broker_position_ids
            for pos_id in removed_ids:
                closed_pos = self.positions.pop(pos_id, None)
                if closed_pos:
                    sync_result["removed"].append({
                        "id": pos_id,
                        "symbol": closed_pos.symbol,
                        "side": closed_pos.side.value,
                        "pnl": closed_pos.pnl,
                        "reason": "Closed by MT5 (SL/TP hit or manual)"
                    })
                    logger.info(f"🔄 SYNC: Removed position {pos_id} ({closed_pos.symbol}) - closed externally")
                    
                    # Trigger callback if exists
                    if self.on_position_closed:
                        result = TradeResult(
                            success=True,
                            position=closed_pos,
                            message="Position closed by MT5 (SL/TP hit)"
                        )
                        self.on_position_closed(result)
            
            # 2. Find new positions opened externally - add to internal
            for broker_pos in broker_positions:
                if broker_pos.id not in internal_position_ids:
                    self.positions[broker_pos.id] = broker_pos
                    sync_result["added"].append({
                        "id": broker_pos.id,
                        "symbol": broker_pos.symbol,
                        "side": broker_pos.side.value,
                        "reason": "Opened externally (manual or other)"
                    })
                    logger.info(f"🔄 SYNC: Added position {broker_pos.id} ({broker_pos.symbol}) - opened externally")
            
            # 3. Update existing positions with current prices
            for broker_pos in broker_positions:
                if broker_pos.id in self.positions:
                    self.positions[broker_pos.id] = broker_pos
                    sync_result["unchanged"].append(broker_pos.id)
        
        except Exception as e:
            logger.error(f"🔄 SYNC ERROR: {e}")
        
        return sync_result


class RiskManager:
    """
    จัดการความเสี่ยงในการเทรด
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 2.0,  # % of balance
        max_daily_loss: float = 5.0,  # % of starting balance
        max_positions: int = 5,
        min_confidence: float = 60.0,
        max_drawdown: float = 10.0  # % of peak
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.max_drawdown = max_drawdown
        
        self.daily_loss = 0.0
        self.starting_balance = 0.0
        self.peak_balance = 0.0
    
    def can_trade(
        self,
        balance: float,
        open_positions: int,
        confidence: float
    ) -> tuple[bool, str]:
        """
        ตรวจสอบว่าสามารถเทรดได้หรือไม่
        
        Returns:
            (can_trade, reason)
        """
        # ตรวจสอบจำนวน Position
        if open_positions >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # ตรวจสอบ Confidence
        if confidence < self.min_confidence:
            return False, f"Confidence too low ({confidence:.1f}% < {self.min_confidence}%)"
        
        # ตรวจสอบ Daily Loss
        if self.starting_balance > 0:
            daily_loss_percent = (self.daily_loss / self.starting_balance) * 100
            if daily_loss_percent >= self.max_daily_loss:
                return False, f"Daily loss limit reached ({daily_loss_percent:.1f}%)"
        
        # ตรวจสอบ Drawdown
        if self.peak_balance > 0:
            drawdown = ((self.peak_balance - balance) / self.peak_balance) * 100
            if drawdown >= self.max_drawdown:
                return False, f"Max drawdown reached ({drawdown:.1f}%)"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        balance: float,
        risk_percent: float = None
    ) -> float:
        """
        คำนวณขนาด Position ตาม Risk
        
        Returns:
            Lot size
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
        
        risk_amount = balance * (risk_percent / 100)
        
        # Simple calculation - should be enhanced with proper lot size calculation
        lot_size = risk_amount / 100  # Simplified
        
        return max(0.01, round(lot_size, 2))
    
    def update_daily_stats(self, pnl: float, balance: float) -> None:
        """อัพเดทสถิติรายวัน"""
        if pnl < 0:
            self.daily_loss += abs(pnl)
        
        if balance > self.peak_balance:
            self.peak_balance = balance
    
    def reset_daily(self, balance: float) -> None:
        """Reset สถิติรายวัน"""
        self.daily_loss = 0.0
        self.starting_balance = balance
        self.peak_balance = balance
