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
    """
    
    def __init__(
        self,
        broker: BaseBroker,
        risk_manager: 'RiskManager',
        max_positions: int = 5,
        enabled: bool = False
    ):
        self.broker = broker
        self.risk_manager = risk_manager
        self.max_positions = max_positions
        self.enabled = enabled
        
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[TradeResult] = []
        
        # Callbacks
        self.on_signal_received: Optional[Callable] = None
        self.on_trade_executed: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        
        self._running = False
    
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
            vote_result: ผลโหวตจาก VotingSystem
            symbol: สัญลักษณ์ที่จะเทรด
        
        Returns:
            TradeResult หากมีการเปิด/ปิด Position
        """
        if not self.enabled:
            logger.info("Trading disabled, skipping signal")
            return None
        
        if self.on_signal_received:
            self.on_signal_received(vote_result, symbol)
        
        signal = vote_result.signal
        confidence = vote_result.confidence
        
        # ตรวจสอบว่าควรเทรดหรือไม่
        if signal == Signal.WAIT:
            logger.info(f"WAIT signal for {symbol}, no action taken")
            return None
        
        # ตรวจสอบ Risk Management
        can_trade, reason = await self.risk_manager.can_open_position(
            symbol=symbol,
            confidence=confidence,
            current_positions=len(self.positions)
        )
        
        if not can_trade:
            logger.info(f"Risk manager rejected trade: {reason}")
            return TradeResult(success=False, message=reason)
        
        # คำนวณ Position Size
        balance = await self.broker.get_balance()
        current_price = await self.broker.get_current_price(symbol)
        
        position_size = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=current_price,
            stop_loss=vote_result.stop_loss or current_price * 0.98,
            risk_percent=self.risk_manager.risk_per_trade
        )
        
        # สร้าง Order
        side = OrderSide.BUY if signal in [Signal.STRONG_BUY, Signal.BUY] else OrderSide.SELL
        
        order = Order(
            id=f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=position_size,
            stop_loss=vote_result.stop_loss,
            take_profit=vote_result.take_profit,
        )
        
        # ส่งคำสั่ง
        result = await self.broker.place_order(order)
        
        if result.success and result.position:
            self.positions[result.position.id] = result.position
            self.trade_history.append(result)
            
            if self.on_trade_executed:
                self.on_trade_executed(result)
            
            logger.info(f"Opened {side.value} position for {symbol}: {position_size} @ {current_price}")
        
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
        while self._running:
            try:
                for position_id, position in list(self.positions.items()):
                    current_price = await self.broker.get_current_price(position.symbol)
                    position.update_pnl(current_price)
                    
                    # ตรวจสอบ SL/TP
                    should_close, reason = self._check_exit_conditions(position, current_price)
                    
                    if should_close:
                        logger.info(f"Closing position {position_id}: {reason}")
                        result = await self.broker.close_position(position_id)
                        
                        if result.success:
                            del self.positions[position_id]
                            
                            if self.on_position_closed:
                                self.on_position_closed(result)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(5)
    
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


class RiskManager:
    """
    Risk Management System
    ควบคุมความเสี่ยงในการเทรด
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 2.0,  # % ของ balance
        max_daily_loss: float = 5.0,  # % ของ balance
        max_positions: int = 5,
        min_confidence: float = 70.0,
        max_drawdown: float = 10.0,  # % ของ balance
    ):
        self.risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.max_drawdown = max_drawdown
        
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
    
    async def can_open_position(
        self,
        symbol: str,
        confidence: float,
        current_positions: int
    ) -> tuple[bool, str]:
        """
        ตรวจสอบว่าสามารถเปิด Position ใหม่ได้หรือไม่
        """
        # Check confidence
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence}% below minimum {self.min_confidence}%"
        
        # Check max positions
        if current_positions >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit ({self.max_daily_loss}%) reached"
        
        # Check drawdown
        if self.current_drawdown >= self.max_drawdown:
            return False, f"Maximum drawdown ({self.max_drawdown}%) reached"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float
    ) -> float:
        """
        คำนวณ Position Size ตาม Risk %
        
        Formula: Position Size = (Balance * Risk%) / |Entry - SL|
        """
        risk_amount = balance * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        
        return round(position_size, 4)
    
    def update_daily_stats(self, pnl: float, current_balance: float) -> None:
        """อัพเดทสถิติรายวัน"""
        self.daily_pnl += pnl
        
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        if self.peak_balance > 0:
            self.current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
    
    def reset_daily_stats(self) -> None:
        """รีเซ็ตสถิติรายวัน (เรียกทุกวัน)"""
        self.daily_pnl = 0.0
    
    def to_dict(self) -> dict:
        return {
            "risk_per_trade": self.risk_per_trade,
            "max_daily_loss": self.max_daily_loss,
            "max_positions": self.max_positions,
            "min_confidence": self.min_confidence,
            "max_drawdown": self.max_drawdown,
            "daily_pnl": round(self.daily_pnl, 2),
            "current_drawdown": round(self.current_drawdown, 2),
        }
