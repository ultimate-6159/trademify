"""
MetaTrader 5 Connector
เชื่อมต่อกับ MT5 สำหรับ Forex และ CFD Trading

แก้ไขปัญหา 3 จุดตาย:
1. Filling Mode - รองรับ IOC, FOK, RETURN
2. Price Precision - Round ราคาตาม symbol digits
3. Volume Validation - เช็ค lot size ตาม min/max/step

🔧 Auto-Reconnect Feature:
- Heartbeat check ทุก 30 วินาที
- Auto reconnect เมื่อหลุด
- Max retry 5 ครั้ง
"""
import asyncio
import logging
import math
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .engine import (
    BaseBroker,
    Order,
    Position,
    TradeResult,
    OrderType,
    OrderSide,
    OrderStatus,
    PositionStatus,
)

logger = logging.getLogger(__name__)


# MT5 Return Codes Reference
MT5_RETCODE_MESSAGES = {
    10004: "Requote - ราคาเปลี่ยน ลองใหม่",
    10006: "Request rejected",
    10007: "Request canceled by trader",
    10008: "Order placed",
    10009: "Request completed - สำเร็จ!",
    10010: "Request partially completed",
    10011: "Request processing error",
    10012: "Request canceled by timeout",
    10013: "Invalid request",
    10014: "Invalid volume",
    10015: "Invalid price - ราคาไม่ถูกต้อง",
    10016: "Invalid stops - SL/TP ไม่ถูกต้อง",
    10017: "Trade disabled",
    10018: "Market closed - ตลาดปิด",
    10019: "Insufficient funds - เงินไม่พอ",
    10020: "Prices changed",
    10021: "No quotes - ไม่มีราคา",
    10022: "Invalid order expiration",
    10023: "Order state changed",
    10024: "Too many requests",
    10025: "No changes in request",
    10026: "Autotrading disabled - เปิด Algo Trading!",
    10027: "Autotrading disabled by client terminal",
    10028: "Request locked",
    10029: "Order or position frozen",
    10030: "Invalid fill type - ต้องเปลี่ยน Filling Mode!",
    10031: "No connection with trade server",
    10032: "Operation allowed only for live accounts",
    10033: "Pending orders limit reached",
    10034: "Volume limit reached",
    10035: "Invalid or prohibited order type",
    10036: "Position with specified ID already closed",
    10038: "Close volume exceeds position volume",
    10039: "Close order already exists",
    10040: "Limit orders limit reached",
    10041: "Pending orders and positions limit reached",
    10042: "Position or order modification timeout",
    10043: "Orders and positions count limit reached",
    10044: "Hedging prohibited",
    10045: "Prohibited by FIFO rule",
}


@dataclass
class MT5Config:
    """MetaTrader 5 Configuration"""
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = ""  # Path to terminal64.exe
    timeout: int = 60000
    # Auto-reconnect settings
    heartbeat_interval: int = 30  # วินาที
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5  # วินาที


@dataclass
class SymbolSpec:
    """Symbol specification for validation"""
    digits: int = 5          # Price precision
    point: float = 0.00001   # Minimum price change
    lot_min: float = 0.01    # Minimum lot size
    lot_max: float = 100.0   # Maximum lot size
    lot_step: float = 0.01   # Lot size step
    volume_limit: float = 0.0
    filling_modes: List[str] = None  # Supported filling modes
    
    def __post_init__(self):
        if self.filling_modes is None:
            self.filling_modes = ["IOC", "FOK"]


class MT5Broker(BaseBroker):
    """
    MetaTrader 5 Connector
    รองรับ Forex, CFD, Commodities, Indices
    
    แก้ไขปัญหา:
    - Filling Mode: ลอง IOC -> FOK -> RETURN
    - Price Precision: Round ตาม digits
    - Volume: Validate ตาม min/max/step
    
    🔧 Auto-Reconnect:
    - Heartbeat check
    - Auto reconnect when disconnected
    """
    
    def __init__(self, config: MT5Config):
        self.config = config
        self._connected = False
        self._mt5 = None
        
        # Cache
        self._account_info: Dict[str, Any] = {}
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._symbol_specs: Dict[str, SymbolSpec] = {}
        
        # Auto-reconnect state
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._reconnect_count = 0
        self._last_heartbeat = time.time()
        self._connection_lock = threading.Lock()
    
    async def connect(self) -> bool:
        """เชื่อมต่อกับ MT5 Terminal - ใช้ connection ที่มีอยู่แล้ว"""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            
            # ตรวจสอบว่า MT5 เชื่อมต่ออยู่แล้วหรือไม่
            terminal_info = mt5.terminal_info()
            if terminal_info:
                # MT5 เชื่อมต่ออยู่แล้ว - ใช้ต่อได้เลย
                self._connected = True
                account = mt5.account_info()
                if account:
                    logger.info(f"✅ MT5 Broker using existing connection: {account.login}@{account.server}")
                return True
            
            # Initialize MT5 ถ้ายังไม่ได้เชื่อมต่อ
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialize failed: {error}")
                return False
            
            # Login - only if credentials provided
            if self.config.login > 0 and self.config.password:
                authorized = mt5.login(
                    login=self.config.login,
                    password=self.config.password,
                    server=self.config.server,
                    timeout=self.config.timeout
                )
                
                if not authorized:
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    return False
            
            self._connected = True
            
            # Log account info
            account = mt5.account_info()
            if account:
                logger.info(f"✅ Connected to MT5: {account.server}")
                logger.info(f"   Account: {account.login} ({account.name})")
                logger.info(f"   Balance: {account.balance} {account.currency}")
                logger.info(f"   Trade Mode: {'Hedge' if account.margin_mode == 0 else 'Netting'}")
            
            # Start heartbeat thread
            self._start_heartbeat()
            
            return True
            
        except ImportError:
            logger.error("MetaTrader5 package not installed!")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            return False
    
    def _start_heartbeat(self):
        """เริ่ม heartbeat thread สำหรับตรวจสอบ connection"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return  # Already running
        
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info(f"🔄 MT5 Heartbeat started (interval: {self.config.heartbeat_interval}s)")
    
    def _stop_heartbeat_thread(self):
        """หยุด heartbeat thread"""
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None
        logger.info("🛑 MT5 Heartbeat stopped")
    
    def _heartbeat_loop(self):
        """Loop ตรวจสอบ connection และ auto-reconnect"""
        while not self._stop_heartbeat.is_set():
            try:
                # Wait for interval
                self._stop_heartbeat.wait(self.config.heartbeat_interval)
                if self._stop_heartbeat.is_set():
                    break
                
                # Check connection
                if not self._check_connection():
                    logger.warning("💔 MT5 connection lost! Attempting reconnect...")
                    self._attempt_reconnect()
                else:
                    self._last_heartbeat = time.time()
                    self._reconnect_count = 0  # Reset on success
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def _check_connection(self) -> bool:
        """ตรวจสอบว่า MT5 ยังเชื่อมต่ออยู่หรือไม่"""
        with self._connection_lock:
            try:
                if not self._mt5:
                    return False
                
                # Try to get terminal info
                terminal_info = self._mt5.terminal_info()
                if not terminal_info:
                    return False
                
                # Try to get account info
                account = self._mt5.account_info()
                if not account:
                    return False
                
                # Connection OK
                return True
                
            except Exception as e:
                logger.error(f"Connection check failed: {e}")
                return False
    
    def _attempt_reconnect(self):
        """พยายาม reconnect กับ MT5"""
        with self._connection_lock:
            if self._reconnect_count >= self.config.max_reconnect_attempts:
                logger.error(f"❌ Max reconnect attempts ({self.config.max_reconnect_attempts}) reached!")
                self._connected = False
                return
            
            self._reconnect_count += 1
            logger.info(f"🔄 Reconnect attempt {self._reconnect_count}/{self.config.max_reconnect_attempts}...")
            
            try:
                # Try shutdown first
                try:
                    self._mt5.shutdown()
                except:
                    pass
                
                # Wait before reconnect
                time.sleep(self.config.reconnect_delay)
                
                # Reinitialize
                if not self._mt5.initialize():
                    logger.error(f"Reconnect failed: MT5 initialize error")
                    return
                
                # Check if connected
                terminal_info = self._mt5.terminal_info()
                if terminal_info:
                    account = self._mt5.account_info()
                    if account:
                        self._connected = True
                        logger.info(f"✅ MT5 Reconnected successfully! Account: {account.login}")
                        self._reconnect_count = 0
                        return
                
                logger.warning(f"Reconnect attempt {self._reconnect_count} failed")
                
            except Exception as e:
                logger.error(f"Reconnect error: {e}")
    
    def ensure_connected(self) -> bool:
        """ตรวจสอบและ reconnect ถ้าจำเป็น - เรียกก่อนทำ operation"""
        if self._connected and self._check_connection():
            return True
        
        logger.warning("MT5 not connected, attempting reconnect...")
        self._attempt_reconnect()
        return self._connected
    
    async def disconnect(self) -> None:
        """ตัดการเชื่อมต่อ - หยุด heartbeat ด้วย"""
        self._stop_heartbeat_thread()
        self._connected = False
        logger.info("MT5 Broker disconnected (MT5 terminal still running)")
    
    def _get_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        """
        ดึง Symbol Specification (สำคัญมาก!)
        ใช้สำหรับ validate price/volume
        """
        if symbol in self._symbol_specs:
            return self._symbol_specs[symbol]
        
        if not self._mt5:
            return SymbolSpec()
        
        info = self._mt5.symbol_info(symbol)
        if not info:
            logger.error(f"Symbol {symbol} not found!")
            return None
        
        # Check if symbol is visible/enabled
        if not info.visible:
            logger.info(f"Enabling symbol {symbol}...")
            if not self._mt5.symbol_select(symbol, True):
                logger.error(f"Cannot enable symbol {symbol}")
                return None
        
        # Determine supported filling modes
        filling_modes = []
        # Check SYMBOL_FILLING_MODE flags
        if info.filling_mode & 1:  # SYMBOL_FILLING_FOK
            filling_modes.append("FOK")
        if info.filling_mode & 2:  # SYMBOL_FILLING_IOC
            filling_modes.append("IOC")
        if info.filling_mode & 4:  # SYMBOL_FILLING_RETURN (Market orders)
            filling_modes.append("RETURN")
        
        if not filling_modes:
            filling_modes = ["IOC", "FOK"]  # Default fallback
        
        spec = SymbolSpec(
            digits=info.digits,
            point=info.point,
            lot_min=info.volume_min,
            lot_max=info.volume_max,
            lot_step=info.volume_step,
            volume_limit=info.volume_limit,
            filling_modes=filling_modes,
        )
        
        self._symbol_specs[symbol] = spec
        logger.info(f"Symbol {symbol}: digits={spec.digits}, lot={spec.lot_min}-{spec.lot_max}, filling={spec.filling_modes}")
        
        return spec
    
    def _normalize_price(self, price: float, symbol: str) -> float:
        """
        Round ราคาให้ตรงกับ digits ของ symbol
        
        จุดตายที่ 2: Invalid Price
        """
        spec = self._get_symbol_spec(symbol)
        if not spec:
            return round(price, 5)
        
        # Round to symbol's digits
        normalized = round(price, spec.digits)
        logger.debug(f"Price normalized: {price} -> {normalized} (digits={spec.digits})")
        return normalized
    
    def _normalize_volume(self, volume: float, symbol: str) -> Tuple[float, str]:
        """
        Validate และ normalize volume ให้ถูกต้อง
        
        จุดตายที่ 3: Invalid Volume
        
        Returns:
            (normalized_volume, error_message or None)
        """
        spec = self._get_symbol_spec(symbol)
        if not spec:
            return volume, None
        
        # Check minimum
        if volume < spec.lot_min:
            return spec.lot_min, f"Volume {volume} < min {spec.lot_min}, adjusted"
        
        # Check maximum
        if volume > spec.lot_max:
            return spec.lot_max, f"Volume {volume} > max {spec.lot_max}, adjusted"
        
        # Round to step
        if spec.lot_step > 0:
            steps = math.floor(volume / spec.lot_step)
            normalized = steps * spec.lot_step
            normalized = round(normalized, 2)  # Avoid floating point issues
            
            if normalized != volume:
                logger.info(f"Volume normalized: {volume} -> {normalized} (step={spec.lot_step})")
            
            return normalized, None
        
        return volume, None
    
    def _get_filling_mode(self, symbol: str, attempt: int = 0):
        """
        เลือก Filling Mode ที่เหมาะสม
        
        จุดตายที่ 1: Invalid Filling
        ลองตามลำดับ: IOC -> FOK -> RETURN
        """
        spec = self._get_symbol_spec(symbol)
        filling_order = ["IOC", "FOK", "RETURN"]
        
        if spec and spec.filling_modes:
            # Use symbol's supported modes first
            filling_order = spec.filling_modes
        
        if attempt >= len(filling_order):
            return None
        
        mode_name = filling_order[attempt]
        
        if mode_name == "IOC":
            return self._mt5.ORDER_FILLING_IOC
        elif mode_name == "FOK":
            return self._mt5.ORDER_FILLING_FOK
        elif mode_name == "RETURN":
            return self._mt5.ORDER_FILLING_RETURN
        
        return self._mt5.ORDER_FILLING_IOC
    
    async def get_account_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลบัญชี - พร้อม auto-reconnect"""
        # Ensure connection before operation
        if not self.ensure_connected():
            logger.warning("MT5 not connected - cannot get account info")
            return {}
        
        try:
            if self._mt5:
                account = self._mt5.account_info()
                if account:
                    self._account_info = {
                        "login": account.login,
                        "name": account.name,
                        "server": account.server,
                        "currency": account.currency,
                        "balance": account.balance,
                        "equity": account.equity,
                        "margin": account.margin,
                        "margin_free": account.margin_free,
                        "margin_level": account.margin_level,
                        "profit": account.profit,
                    }
                    return self._account_info
                else:
                    # Account info failed - try reconnect
                    logger.warning("Failed to get account info, attempting reconnect...")
                    self._attempt_reconnect()
                    if self._connected:
                        account = self._mt5.account_info()
                        if account:
                            self._account_info = {
                                "login": account.login,
                                "name": account.name,
                                "server": account.server,
                                "currency": account.currency,
                                "balance": account.balance,
                                "equity": account.equity,
                                "margin": account.margin,
                                "margin_free": account.margin_free,
                                "margin_level": account.margin_level,
                                "profit": account.profit,
                            }
                            return self._account_info
            
            # MT5 not available
            logger.warning("MT5 not connected - cannot get account info")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    async def get_balance(self) -> float:
        """ดึงยอดเงินคงเหลือ"""
        account = await self.get_account_info()
        return account.get("balance", 10000.0)
    
    async def get_equity(self) -> float:
        """ดึง equity ปัจจุบัน"""
        account = await self.get_account_info()
        return account.get("equity", account.get("balance", 10000.0))
    
    async def place_order(self, order: Order) -> TradeResult:
        """
        ส่งคำสั่งเทรด (แก้ไขแล้ว!)
        
        Features:
        - Auto-retry with different filling modes
        - Price normalization
        - Volume validation
        - Detailed error logging
        - Auto-reconnect if disconnected
        """
        # Ensure connection before trading
        if not self.ensure_connected():
            return TradeResult(success=False, error="MT5 not connected - reconnect failed")
        
        try:
            if self._mt5:
                # Step 1: Get and validate symbol
                symbol_info = self._mt5.symbol_info(order.symbol)
                if not symbol_info:
                    return TradeResult(success=False, error=f"Symbol {order.symbol} not found")
                
                # Enable symbol if not visible
                if not symbol_info.visible:
                    logger.info(f"Enabling symbol {order.symbol}...")
                    if not self._mt5.symbol_select(order.symbol, True):
                        return TradeResult(success=False, error=f"Cannot enable symbol {order.symbol}")
                
                # Step 2: Get current price
                tick = self._mt5.symbol_info_tick(order.symbol)
                if not tick:
                    return TradeResult(success=False, error=f"Cannot get price for {order.symbol}")
                
                price = tick.ask if order.side == OrderSide.BUY else tick.bid
                
                # Step 3: Normalize price (จุดตาย #2)
                price = self._normalize_price(price, order.symbol)
                
                # Step 4: Normalize volume (จุดตาย #3)
                volume, vol_warning = self._normalize_volume(order.quantity, order.symbol)
                if vol_warning:
                    logger.warning(vol_warning)
                
                # Step 5: Normalize SL/TP
                sl = self._normalize_price(order.stop_loss, order.symbol) if order.stop_loss else 0.0
                tp = self._normalize_price(order.take_profit, order.symbol) if order.take_profit else 0.0
                
                # Step 6: Try sending order with different filling modes
                max_attempts = 3
                last_error = ""
                
                for attempt in range(max_attempts):
                    filling_mode = self._get_filling_mode(order.symbol, attempt)
                    if filling_mode is None:
                        break
                    
                    filling_name = ["FOK", "IOC", "RETURN"][attempt] if attempt < 3 else "UNKNOWN"
                    logger.info(f"Attempt {attempt + 1}: Trying filling mode {filling_name}")
                    
                    request = {
                        "action": self._mt5.TRADE_ACTION_DEAL,
                        "symbol": order.symbol,
                        "volume": volume,
                        "type": self._mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else self._mt5.ORDER_TYPE_SELL,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": int(os.getenv("MT5_DEVIATION", "50")),  # 🚀 CHANGED: 20 → 50 for faster execution
                        "magic": 234000,
                        "comment": "Trademify Auto",
                        "type_time": self._mt5.ORDER_TIME_GTC,
                        "type_filling": filling_mode,  # จุดตาย #1
                    }
                    
                    # Log request for debugging
                    logger.info(f"📤 Order Request: {order.side.value} {volume} {order.symbol} @ {price}")
                    logger.debug(f"   Full request: {request}")
                    
                    # Send order
                    result = self._mt5.order_send(request)
                    
                    # Analyze result
                    if result is None:
                        error = self._mt5.last_error()
                        logger.error(f"order_send() returned None: {error}")
                        last_error = f"MT5 Error: {error}"
                        continue
                    
                    retcode = result.retcode
                    retcode_msg = MT5_RETCODE_MESSAGES.get(retcode, f"Unknown code {retcode}")
                    
                    logger.info(f"📥 Result: retcode={retcode} ({retcode_msg})")
                    logger.info(f"   Comment: {result.comment}")
                    
                    # Success!
                    if retcode == self._mt5.TRADE_RETCODE_DONE:
                        logger.info(f"✅ ORDER SUCCESS! Ticket: {result.order}")
                        
                        order.status = OrderStatus.FILLED
                        order.filled_at = datetime.now()
                        order.filled_price = result.price if result.price > 0 else price
                        
                        position = Position(
                            id=str(result.order),
                            symbol=order.symbol,
                            side=order.side,
                            quantity=volume,
                            entry_price=order.filled_price,
                            stop_loss=sl if sl > 0 else None,
                            take_profit=tp if tp > 0 else None,
                        )
                        
                        self._positions[position.id] = position
                        self._orders[order.id] = order
                        
                        return TradeResult(
                            success=True,
                            order=order,
                            position=position,
                            message=f"Order filled at {order.filled_price}"
                        )
                    
                    # Filling mode error - try next mode
                    if retcode == 10030:  # Invalid fill type
                        logger.warning(f"Filling mode {filling_name} not supported, trying next...")
                        last_error = result.comment
                        continue
                    
                    # Other errors - don't retry
                    last_error = f"[{retcode}] {retcode_msg}: {result.comment}"
                    logger.error(f"❌ Order failed: {last_error}")
                    
                    # Specific error handling
                    if retcode == 10026:
                        last_error += " (เปิดปุ่ม Algo Trading ใน MT5!)"
                    elif retcode == 10019:
                        last_error += f" (Balance: {await self.get_balance()})"
                    elif retcode == 10015:
                        last_error += f" (Price sent: {price}, Current: ask={tick.ask}, bid={tick.bid})"
                    
                    break
                
                return TradeResult(success=False, error=last_error)
            
            # MT5 not available - cannot trade
            return TradeResult(
                success=False, 
                error="MT5 not connected - cannot execute order"
            )
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            import traceback
            traceback.print_exc()
            return TradeResult(success=False, error=str(e))
    
    async def cancel_order(self, order_id: str) -> bool:
        """ยกเลิกคำสั่ง"""
        try:
            if self._mt5:
                request = {
                    "action": self._mt5.TRADE_ACTION_REMOVE,
                    "order": int(order_id),
                }
                result = self._mt5.order_send(request)
                
                if result.retcode == self._mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Order {order_id} cancelled")
                    return True
                else:
                    logger.error(f"❌ Cancel failed: {result.comment}")
                    return False
            
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
                return True
                
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
        
        return False
    
    async def get_positions(self) -> List[Position]:
        """ดึง Position ที่เปิดอยู่"""
        try:
            if self._mt5:
                positions = self._mt5.positions_get()
                if positions:
                    result = []
                    for pos in positions:
                        position = Position(
                            id=str(pos.ticket),
                            symbol=pos.symbol,
                            side=OrderSide.BUY if pos.type == 0 else OrderSide.SELL,
                            quantity=pos.volume,
                            entry_price=pos.price_open,
                            current_price=pos.price_current,
                            stop_loss=pos.sl if pos.sl > 0 else None,
                            take_profit=pos.tp if pos.tp > 0 else None,
                            pnl=pos.profit,
                        )
                        result.append(position)
                        self._positions[position.id] = position
                    return result
                    
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
        
        return list(self._positions.values())
    
    async def close_position(self, position_id: str) -> TradeResult:
        """
        ปิด Position (แก้ไขแล้ว!)
        
        - Auto-retry with different filling modes
        - Price normalization
        """
        try:
            if self._mt5:
                position = self._mt5.positions_get(ticket=int(position_id))
                if not position:
                    return TradeResult(success=False, error="Position not found")
                
                pos = position[0]
                close_type = self._mt5.ORDER_TYPE_SELL if pos.type == 0 else self._mt5.ORDER_TYPE_BUY
                
                tick = self._mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    return TradeResult(success=False, error=f"Cannot get price for {pos.symbol}")
                
                price = tick.bid if pos.type == 0 else tick.ask
                price = self._normalize_price(price, pos.symbol)
                
                # Try different filling modes
                max_attempts = 3
                last_error = ""
                
                for attempt in range(max_attempts):
                    filling_mode = self._get_filling_mode(pos.symbol, attempt)
                    if filling_mode is None:
                        break
                    
                    request = {
                        "action": self._mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": close_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": "Trademify Close",
                        "type_time": self._mt5.ORDER_TIME_GTC,
                        "type_filling": filling_mode,
                    }
                    
                    logger.info(f"📤 Close Position {position_id}: {pos.volume} {pos.symbol} @ {price}")
                    
                    result = self._mt5.order_send(request)
                    
                    if result is None:
                        last_error = f"MT5 Error: {self._mt5.last_error()}"
                        continue
                    
                    retcode = result.retcode
                    retcode_msg = MT5_RETCODE_MESSAGES.get(retcode, f"Unknown code {retcode}")
                    
                    if retcode == self._mt5.TRADE_RETCODE_DONE:
                        logger.info(f"✅ Position {position_id} closed at {price}")
                        if position_id in self._positions:
                            del self._positions[position_id]
                        return TradeResult(
                            success=True, 
                            message=f"Position closed at {price}, PnL: {pos.profit:.2f}"
                        )
                    
                    if retcode == 10030:
                        logger.warning(f"Filling mode not supported, trying next...")
                        continue
                    
                    last_error = f"[{retcode}] {retcode_msg}: {result.comment}"
                    logger.error(f"❌ Close failed: {last_error}")
                    break
                
                return TradeResult(success=False, error=last_error)
            
            # MT5 not available - cannot close
            return TradeResult(
                success=False,
                error="MT5 not connected - cannot close position"
            )
                
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            import traceback
            traceback.print_exc()
        
        return TradeResult(success=False, error="Position not found")
    
    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeResult:
        """
        แก้ไข SL/TP ของ Position (แก้ไขแล้ว!)
        
        - Price normalization for SL/TP
        """
        try:
            if self._mt5:
                position = self._mt5.positions_get(ticket=int(position_id))
                if not position:
                    return TradeResult(success=False, error="Position not found")
                
                pos = position[0]
                
                # Normalize SL/TP prices
                new_sl = self._normalize_price(stop_loss, pos.symbol) if stop_loss else pos.sl
                new_tp = self._normalize_price(take_profit, pos.symbol) if take_profit else pos.tp
                
                request = {
                    "action": self._mt5.TRADE_ACTION_SLTP,
                    "symbol": pos.symbol,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": new_tp,
                }
                
                logger.info(f"📤 Modify Position {position_id}: SL={new_sl}, TP={new_tp}")
                
                result = self._mt5.order_send(request)
                
                if result is None:
                    return TradeResult(success=False, error=f"MT5 Error: {self._mt5.last_error()}")
                
                retcode = result.retcode
                retcode_msg = MT5_RETCODE_MESSAGES.get(retcode, f"Unknown code {retcode}")
                
                if retcode == self._mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Position {position_id} modified: SL={new_sl}, TP={new_tp}")
                    return TradeResult(success=True, message="Position modified")
                else:
                    logger.error(f"❌ Modify failed: [{retcode}] {retcode_msg}")
                    return TradeResult(success=False, error=f"[{retcode}] {result.comment}")
            
            # MT5 not available - cannot modify
            return TradeResult(
                success=False,
                error="MT5 not connected - cannot modify position"
            )
                
        except Exception as e:
            logger.error(f"Failed to modify position: {e}")
        
        return TradeResult(success=False, error="Position not found")
    
    async def get_current_price(self, symbol: str) -> float:
        """ดึงราคาปัจจุบันจาก MT5"""
        try:
            if self._mt5:
                tick = self._mt5.symbol_info_tick(symbol)
                if tick:
                    return (tick.ask + tick.bid) / 2
            
            logger.warning(f"Cannot get price for {symbol} - MT5 not connected")
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return 0.0
