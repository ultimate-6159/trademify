"""
Binance Connector
เชื่อมต่อกับ Binance Exchange สำหรับ Crypto Trading
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
import hmac
import time

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


@dataclass
class BinanceConfig:
    """Binance Configuration"""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True  # ใช้ Testnet เป็นค่าเริ่มต้นเพื่อความปลอดภัย
    
    @property
    def base_url(self) -> str:
        if self.testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"
    
    @property
    def ws_url(self) -> str:
        if self.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"


class BinanceBroker(BaseBroker):
    """
    Binance Exchange Connector
    รองรับทั้ง Spot และ Futures
    """
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self._connected = False
        self._session = None
        
        # Cache
        self._account_info: Dict[str, Any] = {}
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._prices: Dict[str, float] = {}
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """สร้าง signature สำหรับ API request"""
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def connect(self) -> bool:
        """เชื่อมต่อกับ Binance"""
        try:
            # ใช้ aiohttp สำหรับ async HTTP
            import aiohttp
            
            self._session = aiohttp.ClientSession(
                headers={
                    "X-MBX-APIKEY": self.config.api_key,
                    "Content-Type": "application/json",
                }
            )
            
            # ทดสอบการเชื่อมต่อ
            async with self._session.get(f"{self.config.base_url}/api/v3/ping") as resp:
                if resp.status == 200:
                    self._connected = True
                    logger.info("Connected to Binance" + (" (Testnet)" if self.config.testnet else ""))
                    return True
                else:
                    logger.error(f"Failed to connect to Binance: {resp.status}")
                    return False
                    
        except ImportError:
            logger.error("aiohttp not installed - cannot connect to Binance")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self) -> None:
        """ตัดการเชื่อมต่อ"""
        if self._session:
            await self._session.close()
        self._connected = False
        logger.info("Disconnected from Binance")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลบัญชี"""
        if not self._connected:
            return {}
        
        try:
            timestamp = int(time.time() * 1000)
            params = {"timestamp": timestamp}
            params["signature"] = self._sign_request(params)
            
            if self._session:
                async with self._session.get(
                    f"{self.config.base_url}/api/v3/account",
                    params=params
                ) as resp:
                    if resp.status == 200:
                        self._account_info = await resp.json()
                        return self._account_info
            
            # Not connected
            logger.warning("Binance API not connected - cannot get account info")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    async def get_balance(self) -> float:
        """ดึงยอดเงินคงเหลือ (USDT)"""
        account = await self.get_account_info()
        
        for balance in account.get("balances", []):
            if balance["asset"] == "USDT":
                return float(balance["free"])
        
        return 0.0  # No balance if not connected
    
    async def place_order(self, order: Order) -> TradeResult:
        """ส่งคำสั่งเทรด"""
        if not self._connected:
            return TradeResult(success=False, error="Not connected")
        
        try:
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": order.symbol.replace("/", ""),
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": order.quantity,
                "timestamp": timestamp,
            }
            
            if order.price and order.order_type != OrderType.MARKET:
                params["price"] = order.price
            
            params["signature"] = self._sign_request(params)
            
            if self._session:
                async with self._session.post(
                    f"{self.config.base_url}/api/v3/order",
                    data=params
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        order.status = OrderStatus.FILLED
                        order.filled_at = datetime.now()
                        order.filled_price = float(data.get("fills", [{}])[0].get("price", 0))
                        
                        # Create position
                        position = Position(
                            id=f"POS-{data['orderId']}",
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            entry_price=order.filled_price,
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                        )
                        
                        self._positions[position.id] = position
                        self._orders[order.id] = order
                        
                        return TradeResult(
                            success=True,
                            order=order,
                            position=position,
                            message="Order filled"
                        )
                    else:
                        error = await resp.text()
                        return TradeResult(success=False, error=error)
            
            # API not available - cannot trade
            return TradeResult(
                success=False,
                error="Binance API not connected - cannot execute order"
            )
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return TradeResult(success=False, error=str(e))
    
    async def cancel_order(self, order_id: str) -> bool:
        """ยกเลิกคำสั่ง"""
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    async def get_positions(self) -> List[Position]:
        """ดึง Position ที่เปิดอยู่"""
        return list(self._positions.values())
    
    async def close_position(self, position_id: str) -> TradeResult:
        """ปิด Position"""
        if position_id not in self._positions:
            return TradeResult(success=False, error="Position not found")
        
        position = self._positions[position_id]
        current_price = await self.get_current_price(position.symbol)
        
        # Create close order
        close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        
        close_order = Order(
            id=f"ORD-CLOSE-{position_id}",
            symbol=position.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
        )
        
        result = await self.place_order(close_order)
        
        if result.success:
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.now()
            position.exit_price = current_price
            position.update_pnl(current_price)
            
            del self._positions[position_id]
            
            return TradeResult(
                success=True,
                order=close_order,
                position=position,
                message=f"Position closed at {current_price}, PnL: {position.pnl:.2f}"
            )
        
        return result
    
    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeResult:
        """แก้ไข SL/TP ของ Position"""
        if position_id not in self._positions:
            return TradeResult(success=False, error="Position not found")
        
        position = self._positions[position_id]
        
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        
        return TradeResult(
            success=True,
            position=position,
            message="Position modified"
        )
    
    async def get_current_price(self, symbol: str) -> float:
        """ดึงราคาปัจจุบัน"""
        try:
            symbol_clean = symbol.replace("/", "")
            
            if self._session:
                async with self._session.get(
                    f"{self.config.base_url}/api/v3/ticker/price",
                    params={"symbol": symbol_clean}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = float(data["price"])
                        self._prices[symbol] = price
                        return price
            
            # Not connected - return cached or 0
            logger.warning(f"Cannot get price for {symbol} - Binance not connected")
            return self._prices.get(symbol, 0.0)
            
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return self._prices.get(symbol, 0.0)
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> List[List]:
        """ดึงข้อมูล OHLCV"""
        try:
            symbol_clean = symbol.replace("/", "")
            
            if self._session:
                async with self._session.get(
                    f"{self.config.base_url}/api/v3/klines",
                    params={
                        "symbol": symbol_clean,
                        "interval": interval,
                        "limit": limit,
                    }
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get klines: {e}")
            return []
