"""
Shared State Service
Sync ข้อมูลระหว่าง VPS หลายเครื่องผ่าน Firebase Realtime Database

Architecture:
    VPS 1 ──┐
    VPS 2 ──┼──► Firebase Realtime DB ◄──► Dashboard
    VPS 3 ──┘
"""
import os
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import uuid

logger = logging.getLogger(__name__)

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("firebase-admin not installed")


class LockStatus(str, Enum):
    """สถานะ Distributed Lock"""
    ACQUIRED = "ACQUIRED"
    WAITING = "WAITING"
    FAILED = "FAILED"


@dataclass
class NodeInfo:
    """ข้อมูล Node/VPS"""
    node_id: str
    hostname: str
    symbols: List[str]
    started_at: str
    last_heartbeat: str
    status: str = "ACTIVE"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SharedPosition:
    """Position ที่ sync ระหว่าง nodes"""
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: float
    pnl_percent: float
    status: str
    node_id: str  # Node ที่เปิด position นี้
    created_at: str
    updated_at: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SharedPosition':
        return cls(**data)


@dataclass 
class SharedTrade:
    """Trade history ที่ sync"""
    id: str
    position_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    node_id: str
    opened_at: str
    closed_at: str
    close_reason: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class SharedStateService:
    """
    Shared State Service
    จัดการ state ที่ต้อง sync ระหว่าง VPS หลายเครื่อง
    
    Features:
    - Position sync: sync positions แบบ real-time
    - Distributed lock: ป้องกัน race condition
    - Node registry: track active nodes
    - Trade history: เก็บประวัติเทรด
    """
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        database_url: Optional[str] = None,
        node_id: Optional[str] = None
    ):
        self.initialized = False
        self._listeners = {}
        self._callbacks: Dict[str, List[Callable]] = {
            'position_added': [],
            'position_updated': [],
            'position_removed': [],
            'node_joined': [],
            'node_left': [],
            'settings_changed': [],
        }
        
        # Generate unique node ID
        self.node_id = node_id or self._generate_node_id()
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'unknown')
        
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase not available - running in local mode")
            self._local_positions: Dict[str, SharedPosition] = {}
            self._local_trades: List[SharedTrade] = []
            return
        
        # Get credentials
        cred_path = credentials_path or os.getenv("FIREBASE_CREDENTIALS_PATH")
        db_url = database_url or os.getenv("FIREBASE_DATABASE_URL")
        
        if not cred_path or not db_url:
            logger.warning("Firebase not configured - running in local mode")
            self._local_positions = {}
            self._local_trades = []
            return
        
        try:
            # Initialize Firebase (if not already)
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {"databaseURL": db_url})
            
            self.initialized = True
            logger.info(f"SharedStateService initialized - Node ID: {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SharedStateService: {e}")
            self._local_positions = {}
            self._local_trades = []
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        unique = f"{hostname}-{os.getpid()}-{datetime.now().timestamp()}"
        return hashlib.md5(unique.encode()).hexdigest()[:12]
    
    # ==================== Node Registry ====================
    
    def register_node(self, symbols: List[str]) -> bool:
        """
        ลงทะเบียน node นี้ใน cluster
        
        Args:
            symbols: List of symbols ที่ node นี้ดูแล
        """
        if not self.initialized:
            logger.info(f"Local mode: Node {self.node_id} registered for {symbols}")
            return True
        
        try:
            node_info = NodeInfo(
                node_id=self.node_id,
                hostname=self.hostname,
                symbols=symbols,
                started_at=datetime.now().isoformat(),
                last_heartbeat=datetime.now().isoformat(),
                status="ACTIVE"
            )
            
            ref = db.reference(f"cluster/nodes/{self.node_id}")
            ref.set(node_info.to_dict())
            
            logger.info(f"Node {self.node_id} registered for symbols: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False
    
    def unregister_node(self) -> bool:
        """ยกเลิกการลงทะเบียน node"""
        if not self.initialized:
            return True
        
        try:
            ref = db.reference(f"cluster/nodes/{self.node_id}")
            ref.delete()
            logger.info(f"Node {self.node_id} unregistered")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister node: {e}")
            return False
    
    def heartbeat(self) -> bool:
        """ส่ง heartbeat เพื่อบอกว่า node ยังทำงานอยู่"""
        if not self.initialized:
            return True
        
        try:
            ref = db.reference(f"cluster/nodes/{self.node_id}/last_heartbeat")
            ref.set(datetime.now().isoformat())
            return True
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            return False
    
    def get_active_nodes(self) -> List[NodeInfo]:
        """ดึงรายการ nodes ที่ active"""
        if not self.initialized:
            return []
        
        try:
            ref = db.reference("cluster/nodes")
            data = ref.get() or {}
            
            nodes = []
            for node_id, info in data.items():
                nodes.append(NodeInfo(**info))
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []
    
    # ==================== Position Management ====================
    
    def save_position(self, position: SharedPosition) -> bool:
        """
        บันทึก/อัพเดท position ไป Firebase
        
        Args:
            position: SharedPosition object
        """
        if not self.initialized:
            self._local_positions[position.id] = position
            logger.debug(f"Local mode: Saved position {position.id}")
            return True
        
        try:
            position.updated_at = datetime.now().isoformat()
            position.node_id = self.node_id
            
            ref = db.reference(f"trading/positions/{position.id}")
            ref.set(position.to_dict())
            
            logger.info(f"Position saved: {position.symbol} {position.side} @ {position.entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
            return False
    
    def get_position(self, position_id: str) -> Optional[SharedPosition]:
        """ดึง position by ID"""
        if not self.initialized:
            return self._local_positions.get(position_id)
        
        try:
            ref = db.reference(f"trading/positions/{position_id}")
            data = ref.get()
            
            if data:
                return SharedPosition.from_dict(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None
    
    def get_all_positions(self, symbol: Optional[str] = None, status: str = "OPEN") -> List[SharedPosition]:
        """
        ดึง positions ทั้งหมด
        
        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (default: OPEN)
        """
        if not self.initialized:
            positions = list(self._local_positions.values())
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
            if status:
                positions = [p for p in positions if p.status == status]
            return positions
        
        try:
            ref = db.reference("trading/positions")
            data = ref.get() or {}
            
            positions = []
            for pos_id, pos_data in data.items():
                pos = SharedPosition.from_dict(pos_data)
                
                if status and pos.status != status:
                    continue
                if symbol and pos.symbol != symbol:
                    continue
                    
                positions.append(pos)
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def delete_position(self, position_id: str) -> bool:
        """ลบ position (ใช้เมื่อปิด position แล้ว)"""
        if not self.initialized:
            if position_id in self._local_positions:
                del self._local_positions[position_id]
            return True
        
        try:
            ref = db.reference(f"trading/positions/{position_id}")
            ref.delete()
            logger.info(f"Position {position_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete position: {e}")
            return False
    
    def update_position_price(self, position_id: str, current_price: float, pnl: float, pnl_percent: float) -> bool:
        """อัพเดทราคาและ PnL ของ position"""
        if not self.initialized:
            if position_id in self._local_positions:
                pos = self._local_positions[position_id]
                pos.current_price = current_price
                pos.pnl = pnl
                pos.pnl_percent = pnl_percent
                pos.updated_at = datetime.now().isoformat()
            return True
        
        try:
            ref = db.reference(f"trading/positions/{position_id}")
            ref.update({
                "current_price": current_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "updated_at": datetime.now().isoformat()
            })
            return True
            
        except Exception as e:
            logger.error(f"Failed to update position price: {e}")
            return False
    
    # ==================== Trade History ====================
    
    def save_trade(self, trade: SharedTrade) -> bool:
        """บันทึก trade ที่ปิดแล้ว"""
        if not self.initialized:
            self._local_trades.append(trade)
            return True
        
        try:
            ref = db.reference(f"trading/history/{trade.id}")
            ref.set(trade.to_dict())
            
            # Update daily stats
            self._update_daily_stats(trade)
            
            logger.info(f"Trade saved: {trade.symbol} PnL: {trade.pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            return False
    
    def get_trade_history(self, limit: int = 100, symbol: Optional[str] = None) -> List[SharedTrade]:
        """ดึงประวัติเทรด"""
        if not self.initialized:
            trades = self._local_trades[-limit:]
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            return trades
        
        try:
            ref = db.reference("trading/history")
            query = ref.order_by_child("closed_at").limit_to_last(limit)
            data = query.get() or {}
            
            trades = []
            for trade_id, trade_data in data.items():
                trade = SharedTrade(**trade_data)
                if symbol and trade.symbol != symbol:
                    continue
                trades.append(trade)
            
            return sorted(trades, key=lambda x: x.closed_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    def _update_daily_stats(self, trade: SharedTrade) -> None:
        """อัพเดทสถิติรายวัน"""
        if not self.initialized:
            return
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            ref = db.reference(f"trading/stats/daily/{today}")
            
            current = ref.get() or {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "symbols": {}
            }
            
            current["total_trades"] += 1
            current["total_pnl"] += trade.pnl
            
            if trade.pnl > 0:
                current["winning_trades"] += 1
            else:
                current["losing_trades"] += 1
            
            # Per-symbol stats
            if trade.symbol not in current["symbols"]:
                current["symbols"][trade.symbol] = {"trades": 0, "pnl": 0.0}
            
            current["symbols"][trade.symbol]["trades"] += 1
            current["symbols"][trade.symbol]["pnl"] += trade.pnl
            
            ref.set(current)
            
        except Exception as e:
            logger.error(f"Failed to update daily stats: {e}")
    
    # ==================== Distributed Lock ====================
    
    def acquire_lock(self, lock_name: str, timeout_seconds: int = 30) -> LockStatus:
        """
        ขอ distributed lock
        
        Args:
            lock_name: ชื่อ lock (เช่น "position_BTCUSDT")
            timeout_seconds: timeout ของ lock
        
        Returns:
            LockStatus
        """
        if not self.initialized:
            return LockStatus.ACQUIRED
        
        try:
            ref = db.reference(f"cluster/locks/{lock_name}")
            current = ref.get()
            
            now = datetime.now()
            
            # Check if lock exists and is still valid
            if current:
                lock_time = datetime.fromisoformat(current.get("acquired_at", "2000-01-01"))
                if (now - lock_time).total_seconds() < current.get("timeout", timeout_seconds):
                    if current.get("node_id") != self.node_id:
                        return LockStatus.WAITING
            
            # Acquire lock
            ref.set({
                "node_id": self.node_id,
                "acquired_at": now.isoformat(),
                "timeout": timeout_seconds
            })
            
            return LockStatus.ACQUIRED
            
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return LockStatus.FAILED
    
    def release_lock(self, lock_name: str) -> bool:
        """ปล่อย lock"""
        if not self.initialized:
            return True
        
        try:
            ref = db.reference(f"cluster/locks/{lock_name}")
            current = ref.get()
            
            if current and current.get("node_id") == self.node_id:
                ref.delete()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False
    
    # ==================== Settings Sync ====================
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """บันทึก settings ที่ใช้ร่วมกัน"""
        if not self.initialized:
            return True
        
        try:
            ref = db.reference("trading/settings")
            settings["updated_at"] = datetime.now().isoformat()
            settings["updated_by"] = self.node_id
            ref.set(settings)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def get_settings(self) -> Dict[str, Any]:
        """ดึง settings"""
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("trading/settings")
            return ref.get() or {}
            
        except Exception as e:
            logger.error(f"Failed to get settings: {e}")
            return {}
    
    # ==================== Real-time Listeners ====================
    
    def start_listening(self) -> None:
        """เริ่ม listen การเปลี่ยนแปลงจาก Firebase"""
        if not self.initialized:
            return
        
        # Listen for position changes
        positions_ref = db.reference("trading/positions")
        self._listeners['positions'] = positions_ref.listen(self._on_positions_changed)
        
        # Listen for settings changes
        settings_ref = db.reference("trading/settings")
        self._listeners['settings'] = settings_ref.listen(self._on_settings_changed)
        
        # Listen for node changes
        nodes_ref = db.reference("cluster/nodes")
        self._listeners['nodes'] = nodes_ref.listen(self._on_nodes_changed)
        
        logger.info("Started listening for real-time updates")
    
    def stop_listening(self) -> None:
        """หยุด listen"""
        for name, listener in self._listeners.items():
            if listener:
                listener.close()
        self._listeners.clear()
        logger.info("Stopped listening")
    
    def _on_positions_changed(self, event) -> None:
        """Callback เมื่อ positions เปลี่ยน"""
        if event.event_type == 'put':
            if event.data is None:
                # Position removed
                for callback in self._callbacks['position_removed']:
                    callback(event.path)
            elif isinstance(event.data, dict):
                if 'symbol' in event.data:
                    # Single position updated
                    pos = SharedPosition.from_dict(event.data)
                    for callback in self._callbacks['position_updated']:
                        callback(pos)
    
    def _on_settings_changed(self, event) -> None:
        """Callback เมื่อ settings เปลี่ยน"""
        if event.event_type == 'put' and event.data:
            for callback in self._callbacks['settings_changed']:
                callback(event.data)
    
    def _on_nodes_changed(self, event) -> None:
        """Callback เมื่อ nodes เปลี่ยน"""
        pass  # Implement if needed
    
    def on(self, event_name: str, callback: Callable) -> None:
        """
        ลงทะเบียน callback สำหรับ event
        
        Args:
            event_name: ชื่อ event (position_added, position_updated, etc.)
            callback: Function to call
        """
        if event_name in self._callbacks:
            self._callbacks[event_name].append(callback)
    
    # ==================== Summary & Stats ====================
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """ดึงสรุปข้อมูล cluster"""
        nodes = self.get_active_nodes()
        positions = self.get_all_positions()
        
        total_pnl = sum(p.pnl for p in positions)
        
        return {
            "total_nodes": len(nodes),
            "nodes": [n.to_dict() for n in nodes],
            "total_positions": len(positions),
            "total_pnl": total_pnl,
            "positions_by_symbol": self._group_positions_by_symbol(positions),
            "this_node": self.node_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _group_positions_by_symbol(self, positions: List[SharedPosition]) -> Dict[str, int]:
        """Group positions by symbol"""
        result = {}
        for pos in positions:
            result[pos.symbol] = result.get(pos.symbol, 0) + 1
        return result


# Singleton instance
_shared_state: Optional[SharedStateService] = None


def get_shared_state() -> SharedStateService:
    """Get singleton SharedStateService instance"""
    global _shared_state
    if _shared_state is None:
        _shared_state = SharedStateService()
    return _shared_state
