"""
Firebase Service - The Glue
รับผลโหวตจาก Python ส่งไปหน้าเว็บ

Firebase Realtime Database สำหรับ real-time sync
"""
import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("firebase-admin not installed. Firebase features will be disabled.")


class FirebaseRESTService:
    """
    Firebase REST API Service - ไม่ต้องใช้ Service Account
    ใช้ Database URL + API Key แทน (เหมือน Frontend)
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.database_url = database_url or os.getenv("FIREBASE_DATABASE_URL")
        self.api_key = api_key or os.getenv("FIREBASE_API_KEY")
        self.initialized = bool(self.database_url)
        
        if self.initialized:
            # Remove trailing slash
            self.database_url = self.database_url.rstrip('/')
            logger.info(f"Firebase REST Service initialized: {self.database_url}")
        else:
            logger.warning("Firebase REST Service: No database URL configured")
    
    def _make_url(self, path: str) -> str:
        """Build Firebase REST URL"""
        url = f"{self.database_url}/{path}.json"
        if self.api_key:
            url += f"?auth={self.api_key}"
        return url
    
    def _put(self, path: str, data: Any) -> bool:
        """PUT data to Firebase"""
        try:
            url = self._make_url(path)
            response = requests.put(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Firebase PUT error: {e}")
            return False
    
    def _get(self, path: str) -> Optional[Any]:
        """GET data from Firebase"""
        try:
            url = self._make_url(path)
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Firebase GET error: {e}")
            return None
    
    def _post(self, path: str, data: Any) -> Optional[str]:
        """POST data to Firebase (creates new key)"""
        try:
            url = self._make_url(path)
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get("name")  # Returns the generated key
            return None
        except Exception as e:
            logger.error(f"Firebase POST error: {e}")
            return None
    
    def _delete(self, path: str) -> bool:
        """DELETE data from Firebase"""
        try:
            url = self._make_url(path)
            response = requests.delete(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Firebase DELETE error: {e}")
            return False
    
    # === Signal Methods ===
    
    def push_signal(self, symbol: str, timeframe: str, signal_data: Dict) -> Optional[str]:
        if not self.initialized:
            return None
        signal_data["timestamp"] = datetime.now().isoformat()
        return self._post(f"signals/{symbol}/{timeframe}", signal_data)
    
    def update_current_signal(self, symbol: str, timeframe: str, signal_data: Dict) -> bool:
        if not self.initialized:
            return False
        signal_data["timestamp"] = datetime.now().isoformat()
        return self._put(f"current/{symbol}_{timeframe}", signal_data)
    
    def get_current_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        if not self.initialized:
            return None
        return self._get(f"current/{symbol}_{timeframe}")
    
    # === Trading Methods ===
    
    def update_trading_status(self, status_data: Dict) -> bool:
        if not self.initialized:
            return False
        status_data["updated_at"] = datetime.now().isoformat()
        return self._put("trading/status", status_data)
    
    def get_trading_status(self) -> Optional[Dict]:
        if not self.initialized:
            return None
        return self._get("trading/status")
    
    def push_trade(self, trade_data: Dict) -> Optional[str]:
        if not self.initialized:
            return None
        trade_data["timestamp"] = datetime.now().isoformat()
        return self._post("trading/history", trade_data)
    
    def get_trade_history(self, limit: int = 100) -> list:
        if not self.initialized:
            return []
        data = self._get("trading/history")
        if data and isinstance(data, dict):
            trades = list(data.values())
            return sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
        return []
    
    def update_position(self, position_id: str, position_data: Dict) -> bool:
        if not self.initialized:
            return False
        return self._put(f"trading/positions/{position_id}", position_data)
    
    def remove_position(self, position_id: str) -> bool:
        if not self.initialized:
            return False
        return self._delete(f"trading/positions/{position_id}")
    
    def get_positions(self) -> Dict:
        if not self.initialized:
            return {}
        return self._get("trading/positions") or {}
    
    # === Bot Status Methods ===
    
    def update_bot_status(self, status_data: Dict) -> bool:
        if not self.initialized:
            return False
        status_data["updated_at"] = datetime.now().isoformat()
        return self._put("bot/status", status_data)
    
    def get_bot_status(self) -> Optional[Dict]:
        if not self.initialized:
            return None
        return self._get("bot/status")
    
    # === Learning Methods ===
    
    def save_learning_state(self, state: Dict) -> bool:
        if not self.initialized:
            return False
        return self._put("learning/state", state)
    
    def load_learning_state(self) -> Optional[Dict]:
        if not self.initialized:
            return None
        return self._get("learning/state")
    
    def save_factor_weights(self, weights: Dict) -> bool:
        if not self.initialized:
            return False
        return self._put("learning/factor_weights", weights)
    
    def load_factor_weights(self) -> Optional[Dict]:
        if not self.initialized:
            return None
        return self._get("learning/factor_weights")
    
    def save_optimized_params(self, params: Dict) -> bool:
        if not self.initialized:
            return False
        return self._put("learning/optimized_params", params)
    
    def load_optimized_params(self) -> Optional[Dict]:
        if not self.initialized:
            return None
        return self._get("learning/optimized_params")
    
    def save_market_cycle(self, cycle_data: Dict) -> bool:
        if not self.initialized:
            return False
        return self._put("learning/market_cycle", cycle_data)
    
    def get_learning_summary(self) -> Optional[Dict]:
        if not self.initialized:
            return None
        return self._get("learning")


class FirebaseService:
    """
    Firebase Realtime Database Service
    ส่งผลวิเคราะห์ไปแสดงบน Vue.js แบบ Real-time
    """
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        database_url: Optional[str] = None
    ):
        """
        Initialize Firebase Service
        
        Args:
            credentials_path: Path to Firebase credentials JSON
            database_url: Firebase Realtime Database URL
        """
        self.initialized = False
        
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase Admin SDK not available")
            return
        
        # Get credentials from environment or parameter
        cred_path = credentials_path or os.getenv("FIREBASE_CREDENTIALS_PATH")
        db_url = database_url or os.getenv("FIREBASE_DATABASE_URL")
        
        if not cred_path or not db_url:
            logger.warning("Firebase credentials or database URL not configured")
            return
        
        try:
            # Check if already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    "databaseURL": db_url
                })
            
            self.initialized = True
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
    
    def push_signal(
        self,
        symbol: str,
        timeframe: str,
        signal_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Push trading signal to Firebase
        
        Args:
            symbol: Trading symbol (e.g., EURUSD)
            timeframe: Timeframe (e.g., H1)
            signal_data: Signal data from voting system
        
        Returns:
            Firebase key if successful, None otherwise
        """
        if not self.initialized:
            logger.warning("Firebase not initialized. Skipping push.")
            return None
        
        try:
            ref = db.reference(f"signals/{symbol}/{timeframe}")
            
            # Add timestamp
            signal_data["timestamp"] = datetime.now().isoformat()
            signal_data["symbol"] = symbol
            signal_data["timeframe"] = timeframe
            
            # Push to Firebase
            new_ref = ref.push(signal_data)
            
            logger.info(f"Pushed signal to Firebase: {new_ref.key}")
            return new_ref.key
            
        except Exception as e:
            logger.error(f"Failed to push signal to Firebase: {e}")
            return None
    
    def update_current_signal(
        self,
        symbol: str,
        timeframe: str,
        signal_data: Dict[str, Any]
    ) -> bool:
        """
        Update current signal (overwrite)
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal_data: Signal data
        
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.warning("Firebase not initialized. Skipping update.")
            return False
        
        try:
            ref = db.reference(f"current_signals/{symbol}/{timeframe}")
            
            signal_data["timestamp"] = datetime.now().isoformat()
            signal_data["symbol"] = symbol
            signal_data["timeframe"] = timeframe
            
            ref.set(signal_data)
            
            logger.info(f"Updated current signal for {symbol}/{timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update signal: {e}")
            return False
    
    def push_pattern_match(
        self,
        symbol: str,
        timeframe: str,
        matched_patterns: list,
        current_pattern: list,
        projected_patterns: list
    ) -> Optional[str]:
        """
        Push matched patterns for visualization
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            matched_patterns: List of matched historical patterns
            current_pattern: Current price pattern
            projected_patterns: Projected future patterns
        
        Returns:
            Firebase key if successful
        """
        if not self.initialized:
            return None
        
        try:
            ref = db.reference(f"patterns/{symbol}/{timeframe}")
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "current_pattern": current_pattern,
                "matched_patterns": matched_patterns,
                "projected_patterns": projected_patterns,
            }
            
            new_ref = ref.push(data)
            
            # Also update current
            current_ref = db.reference(f"current_patterns/{symbol}/{timeframe}")
            current_ref.set(data)
            
            return new_ref.key
            
        except Exception as e:
            logger.error(f"Failed to push patterns: {e}")
            return None
    
    def get_signal_history(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> list:
        """
        Get signal history from Firebase
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Maximum number of signals
        
        Returns:
            List of historical signals
        """
        if not self.initialized:
            return []
        
        try:
            ref = db.reference(f"signals/{symbol}/{timeframe}")
            signals = ref.order_by_child("timestamp").limit_to_last(limit).get()
            
            if signals:
                return list(signals.values())
            return []
            
        except Exception as e:
            logger.error(f"Failed to get signal history: {e}")
            return []
    
    # =========================================================================
    # SMART BRAIN DATA - Trade Journal & Pattern Memory
    # =========================================================================
    
    def save_trade_journal(self, trades: list) -> bool:
        """
        Save trade journal to Firebase
        
        Args:
            trades: List of trade records
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("smart_brain/trade_journal")
            ref.set({
                "trades": trades,
                "updated_at": datetime.now().isoformat(),
                "total_trades": len(trades)
            })
            logger.info(f"Saved {len(trades)} trades to Firebase")
            return True
        except Exception as e:
            logger.error(f"Failed to save trade journal: {e}")
            return False
    
    def load_trade_journal(self) -> list:
        """
        Load trade journal from Firebase
        
        Returns:
            List of trade records
        """
        if not self.initialized:
            return []
        
        try:
            ref = db.reference("smart_brain/trade_journal/trades")
            trades = ref.get()
            if trades:
                logger.info(f"Loaded {len(trades)} trades from Firebase")
                return trades
            return []
        except Exception as e:
            logger.error(f"Failed to load trade journal: {e}")
            return []
    
    def add_trade(self, trade: dict) -> bool:
        """
        Add single trade to journal (append)
        
        Args:
            trade: Trade record
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("smart_brain/trade_journal/trades")
            ref.push(trade)
            
            # Also update trade history (for Dashboard)
            self.add_trade_history(trade)
            
            logger.info(f"Added trade {trade.get('trade_id', 'unknown')} to Firebase")
            return True
        except Exception as e:
            logger.error(f"Failed to add trade: {e}")
            return False
    
    def update_trade(self, trade_id: str, updates: dict) -> bool:
        """
        Update existing trade (e.g., when closed)
        
        Args:
            trade_id: Trade ID to update
            updates: Fields to update
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("smart_brain/trade_journal/trades")
            trades = ref.get()
            
            if trades:
                for key, trade in trades.items():
                    if trade and trade.get("trade_id") == trade_id:
                        for k, v in updates.items():
                            ref.child(key).child(k).set(v)
                        logger.info(f"Updated trade {trade_id} in Firebase")
                        return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")
            return False
    
    def save_pattern_memory(self, patterns: dict) -> bool:
        """
        Save pattern memory to Firebase
        
        Args:
            patterns: Dict of pattern_hash -> PatternMemory
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("smart_brain/pattern_memory")
            ref.set({
                "patterns": patterns,
                "updated_at": datetime.now().isoformat(),
                "total_patterns": len(patterns)
            })
            logger.info(f"Saved {len(patterns)} patterns to Firebase")
            return True
        except Exception as e:
            logger.error(f"Failed to save pattern memory: {e}")
            return False
    
    def load_pattern_memory(self) -> dict:
        """
        Load pattern memory from Firebase
        
        Returns:
            Dict of pattern_hash -> PatternMemory data
        """
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("smart_brain/pattern_memory/patterns")
            patterns = ref.get()
            if patterns:
                logger.info(f"Loaded {len(patterns)} patterns from Firebase")
                return patterns
            return {}
        except Exception as e:
            logger.error(f"Failed to load pattern memory: {e}")
            return {}
    
    def save_time_stats(self, hourly: dict, daily: dict) -> bool:
        """
        Save time analysis stats to Firebase
        
        Args:
            hourly: Hourly performance stats
            daily: Daily performance stats
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("smart_brain/time_stats")
            ref.set({
                "hourly": hourly,
                "daily": daily,
                "updated_at": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to save time stats: {e}")
            return False
    
    def load_time_stats(self) -> tuple:
        """
        Load time analysis stats from Firebase
        
        Returns:
            (hourly_stats, daily_stats)
        """
        if not self.initialized:
            return {}, {}
        
        try:
            ref = db.reference("smart_brain/time_stats")
            data = ref.get()
            if data:
                return data.get("hourly", {}), data.get("daily", {})
            return {}, {}
        except Exception as e:
            logger.error(f"Failed to load time stats: {e}")
            return {}, {}
    
    def save_symbol_stats(self, stats: dict) -> bool:
        """
        Save symbol performance stats to Firebase
        
        Args:
            stats: Symbol performance stats
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("smart_brain/symbol_stats")
            ref.set({
                "stats": stats,
                "updated_at": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to save symbol stats: {e}")
            return False
    
    def load_symbol_stats(self) -> dict:
        """
        Load symbol performance stats from Firebase
        
        Returns:
            Symbol stats dict
        """
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("smart_brain/symbol_stats/stats")
            stats = ref.get()
            return stats if stats else {}
        except Exception as e:
            logger.error(f"Failed to load symbol stats: {e}")
            return {}
    
    def add_trade_history(self, trade: dict) -> bool:
        """
        Add trade to history (for Dashboard)
        
        Args:
            trade: Trade record
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("trade_history")
            ref.push({
                **trade,
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to add trade history: {e}")
            return False
    
    def get_smart_brain_summary(self) -> dict:
        """
        Get Smart Brain summary for Dashboard
        
        Returns:
            Summary dict with stats
        """
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("smart_brain")
            data = ref.get()
            
            if not data:
                return {}
            
            trades = data.get("trade_journal", {}).get("trades", [])
            patterns = data.get("pattern_memory", {}).get("patterns", {})
            
            # Calculate stats
            if trades:
                if isinstance(trades, dict):
                    trades = list(trades.values())
                
                closed_trades = [t for t in trades if t and t.get("exit_price")]
                wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
                
                return {
                    "total_trades": len(trades),
                    "closed_trades": len(closed_trades),
                    "wins": len(wins),
                    "losses": len(closed_trades) - len(wins),
                    "win_rate": len(wins) / len(closed_trades) * 100 if closed_trades else 0,
                    "total_pnl": sum(t.get("pnl_percent", 0) for t in closed_trades),
                    "patterns_learned": len(patterns),
                    "updated_at": data.get("trade_journal", {}).get("updated_at", "")
                }
            
            return {"total_trades": 0, "patterns_learned": len(patterns)}
            
        except Exception as e:
            logger.error(f"Failed to get smart brain summary: {e}")
            return {}
    
    # =====================
    # Continuous Learning Methods
    # =====================
    
    def save_learning_state(self, state: dict) -> bool:
        """
        Save continuous learning state to Firebase
        
        Args:
            state: Learning state dict
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("continuous_learning/state")
            ref.set({
                **state,
                "saved_at": datetime.now().isoformat()
            })
            logger.info("Saved learning state to Firebase")
            return True
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
            return False
    
    def load_learning_state(self) -> dict:
        """
        Load continuous learning state from Firebase
        
        Returns:
            Learning state dict
        """
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("continuous_learning/state")
            state = ref.get()
            if state:
                logger.info("Loaded learning state from Firebase")
                return state
            return {}
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")
            return {}
    
    def save_factor_weights(self, weights: dict) -> bool:
        """
        Save learned factor weights
        
        Args:
            weights: Factor importance weights
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("continuous_learning/factor_weights")
            ref.set({
                "weights": weights,
                "updated_at": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to save factor weights: {e}")
            return False
    
    def load_factor_weights(self) -> dict:
        """Load learned factor weights"""
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("continuous_learning/factor_weights/weights")
            weights = ref.get()
            return weights if weights else {}
        except Exception as e:
            logger.error(f"Failed to load factor weights: {e}")
            return {}
    
    def save_optimized_params(self, params: dict) -> bool:
        """
        Save auto-optimized strategy parameters
        
        Args:
            params: Optimized strategy params
        
        Returns:
            True if successful
        """
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("continuous_learning/optimized_params")
            ref.set({
                **params,
                "optimized_at": datetime.now().isoformat()
            })
            logger.info("Saved optimized params to Firebase")
            return True
        except Exception as e:
            logger.error(f"Failed to save optimized params: {e}")
            return False
    
    def load_optimized_params(self) -> dict:
        """Load auto-optimized strategy parameters"""
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("continuous_learning/optimized_params")
            params = ref.get()
            return params if params else {}
        except Exception as e:
            logger.error(f"Failed to load optimized params: {e}")
            return {}
    
    def save_market_cycle(self, cycle_data: dict) -> bool:
        """Save market cycle detection data"""
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("continuous_learning/market_cycle")
            ref.set({
                **cycle_data,
                "detected_at": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to save market cycle: {e}")
            return False
    
    def get_learning_summary(self) -> dict:
        """
        Get summary of all continuous learning data
        
        Returns:
            Summary of learning progress
        """
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("continuous_learning")
            data = ref.get()
            
            if not data:
                return {"status": "no_data"}
            
            state = data.get("state", {})
            factor_weights = data.get("factor_weights", {}).get("weights", {})
            params = data.get("optimized_params", {})
            
            return {
                "sample_count": state.get("online_learner", {}).get("sample_count", 0),
                "ema_win_rate": state.get("online_learner", {}).get("ema_win_rate", 0.5) * 100,
                "top_factors": sorted(
                    factor_weights.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5] if factor_weights else [],
                "optimized_params": {
                    "min_confidence": params.get("min_confidence", 70),
                    "min_confluence": params.get("min_confluence", 3),
                },
                "last_saved": state.get("saved_at", ""),
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning summary: {e}")
            return {}
    
    # =====================
    # Bot Status Methods (for Dashboard)
    # =====================
    
    def update_bot_status(self, status_data: dict) -> bool:
        """Update bot status in Firebase"""
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("bot/status")
            ref.set({
                **status_data,
                "updated_at": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to update bot status: {e}")
            return False
    
    def get_bot_status(self) -> dict:
        """Get bot status from Firebase"""
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("bot/status")
            return ref.get() or {}
        except Exception as e:
            logger.error(f"Failed to get bot status: {e}")
            return {}
    
    # =====================
    # Trading Status Methods
    # =====================
    
    def update_trading_status(self, status_data: dict) -> bool:
        """Update trading status"""
        if not self.initialized:
            return False
        
        try:
            ref = db.reference("trading/status")
            ref.set({
                **status_data,
                "updated_at": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to update trading status: {e}")
            return False
    
    def get_trading_status(self) -> dict:
        """Get trading status"""
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("trading/status")
            return ref.get() or {}
        except Exception as e:
            logger.error(f"Failed to get trading status: {e}")
            return {}
    
    def update_position(self, position_id: str, position_data: dict) -> bool:
        """Update position data"""
        if not self.initialized:
            return False
        
        try:
            ref = db.reference(f"trading/positions/{position_id}")
            ref.set(position_data)
            return True
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
            return False
    
    def remove_position(self, position_id: str) -> bool:
        """Remove position"""
        if not self.initialized:
            return False
        
        try:
            ref = db.reference(f"trading/positions/{position_id}")
            ref.delete()
            return True
        except Exception as e:
            logger.error(f"Failed to remove position: {e}")
            return False
    
    def get_positions(self) -> dict:
        """Get all positions"""
        if not self.initialized:
            return {}
        
        try:
            ref = db.reference("trading/positions")
            return ref.get() or {}
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    def get_trade_history(self, limit: int = 100) -> list:
        """Get trade history"""
        if not self.initialized:
            return []
        
        try:
            ref = db.reference("trading/history")
            data = ref.order_by_child("timestamp").limit_to_last(limit).get()
            if data:
                trades = list(data.values())
                return sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
            return []
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    def push_trade(self, trade_data: dict) -> str:
        """Push new trade to history"""
        if not self.initialized:
            return None
        
        try:
            ref = db.reference("trading/history")
            trade_data["timestamp"] = datetime.now().isoformat()
            new_ref = ref.push(trade_data)
            return new_ref.key
        except Exception as e:
            logger.error(f"Failed to push trade: {e}")
            return None


class MockFirebaseService:
    """
    Mock Firebase Service for development/testing
    ใช้ตอนที่ไม่มี Firebase จริง
    """
    
    def __init__(self):
        self.signals = {}
        self.patterns = {}
        self.trade_journal = []
        self.pattern_memory = {}
        self.time_stats = {"hourly": {}, "daily": {}}
        self.symbol_stats = {}
        self.initialized = True
        logger.info("Mock Firebase Service initialized")
    
    def push_signal(self, symbol: str, timeframe: str, signal_data: Dict[str, Any]) -> str:
        key = f"{symbol}_{timeframe}_{datetime.now().timestamp()}"
        
        if symbol not in self.signals:
            self.signals[symbol] = {}
        if timeframe not in self.signals[symbol]:
            self.signals[symbol][timeframe] = []
        
        signal_data["key"] = key
        signal_data["timestamp"] = datetime.now().isoformat()
        self.signals[symbol][timeframe].append(signal_data)
        
        logger.info(f"[Mock] Pushed signal: {key}")
        return key
    
    def update_current_signal(self, symbol: str, timeframe: str, signal_data: Dict[str, Any]) -> bool:
        signal_data["timestamp"] = datetime.now().isoformat()
        
        if symbol not in self.signals:
            self.signals[symbol] = {}
        self.signals[symbol][f"{timeframe}_current"] = signal_data
        
        logger.info(f"[Mock] Updated current signal for {symbol}/{timeframe}")
        return True
    
    def push_pattern_match(self, symbol: str, timeframe: str, *args, **kwargs) -> str:
        key = f"pattern_{symbol}_{timeframe}_{datetime.now().timestamp()}"
        logger.info(f"[Mock] Pushed pattern match: {key}")
        return key
    
    def get_signal_history(self, symbol: str, timeframe: str, limit: int = 100) -> list:
        if symbol in self.signals and timeframe in self.signals[symbol]:
            return self.signals[symbol][timeframe][-limit:]
        return []
    
    # Smart Brain methods
    def save_trade_journal(self, trades: list) -> bool:
        self.trade_journal = trades
        logger.info(f"[Mock] Saved {len(trades)} trades")
        return True
    
    def load_trade_journal(self) -> list:
        return self.trade_journal
    
    def add_trade(self, trade: dict) -> bool:
        self.trade_journal.append(trade)
        logger.info(f"[Mock] Added trade {trade.get('trade_id', 'unknown')}")
        return True
    
    def update_trade(self, trade_id: str, updates: dict) -> bool:
        for trade in self.trade_journal:
            if trade.get("trade_id") == trade_id:
                trade.update(updates)
                logger.info(f"[Mock] Updated trade {trade_id}")
                return True
        return False
    
    def save_pattern_memory(self, patterns: dict) -> bool:
        self.pattern_memory = patterns
        logger.info(f"[Mock] Saved {len(patterns)} patterns")
        return True
    
    def load_pattern_memory(self) -> dict:
        return self.pattern_memory
    
    def save_time_stats(self, hourly: dict, daily: dict) -> bool:
        self.time_stats = {"hourly": hourly, "daily": daily}
        return True
    
    def load_time_stats(self) -> tuple:
        return self.time_stats.get("hourly", {}), self.time_stats.get("daily", {})
    
    def save_symbol_stats(self, stats: dict) -> bool:
        self.symbol_stats = stats
        return True
    
    def load_symbol_stats(self) -> dict:
        return self.symbol_stats
    
    def add_trade_history(self, trade: dict) -> bool:
        return True
    
    def get_smart_brain_summary(self) -> dict:
        closed = [t for t in self.trade_journal if t.get("exit_price")]
        wins = [t for t in closed if t.get("pnl", 0) > 0]
        return {
            "total_trades": len(self.trade_journal),
            "closed_trades": len(closed),
            "wins": len(wins),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0,
            "patterns_learned": len(self.pattern_memory)
        }
    
    # Continuous Learning methods (Mock)
    def save_learning_state(self, state: dict) -> bool:
        self._learning_state = state
        logger.info(f"[Mock] Saved learning state")
        return True
    
    def load_learning_state(self) -> dict:
        return getattr(self, '_learning_state', {})
    
    def save_factor_weights(self, weights: dict) -> bool:
        self._factor_weights = weights
        return True
    
    def load_factor_weights(self) -> dict:
        return getattr(self, '_factor_weights', {})
    
    def save_optimized_params(self, params: dict) -> bool:
        self._optimized_params = params
        return True
    
    def load_optimized_params(self) -> dict:
        return getattr(self, '_optimized_params', {})
    
    def save_market_cycle(self, cycle_data: dict) -> bool:
        self._market_cycle = cycle_data
        return True
    
    def get_learning_summary(self) -> dict:
        state = getattr(self, '_learning_state', {})
        return {
            "sample_count": state.get("online_learner", {}).get("sample_count", 0),
            "ema_win_rate": state.get("online_learner", {}).get("ema_win_rate", 0.5) * 100,
            "status": "mock",
        }


def get_firebase_service(mock: bool = False):
    """
    Factory function to get appropriate Firebase service
    
    Priority:
    1. Admin SDK (with Service Account credentials)
    2. REST API (with Database URL only)
    3. Mock Service (fallback)
    
    Args:
        mock: Whether to force mock service
    
    Returns:
        Firebase service instance (FirebaseService, FirebaseRESTService, or MockFirebaseService)
    """
    if mock:
        logger.info("Using Mock Firebase Service (forced)")
        return MockFirebaseService()
    
    # Try Admin SDK first (needs credentials file)
    if FIREBASE_AVAILABLE:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        db_url = os.getenv("FIREBASE_DATABASE_URL")
        
        if cred_path and db_url and os.path.exists(cred_path):
            service = FirebaseService(credentials_path=cred_path, database_url=db_url)
            if service.initialized:
                logger.info("Using Firebase Admin SDK")
                return service
    
    # Try REST API (needs only database URL)
    db_url = os.getenv("FIREBASE_DATABASE_URL")
    if db_url:
        rest_service = FirebaseRESTService(database_url=db_url)
        if rest_service.initialized:
            logger.info("Using Firebase REST API Service")
            return rest_service
    
    # Fallback to Mock
    logger.info("Falling back to Mock Firebase Service")
    return MockFirebaseService()


if __name__ == "__main__":
    # Example usage with mock service
    print("=" * 50)
    print("Trademify Firebase Service - Example Usage")
    print("=" * 50)
    
    # Use mock service for testing
    firebase = get_firebase_service(mock=True)
    
    # Push a signal
    signal_data = {
        "signal": "STRONG_BUY",
        "confidence": 85.0,
        "vote_details": {
            "bullish": 8,
            "bearish": 2,
            "total": 10
        },
        "price_projection": {
            "current": 1.0850,
            "projected": 1.0900,
            "stop_loss": 1.0800,
            "take_profit": 1.0950
        }
    }
    
    key = firebase.push_signal("EURUSD", "H1", signal_data)
    print(f"\nPushed signal with key: {key}")
    
    # Update current signal
    firebase.update_current_signal("EURUSD", "H1", signal_data)
    
    # Get history
    history = firebase.get_signal_history("EURUSD", "H1")
    print(f"\nSignal history count: {len(history)}")
