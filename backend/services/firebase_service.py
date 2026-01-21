"""
Firebase Service - The Glue
รับผลโหวตจาก Python ส่งไปหน้าเว็บ

Firebase Realtime Database สำหรับ real-time sync
"""
import os
import json
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


class MockFirebaseService:
    """
    Mock Firebase Service for development/testing
    ใช้ตอนที่ไม่มี Firebase จริง
    """
    
    def __init__(self):
        self.signals = {}
        self.patterns = {}
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


def get_firebase_service(mock: bool = False) -> FirebaseService:
    """
    Factory function to get appropriate Firebase service
    
    Args:
        mock: Whether to use mock service
    
    Returns:
        Firebase service instance
    """
    if mock or not FIREBASE_AVAILABLE:
        return MockFirebaseService()
    
    service = FirebaseService()
    if not service.initialized:
        logger.info("Falling back to Mock Firebase Service")
        return MockFirebaseService()
    
    return service


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
