"""
Trading Settings Configuration
การตั้งค่าสำหรับระบบเทรดอัตโนมัติ
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from pathlib import Path


# Config file path - จะเก็บไว้ในโฟลเดอร์ config
CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "trading_config.json"


class BrokerType(str, Enum):
    """ประเภท Broker"""
    BINANCE = "BINANCE"
    BINANCE_FUTURES = "BINANCE_FUTURES"
    MT5 = "MT5"


@dataclass
class TradingConfig:
    """
    Configuration หลักสำหรับระบบเทรดอัตโนมัติ
    🚨 PRODUCTION: Exness MT5 on Windows VPS only
    """
    # Trading Control
    enabled: bool = True   # เปิดระบบเทรดอัตโนมัติ
    
    # Broker Settings - MT5 only
    broker_type: BrokerType = BrokerType.MT5
    
    # Binance Settings
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = False
    
    # MT5 Settings
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: str = ""
    
    # Risk Management - MATCHED WITH BACKTEST
    max_risk_per_trade: float = 2.0  # % ของ balance (backtest verified)
    max_daily_loss: float = 5.0  # % ของ balance (backtest verified)
    max_positions: int = 5
    max_drawdown: float = 15.0  # % ของ balance (backtest: 14.1%)
    
    # Signal Filtering - MATCHED WITH BACKTEST BALANCED MODE
    min_confidence: float = 70.0  # % ความมั่นใจขั้นต่ำ (backtest: 70%)
    min_quality: str = "HIGH"  # 🥇 BALANCED MODE uses HIGH quality
    # BUY/SELL included for more opportunities
    allowed_signals: List[str] = field(default_factory=lambda: ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL"])
    ignore_entry_timing_for_strong: bool = True  # STRONG signals เทรดทันทีโดยไม่รอ timing
    
    # Quality to confidence mapping
    QUALITY_THRESHOLDS: Dict = field(default_factory=lambda: {
        "PREMIUM": 85.0,
        "HIGH": 75.0,
        "MEDIUM": 65.0,
        "LOW": 50.0,
        "SKIP": 0.0,
    })
    
    def get_min_confidence_from_quality(self) -> float:
        """Get min_confidence based on min_quality setting"""
        return self.QUALITY_THRESHOLDS.get(self.min_quality.upper(), 70.0)
    
    # Position Settings
    default_stop_loss_percent: float = 2.0  # % จากราคาเข้า
    default_take_profit_percent: float = 4.0  # % จากราคาเข้า
    max_holding_hours: Optional[int] = None  # ชั่วโมงสูงสุดที่ถือ
    
    # Trailing Stop
    trailing_stop_enabled: bool = False
    trailing_stop_percent: float = 1.0  # %
    trailing_stop_activation: float = 2.0  # เริ่มทำงานเมื่อ profit ถึง %
    
    # Break-Even
    break_even_enabled: bool = False
    break_even_at_percent: float = 1.0  # ย้าย SL ไป break-even เมื่อ profit ถึง %
    
    # Trading Hours (UTC)
    trading_hours_enabled: bool = False
    trading_start_hour: int = 8
    trading_end_hour: int = 22
    
    # Symbol Filtering
    allowed_symbols: List[str] = field(default_factory=list)  # ว่าง = ทุก symbol
    blocked_symbols: List[str] = field(default_factory=list)
    
    # Pattern Settings (for reference from frontend)
    top_k_patterns: int = 16  # จำนวน patterns ที่หา (5-100)
    min_correlation: float = 0.85  # correlation threshold
    window_size: int = 60  # candles per pattern
    # 🥇 GOLD ONLY - Best performance in backtest (88.7% win rate)
    symbols: List[str] = field(default_factory=lambda: ["XAUUSDm"])  # Gold only
    timeframe: str = "H1"
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "broker_type": self.broker_type.value,
            "risk": {
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_daily_loss": self.max_daily_loss,
                "max_positions": self.max_positions,
                "max_drawdown": self.max_drawdown,
            },
            "signals": {
                "min_confidence": self.get_min_confidence_from_quality(),  # calculated from quality
                "min_quality": self.min_quality,
                "allowed_signals": self.allowed_signals,
            },
            "position": {
                "default_stop_loss_percent": self.default_stop_loss_percent,
                "default_take_profit_percent": self.default_take_profit_percent,
                "max_holding_hours": self.max_holding_hours,
            },
            "trailing_stop": {
                "enabled": self.trailing_stop_enabled,
                "percent": self.trailing_stop_percent,
                "activation": self.trailing_stop_activation,
            },
            "break_even": {
                "enabled": self.break_even_enabled,
                "at_percent": self.break_even_at_percent,
            },
            "trading_hours": {
                "enabled": self.trading_hours_enabled,
                "start_hour": self.trading_start_hour,
                "end_hour": self.trading_end_hour,
            },
            "symbols": {
                "allowed": self.allowed_symbols,
                "blocked": self.blocked_symbols,
            },
            "pattern": {
                "top_k": self.top_k_patterns,
                "min_correlation": self.min_correlation,
                "window_size": self.window_size,
            },
            "trading_symbols": self.symbols,
            "timeframe": self.timeframe,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TradingConfig':
        """สร้าง config จาก dict"""
        config = cls()
        
        config.enabled = data.get("enabled", True)
        
        if "broker_type" in data:
            config.broker_type = BrokerType(data["broker_type"])
        
        # Risk settings
        risk = data.get("risk", {})
        config.max_risk_per_trade = risk.get("max_risk_per_trade", 2.0)
        config.max_daily_loss = risk.get("max_daily_loss", 5.0)
        config.max_positions = risk.get("max_positions", 5)
        config.max_drawdown = risk.get("max_drawdown", 10.0)
        
        # Signal settings
        signals = data.get("signals", {})
        config.min_quality = signals.get("min_quality", "HIGH")
        # min_confidence is auto-calculated from min_quality, but can be overridden
        if "min_confidence" in signals:
            config.min_confidence = signals.get("min_confidence", 70.0)
        else:
            config.min_confidence = config.get_min_confidence_from_quality()
        config.allowed_signals = signals.get("allowed_signals", ["STRONG_BUY", "STRONG_SELL"])
        
        # Position settings
        position = data.get("position", {})
        config.default_stop_loss_percent = position.get("default_stop_loss_percent", 2.0)
        config.default_take_profit_percent = position.get("default_take_profit_percent", 4.0)
        config.max_holding_hours = position.get("max_holding_hours")
        
        # Trailing stop
        ts = data.get("trailing_stop", {})
        config.trailing_stop_enabled = ts.get("enabled", False)
        config.trailing_stop_percent = ts.get("percent", 1.0)
        config.trailing_stop_activation = ts.get("activation", 2.0)
        
        # Break-even
        be = data.get("break_even", {})
        config.break_even_enabled = be.get("enabled", False)
        config.break_even_at_percent = be.get("at_percent", 1.0)
        
        # Trading hours
        th = data.get("trading_hours", {})
        config.trading_hours_enabled = th.get("enabled", False)
        config.trading_start_hour = th.get("start_hour", 8)
        config.trading_end_hour = th.get("end_hour", 22)
        
        # Symbols
        symbols = data.get("symbols", {})
        config.allowed_symbols = symbols.get("allowed", [])
        config.blocked_symbols = symbols.get("blocked", [])
        
        # Pattern settings
        pattern = data.get("pattern", {})
        config.top_k_patterns = pattern.get("top_k", 10)
        config.min_correlation = pattern.get("min_correlation", 0.85)
        config.window_size = pattern.get("window_size", 60)
        
        # Trading symbols and timeframe
        config.symbols = data.get("trading_symbols", ["EURUSD", "GBPUSD", "XAUUSD"])
        config.timeframe = data.get("timeframe", "H1")
        
        return config
    
    def save_to_file(self, filepath: Path = None) -> bool:
        """บันทึก config ลงไฟล์ JSON"""
        filepath = filepath or CONFIG_FILE
        try:
            # สร้างโฟลเดอร์ถ้าไม่มี
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"[Config] Saved to {filepath}")
            return True
        except Exception as e:
            print(f"[Config] Failed to save: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: Path = None) -> 'TradingConfig':
        """โหลด config จากไฟล์ JSON ถ้ามี ไม่มีก็ใช้ค่า default"""
        filepath = filepath or CONFIG_FILE
        try:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"[Config] Loaded from {filepath}")
                return cls.from_dict(data)
            else:
                print(f"[Config] No config file found, using defaults")
                return cls()
        except Exception as e:
            print(f"[Config] Failed to load: {e}, using defaults")
            return cls()


# Default configurations for different trading styles
CONSERVATIVE_CONFIG = TradingConfig(
    max_risk_per_trade=1.0,
    max_daily_loss=3.0,
    max_positions=3,
    min_confidence=80.0,
    allowed_signals=["STRONG_BUY", "STRONG_SELL"],
    trailing_stop_enabled=True,
    trailing_stop_percent=0.5,
    break_even_enabled=True,
    break_even_at_percent=1.0,
)

MODERATE_CONFIG = TradingConfig(
    max_risk_per_trade=2.0,
    max_daily_loss=5.0,
    max_positions=5,
    min_confidence=70.0,
    allowed_signals=["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"],
    trailing_stop_enabled=True,
    trailing_stop_percent=1.0,
)

AGGRESSIVE_CONFIG = TradingConfig(
    max_risk_per_trade=3.0,
    max_daily_loss=10.0,
    max_positions=10,
    min_confidence=60.0,
    allowed_signals=["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"],
)
