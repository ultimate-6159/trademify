"""
Enhanced Multi-Factor Analysis Configuration
ตั้งค่าสำหรับระบบวิเคราะห์หลายปัจจัยเพื่อ Win Rate สูงสุด
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class TradingMode(str, Enum):
    """โหมดการเทรด"""
    AGGRESSIVE = "AGGRESSIVE"      # Win Rate ~60%, High Frequency
    BALANCED = "BALANCED"          # Win Rate ~70%, Normal Frequency
    CONSERVATIVE = "CONSERVATIVE"  # Win Rate ~80%, Low Frequency
    SNIPER = "SNIPER"              # Win Rate ~85%+, Very Low Frequency


@dataclass
class TrendFilterConfig:
    """Trend Filter Configuration"""
    enabled: bool = True
    fast_ema: int = 20
    slow_ema: int = 50
    trend_ema: int = 200
    weight: float = 0.20
    
    # Scoring
    aligned_score: float = 1.0      # Signal ไปทางเดียวกับ trend
    neutral_score: float = 0.5      # Sideways market
    counter_score: float = 0.2      # Signal สวน trend (penalize)
    
    @classmethod
    def from_env(cls) -> "TrendFilterConfig":
        return cls(
            enabled=os.getenv("TREND_FILTER_ENABLED", "true").lower() == "true",
            fast_ema=int(os.getenv("TREND_FAST_EMA", "20")),
            slow_ema=int(os.getenv("TREND_SLOW_EMA", "50")),
            trend_ema=int(os.getenv("TREND_EMA", "200")),
            weight=float(os.getenv("TREND_WEIGHT", "0.20")),
        )


@dataclass
class VolumeFilterConfig:
    """Volume Confirmation Configuration"""
    enabled: bool = True
    lookback_period: int = 20
    confirmation_threshold: float = 1.2  # Volume > 1.2x average
    spike_threshold: float = 2.0         # Volume spike detection
    weight: float = 0.15
    
    # OBV settings
    obv_enabled: bool = True
    obv_period: int = 14
    
    @classmethod
    def from_env(cls) -> "VolumeFilterConfig":
        return cls(
            enabled=os.getenv("VOLUME_FILTER_ENABLED", "true").lower() == "true",
            lookback_period=int(os.getenv("VOLUME_LOOKBACK", "20")),
            confirmation_threshold=float(os.getenv("VOLUME_THRESHOLD", "1.2")),
            spike_threshold=float(os.getenv("VOLUME_SPIKE_THRESHOLD", "2.0")),
            weight=float(os.getenv("VOLUME_WEIGHT", "0.15")),
        )


@dataclass
class RecencyFilterConfig:
    """Pattern Recency Weighting - Patterns ล่าสุดมีน้ำหนักมากกว่า"""
    enabled: bool = True
    decay_years: float = 3.0  # Half-life 3 years
    weight: float = 0.10
    
    # Scoring by age
    # 1 year: score = 0.72
    # 3 years: score = 0.37
    # 5 years: score = 0.19
    
    @classmethod
    def from_env(cls) -> "RecencyFilterConfig":
        return cls(
            enabled=os.getenv("RECENCY_FILTER_ENABLED", "true").lower() == "true",
            decay_years=float(os.getenv("RECENCY_DECAY_YEARS", "3.0")),
            weight=float(os.getenv("RECENCY_WEIGHT", "0.10")),
        )


@dataclass
class VolatilityFilterConfig:
    """Volatility Filter - ไม่เทรดเมื่อ volatility ผิดปกติ"""
    enabled: bool = True
    atr_period: int = 14
    min_percentile: float = 20.0   # Skip if ATR < 20th percentile
    max_percentile: float = 90.0   # Skip if ATR > 90th percentile
    optimal_range: tuple = (30.0, 70.0)  # Best volatility range
    weight: float = 0.10
    
    @classmethod
    def from_env(cls) -> "VolatilityFilterConfig":
        return cls(
            enabled=os.getenv("VOLATILITY_FILTER_ENABLED", "true").lower() == "true",
            atr_period=int(os.getenv("ATR_PERIOD", "14")),
            min_percentile=float(os.getenv("VOLATILITY_MIN_PERCENTILE", "20.0")),
            max_percentile=float(os.getenv("VOLATILITY_MAX_PERCENTILE", "90.0")),
            weight=float(os.getenv("VOLATILITY_WEIGHT", "0.10")),
        )


@dataclass
class SessionFilterConfig:
    """Trading Session Filter - เทรดเฉพาะช่วงที่ดี"""
    enabled: bool = True
    timezone: str = "UTC"
    
    # Session times (UTC)
    sessions: Dict = field(default_factory=lambda: {
        "SYDNEY": {"start": "21:00", "end": "06:00", "score": 0.6},
        "TOKYO": {"start": "00:00", "end": "09:00", "score": 0.7},
        "LONDON": {"start": "07:00", "end": "16:00", "score": 1.0},  # Best
        "NEWYORK": {"start": "12:00", "end": "21:00", "score": 0.95},
        "OVERLAP_LN": {"start": "12:00", "end": "16:00", "score": 1.0},  # London-NY overlap
    })
    
    # Avoid periods
    avoid_news_minutes: int = 30     # หยุด 30 นาทีก่อน/หลังข่าว
    weekend_close_hours: int = 4     # ไม่เปิด position 4 ชม.ก่อนปิดตลาด
    monday_open_skip_hours: int = 2  # Skip 2 ชม.แรกวันจันทร์
    
    weight: float = 0.10
    
    @classmethod
    def from_env(cls) -> "SessionFilterConfig":
        return cls(
            enabled=os.getenv("SESSION_FILTER_ENABLED", "true").lower() == "true",
            avoid_news_minutes=int(os.getenv("AVOID_NEWS_MINUTES", "30")),
            weekend_close_hours=int(os.getenv("WEEKEND_CLOSE_HOURS", "4")),
            weight=float(os.getenv("SESSION_WEIGHT", "0.10")),
        )


@dataclass
class MomentumFilterConfig:
    """Momentum Confirmation - RSI, MACD confluence"""
    enabled: bool = True
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    weight: float = 0.10
    
    @classmethod
    def from_env(cls) -> "MomentumFilterConfig":
        return cls(
            enabled=os.getenv("MOMENTUM_FILTER_ENABLED", "true").lower() == "true",
            rsi_period=int(os.getenv("RSI_PERIOD", "14")),
            rsi_overbought=float(os.getenv("RSI_OVERBOUGHT", "70.0")),
            rsi_oversold=float(os.getenv("RSI_OVERSOLD", "30.0")),
            weight=float(os.getenv("MOMENTUM_WEIGHT", "0.10")),
        )


@dataclass
class DrawdownProtectionConfig:
    """Drawdown Protection - หยุดเมื่อขาดทุนมาก"""
    enabled: bool = True
    max_drawdown_percent: float = 10.0
    consecutive_loss_limit: int = 3
    reduce_size_multiplier: float = 0.5  # ลด size 50% หลังขาดทุนติดกัน
    daily_loss_limit: float = 5.0
    daily_profit_target: float = 3.0     # หยุดเมื่อกำไร 3%
    
    @classmethod
    def from_env(cls) -> "DrawdownProtectionConfig":
        return cls(
            enabled=os.getenv("DRAWDOWN_PROTECTION_ENABLED", "true").lower() == "true",
            max_drawdown_percent=float(os.getenv("MAX_DRAWDOWN", "10.0")),
            consecutive_loss_limit=int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3")),
            reduce_size_multiplier=float(os.getenv("REDUCE_SIZE_MULTIPLIER", "0.5")),
            daily_loss_limit=float(os.getenv("MAX_DAILY_LOSS", "5.0")),
            daily_profit_target=float(os.getenv("DAILY_PROFIT_TARGET", "3.0")),
        )


@dataclass
class AdaptiveThresholdConfig:
    """Adaptive Thresholds - ปรับตาม market conditions"""
    enabled: bool = True
    
    # Base thresholds
    base_min_confidence: float = 70.0
    base_strong_confidence: float = 80.0
    base_min_correlation: float = 0.85
    
    # Adjustments based on market regime
    trending_confidence_bonus: float = -5.0   # ลด threshold ใน trend (เทรดง่ายขึ้น)
    ranging_confidence_penalty: float = 10.0  # เพิ่ม threshold ใน range (เทรดยากขึ้น)
    volatile_confidence_penalty: float = 15.0 # เพิ่ม threshold ใน volatile (ระวังมากขึ้น)
    
    @classmethod
    def from_env(cls) -> "AdaptiveThresholdConfig":
        return cls(
            enabled=os.getenv("ADAPTIVE_THRESHOLD_ENABLED", "true").lower() == "true",
            base_min_confidence=float(os.getenv("MIN_CONFIDENCE", "70.0")),
            base_strong_confidence=float(os.getenv("STRONG_CONFIDENCE", "80.0")),
            base_min_correlation=float(os.getenv("MIN_CORRELATION", "0.85")),
        )


@dataclass
class EnhancedAnalysisConfig:
    """Master Configuration for Multi-Factor AI Analysis"""
    
    # Trading mode
    mode: TradingMode = TradingMode.BALANCED
    
    # Base pattern matching weight
    pattern_weight: float = 0.25
    
    # Factor configs
    trend: TrendFilterConfig = field(default_factory=TrendFilterConfig)
    volume: VolumeFilterConfig = field(default_factory=VolumeFilterConfig)
    recency: RecencyFilterConfig = field(default_factory=RecencyFilterConfig)
    volatility: VolatilityFilterConfig = field(default_factory=VolatilityFilterConfig)
    session: SessionFilterConfig = field(default_factory=SessionFilterConfig)
    momentum: MomentumFilterConfig = field(default_factory=MomentumFilterConfig)
    drawdown: DrawdownProtectionConfig = field(default_factory=DrawdownProtectionConfig)
    adaptive: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    
    # Quality thresholds (adjusted by mode)
    quality_thresholds: Dict = field(default_factory=lambda: {
        "PREMIUM": 85,   # >= 85 = Premium quality
        "HIGH": 75,      # >= 75 = High quality
        "MEDIUM": 65,    # >= 65 = Medium quality
        "LOW": 50,       # >= 50 = Low quality (skip in conservative mode)
    })
    
    # Mode-specific settings
    mode_settings: Dict = field(default_factory=lambda: {
        TradingMode.AGGRESSIVE: {
            "min_quality": "LOW",
            "min_final_score": 55.0,
            "strong_signal_score": 70.0,
            "position_size_multiplier": 1.2,
        },
        TradingMode.BALANCED: {
            "min_quality": "MEDIUM",
            "min_final_score": 65.0,
            "strong_signal_score": 80.0,
            "position_size_multiplier": 1.0,
        },
        TradingMode.CONSERVATIVE: {
            "min_quality": "HIGH",
            "min_final_score": 75.0,
            "strong_signal_score": 85.0,
            "position_size_multiplier": 0.8,
        },
        TradingMode.SNIPER: {
            "min_quality": "PREMIUM",
            "min_final_score": 85.0,
            "strong_signal_score": 90.0,
            "position_size_multiplier": 1.5,  # Higher size for high-confidence trades
        },
    })
    
    @classmethod
    def from_env(cls) -> "EnhancedAnalysisConfig":
        """Load configuration from environment variables"""
        mode_str = os.getenv("TRADING_MODE", "BALANCED").upper()
        try:
            mode = TradingMode(mode_str)
        except ValueError:
            mode = TradingMode.BALANCED
        
        return cls(
            mode=mode,
            pattern_weight=float(os.getenv("PATTERN_WEIGHT", "0.25")),
            trend=TrendFilterConfig.from_env(),
            volume=VolumeFilterConfig.from_env(),
            recency=RecencyFilterConfig.from_env(),
            volatility=VolatilityFilterConfig.from_env(),
            session=SessionFilterConfig.from_env(),
            momentum=MomentumFilterConfig.from_env(),
            drawdown=DrawdownProtectionConfig.from_env(),
            adaptive=AdaptiveThresholdConfig.from_env(),
        )
    
    def get_mode_setting(self, key: str):
        """Get setting for current trading mode"""
        return self.mode_settings[self.mode].get(key)
    
    def get_min_final_score(self) -> float:
        """Get minimum final score for current mode"""
        return self.mode_settings[self.mode]["min_final_score"]
    
    def get_strong_signal_score(self) -> float:
        """Get strong signal score for current mode"""
        return self.mode_settings[self.mode]["strong_signal_score"]
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier for current mode"""
        return self.mode_settings[self.mode]["position_size_multiplier"]
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "mode": self.mode.value,
            "pattern_weight": self.pattern_weight,
            "filters": {
                "trend": {
                    "enabled": self.trend.enabled,
                    "weight": self.trend.weight,
                    "fast_ema": self.trend.fast_ema,
                    "slow_ema": self.trend.slow_ema,
                },
                "volume": {
                    "enabled": self.volume.enabled,
                    "weight": self.volume.weight,
                    "threshold": self.volume.confirmation_threshold,
                },
                "recency": {
                    "enabled": self.recency.enabled,
                    "weight": self.recency.weight,
                    "decay_years": self.recency.decay_years,
                },
                "volatility": {
                    "enabled": self.volatility.enabled,
                    "weight": self.volatility.weight,
                },
                "session": {
                    "enabled": self.session.enabled,
                    "weight": self.session.weight,
                },
                "momentum": {
                    "enabled": self.momentum.enabled,
                    "weight": self.momentum.weight,
                },
            },
            "thresholds": {
                "min_final_score": self.get_min_final_score(),
                "strong_signal_score": self.get_strong_signal_score(),
            },
            "quality_thresholds": self.quality_thresholds,
        }
