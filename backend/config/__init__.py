"""
Configuration package
"""
from .settings import (
    DataConfig,
    PatternConfig,
    NormConfig,
    VotingConfig,
    FirebaseConfig,
    APIConfig,
    RiskConfig,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    INDEX_DIR,
    MODELS_DIR,
)

# ?? High Frequency Trading Config (10-15 trades/day)
from .high_frequency_trading import (
    HighFrequencyConfig,
    TradingFrequencyMode,
    CONSERVATIVE_CONFIG,
    BALANCED_CONFIG,
    ACTIVE_CONFIG,
    HIGH_FREQUENCY_CONFIG,
    AGGRESSIVE_CONFIG,
    get_config_for_mode,
)

__all__ = [
    "DataConfig",
    "PatternConfig",
    "NormConfig",
    "VotingConfig",
    "FirebaseConfig",
    "APIConfig",
    "RiskConfig",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "INDEX_DIR",
    "MODELS_DIR",
    # High Frequency Trading
    "HighFrequencyConfig",
    "TradingFrequencyMode",
    "CONSERVATIVE_CONFIG",
    "BALANCED_CONFIG", 
    "ACTIVE_CONFIG",
    "HIGH_FREQUENCY_CONFIG",
    "AGGRESSIVE_CONFIG",
    "get_config_for_mode",
]
