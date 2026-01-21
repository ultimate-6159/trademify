"""
Analysis Package
"""
from .voting_system import (
    VotingSystem,
    PatternAnalyzer,
    VoteResult,
    Signal,
    SignalDuration,
    SignalStrength,
    SignalDurationEstimator,
    analyze_patterns
)

from .enhanced_analyzer import (
    EnhancedAnalyzer,
    EnhancedSignalResult,
    TechnicalIndicators,
    VolumeAnalysis,
    MultiTimeframeAnalysis,
    MarketRegime,
    SignalQuality,
    TechnicalIndicatorCalculator,
    get_enhanced_analyzer,
)

__all__ = [
    # Voting System
    "VotingSystem",
    "PatternAnalyzer",
    "VoteResult",
    "Signal",
    "SignalDuration",
    "SignalStrength",
    "SignalDurationEstimator",
    "analyze_patterns",
    # Enhanced Analyzer
    "EnhancedAnalyzer",
    "EnhancedSignalResult",
    "TechnicalIndicators",
    "VolumeAnalysis",
    "MultiTimeframeAnalysis",
    "MarketRegime",
    "SignalQuality",
    "TechnicalIndicatorCalculator",
    "get_enhanced_analyzer",
]
