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

from .sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentSignal,
    SentimentData,
    AggregatedSentiment,
    get_sentiment_analyzer,
    analyze_sentiment,
)

from .smart_money_analyzer import (
    SmartMoneyAnalyzer,
    SmartMoneySignal,
    SmartMoneyAnalysis,
    MarketStructure,
    LiquidityZone,
    OrderBlock,
    FairValueGap,
    get_smart_money_analyzer,
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
    # Sentiment Analyzer (Contrarian)
    "SentimentAnalyzer",
    "SentimentSignal",
    "SentimentData",
    "AggregatedSentiment",
    "get_sentiment_analyzer",
    "analyze_sentiment",
    # Smart Money Analyzer
    "SmartMoneyAnalyzer",
    "SmartMoneySignal",
    "SmartMoneyAnalysis",
    "MarketStructure",
    "LiquidityZone",
    "OrderBlock",
    "FairValueGap",
    "get_smart_money_analyzer",
]
