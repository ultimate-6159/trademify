"""
🏛️⚔️ TITAN CORE - Ultimate Meta-Intelligence Synthesis
========================================================

The apex of trading AI - synthesizes all intelligence modules:
- Meta-Learning Ensemble (combine all module insights)
- Confidence Calibrator (adjust based on historical accuracy)
- Dynamic Weight Optimizer (learn optimal module weights)
- Consensus Engine (when modules agree = high confidence)
- Prediction Ensemble (combine multiple predictions)
- Self-Improvement Loop (learn from mistakes)

This is the FINAL BOSS of trading intelligence.
"""
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import json
import hashlib

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATA CLASSES
# ============================================================

class TitanGrade(Enum):
    """Titan Core ultimate grade"""
    TITAN_SUPREME = "🏛️ TITAN SUPREME"   # Perfect consensus
    TITAN_ELITE = "⚔️ TITAN ELITE"       # Excellent alignment
    TITAN_PRIME = "🔱 TITAN PRIME"       # Very strong
    TITAN_CORE = "💎 TITAN CORE"         # Strong
    TITAN_BASE = "🛡️ TITAN BASE"         # Moderate
    MORTAL = "👤 MORTAL"                 # Weak
    REJECT = "❌ REJECT"                 # Do not trade


class ConsensusLevel(Enum):
    """Module consensus level"""
    UNANIMOUS = "unanimous"        # All modules agree
    STRONG = "strong"              # 80%+ agree
    MODERATE = "moderate"          # 60%+ agree
    WEAK = "weak"                  # 40%+ agree
    CONFLICT = "conflict"          # Less than 40% agree


class PredictionMethod(Enum):
    """Prediction methods"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    PATTERN_BASED = "pattern_based"
    ENSEMBLE = "ensemble"


class MarketCondition(Enum):
    """Overall market condition"""
    HIGHLY_FAVORABLE = "highly_favorable"
    FAVORABLE = "favorable"
    NEUTRAL = "neutral"
    UNFAVORABLE = "unfavorable"
    HIGHLY_UNFAVORABLE = "highly_unfavorable"


@dataclass
class ModuleSignal:
    """Signal from an intelligence module"""
    module_name: str
    should_trade: bool
    direction: str          # "BUY" or "SELL"
    confidence: float       # 0-100
    multiplier: float       # Position size multiplier
    score: float            # Module-specific score
    reasons: List[str]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """Result of consensus analysis"""
    level: ConsensusLevel
    agreement_ratio: float          # 0-1
    bullish_modules: List[str]
    bearish_modules: List[str]
    neutral_modules: List[str]
    blocking_modules: List[str]     # Modules that blocked trade
    dominant_direction: str
    confidence_boost: float         # Boost from consensus


@dataclass
class PredictionEnsemble:
    """Ensemble of predictions"""
    final_prediction: str           # "UP", "DOWN", "NEUTRAL"
    predicted_move: float           # Expected price move %
    prediction_confidence: float    # 0-100
    method_predictions: Dict[str, float]  # Method -> predicted move
    method_weights: Dict[str, float]      # Method -> weight
    time_horizon_bars: int


@dataclass
class CalibrationData:
    """Confidence calibration data"""
    raw_confidence: float
    calibrated_confidence: float
    historical_accuracy: float      # Based on past predictions
    adjustment_factor: float
    sample_size: int               # Number of historical samples


@dataclass
class ModuleWeight:
    """Dynamic module weight"""
    module_name: str
    current_weight: float
    historical_accuracy: float
    recent_performance: float       # Last N trades
    adjustment_momentum: float      # How fast weight is changing


@dataclass
class SelfImprovementInsight:
    """Insight from self-improvement analysis"""
    insight_type: str
    description: str
    suggested_action: str
    impact_score: float            # 0-100
    confidence: float


@dataclass
class TitanDecision:
    """Final Titan Core decision"""
    should_trade: bool
    direction: str
    grade: TitanGrade
    titan_score: float              # 0-100
    confidence: float
    position_multiplier: float
    
    # Synthesis results
    consensus: ConsensusResult
    prediction: PredictionEnsemble
    calibration: CalibrationData
    module_weights: Dict[str, float]
    market_condition: MarketCondition
    
    # Aggregated signals
    total_modules: int
    agreeing_modules: int
    blocking_modules: int
    
    # Entry/Exit
    optimal_entry: float
    stop_loss: float
    targets: List[float]
    risk_reward: float
    max_risk_percent: float
    
    # Meta-analysis
    edge_factors: List[str]
    risk_factors: List[str]
    improvement_insights: List[SelfImprovementInsight]
    final_verdict: str
    
    # Performance tracking
    decision_id: str
    module_signals: List[ModuleSignal]


# ============================================================
# CONSENSUS ENGINE
# ============================================================

class ConsensusEngine:
    """
    Analyze consensus across all intelligence modules.
    High consensus = high confidence trade.
    """
    
    def __init__(self):
        self.consensus_history = deque(maxlen=200)
    
    def analyze(
        self,
        signals: List[ModuleSignal],
        direction: str
    ) -> ConsensusResult:
        """Analyze module consensus"""
        
        if not signals:
            return ConsensusResult(
                level=ConsensusLevel.CONFLICT,
                agreement_ratio=0,
                bullish_modules=[],
                bearish_modules=[],
                neutral_modules=[],
                blocking_modules=[],
                dominant_direction="NEUTRAL",
                confidence_boost=0
            )
        
        bullish = []
        bearish = []
        neutral = []
        blocking = []
        
        for sig in signals:
            if not sig.should_trade:
                blocking.append(sig.module_name)
            elif sig.direction == "BUY":
                bullish.append(sig.module_name)
            elif sig.direction == "SELL":
                bearish.append(sig.module_name)
            else:
                neutral.append(sig.module_name)
        
        total_tradeable = len(bullish) + len(bearish) + len(neutral)
        
        # Determine dominant direction
        if len(bullish) > len(bearish):
            dominant = "BUY"
            agreeing = len(bullish)
        elif len(bearish) > len(bullish):
            dominant = "SELL"
            agreeing = len(bearish)
        else:
            dominant = direction  # Use requested direction as tiebreaker
            agreeing = len(bullish) if direction == "BUY" else len(bearish)
        
        # Calculate agreement ratio
        agreement_ratio = agreeing / max(total_tradeable, 1)
        
        # Determine consensus level
        if agreement_ratio >= 0.9 and len(blocking) == 0:
            level = ConsensusLevel.UNANIMOUS
            confidence_boost = 25
        elif agreement_ratio >= 0.8:
            level = ConsensusLevel.STRONG
            confidence_boost = 15
        elif agreement_ratio >= 0.6:
            level = ConsensusLevel.MODERATE
            confidence_boost = 5
        elif agreement_ratio >= 0.4:
            level = ConsensusLevel.WEAK
            confidence_boost = 0
        else:
            level = ConsensusLevel.CONFLICT
            confidence_boost = -10
        
        # Reduce boost if there are blocking modules
        if blocking:
            confidence_boost -= len(blocking) * 5
        
        result = ConsensusResult(
            level=level,
            agreement_ratio=agreement_ratio,
            bullish_modules=bullish,
            bearish_modules=bearish,
            neutral_modules=neutral,
            blocking_modules=blocking,
            dominant_direction=dominant,
            confidence_boost=confidence_boost
        )
        
        self.consensus_history.append({
            "level": level.value,
            "ratio": agreement_ratio,
            "timestamp": datetime.now().isoformat()
        })
        
        return result


# ============================================================
# PREDICTION ENSEMBLE
# ============================================================

class PredictionEnsembleEngine:
    """
    Combine multiple prediction methods for robust forecasting.
    """
    
    def __init__(self):
        self.method_accuracies = {
            PredictionMethod.MOMENTUM: 0.55,
            PredictionMethod.MEAN_REVERSION: 0.52,
            PredictionMethod.TREND_FOLLOWING: 0.58,
            PredictionMethod.PATTERN_BASED: 0.54,
        }
        self.prediction_history = deque(maxlen=500)
    
    def predict(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        horizon_bars: int = 5
    ) -> PredictionEnsemble:
        """Generate ensemble prediction"""
        
        current_price = closes[-1]
        predictions = {}
        
        # 1. Momentum prediction
        if len(closes) >= 20:
            momentum = (closes[-1] - closes[-10]) / closes[-10]
            # Project forward
            predictions[PredictionMethod.MOMENTUM] = momentum * horizon_bars / 10
        else:
            predictions[PredictionMethod.MOMENTUM] = 0
        
        # 2. Mean reversion prediction
        if len(closes) >= 50:
            ma50 = np.mean(closes[-50:])
            deviation = (closes[-1] - ma50) / ma50
            # Expect reversion
            predictions[PredictionMethod.MEAN_REVERSION] = -deviation * 0.5
        else:
            predictions[PredictionMethod.MEAN_REVERSION] = 0
        
        # 3. Trend following
        if len(closes) >= 20:
            ma20 = np.mean(closes[-20:])
            ma10 = np.mean(closes[-10:])
            trend_strength = (ma10 - ma20) / ma20
            predictions[PredictionMethod.TREND_FOLLOWING] = trend_strength * horizon_bars
        else:
            predictions[PredictionMethod.TREND_FOLLOWING] = 0
        
        # 4. Pattern-based (simplified)
        if len(closes) >= 30:
            # Look for recent patterns
            recent_high = np.max(highs[-20:])
            recent_low = np.min(lows[-20:])
            price_position = (closes[-1] - recent_low) / (recent_high - recent_low + 1e-10)
            
            if price_position > 0.8:
                predictions[PredictionMethod.PATTERN_BASED] = -0.01  # Near high, expect pullback
            elif price_position < 0.2:
                predictions[PredictionMethod.PATTERN_BASED] = 0.01   # Near low, expect bounce
            else:
                predictions[PredictionMethod.PATTERN_BASED] = 0
        else:
            predictions[PredictionMethod.PATTERN_BASED] = 0
        
        # Calculate weights based on accuracy
        total_accuracy = sum(self.method_accuracies.values())
        weights = {
            method: acc / total_accuracy
            for method, acc in self.method_accuracies.items()
        }
        
        # Weighted ensemble
        final_move = sum(
            predictions[method] * weights[method]
            for method in predictions
        )
        
        # Determine direction
        if final_move > 0.001:
            direction = "UP"
        elif final_move < -0.001:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        # Confidence based on agreement
        pred_signs = [1 if p > 0 else -1 if p < 0 else 0 for p in predictions.values()]
        agreement = abs(sum(pred_signs)) / len(pred_signs)
        confidence = 40 + agreement * 40 + abs(final_move) * 1000
        confidence = min(95, confidence)
        
        return PredictionEnsemble(
            final_prediction=direction,
            predicted_move=final_move * 100,  # As percentage
            prediction_confidence=confidence,
            method_predictions={m.value: p * 100 for m, p in predictions.items()},
            method_weights={m.value: w for m, w in weights.items()},
            time_horizon_bars=horizon_bars
        )


# ============================================================
# CONFIDENCE CALIBRATOR
# ============================================================

class ConfidenceCalibrator:
    """
    Calibrate confidence based on historical accuracy.
    If we've been overconfident, reduce. If underconfident, increase.
    """
    
    def __init__(self, calibration_file: str = "data/calibration.json"):
        self.calibration_file = calibration_file
        self.prediction_outcomes: deque = deque(maxlen=1000)
        self.calibration_curve = {}  # raw_confidence -> actual_accuracy
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration data from file"""
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                self.calibration_curve = data.get("curve", {})
        except:
            # Default curve (slightly conservative)
            self.calibration_curve = {
                "90": 75,
                "80": 68,
                "70": 60,
                "60": 52,
                "50": 48,
            }
    
    def _save_calibration(self):
        """Save calibration data"""
        try:
            import os
            os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
            with open(self.calibration_file, 'w') as f:
                json.dump({"curve": self.calibration_curve}, f)
        except:
            pass
    
    def calibrate(self, raw_confidence: float) -> CalibrationData:
        """Calibrate raw confidence"""
        
        # Find nearest calibration points
        conf_key = str(int(raw_confidence / 10) * 10)
        
        if conf_key in self.calibration_curve:
            actual_accuracy = self.calibration_curve[conf_key]
            # Apply calibration adjustment
            adjustment_factor = actual_accuracy / max(raw_confidence, 1)
            calibrated = raw_confidence * adjustment_factor
        else:
            # 🔧 FIX: No calibration data - use raw confidence directly
            # Don't penalize confidence when we don't have historical data
            actual_accuracy = raw_confidence
            adjustment_factor = 1.0
            calibrated = raw_confidence  # Use raw confidence as-is
        
        calibrated = np.clip(calibrated, 0, 95)
        
        # Calculate historical accuracy from outcomes
        if self.prediction_outcomes:
            correct = sum(1 for o in self.prediction_outcomes if o["correct"])
            historical_accuracy = correct / len(self.prediction_outcomes) * 100
        else:
            historical_accuracy = 50
        
        return CalibrationData(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            historical_accuracy=historical_accuracy,
            adjustment_factor=adjustment_factor,
            sample_size=len(self.prediction_outcomes)
        )
    
    def record_outcome(self, confidence: float, was_correct: bool):
        """Record prediction outcome for future calibration"""
        self.prediction_outcomes.append({
            "confidence": confidence,
            "correct": was_correct,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update calibration curve periodically
        if len(self.prediction_outcomes) % 50 == 0:
            self._update_curve()
    
    def _update_curve(self):
        """Update calibration curve based on outcomes"""
        # Group by confidence bucket
        buckets = {}
        for outcome in self.prediction_outcomes:
            bucket = str(int(outcome["confidence"] / 10) * 10)
            if bucket not in buckets:
                buckets[bucket] = {"correct": 0, "total": 0}
            buckets[bucket]["total"] += 1
            if outcome["correct"]:
                buckets[bucket]["correct"] += 1
        
        # Calculate actual accuracy per bucket
        for bucket, data in buckets.items():
            if data["total"] >= 10:  # Minimum samples
                accuracy = data["correct"] / data["total"] * 100
                self.calibration_curve[bucket] = accuracy
        
        self._save_calibration()


# ============================================================
# DYNAMIC WEIGHT OPTIMIZER
# ============================================================

class DynamicWeightOptimizer:
    """
    Learn optimal weights for each intelligence module
    based on their historical performance.
    """
    
    def __init__(self):
        self.module_performances: Dict[str, deque] = {}
        self.base_weights = {
            "RiskGuardian": 1.0,
            "ProFeatures": 0.9,
            "SmartBrain": 1.0,
            "Intelligence": 0.95,
            "Learning": 0.85,
            "NeuralBrain": 0.9,
            "DeepIntelligence": 0.95,
            "QuantumStrategy": 0.9,
            "AlphaEngine": 0.95,
            "OmegaBrain": 1.0,
        }
        self.current_weights = self.base_weights.copy()
        self.learning_rate = 0.1
    
    def get_weights(self) -> Dict[str, ModuleWeight]:
        """Get current module weights"""
        weights = {}
        
        for module_name, base_weight in self.base_weights.items():
            perf_history = self.module_performances.get(module_name, deque(maxlen=100))
            
            if perf_history:
                recent = list(perf_history)[-20:]
                recent_perf = sum(recent) / len(recent)
                historical_acc = sum(perf_history) / len(perf_history)
            else:
                recent_perf = 0.5
                historical_acc = 0.5
            
            current = self.current_weights.get(module_name, base_weight)
            
            # Calculate adjustment momentum
            momentum = (recent_perf - 0.5) * self.learning_rate
            
            weights[module_name] = ModuleWeight(
                module_name=module_name,
                current_weight=current,
                historical_accuracy=historical_acc,
                recent_performance=recent_perf,
                adjustment_momentum=momentum
            )
        
        return weights
    
    def record_performance(self, module_name: str, was_correct: bool):
        """Record module performance"""
        if module_name not in self.module_performances:
            self.module_performances[module_name] = deque(maxlen=100)
        
        self.module_performances[module_name].append(1.0 if was_correct else 0.0)
        
        # Update weight
        perf = self.module_performances[module_name]
        if len(perf) >= 10:
            recent_accuracy = sum(list(perf)[-10:]) / 10
            base = self.base_weights.get(module_name, 1.0)
            
            # Adjust weight based on performance
            if recent_accuracy > 0.6:
                self.current_weights[module_name] = min(1.2, base * (1 + (recent_accuracy - 0.5) * 0.5))
            elif recent_accuracy < 0.4:
                self.current_weights[module_name] = max(0.5, base * (1 - (0.5 - recent_accuracy) * 0.5))
            else:
                # Regression to base
                self.current_weights[module_name] = base * 0.9 + self.current_weights.get(module_name, base) * 0.1
    
    def get_module_weight(self, module_name: str) -> float:
        """Get weight for specific module"""
        return self.current_weights.get(module_name, 1.0)


# ============================================================
# SELF-IMPROVEMENT ENGINE
# ============================================================

class SelfImprovementEngine:
    """
    Learn from mistakes and generate improvement insights.
    The bot gets smarter over time.
    """
    
    def __init__(self):
        self.trade_outcomes: deque = deque(maxlen=500)
        self.pattern_library: Dict[str, Dict] = {}
        self.insights_generated = 0
    
    def analyze(
        self,
        current_setup: Dict[str, Any],
        signals: List[ModuleSignal]
    ) -> List[SelfImprovementInsight]:
        """Generate improvement insights"""
        
        insights = []
        
        # 1. Check for common failure patterns
        blocking_count = sum(1 for s in signals if not s.should_trade)
        
        if blocking_count > len(signals) * 0.5:
            insights.append(SelfImprovementInsight(
                insight_type="high_blocking",
                description=f"{blocking_count}/{len(signals)} modules blocking",
                suggested_action="Consider relaxing filter thresholds or waiting for better setup",
                impact_score=70,
                confidence=80
            ))
        
        # 2. Check for low consensus
        directions = [s.direction for s in signals if s.should_trade]
        if directions:
            buy_ratio = directions.count("BUY") / len(directions)
            if 0.4 < buy_ratio < 0.6:
                insights.append(SelfImprovementInsight(
                    insight_type="low_consensus",
                    description="Modules have conflicting signals",
                    suggested_action="Wait for clearer direction or reduce position size",
                    impact_score=60,
                    confidence=75
                ))
        
        # 3. Check confidence distribution
        confidences = [s.confidence for s in signals]
        if confidences:
            avg_conf = np.mean(confidences)
            conf_std = np.std(confidences)
            
            if conf_std > 20:
                insights.append(SelfImprovementInsight(
                    insight_type="confidence_divergence",
                    description=f"Large confidence spread (std={conf_std:.1f})",
                    suggested_action="Investigate why modules disagree on confidence",
                    impact_score=50,
                    confidence=70
                ))
            
            if avg_conf < 50:
                insights.append(SelfImprovementInsight(
                    insight_type="low_confidence",
                    description=f"Average confidence is low ({avg_conf:.1f}%)",
                    suggested_action="Consider skipping this trade",
                    impact_score=65,
                    confidence=85
                ))
        
        # 4. Check for warning accumulation
        total_warnings = sum(len(s.warnings) for s in signals)
        if total_warnings > 5:
            insights.append(SelfImprovementInsight(
                insight_type="warning_accumulation",
                description=f"High warning count: {total_warnings}",
                suggested_action="Review all warnings before proceeding",
                impact_score=55,
                confidence=80
            ))
        
        # 5. Historical pattern matching (simplified)
        setup_hash = self._hash_setup(current_setup)
        if setup_hash in self.pattern_library:
            past_data = self.pattern_library[setup_hash]
            if past_data["losses"] > past_data["wins"]:
                insights.append(SelfImprovementInsight(
                    insight_type="negative_history",
                    description=f"Similar setup has {past_data['losses']}/{past_data['wins']+past_data['losses']} losses",
                    suggested_action="Extra caution recommended",
                    impact_score=75,
                    confidence=65
                ))
        
        self.insights_generated += len(insights)
        return insights
    
    def _hash_setup(self, setup: Dict) -> str:
        """Create hash of setup for pattern matching"""
        key_features = {
            "direction": setup.get("direction"),
            "regime": setup.get("regime"),
            "consensus_level": setup.get("consensus_level"),
        }
        return hashlib.md5(json.dumps(key_features, sort_keys=True).encode()).hexdigest()[:8]
    
    def record_outcome(self, setup: Dict, was_profitable: bool):
        """Record trade outcome for learning"""
        setup_hash = self._hash_setup(setup)
        
        if setup_hash not in self.pattern_library:
            self.pattern_library[setup_hash] = {"wins": 0, "losses": 0}
        
        if was_profitable:
            self.pattern_library[setup_hash]["wins"] += 1
        else:
            self.pattern_library[setup_hash]["losses"] += 1
        
        self.trade_outcomes.append({
            "hash": setup_hash,
            "profitable": was_profitable,
            "timestamp": datetime.now().isoformat()
        })


# ============================================================
# MARKET CONDITION ANALYZER
# ============================================================

class MarketConditionAnalyzer:
    """
    Assess overall market condition for trading.
    """
    
    def analyze(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        consensus: ConsensusResult,
        prediction: PredictionEnsemble
    ) -> MarketCondition:
        """Analyze market condition"""
        
        score = 50  # Base score
        
        # 1. Trend clarity
        if len(closes) >= 50:
            ma20 = np.mean(closes[-20:])
            ma50 = np.mean(closes[-50:])
            trend_clarity = abs(ma20 - ma50) / ma50
            score += trend_clarity * 500  # Max +10 points
        
        # 2. Consensus level
        consensus_bonus = {
            ConsensusLevel.UNANIMOUS: 20,
            ConsensusLevel.STRONG: 15,
            ConsensusLevel.MODERATE: 5,
            ConsensusLevel.WEAK: -5,
            ConsensusLevel.CONFLICT: -15,
        }
        score += consensus_bonus.get(consensus.level, 0)
        
        # 3. Prediction confidence
        score += (prediction.prediction_confidence - 50) * 0.3
        
        # 4. Volume health
        if len(volumes) >= 20:
            vol_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-20:])
            if vol_ratio > 1.2:
                score += 5  # Good volume
            elif vol_ratio < 0.8:
                score -= 5  # Low volume
        
        # 5. Volatility check
        if len(closes) >= 20:
            returns = np.diff(np.log(closes[-21:]))
            volatility = np.std(returns)
            if volatility > 0.02:  # High volatility
                score -= 10
            elif volatility < 0.005:  # Very low volatility
                score -= 5
        
        # Classify condition
        if score >= 75:
            return MarketCondition.HIGHLY_FAVORABLE
        elif score >= 60:
            return MarketCondition.FAVORABLE
        elif score >= 40:
            return MarketCondition.NEUTRAL
        elif score >= 25:
            return MarketCondition.UNFAVORABLE
        else:
            return MarketCondition.HIGHLY_UNFAVORABLE


# ============================================================
# TITAN CORE - MAIN CLASS
# ============================================================

class TitanCore:
    """
    🏛️⚔️ TITAN CORE - Ultimate Meta-Intelligence Synthesis
    
    The apex of trading AI that synthesizes all intelligence:
    - Meta-Learning Ensemble
    - Confidence Calibrator
    - Dynamic Weight Optimizer
    - Consensus Engine
    - Prediction Ensemble
    - Self-Improvement Loop
    
    This is the FINAL BOSS of trading intelligence.
    """
    
    def __init__(
        self,
        min_titan_score: float = 60.0,
        min_consensus: ConsensusLevel = ConsensusLevel.WEAK,
        enable_calibration: bool = True,
        enable_self_improvement: bool = True
    ):
        self.min_titan_score = min_titan_score
        self.min_consensus = min_consensus
        self.enable_calibration = enable_calibration
        self.enable_self_improvement = enable_self_improvement
        
        # Initialize components
        self.consensus_engine = ConsensusEngine()
        self.prediction_engine = PredictionEnsembleEngine()
        self.calibrator = ConfidenceCalibrator()
        self.weight_optimizer = DynamicWeightOptimizer()
        self.improvement_engine = SelfImprovementEngine()
        self.condition_analyzer = MarketConditionAnalyzer()
        
        # Tracking
        self.decision_count = 0
        self.decision_history: deque = deque(maxlen=1000)
        
        logger.info("🏛️⚔️ TITAN CORE initialized")
        logger.info("   - Consensus Engine: ✓")
        logger.info("   - Prediction Ensemble: ✓")
        logger.info("   - Confidence Calibrator: ✓")
        logger.info("   - Dynamic Weight Optimizer: ✓")
        logger.info("   - Self-Improvement Engine: ✓")
        logger.info("   - Market Condition Analyzer: ✓")
    
    def synthesize(
        self,
        symbol: str,
        signal_direction: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        module_signals: List[ModuleSignal],
        current_price: float = None
    ) -> TitanDecision:
        """
        Synthesize all module signals into final Titan decision.
        This is the ultimate trading decision.
        """
        
        if current_price is None:
            current_price = closes[-1]
        
        self.decision_count += 1
        decision_id = f"TITAN-{self.decision_count:06d}"
        
        edge_factors = []
        risk_factors = []
        
        # 1. Consensus Analysis
        consensus = self.consensus_engine.analyze(module_signals, signal_direction)
        
        if consensus.level == ConsensusLevel.UNANIMOUS:
            edge_factors.append("🏛️ Unanimous module consensus!")
        elif consensus.level == ConsensusLevel.STRONG:
            edge_factors.append(f"⚔️ Strong consensus ({consensus.agreement_ratio:.0%})")
        elif consensus.level == ConsensusLevel.CONFLICT:
            risk_factors.append(f"❌ Module conflict ({consensus.agreement_ratio:.0%} agreement)")
        
        if consensus.blocking_modules:
            risk_factors.append(f"🚫 {len(consensus.blocking_modules)} module(s) blocking")
        
        # 2. Prediction Ensemble
        prediction = self.prediction_engine.predict(closes, highs, lows, volumes)
        
        pred_aligned = (
            (signal_direction == "BUY" and prediction.final_prediction == "UP") or
            (signal_direction == "SELL" and prediction.final_prediction == "DOWN")
        )
        
        if pred_aligned:
            edge_factors.append(f"📈 Ensemble predicts {prediction.predicted_move:+.2f}% move")
        elif prediction.final_prediction != "NEUTRAL":
            risk_factors.append(f"📉 Ensemble predicts opposite: {prediction.final_prediction}")
        
        # 3. Calculate raw confidence
        if module_signals:
            tradeable_signals = [s for s in module_signals if s.should_trade]
            if tradeable_signals:
                # Weighted average confidence
                weights = self.weight_optimizer.get_weights()
                total_weight = 0
                weighted_conf = 0
                
                for sig in tradeable_signals:
                    w = weights.get(sig.module_name, ModuleWeight(sig.module_name, 1.0, 0.5, 0.5, 0)).current_weight
                    weighted_conf += sig.confidence * w
                    total_weight += w
                
                raw_confidence = weighted_conf / total_weight if total_weight > 0 else 50
            else:
                raw_confidence = 30
        else:
            raw_confidence = 50
        
        # Add consensus boost
        raw_confidence += consensus.confidence_boost
        raw_confidence = np.clip(raw_confidence, 0, 100)
        
        # 4. Calibrate confidence
        if self.enable_calibration:
            calibration = self.calibrator.calibrate(raw_confidence)
        else:
            calibration = CalibrationData(
                raw_confidence=raw_confidence,
                calibrated_confidence=raw_confidence,
                historical_accuracy=50,
                adjustment_factor=1.0,
                sample_size=0
            )
        
        # 5. Market condition
        market_condition = self.condition_analyzer.analyze(
            closes, volumes, consensus, prediction
        )
        
        condition_multiplier = {
            MarketCondition.HIGHLY_FAVORABLE: 1.2,
            MarketCondition.FAVORABLE: 1.0,
            MarketCondition.NEUTRAL: 0.8,
            MarketCondition.UNFAVORABLE: 0.5,
            MarketCondition.HIGHLY_UNFAVORABLE: 0.3,
        }
        
        if market_condition in [MarketCondition.HIGHLY_FAVORABLE, MarketCondition.FAVORABLE]:
            edge_factors.append(f"🌟 {market_condition.value} market condition")
        elif market_condition in [MarketCondition.UNFAVORABLE, MarketCondition.HIGHLY_UNFAVORABLE]:
            risk_factors.append(f"⚠️ {market_condition.value} market condition")
        
        # 6. Calculate Titan Score
        titan_score = self._calculate_titan_score(
            consensus=consensus,
            prediction=prediction,
            calibration=calibration,
            market_condition=market_condition,
            module_signals=module_signals
        )
        
        # 7. Determine grade
        grade = self._determine_grade(
            titan_score=titan_score,
            consensus=consensus,
            edge_count=len(edge_factors),
            risk_count=len(risk_factors)
        )
        
        # 8. Self-improvement insights
        if self.enable_self_improvement:
            setup_info = {
                "direction": signal_direction,
                "regime": market_condition.value,
                "consensus_level": consensus.level.value,
            }
            insights = self.improvement_engine.analyze(setup_info, module_signals)
        else:
            insights = []
        
        # 9. Final decision
        consensus_ok = self._consensus_meets_minimum(consensus.level)
        
        should_trade = (
            titan_score >= self.min_titan_score and
            consensus_ok and
            grade not in [TitanGrade.MORTAL, TitanGrade.REJECT] and
            len(consensus.blocking_modules) < 3
        )
        
        # 10. Position multiplier
        grade_multipliers = {
            TitanGrade.TITAN_SUPREME: 1.0,
            TitanGrade.TITAN_ELITE: 0.9,
            TitanGrade.TITAN_PRIME: 0.8,
            TitanGrade.TITAN_CORE: 0.7,
            TitanGrade.TITAN_BASE: 0.5,
            TitanGrade.MORTAL: 0.3,
            TitanGrade.REJECT: 0.0,
        }
        
        position_multiplier = grade_multipliers.get(grade, 0.5)
        position_multiplier *= condition_multiplier.get(market_condition, 0.8)
        position_multiplier = min(1.0, position_multiplier)
        
        # 11. Entry/Exit calculation
        atr = np.mean(highs[-20:] - lows[-20:]) if len(highs) >= 20 else current_price * 0.01
        
        if signal_direction == "BUY":
            optimal_entry = current_price - atr * 0.2
            stop_loss = current_price - atr * 2.0
            targets = [
                current_price + atr * 2.0,
                current_price + atr * 3.5,
                current_price + atr * 5.0,
            ]
        else:
            optimal_entry = current_price + atr * 0.2
            stop_loss = current_price + atr * 2.0
            targets = [
                current_price - atr * 2.0,
                current_price - atr * 3.5,
                current_price - atr * 5.0,
            ]
        
        risk = abs(optimal_entry - stop_loss)
        reward = abs(targets[0] - optimal_entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        # 12. Get module weights
        module_weights = {
            mw.module_name: mw.current_weight
            for mw in self.weight_optimizer.get_weights().values()
        }
        
        # 13. Final verdict
        if grade == TitanGrade.TITAN_SUPREME:
            verdict = f"🏛️ TITAN SUPREME: {signal_direction} with maximum conviction!"
        elif grade == TitanGrade.TITAN_ELITE:
            verdict = f"⚔️ TITAN ELITE: Strong {signal_direction} recommended"
        elif grade == TitanGrade.TITAN_PRIME:
            verdict = f"🔱 TITAN PRIME: {signal_direction} with good probability"
        elif grade == TitanGrade.TITAN_CORE:
            verdict = f"💎 TITAN CORE: {signal_direction} acceptable"
        elif grade == TitanGrade.TITAN_BASE:
            verdict = f"🛡️ TITAN BASE: Weak {signal_direction} - reduce size"
        else:
            verdict = f"❌ REJECT: Do not trade this setup"
        
        decision = TitanDecision(
            should_trade=should_trade,
            direction=signal_direction,
            grade=grade,
            titan_score=titan_score,
            confidence=calibration.calibrated_confidence,
            position_multiplier=position_multiplier,
            consensus=consensus,
            prediction=prediction,
            calibration=calibration,
            module_weights=module_weights,
            market_condition=market_condition,
            total_modules=len(module_signals),
            agreeing_modules=len(consensus.bullish_modules) if signal_direction == "BUY" else len(consensus.bearish_modules),
            blocking_modules=len(consensus.blocking_modules),
            optimal_entry=optimal_entry,
            stop_loss=stop_loss,
            targets=targets,
            risk_reward=risk_reward,
            max_risk_percent=2.0,
            edge_factors=edge_factors,
            risk_factors=risk_factors,
            improvement_insights=insights,
            final_verdict=verdict,
            decision_id=decision_id,
            module_signals=module_signals
        )
        
        self.decision_history.append({
            "id": decision_id,
            "grade": grade.value,
            "score": titan_score,
            "timestamp": datetime.now().isoformat()
        })
        
        return decision
    
    def _calculate_titan_score(
        self,
        consensus: ConsensusResult,
        prediction: PredictionEnsemble,
        calibration: CalibrationData,
        market_condition: MarketCondition,
        module_signals: List[ModuleSignal]
    ) -> float:
        """Calculate comprehensive Titan Score"""
        
        score = 40  # Base score
        
        # Consensus contribution (0-25 points)
        consensus_points = {
            ConsensusLevel.UNANIMOUS: 25,
            ConsensusLevel.STRONG: 20,
            ConsensusLevel.MODERATE: 12,
            ConsensusLevel.WEAK: 5,
            ConsensusLevel.CONFLICT: 0,
        }
        score += consensus_points.get(consensus.level, 0)
        
        # Prediction contribution (0-15 points)
        score += prediction.prediction_confidence * 0.15
        
        # Calibrated confidence contribution (0-15 points)
        score += calibration.calibrated_confidence * 0.15
        
        # Market condition (0-10 points)
        condition_points = {
            MarketCondition.HIGHLY_FAVORABLE: 10,
            MarketCondition.FAVORABLE: 7,
            MarketCondition.NEUTRAL: 4,
            MarketCondition.UNFAVORABLE: 1,
            MarketCondition.HIGHLY_UNFAVORABLE: 0,
        }
        score += condition_points.get(market_condition, 0)
        
        # Module quality contribution
        if module_signals:
            avg_module_score = np.mean([s.score for s in module_signals])
            score += avg_module_score * 0.1
        
        # Penalty for blocking modules
        score -= len(consensus.blocking_modules) * 5
        
        return np.clip(score, 0, 100)
    
    def _determine_grade(
        self,
        titan_score: float,
        consensus: ConsensusResult,
        edge_count: int,
        risk_count: int
    ) -> TitanGrade:
        """Determine Titan Grade"""
        
        net_factors = edge_count - risk_count
        
        if titan_score >= 90 and consensus.level == ConsensusLevel.UNANIMOUS:
            return TitanGrade.TITAN_SUPREME
        elif titan_score >= 80 and consensus.level in [ConsensusLevel.UNANIMOUS, ConsensusLevel.STRONG]:
            return TitanGrade.TITAN_ELITE
        elif titan_score >= 70 and net_factors >= 1:
            return TitanGrade.TITAN_PRIME
        elif titan_score >= 60 and net_factors >= 0:
            return TitanGrade.TITAN_CORE
        elif titan_score >= 50:
            return TitanGrade.TITAN_BASE
        elif titan_score >= 35:
            return TitanGrade.MORTAL
        else:
            return TitanGrade.REJECT
    
    def _consensus_meets_minimum(self, level: ConsensusLevel) -> bool:
        """Check if consensus meets minimum requirement"""
        levels_order = [
            ConsensusLevel.CONFLICT,
            ConsensusLevel.WEAK,
            ConsensusLevel.MODERATE,
            ConsensusLevel.STRONG,
            ConsensusLevel.UNANIMOUS,
        ]
        
        current_idx = levels_order.index(level)
        min_idx = levels_order.index(self.min_consensus)
        
        return current_idx >= min_idx
    
    def record_trade_outcome(self, decision_id: str, was_profitable: bool):
        """Record trade outcome for learning"""
        # Update calibrator
        for hist in self.decision_history:
            if hist["id"] == decision_id:
                self.calibrator.record_outcome(hist["score"], was_profitable)
                break
        
        # Update improvement engine
        self.improvement_engine.record_outcome(
            {"decision_id": decision_id},
            was_profitable
        )


# ============================================================
# SINGLETON & HELPERS
# ============================================================

_titan_core_instance: Optional[TitanCore] = None


def get_titan_core() -> TitanCore:
    """Get or create TitanCore singleton"""
    global _titan_core_instance
    if _titan_core_instance is None:
        _titan_core_instance = TitanCore()
    return _titan_core_instance


def init_titan_core(
    min_titan_score: float = 60.0,
    min_consensus: ConsensusLevel = ConsensusLevel.WEAK,
    enable_calibration: bool = True,
    enable_self_improvement: bool = True
) -> TitanCore:
    """Initialize TitanCore with custom settings"""
    global _titan_core_instance
    _titan_core_instance = TitanCore(
        min_titan_score=min_titan_score,
        min_consensus=min_consensus,
        enable_calibration=enable_calibration,
        enable_self_improvement=enable_self_improvement
    )
    return _titan_core_instance
