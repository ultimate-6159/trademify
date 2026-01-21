"""
🧠 Neural Brain - ระบบสมองประสาทเทียมสำหรับ Trading
Deep Pattern Understanding + Adaptive Decision Making

Features:
1. Pattern DNA - จำ DNA ของ pattern ที่ทำกำไร
2. Market State Machine - รู้ว่าตลาดอยู่ในสถานะไหน
3. Predictive Confidence - ทำนายความมั่นใจจาก features
4. Self-Evaluation - ประเมินตัวเองและปรับปรุง
5. Anomaly Detection - ตรวจจับสถานการณ์ผิดปกติ
6. Risk Intelligence - ฉลาดเรื่อง risk แบบลึก

Resource Efficient:
- ใช้ numpy แทน ML library หนักๆ
- Incremental learning (ไม่ต้อง retrain ทั้งหมด)
- Memory-efficient data structures
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import hashlib
import json
import os
import logging

logger = logging.getLogger(__name__)


# =====================
# Enums & Data Classes
# =====================

class MarketState(Enum):
    """สถานะตลาด 7 แบบ"""
    STRONG_UPTREND = "strong_uptrend"      # ขาขึ้นแรง
    WEAK_UPTREND = "weak_uptrend"          # ขาขึ้นอ่อน
    STRONG_DOWNTREND = "strong_downtrend"  # ขาลงแรง
    WEAK_DOWNTREND = "weak_downtrend"      # ขาลงอ่อน
    RANGING = "ranging"                     # Sideway
    VOLATILE = "volatile"                   # ผันผวนสูง
    TRANSITIONING = "transitioning"         # กำลังเปลี่ยน


class RiskLevel(Enum):
    """ระดับความเสี่ยง"""
    VERY_LOW = "very_low"      # เสี่ยงน้อยมาก
    LOW = "low"                # เสี่ยงน้อย
    MEDIUM = "medium"          # ปานกลาง
    HIGH = "high"              # เสี่ยงสูง
    EXTREME = "extreme"        # เสี่ยงมากสุด


@dataclass
class PatternDNA:
    """DNA ของ pattern - ลักษณะเฉพาะที่ทำให้ pattern ทำกำไร"""
    hash: str
    features: Dict[str, float]  # Normalized features
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    avg_duration_hours: float = 0.0
    best_entry_hour: int = 0
    best_exit_hour: int = 0
    preferred_session: str = "london"
    last_seen: str = ""
    
    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5
    
    @property
    def expected_value(self) -> float:
        """Expected value per trade"""
        total = self.win_count + self.loss_count
        return self.total_pnl / total if total > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "features": self.features,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "expected_value": round(self.expected_value, 2),
            "avg_duration_hours": round(self.avg_duration_hours, 1),
            "best_entry_hour": self.best_entry_hour,
            "preferred_session": self.preferred_session,
        }


@dataclass
class NeuralDecision:
    """ผลการตัดสินใจจาก Neural Brain"""
    can_trade: bool
    confidence: float  # 0-100
    risk_level: RiskLevel
    market_state: MarketState
    position_size_factor: float  # 0.0 - 2.0
    optimal_entry_delay: int  # minutes to wait
    pattern_quality: str  # "excellent", "good", "average", "poor"
    anomaly_detected: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "can_trade": self.can_trade,
            "confidence": round(self.confidence, 1),
            "risk_level": self.risk_level.value,
            "market_state": self.market_state.value,
            "position_size_factor": round(self.position_size_factor, 2),
            "optimal_entry_delay": self.optimal_entry_delay,
            "pattern_quality": self.pattern_quality,
            "anomaly_detected": self.anomaly_detected,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


# =====================
# Pattern DNA Analyzer
# =====================

class PatternDNAAnalyzer:
    """
    วิเคราะห์ DNA ของ pattern
    จำลักษณะเฉพาะที่ทำให้ pattern ทำกำไร
    """
    
    def __init__(self, max_patterns: int = 1000):
        self.max_patterns = max_patterns
        self.pattern_database: Dict[str, PatternDNA] = {}
        
        # Feature extractors
        self.feature_names = [
            "trend_strength",      # ความแข็งแกร่ง trend
            "volatility",          # ความผันผวน
            "momentum",            # momentum
            "volume_profile",      # ลักษณะ volume
            "pattern_complexity",  # ความซับซ้อน pattern
            "price_position",      # ตำแหน่งราคา (near high/low)
            "time_of_day",         # ช่วงเวลา
            "day_of_week",         # วันในสัปดาห์
        ]
    
    def extract_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None,
        current_hour: int = 12,
        day_of_week: int = 2,
    ) -> Dict[str, float]:
        """Extract normalized features from price data"""
        features = {}
        
        if len(prices) < 10:
            return {name: 0.5 for name in self.feature_names}
        
        # 1. Trend Strength (0-1)
        returns = np.diff(prices) / prices[:-1]
        trend = np.sum(returns)
        features["trend_strength"] = min(1.0, max(0.0, (trend + 0.1) / 0.2))
        
        # 2. Volatility (0-1)
        volatility = np.std(returns)
        features["volatility"] = min(1.0, volatility / 0.02)
        
        # 3. Momentum (0-1)
        recent_returns = returns[-10:] if len(returns) >= 10 else returns
        momentum = np.mean(recent_returns)
        features["momentum"] = min(1.0, max(0.0, (momentum + 0.01) / 0.02))
        
        # 4. Volume Profile (0-1)
        if volumes is not None and len(volumes) > 0:
            vol_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
            features["volume_profile"] = min(1.0, vol_ratio / 2.0)
        else:
            features["volume_profile"] = 0.5
        
        # 5. Pattern Complexity (0-1)
        # Based on number of direction changes
        direction_changes = np.sum(np.diff(np.sign(returns)) != 0)
        features["pattern_complexity"] = min(1.0, direction_changes / len(returns))
        
        # 6. Price Position (0-1)
        price_range = np.max(prices) - np.min(prices)
        if price_range > 0:
            position = (prices[-1] - np.min(prices)) / price_range
        else:
            position = 0.5
        features["price_position"] = position
        
        # 7. Time of Day (0-1)
        # Best hours: 8-12 (London), 13-17 (NY)
        if 8 <= current_hour <= 12:
            features["time_of_day"] = 0.9
        elif 13 <= current_hour <= 17:
            features["time_of_day"] = 0.8
        elif 0 <= current_hour <= 7:
            features["time_of_day"] = 0.3  # Asian
        else:
            features["time_of_day"] = 0.5
        
        # 8. Day of Week (0-1)
        # Best: Tue-Thu, Worst: Fri, Mon
        day_scores = {0: 0.5, 1: 0.7, 2: 0.9, 3: 0.9, 4: 0.4, 5: 0.1, 6: 0.1}
        features["day_of_week"] = day_scores.get(day_of_week, 0.5)
        
        return features
    
    def compute_dna_hash(self, features: Dict[str, float]) -> str:
        """Compute unique hash for pattern DNA"""
        # Quantize features to create discrete hash
        quantized = {k: round(v * 10) / 10 for k, v in features.items()}
        feature_str = json.dumps(quantized, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:12]
    
    def get_or_create_dna(self, features: Dict[str, float]) -> PatternDNA:
        """Get existing DNA or create new one"""
        dna_hash = self.compute_dna_hash(features)
        
        if dna_hash not in self.pattern_database:
            self.pattern_database[dna_hash] = PatternDNA(
                hash=dna_hash,
                features=features,
                last_seen=datetime.now().isoformat()
            )
            self._prune_if_needed()
        
        return self.pattern_database[dna_hash]
    
    def record_trade_result(
        self,
        features: Dict[str, float],
        is_win: bool,
        pnl_percent: float,
        duration_hours: float,
        entry_hour: int,
        exit_hour: int,
        session: str = "london"
    ):
        """Record trade result for pattern DNA"""
        dna = self.get_or_create_dna(features)
        
        if is_win:
            dna.win_count += 1
        else:
            dna.loss_count += 1
        
        dna.total_pnl += pnl_percent
        
        # Update average duration (EMA)
        alpha = 0.3
        dna.avg_duration_hours = (1 - alpha) * dna.avg_duration_hours + alpha * duration_hours
        
        # Track best entry hour
        if is_win:
            dna.best_entry_hour = entry_hour
            dna.best_exit_hour = exit_hour
            dna.preferred_session = session
        
        dna.last_seen = datetime.now().isoformat()
    
    def predict_win_probability(self, features: Dict[str, float]) -> Tuple[float, str]:
        """Predict win probability based on DNA database"""
        dna_hash = self.compute_dna_hash(features)
        
        # Exact match
        if dna_hash in self.pattern_database:
            dna = self.pattern_database[dna_hash]
            total_trades = dna.win_count + dna.loss_count
            if total_trades >= 5:
                return dna.win_rate, f"Exact DNA match ({total_trades} trades)"
        
        # Find similar patterns
        similar_patterns = self._find_similar_patterns(features, top_k=5)
        
        if similar_patterns:
            total_wins = sum(p.win_count for p in similar_patterns)
            total_losses = sum(p.loss_count for p in similar_patterns)
            total = total_wins + total_losses
            
            if total >= 10:
                win_rate = total_wins / total
                return win_rate, f"Similar DNA patterns ({total} trades)"
        
        # No data - return neutral
        return 0.5, "No matching DNA patterns"
    
    def _find_similar_patterns(
        self, 
        features: Dict[str, float], 
        top_k: int = 5
    ) -> List[PatternDNA]:
        """Find patterns with similar features"""
        if not self.pattern_database:
            return []
        
        similarities = []
        for dna in self.pattern_database.values():
            # Compute Euclidean distance
            distance = 0
            for name in self.feature_names:
                diff = features.get(name, 0.5) - dna.features.get(name, 0.5)
                distance += diff ** 2
            similarity = 1 / (1 + np.sqrt(distance))
            similarities.append((similarity, dna))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [dna for _, dna in similarities[:top_k]]
    
    def _prune_if_needed(self):
        """Remove low-value patterns if database too large"""
        if len(self.pattern_database) <= self.max_patterns:
            return
        
        # Sort by value (trades * win_rate)
        patterns = list(self.pattern_database.items())
        patterns.sort(
            key=lambda x: (x[1].win_count + x[1].loss_count) * x[1].win_rate,
            reverse=True
        )
        
        # Keep top patterns
        keep_count = int(self.max_patterns * 0.8)
        self.pattern_database = dict(patterns[:keep_count])
        
        logger.info(f"🧬 Pruned DNA database to {len(self.pattern_database)} patterns")
    
    def get_best_patterns(self, min_trades: int = 5, top_k: int = 10) -> List[dict]:
        """Get best performing pattern DNAs"""
        qualified = [
            dna for dna in self.pattern_database.values()
            if (dna.win_count + dna.loss_count) >= min_trades
        ]
        
        # Sort by expected value
        qualified.sort(key=lambda x: x.expected_value, reverse=True)
        
        return [dna.to_dict() for dna in qualified[:top_k]]


# =====================
# Market State Machine
# =====================

class MarketStateMachine:
    """
    State Machine ที่ track สถานะตลาด
    และ transition probabilities
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.state_history: deque = deque(maxlen=lookback)
        self.current_state = MarketState.RANGING
        
        # Transition matrix (learned from data)
        self.transitions: Dict[str, Dict[str, int]] = {
            state.value: {s.value: 0 for s in MarketState}
            for state in MarketState
        }
        
        # State performance (win rate per state)
        self.state_performance: Dict[str, Dict[str, int]] = {
            state.value: {"wins": 0, "losses": 0}
            for state in MarketState
        }
    
    def detect_state(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None,
        atr: float = None
    ) -> MarketState:
        """Detect current market state"""
        if len(prices) < 20:
            return MarketState.RANGING
        
        # Calculate metrics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Trend strength
        sma20 = np.mean(prices[-20:])
        sma50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma20
        trend = (sma20 - sma50) / sma50 if sma50 > 0 else 0
        
        # Recent momentum
        recent_returns = returns[-10:]
        momentum = np.mean(recent_returns)
        
        # Determine state
        high_vol = volatility > 0.015  # > 1.5% std
        
        if high_vol:
            new_state = MarketState.VOLATILE
        elif trend > 0.02 and momentum > 0:
            new_state = MarketState.STRONG_UPTREND
        elif trend > 0.005 and momentum > 0:
            new_state = MarketState.WEAK_UPTREND
        elif trend < -0.02 and momentum < 0:
            new_state = MarketState.STRONG_DOWNTREND
        elif trend < -0.005 and momentum < 0:
            new_state = MarketState.WEAK_DOWNTREND
        elif abs(trend) > 0.005 and np.sign(trend) != np.sign(momentum):
            new_state = MarketState.TRANSITIONING
        else:
            new_state = MarketState.RANGING
        
        # Record transition
        self._record_transition(self.current_state, new_state)
        self.current_state = new_state
        self.state_history.append(new_state)
        
        return new_state
    
    def _record_transition(self, from_state: MarketState, to_state: MarketState):
        """Record state transition"""
        self.transitions[from_state.value][to_state.value] += 1
    
    def get_next_state_probability(self) -> Dict[str, float]:
        """Get probability distribution for next state"""
        current = self.current_state.value
        counts = self.transitions[current]
        total = sum(counts.values())
        
        if total == 0:
            # Equal probability
            return {s.value: 1/len(MarketState) for s in MarketState}
        
        return {state: count/total for state, count in counts.items()}
    
    def record_trade_result(self, is_win: bool):
        """Record trade result for current state"""
        state = self.current_state.value
        if is_win:
            self.state_performance[state]["wins"] += 1
        else:
            self.state_performance[state]["losses"] += 1
    
    def get_state_win_rate(self, state: MarketState = None) -> float:
        """Get win rate for a market state"""
        if state is None:
            state = self.current_state
        
        perf = self.state_performance[state.value]
        total = perf["wins"] + perf["losses"]
        
        if total < 5:
            return 0.5  # Not enough data
        
        return perf["wins"] / total
    
    def should_trade_in_state(self, state: MarketState = None) -> Tuple[bool, str]:
        """Determine if we should trade in current state"""
        if state is None:
            state = self.current_state
        
        win_rate = self.get_state_win_rate(state)
        
        # States to avoid
        if state == MarketState.VOLATILE:
            return False, "Market too volatile"
        
        if state == MarketState.TRANSITIONING:
            return False, "Market transitioning - wait for clarity"
        
        if win_rate < 0.4:
            return False, f"Low win rate in {state.value} ({win_rate*100:.0f}%)"
        
        return True, f"{state.value} is tradeable (WR: {win_rate*100:.0f}%)"


# =====================
# Anomaly Detector
# =====================

class AnomalyDetector:
    """
    ตรวจจับสถานการณ์ผิดปกติ
    - Price spikes
    - Volume anomalies
    - Unusual patterns
    """
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly
        
        # Rolling statistics
        self.price_returns_mean: deque = deque(maxlen=1000)
        self.price_returns_std: deque = deque(maxlen=1000)
        self.volume_mean: deque = deque(maxlen=1000)
    
    def update(self, price_return: float, volume: float = 0):
        """Update rolling statistics"""
        self.price_returns_mean.append(price_return)
        if volume > 0:
            self.volume_mean.append(volume)
    
    def detect(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None
    ) -> Tuple[bool, List[str]]:
        """
        Detect anomalies
        
        Returns: (is_anomaly, list of anomaly types)
        """
        anomalies = []
        
        if len(prices) < 20:
            return False, anomalies
        
        # 1. Price spike detection
        returns = np.diff(prices) / prices[:-1]
        last_return = returns[-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            z_score = (last_return - mean_return) / std_return
            if abs(z_score) > self.sensitivity:
                anomalies.append(f"Price spike (z={z_score:.1f})")
        
        # 2. Volatility spike
        recent_vol = np.std(returns[-10:])
        historical_vol = np.std(returns[:-10]) if len(returns) > 10 else recent_vol
        
        if historical_vol > 0:
            vol_ratio = recent_vol / historical_vol
            if vol_ratio > 2.0:
                anomalies.append(f"Volatility spike ({vol_ratio:.1f}x)")
        
        # 3. Volume anomaly
        if volumes is not None and len(volumes) > 20:
            avg_vol = np.mean(volumes[:-1])
            if avg_vol > 0:
                vol_spike = volumes[-1] / avg_vol
                if vol_spike > 3.0:
                    anomalies.append(f"Volume spike ({vol_spike:.1f}x)")
        
        # 4. Gap detection
        if len(prices) >= 2:
            gap_percent = abs(prices[-1] - prices[-2]) / prices[-2] * 100
            if gap_percent > 0.5:  # 0.5% gap
                anomalies.append(f"Price gap ({gap_percent:.2f}%)")
        
        return len(anomalies) > 0, anomalies


# =====================
# Risk Intelligence
# =====================

class RiskIntelligence:
    """
    ระบบวิเคราะห์ risk แบบฉลาด
    """
    
    def __init__(self):
        # Track risk metrics over time
        self.daily_pnl: deque = deque(maxlen=30)
        self.trade_results: deque = deque(maxlen=100)
        self.drawdown_history: deque = deque(maxlen=100)
        
        self.peak_balance = 10000.0  # Will be updated
        self.current_balance = 10000.0
    
    def update_balance(self, balance: float):
        """Update balance and track drawdown"""
        self.current_balance = balance
        
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        drawdown = (self.peak_balance - balance) / self.peak_balance * 100
        self.drawdown_history.append(drawdown)
    
    def add_trade_result(self, pnl_percent: float, is_win: bool):
        """Add trade result"""
        self.trade_results.append({
            "pnl": pnl_percent,
            "win": is_win,
            "time": datetime.now().isoformat()
        })
    
    def calculate_risk_level(self) -> RiskLevel:
        """Calculate current risk level"""
        if len(self.drawdown_history) == 0:
            return RiskLevel.MEDIUM
        
        current_dd = self.drawdown_history[-1] if self.drawdown_history else 0
        
        # Recent performance
        recent_trades = list(self.trade_results)[-10:]
        recent_losses = sum(1 for t in recent_trades if not t["win"])
        
        # Calculate risk level
        if current_dd > 15 or recent_losses >= 5:
            return RiskLevel.EXTREME
        elif current_dd > 10 or recent_losses >= 4:
            return RiskLevel.HIGH
        elif current_dd > 5 or recent_losses >= 3:
            return RiskLevel.MEDIUM
        elif current_dd > 2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def get_recommended_size_factor(self) -> float:
        """Get recommended position size multiplier"""
        risk_level = self.calculate_risk_level()
        
        size_factors = {
            RiskLevel.VERY_LOW: 1.5,   # Can increase size
            RiskLevel.LOW: 1.2,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.5,
            RiskLevel.EXTREME: 0.0,    # Stop trading
        }
        
        return size_factors[risk_level]
    
    def get_losing_streak(self) -> int:
        """Get current losing streak"""
        streak = 0
        for trade in reversed(list(self.trade_results)):
            if not trade["win"]:
                streak += 1
            else:
                break
        return streak
    
    def get_winning_streak(self) -> int:
        """Get current winning streak"""
        streak = 0
        for trade in reversed(list(self.trade_results)):
            if trade["win"]:
                streak += 1
            else:
                break
        return streak
    
    def should_pause_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be paused"""
        risk_level = self.calculate_risk_level()
        losing_streak = self.get_losing_streak()
        
        if risk_level == RiskLevel.EXTREME:
            return True, "Risk level EXTREME - stop trading"
        
        if losing_streak >= 5:
            return True, f"Losing streak {losing_streak} - take a break"
        
        current_dd = self.drawdown_history[-1] if self.drawdown_history else 0
        if current_dd > 10:
            return True, f"Drawdown {current_dd:.1f}% - reduce exposure"
        
        return False, "OK to trade"


# =====================
# Neural Brain (Main Class)
# =====================

class NeuralBrain:
    """
    🧠 Neural Brain - ระบบสมองประสาทเทียมหลัก
    รวมทุก component เข้าด้วยกัน
    """
    
    def __init__(
        self,
        data_dir: str = "data/neural",
        firebase_service = None,
        enable_dna: bool = True,
        enable_state_machine: bool = True,
        enable_anomaly: bool = True,
        enable_risk_intel: bool = True,
    ):
        self.data_dir = data_dir
        self.firebase = firebase_service
        os.makedirs(data_dir, exist_ok=True)
        
        # Components
        self.dna_analyzer = PatternDNAAnalyzer() if enable_dna else None
        self.state_machine = MarketStateMachine() if enable_state_machine else None
        self.anomaly_detector = AnomalyDetector() if enable_anomaly else None
        self.risk_intel = RiskIntelligence() if enable_risk_intel else None
        
        # Settings
        self.min_confidence = 60.0
        self.min_dna_trades = 3
        
        # Load saved state
        self._load_state()
        
        logger.info("🧠 Neural Brain initialized")
        logger.info(f"   - Pattern DNA: {'✓' if enable_dna else '✗'}")
        logger.info(f"   - State Machine: {'✓' if enable_state_machine else '✗'}")
        logger.info(f"   - Anomaly Detection: {'✓' if enable_anomaly else '✗'}")
        logger.info(f"   - Risk Intelligence: {'✓' if enable_risk_intel else '✗'}")
    
    def analyze(
        self,
        signal_side: str,  # "BUY" or "SELL"
        prices: np.ndarray,
        volumes: np.ndarray = None,
        current_hour: int = None,
        day_of_week: int = None,
        balance: float = None,
    ) -> NeuralDecision:
        """
        Main analysis function
        
        Returns comprehensive NeuralDecision
        """
        if current_hour is None:
            current_hour = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        
        reasons = []
        warnings = []
        confidence = 50.0
        can_trade = True
        position_size = 1.0
        optimal_delay = 0
        pattern_quality = "average"
        anomaly_detected = False
        
        # 1. Update balance if provided
        if balance and self.risk_intel:
            self.risk_intel.update_balance(balance)
        
        # 2. Detect market state
        market_state = MarketState.RANGING
        if self.state_machine:
            market_state = self.state_machine.detect_state(prices, volumes)
            
            should_trade, state_reason = self.state_machine.should_trade_in_state()
            if not should_trade:
                can_trade = False
                warnings.append(state_reason)
            else:
                reasons.append(state_reason)
            
            # Adjust confidence based on state win rate
            state_wr = self.state_machine.get_state_win_rate()
            confidence += (state_wr - 0.5) * 20  # ±10 based on state
        
        # 3. Check for anomalies
        if self.anomaly_detector:
            anomaly_detected, anomaly_types = self.anomaly_detector.detect(prices, volumes)
            
            if anomaly_detected:
                warnings.extend([f"⚠️ {a}" for a in anomaly_types])
                confidence -= 15
                
                if len(anomaly_types) >= 2:
                    can_trade = False
                    warnings.append("Multiple anomalies - skip trade")
        
        # 4. Analyze pattern DNA
        if self.dna_analyzer:
            features = self.dna_analyzer.extract_features(
                prices, volumes, current_hour, day_of_week
            )
            
            win_prob, dna_reason = self.dna_analyzer.predict_win_probability(features)
            
            if "Exact DNA" in dna_reason:
                confidence += (win_prob - 0.5) * 30  # Strong adjustment
                pattern_quality = "excellent" if win_prob > 0.7 else "good"
                reasons.append(f"🧬 {dna_reason}: {win_prob*100:.0f}% WR")
            elif "Similar DNA" in dna_reason:
                confidence += (win_prob - 0.5) * 15  # Moderate adjustment
                pattern_quality = "good" if win_prob > 0.6 else "average"
                reasons.append(f"🧬 {dna_reason}: {win_prob*100:.0f}% WR")
            else:
                pattern_quality = "unknown"
                warnings.append("New pattern - no historical data")
            
            # Check time preference
            dna = self.dna_analyzer.get_or_create_dna(features)
            if dna.win_count + dna.loss_count >= self.min_dna_trades:
                if abs(current_hour - dna.best_entry_hour) <= 2:
                    confidence += 5
                    reasons.append(f"⏰ Good entry time (best: {dna.best_entry_hour}:00)")
                else:
                    optimal_delay = (dna.best_entry_hour - current_hour) * 60
                    if optimal_delay < 0:
                        optimal_delay = 0
        
        # 5. Risk intelligence check
        risk_level = RiskLevel.MEDIUM
        if self.risk_intel:
            risk_level = self.risk_intel.calculate_risk_level()
            position_size = self.risk_intel.get_recommended_size_factor()
            
            should_pause, pause_reason = self.risk_intel.should_pause_trading()
            if should_pause:
                can_trade = False
                warnings.append(f"🛑 {pause_reason}")
            
            losing_streak = self.risk_intel.get_losing_streak()
            if losing_streak >= 3:
                confidence -= losing_streak * 5
                warnings.append(f"📉 Losing streak: {losing_streak}")
            
            winning_streak = self.risk_intel.get_winning_streak()
            if winning_streak >= 3:
                confidence += min(winning_streak * 2, 10)
                reasons.append(f"📈 Winning streak: {winning_streak}")
        
        # 6. Final confidence bounds
        confidence = max(0, min(100, confidence))
        
        # 7. Final decision
        if confidence < self.min_confidence:
            can_trade = False
            warnings.append(f"Confidence too low: {confidence:.0f}% < {self.min_confidence}%")
        
        return NeuralDecision(
            can_trade=can_trade,
            confidence=confidence,
            risk_level=risk_level,
            market_state=market_state,
            position_size_factor=position_size,
            optimal_entry_delay=max(0, optimal_delay),
            pattern_quality=pattern_quality,
            anomaly_detected=anomaly_detected,
            reasons=reasons,
            warnings=warnings,
        )
    
    def record_trade(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        is_win: bool,
        pnl_percent: float,
        duration_hours: float,
        entry_hour: int,
        exit_hour: int,
        session: str = "london"
    ):
        """Record trade result for learning"""
        # Record in pattern DNA
        if self.dna_analyzer:
            features = self.dna_analyzer.extract_features(
                prices, volumes, entry_hour, datetime.now().weekday()
            )
            self.dna_analyzer.record_trade_result(
                features, is_win, pnl_percent, 
                duration_hours, entry_hour, exit_hour, session
            )
        
        # Record in state machine
        if self.state_machine:
            self.state_machine.record_trade_result(is_win)
        
        # Record in risk intelligence
        if self.risk_intel:
            self.risk_intel.add_trade_result(pnl_percent, is_win)
        
        # Save state periodically
        total_trades = len(self.risk_intel.trade_results) if self.risk_intel else 0
        if total_trades % 5 == 0:
            self._save_state()
    
    def get_brain_stats(self) -> dict:
        """Get comprehensive brain statistics"""
        stats = {
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.dna_analyzer:
            stats["pattern_dna"] = {
                "total_patterns": len(self.dna_analyzer.pattern_database),
                "best_patterns": self.dna_analyzer.get_best_patterns(top_k=5),
            }
        
        if self.state_machine:
            stats["market_state"] = {
                "current": self.state_machine.current_state.value,
                "next_probabilities": self.state_machine.get_next_state_probability(),
                "state_performance": self.state_machine.state_performance,
            }
        
        if self.risk_intel:
            stats["risk"] = {
                "level": self.risk_intel.calculate_risk_level().value,
                "size_factor": self.risk_intel.get_recommended_size_factor(),
                "losing_streak": self.risk_intel.get_losing_streak(),
                "winning_streak": self.risk_intel.get_winning_streak(),
            }
        
        return stats
    
    def _save_state(self):
        """Save brain state to file and Firebase"""
        try:
            state = {
                "saved_at": datetime.now().isoformat(),
            }
            
            if self.dna_analyzer:
                state["pattern_database"] = {
                    h: {
                        "features": dna.features,
                        "win_count": dna.win_count,
                        "loss_count": dna.loss_count,
                        "total_pnl": dna.total_pnl,
                        "avg_duration_hours": dna.avg_duration_hours,
                        "best_entry_hour": dna.best_entry_hour,
                    }
                    for h, dna in list(self.dna_analyzer.pattern_database.items())[:200]
                }
            
            if self.state_machine:
                state["transitions"] = self.state_machine.transitions
                state["state_performance"] = self.state_machine.state_performance
            
            # Save locally
            filepath = os.path.join(self.data_dir, "neural_state.json")
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save to Firebase
            if self.firebase:
                try:
                    self.firebase.save_learning_state({
                        "neural_brain": state
                    })
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to save neural state: {e}")
    
    def _load_state(self):
        """Load brain state from file or Firebase"""
        try:
            state = None
            
            # Try Firebase first
            if self.firebase:
                try:
                    data = self.firebase.load_learning_state()
                    state = data.get("neural_brain") if data else None
                except:
                    pass
            
            # Try local file
            if not state:
                filepath = os.path.join(self.data_dir, "neural_state.json")
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        state = json.load(f)
            
            if state:
                # Restore pattern database
                if self.dna_analyzer and "pattern_database" in state:
                    for h, data in state["pattern_database"].items():
                        self.dna_analyzer.pattern_database[h] = PatternDNA(
                            hash=h,
                            features=data.get("features", {}),
                            win_count=data.get("win_count", 0),
                            loss_count=data.get("loss_count", 0),
                            total_pnl=data.get("total_pnl", 0),
                            avg_duration_hours=data.get("avg_duration_hours", 0),
                            best_entry_hour=data.get("best_entry_hour", 12),
                        )
                
                # Restore state machine
                if self.state_machine:
                    if "transitions" in state:
                        self.state_machine.transitions = state["transitions"]
                    if "state_performance" in state:
                        self.state_machine.state_performance = state["state_performance"]
                
                logger.info(f"🧠 Loaded neural state: {len(self.dna_analyzer.pattern_database) if self.dna_analyzer else 0} patterns")
        
        except Exception as e:
            logger.warning(f"Could not load neural state: {e}")


# =====================
# Singleton
# =====================

_neural_brain: Optional[NeuralBrain] = None


def get_neural_brain(firebase_service=None) -> NeuralBrain:
    """Get or create neural brain instance"""
    global _neural_brain
    
    if _neural_brain is None:
        _neural_brain = NeuralBrain(
            firebase_service=firebase_service
        )
    
    return _neural_brain


def init_neural_brain(
    firebase_service=None,
    **kwargs
) -> NeuralBrain:
    """Initialize neural brain with custom settings"""
    global _neural_brain
    
    _neural_brain = NeuralBrain(
        firebase_service=firebase_service,
        **kwargs
    )
    
    return _neural_brain
