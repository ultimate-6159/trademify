"""
Voting System - Phase 3: The Voting System
ระบบโหวตเพื่อความแม่นยำ

เมื่อได้ 10 Patterns แล้ว ต้องเข้าสู่กระบวนการ "คัดกรอง"
- นับคะแนน BUY vs SELL
- คำนวณ Confidence
- ตัดสินใจ: Trade หรือ Wait
- ประมาณระยะเวลาที่สัญญาณจะคงอยู่
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from config import VotingConfig, PatternConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Signal(str, Enum):
    """Trading signals"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WAIT = "WAIT"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(str, Enum):
    """Signal strength for duration estimation"""
    VERY_STRONG = "VERY_STRONG"   # >90% confidence
    STRONG = "STRONG"             # >80% confidence  
    MODERATE = "MODERATE"         # >70% confidence
    WEAK = "WEAK"                 # <70% confidence


@dataclass
class SignalDuration:
    """Signal duration estimation"""
    estimated_minutes: int          # Estimated duration in minutes
    min_minutes: int                # Minimum expected duration
    max_minutes: int                # Maximum expected duration
    confidence_decay_rate: float    # How fast confidence drops (% per minute)
    strength: SignalStrength        # Signal strength category
    expires_at: Optional[datetime] = None  # When signal is expected to expire
    warning_at: Optional[datetime] = None  # When to warn user (50% time left)
    created_at: Optional[datetime] = None  # When signal was created
    
    def to_dict(self) -> dict:
        return {
            "estimated_minutes": self.estimated_minutes,
            "min_minutes": self.min_minutes,
            "max_minutes": self.max_minutes,
            "confidence_decay_rate": round(self.confidence_decay_rate, 3),
            "strength": self.strength.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "warning_at": self.warning_at.isoformat() if self.warning_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "remaining_seconds": self.remaining_seconds,
            "is_expired": self.is_expired,
            "time_display": self.time_display,
        }
    
    @property
    def remaining_seconds(self) -> int:
        """Get remaining seconds until expiry"""
        if not self.expires_at:
            return 0
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, int(remaining))
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def time_display(self) -> str:
        """Human-readable time remaining"""
        remaining = self.remaining_seconds
        if remaining <= 0:
            return "หมดอายุ"
        elif remaining < 60:
            return f"{remaining} วินาที"
        elif remaining < 3600:
            mins = remaining // 60
            secs = remaining % 60
            return f"{mins}:{secs:02d} นาที"
        else:
            hours = remaining // 3600
            mins = (remaining % 3600) // 60
            return f"{hours}:{mins:02d} ชั่วโมง"


@dataclass
class VoteResult:
    """Result of voting analysis"""
    signal: Signal
    confidence: float
    bullish_votes: int
    bearish_votes: int
    total_votes: int
    average_movement: np.ndarray
    projected_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    duration: Optional[SignalDuration] = None  # Signal duration estimation
    
    def to_dict(self) -> dict:
        return {
            "signal": self.signal.value,
            "confidence": round(self.confidence, 2),
            "bullish_votes": self.bullish_votes,
            "bearish_votes": self.bearish_votes,
            "total_votes": self.total_votes,
            "average_movement": self.average_movement.tolist() if self.average_movement is not None else None,
            "projected_price": self.projected_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "duration": self.duration.to_dict() if self.duration else None,
        }


class VotingSystem:
    """
    ระบบโหวตจาก Pattern ในอดีต
    นับคะแนนและคำนวณ Probability
    """
    
    def __init__(
        self,
        confidence_threshold: float = 70.0,
        strong_signal_threshold: float = 80.0,
        min_patterns: int = 5,
        timeframe: str = "M5"
    ):
        """
        Initialize Voting System
        
        Args:
            confidence_threshold: Minimum confidence to generate signal (%)
            strong_signal_threshold: Threshold for strong signal (%)
            min_patterns: Minimum number of valid patterns required
            timeframe: Trading timeframe for duration estimation
        """
        self.confidence_threshold = confidence_threshold
        self.strong_signal_threshold = strong_signal_threshold
        self.min_patterns = min_patterns
        self.timeframe = timeframe
        self.duration_estimator = SignalDurationEstimator(timeframe)
    
    def analyze_patterns(
        self,
        future_movements: List[np.ndarray],
        current_price: Optional[float] = None,
        timeframe: Optional[str] = None
    ) -> VoteResult:
        """
        Analyze patterns and generate trading signal
        
        วิธีคิด:
        1. ดูราคาปิดสุดท้ายของแต่ละ pattern
        2. ถ้าสูงกว่าราคาเริ่ม = Bullish vote
        3. ถ้าต่ำกว่าราคาเริ่ม = Bearish vote
        4. คำนวณ Confidence = max(bullish, bearish) / total * 100
        
        Args:
            future_movements: List of future price arrays from matched patterns
            current_price: Current price for projection
        
        Returns:
            VoteResult with signal and analysis
        """
        if len(future_movements) < self.min_patterns:
            logger.warning(f"Not enough patterns: {len(future_movements)} < {self.min_patterns}")
            return VoteResult(
                signal=Signal.WAIT,
                confidence=0.0,
                bullish_votes=0,
                bearish_votes=0,
                total_votes=len(future_movements),
                average_movement=np.array([]),
            )
        
        bullish_votes = 0
        bearish_votes = 0
        valid_movements = []
        
        for movement in future_movements:
            if len(movement) < 2:
                continue
            
            # Compare last price with first price
            first_price = movement[0]
            last_price = movement[-1]
            
            if last_price > first_price:
                bullish_votes += 1
            else:
                bearish_votes += 1
            
            valid_movements.append(movement)
        
        total_votes = bullish_votes + bearish_votes
        
        if total_votes == 0:
            return VoteResult(
                signal=Signal.WAIT,
                confidence=0.0,
                bullish_votes=0,
                bearish_votes=0,
                total_votes=0,
                average_movement=np.array([]),
            )
        
        # Calculate confidence
        confidence = max(bullish_votes, bearish_votes) / total_votes * 100
        
        # Calculate average movement (เส้นค่าเฉลี่ยของทุก pattern)
        # Normalize movements to percentage change for proper averaging
        normalized_movements = []
        for movement in valid_movements:
            base_price = movement[0]
            if base_price > 0:
                pct_movement = (movement - base_price) / base_price * 100
                normalized_movements.append(pct_movement)
        
        if normalized_movements:
            avg_movement = np.mean(normalized_movements, axis=0)
        else:
            avg_movement = np.array([])
        
        # Determine signal
        signal = self._determine_signal(
            bullish_votes,
            bearish_votes,
            confidence
        )
        
        # Calculate projected price and SL/TP
        projected_price = None
        stop_loss = None
        take_profit = None
        
        if current_price is not None and len(avg_movement) > 0:
            projected_change = avg_movement[-1] / 100  # Convert back from percentage
            projected_price = current_price * (1 + projected_change)
            
            # Calculate SL/TP based on historical volatility
            stop_loss, take_profit = self._calculate_sl_tp(
                current_price,
                valid_movements,
                bullish_votes > bearish_votes
            )
        
        # Calculate signal duration
        # Use timeframe override if provided
        if timeframe and timeframe != self.timeframe:
            duration_estimator = SignalDurationEstimator(timeframe)
        else:
            duration_estimator = self.duration_estimator
        
        vote_ratio = max(bullish_votes, bearish_votes) / total_votes if total_votes > 0 else 0
        duration = duration_estimator.estimate_duration(
            signal=signal,
            confidence=confidence,
            future_movements=valid_movements,
            vote_ratio=vote_ratio
        )
        
        return VoteResult(
            signal=signal,
            confidence=confidence,
            bullish_votes=bullish_votes,
            bearish_votes=bearish_votes,
            total_votes=total_votes,
            average_movement=avg_movement,
            projected_price=projected_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            duration=duration,
        )
    
    def _determine_signal(
        self,
        bullish: int,
        bearish: int,
        confidence: float
    ) -> Signal:
        """
        Determine trading signal based on votes and confidence
        
        กฎเหล็ก: ต้องชนะขาดลอยถึงจะเข้า
        """
        if confidence < self.confidence_threshold:
            return Signal.WAIT
        
        is_bullish = bullish > bearish
        
        if confidence >= self.strong_signal_threshold:
            return Signal.STRONG_BUY if is_bullish else Signal.STRONG_SELL
        else:
            return Signal.BUY if is_bullish else Signal.SELL
    
    def _calculate_sl_tp(
        self,
        current_price: float,
        movements: List[np.ndarray],
        is_bullish: bool
    ) -> Tuple[float, float]:
        """
        Calculate Stop Loss and Take Profit based on historical patterns
        
        Stop Loss: ดูจาก pattern ที่ผิดทาง ว่าลากผิดไปไกลสุดเท่าไหร่
        Take Profit: ดูจาก pattern ที่ถูกทาง ว่าวิ่งไปไกลสุดเท่าไหร่
        """
        if not movements:
            # Default 2% SL, 4% TP
            if is_bullish:
                return current_price * 0.98, current_price * 1.04
            else:
                return current_price * 1.02, current_price * 0.96
        
        adverse_moves = []
        favorable_moves = []
        
        for movement in movements:
            base = movement[0]
            if base == 0:
                continue
            
            pct_changes = (movement - base) / base * 100
            
            if is_bullish:
                # For bullish: adverse = lowest point, favorable = highest point
                adverse_moves.append(min(pct_changes))
                favorable_moves.append(max(pct_changes))
            else:
                # For bearish: adverse = highest point, favorable = lowest point
                adverse_moves.append(max(pct_changes))
                favorable_moves.append(min(pct_changes))
        
        if not adverse_moves:
            if is_bullish:
                return current_price * 0.98, current_price * 1.04
            else:
                return current_price * 1.02, current_price * 0.96
        
        # SL: worst adverse movement + buffer
        worst_adverse = np.percentile(np.abs(adverse_moves), 90)  # 90th percentile
        sl_distance = worst_adverse * 1.1 / 100  # 10% buffer
        
        # TP: average favorable movement
        avg_favorable = np.mean(np.abs(favorable_moves)) / 100
        
        if is_bullish:
            stop_loss = current_price * (1 - sl_distance)
            take_profit = current_price * (1 + avg_favorable)
        else:
            stop_loss = current_price * (1 + sl_distance)
            take_profit = current_price * (1 - avg_favorable)
        
        return stop_loss, take_profit


class SignalDurationEstimator:
    """
    Signal Duration Estimator
    ประมาณว่าสัญญาณจะคงอยู่นานเท่าไร
    
    วิเคราะห์จาก:
    1. Timeframe - M1, M5, M15, H1, H4, D1
    2. Signal Strength - confidence level
    3. Pattern Consistency - ความสม่ำเสมอของ pattern ในอดีต
    4. Momentum Stability - ความเสถียรของ momentum
    """
    
    # Base duration by timeframe (in minutes)
    # สัญญาณโดยทั่วไปจะอยู่ประมาณ 2-5 แท่งเทียน
    TIMEFRAME_BASE_DURATION = {
        "M1": 3,      # 3 minutes (2-5 candles)
        "M5": 15,     # 15 minutes
        "M15": 45,    # 45 minutes
        "M30": 90,    # 1.5 hours
        "H1": 180,    # 3 hours
        "H4": 720,    # 12 hours
        "D1": 2880,   # 2 days
        "W1": 10080,  # 1 week
    }
    
    # Duration multipliers by signal strength
    STRENGTH_MULTIPLIERS = {
        SignalStrength.VERY_STRONG: 1.5,   # Very strong signals last longer
        SignalStrength.STRONG: 1.2,
        SignalStrength.MODERATE: 1.0,
        SignalStrength.WEAK: 0.6,
    }
    
    # Confidence decay rates (% per minute)
    BASE_DECAY_RATES = {
        "M1": 5.0,    # Fast decay for small timeframes
        "M5": 2.0,
        "M15": 0.8,
        "M30": 0.4,
        "H1": 0.2,
        "H4": 0.05,
        "D1": 0.02,
        "W1": 0.005,
    }
    
    def __init__(self, timeframe: str = "M5"):
        """
        Initialize Signal Duration Estimator
        
        Args:
            timeframe: Trading timeframe (M1, M5, M15, H1, etc.)
        """
        self.timeframe = timeframe.upper()
        self.base_duration = self.TIMEFRAME_BASE_DURATION.get(self.timeframe, 15)
        self.base_decay_rate = self.BASE_DECAY_RATES.get(self.timeframe, 1.0)
    
    def estimate_duration(
        self,
        signal: Signal,
        confidence: float,
        future_movements: List[np.ndarray],
        vote_ratio: float = None,
    ) -> SignalDuration:
        """
        Estimate how long the signal will remain valid
        
        Args:
            signal: The trading signal (BUY/SELL/STRONG_BUY/etc.)
            confidence: Signal confidence (0-100)
            future_movements: Future movements from matched patterns
            vote_ratio: Ratio of winning votes (e.g., 8/10 = 0.8)
        
        Returns:
            SignalDuration with estimation details
        """
        now = datetime.now()
        
        # WAIT signals have no duration
        if signal == Signal.WAIT:
            return SignalDuration(
                estimated_minutes=0,
                min_minutes=0,
                max_minutes=0,
                confidence_decay_rate=0,
                strength=SignalStrength.WEAK,
                expires_at=now,
                warning_at=now,
                created_at=now,
            )
        
        # 1. Determine signal strength
        strength = self._determine_strength(confidence)
        
        # 2. Calculate pattern consistency (how similar are the patterns?)
        consistency_factor = self._calculate_consistency(future_movements)
        
        # 3. Calculate momentum stability
        momentum_factor = self._calculate_momentum_stability(future_movements)
        
        # 4. Calculate vote factor (higher vote ratio = longer duration)
        vote_factor = 1.0 + (vote_ratio - 0.7) if vote_ratio else 1.0
        vote_factor = max(0.7, min(1.3, vote_factor))
        
        # 5. Calculate estimated duration
        strength_multiplier = self.STRENGTH_MULTIPLIERS.get(strength, 1.0)
        
        estimated_minutes = int(
            self.base_duration 
            * strength_multiplier 
            * consistency_factor 
            * momentum_factor
            * vote_factor
        )
        
        # 6. Calculate min/max range (±30%)
        min_minutes = int(estimated_minutes * 0.7)
        max_minutes = int(estimated_minutes * 1.4)
        
        # 7. Calculate decay rate (adjusted for consistency)
        decay_rate = self.base_decay_rate / consistency_factor
        
        # 8. Calculate expiry times
        expires_at = now + timedelta(minutes=estimated_minutes)
        warning_at = now + timedelta(minutes=estimated_minutes // 2)
        
        return SignalDuration(
            estimated_minutes=estimated_minutes,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            confidence_decay_rate=decay_rate,
            strength=strength,
            expires_at=expires_at,
            warning_at=warning_at,
            created_at=now,
        )
    
    def _determine_strength(self, confidence: float) -> SignalStrength:
        """Determine signal strength from confidence"""
        if confidence >= 90:
            return SignalStrength.VERY_STRONG
        elif confidence >= 80:
            return SignalStrength.STRONG
        elif confidence >= 70:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_consistency(self, movements: List[np.ndarray]) -> float:
        """
        Calculate pattern consistency
        ดูว่า pattern ในอดีตมีความสม่ำเสมอแค่ไหน
        
        Returns:
            Consistency factor (0.5 - 1.5)
            - 1.5 = Very consistent (patterns move similarly)
            - 1.0 = Normal
            - 0.5 = Inconsistent (patterns vary a lot)
        """
        if not movements or len(movements) < 2:
            return 1.0
        
        try:
            # Normalize movements to percentage changes
            pct_changes = []
            for m in movements:
                if len(m) > 1 and m[0] != 0:
                    final_change = (m[-1] - m[0]) / m[0] * 100
                    pct_changes.append(final_change)
            
            if len(pct_changes) < 2:
                return 1.0
            
            # Calculate coefficient of variation
            mean_change = np.mean(np.abs(pct_changes))
            std_change = np.std(pct_changes)
            
            if mean_change == 0:
                return 1.0
            
            cv = std_change / mean_change
            
            # Convert to factor (lower CV = higher consistency)
            # CV of 0.3 or less = very consistent (factor 1.5)
            # CV of 1.0 or more = inconsistent (factor 0.5)
            consistency_factor = 1.5 - min(1.0, cv) * 1.0
            
            return max(0.5, min(1.5, consistency_factor))
        
        except Exception:
            return 1.0
    
    def _calculate_momentum_stability(self, movements: List[np.ndarray]) -> float:
        """
        Calculate momentum stability
        ดูว่าราคาวิ่งต่อเนื่องแค่ไหน หรือกลับตัวบ่อย
        
        Returns:
            Momentum factor (0.6 - 1.3)
            - 1.3 = Smooth trending (signal lasts longer)
            - 1.0 = Normal
            - 0.6 = Choppy (signal may reverse quickly)
        """
        if not movements or len(movements) < 2:
            return 1.0
        
        try:
            reversal_counts = []
            
            for m in movements:
                if len(m) < 3:
                    continue
                
                # Count direction changes
                diffs = np.diff(m)
                signs = np.sign(diffs)
                sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
                
                # Normalize by number of points
                reversal_ratio = sign_changes / (len(m) - 2)
                reversal_counts.append(reversal_ratio)
            
            if not reversal_counts:
                return 1.0
            
            avg_reversal = np.mean(reversal_counts)
            
            # Convert to factor
            # Low reversal (< 0.3) = smooth trend = factor 1.3
            # High reversal (> 0.6) = choppy = factor 0.6
            if avg_reversal < 0.3:
                return 1.3
            elif avg_reversal > 0.6:
                return 0.6
            else:
                # Linear interpolation
                return 1.3 - (avg_reversal - 0.3) / 0.3 * 0.7
        
        except Exception:
            return 1.0


class PatternAnalyzer:
    """
    Complete Pattern Analysis Pipeline
    รวมทุกอย่าง: Search + Vote + Risk Management
    """
    
    def __init__(
        self,
        similarity_engine,
        voting_system: Optional[VotingSystem] = None,
        min_correlation: float = 0.85
    ):
        """
        Initialize Pattern Analyzer
        
        Args:
            similarity_engine: FAISS engine or PatternMatcher
            voting_system: Voting system (creates default if None)
            min_correlation: Minimum correlation to accept pattern
        """
        self.similarity_engine = similarity_engine
        self.voting_system = voting_system or VotingSystem()
        self.min_correlation = min_correlation
    
    def analyze(
        self,
        query_pattern: np.ndarray,
        current_price: float,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline
        
        Args:
            query_pattern: Current price pattern (normalized)
            current_price: Current actual price
            k: Number of similar patterns to find
        
        Returns:
            Complete analysis result
        """
        # Step 1: Find similar patterns
        matches = self.similarity_engine.find_matches(query_pattern, k=k)
        
        if not matches.get("valid", True) or matches.get("n_matches", 0) == 0:
            return {
                "status": "insufficient_data",
                "message": "ไม่พบ Pattern ที่เหมือนกันเพียงพอ",
                "n_matches": matches.get("n_matches", 0),
                "signal": Signal.WAIT.value,
                "confidence": 0,
            }
        
        # Step 2: Check correlation threshold
        valid_matches = []
        for match in matches["matches"]:
            if match.get("correlation", 0) >= self.min_correlation:
                valid_matches.append(match)
        
        if len(valid_matches) < self.voting_system.min_patterns:
            return {
                "status": "low_correlation",
                "message": f"Pattern ที่เจอมีความเหมือนต่ำกว่า {self.min_correlation * 100}%",
                "n_matches": len(valid_matches),
                "signal": Signal.WAIT.value,
                "confidence": 0,
            }
        
        # Step 3: Extract future movements
        future_movements = [
            match["future"] for match in valid_matches
            if match.get("future") is not None
        ]
        
        if len(future_movements) < self.voting_system.min_patterns:
            return {
                "status": "no_future_data",
                "message": "ไม่มีข้อมูลอนาคตของ Pattern",
                "n_matches": len(valid_matches),
                "signal": Signal.WAIT.value,
                "confidence": 0,
            }
        
        # Step 4: Vote
        vote_result = self.voting_system.analyze_patterns(
            future_movements,
            current_price
        )
        
        # Step 5: Compile results
        return {
            "status": "success",
            "signal": vote_result.signal.value,
            "confidence": vote_result.confidence,
            "vote_details": {
                "bullish": vote_result.bullish_votes,
                "bearish": vote_result.bearish_votes,
                "total": vote_result.total_votes,
            },
            "price_projection": {
                "current": current_price,
                "projected": vote_result.projected_price,
                "stop_loss": vote_result.stop_loss,
                "take_profit": vote_result.take_profit,
            },
            "average_movement": vote_result.average_movement.tolist(),
            "matched_patterns": [
                {
                    "index": m["index"],
                    "correlation": m.get("correlation", 0),
                    "distance": m.get("distance", 0),
                }
                for m in valid_matches
            ],
            "n_matches": len(valid_matches),
            "duration": vote_result.duration.to_dict() if vote_result.duration else None,
        }


def analyze_patterns(top_10_future_movements: List[np.ndarray]) -> Tuple[str, float, np.ndarray]:
    """
    Original function from the spec
    
    Args:
        top_10_future_movements: Future movements from 10 matched patterns
    
    Returns:
        Tuple of (signal, confidence, average_movement)
    """
    voting_system = VotingSystem(
        confidence_threshold=VotingConfig.MIN_CONFIDENCE,
        strong_signal_threshold=VotingConfig.STRONG_SIGNAL
    )
    
    result = voting_system.analyze_patterns(top_10_future_movements)
    
    signal_str = result.signal.value
    if result.signal == Signal.STRONG_BUY:
        signal_str = "BUY"
    elif result.signal == Signal.STRONG_SELL:
        signal_str = "SELL"
    elif result.signal == Signal.BUY:
        signal_str = "BUY"
    elif result.signal == Signal.SELL:
        signal_str = "SELL"
    else:
        signal_str = "WAIT"
    
    return signal_str, result.confidence, result.average_movement


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("Trademify Voting System - Example Usage")
    print("=" * 50)
    
    # Create sample future movements
    np.random.seed(42)
    
    # 8 bullish patterns, 2 bearish patterns
    future_movements = []
    
    # Bullish patterns (end higher than start)
    for i in range(8):
        start = 100
        movement = start + np.cumsum(np.random.randn(10) * 0.5 + 0.3)  # Upward bias
        future_movements.append(movement)
    
    # Bearish patterns (end lower than start)
    for i in range(2):
        start = 100
        movement = start + np.cumsum(np.random.randn(10) * 0.5 - 0.5)  # Downward bias
        future_movements.append(movement)
    
    # Analyze
    voting_system = VotingSystem()
    result = voting_system.analyze_patterns(future_movements, current_price=100.0)
    
    print(f"\nVoting Result:")
    print(f"  Signal: {result.signal.value}")
    print(f"  Confidence: {result.confidence:.1f}%")
    print(f"  Bullish votes: {result.bullish_votes}")
    print(f"  Bearish votes: {result.bearish_votes}")
    print(f"  Projected price: {result.projected_price:.2f}")
    print(f"  Stop Loss: {result.stop_loss:.2f}")
    print(f"  Take Profit: {result.take_profit:.2f}")
    
    # Test original function
    print("\n" + "=" * 50)
    print("Testing original analyze_patterns function")
    print("=" * 50)
    
    signal, confidence, avg_move = analyze_patterns(future_movements)
    print(f"\nSignal: {signal}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Average movement shape: {avg_move.shape}")
    
    # Test with uncertain patterns (5-5 split)
    print("\n" + "=" * 50)
    print("Testing with uncertain patterns (5-5 split)")
    print("=" * 50)
    
    uncertain_movements = []
    for i in range(5):
        start = 100
        movement = start + np.cumsum(np.random.randn(10) * 0.5 + 0.3)
        uncertain_movements.append(movement)
    for i in range(5):
        start = 100
        movement = start + np.cumsum(np.random.randn(10) * 0.5 - 0.3)
        uncertain_movements.append(movement)
    
    signal2, confidence2, _ = analyze_patterns(uncertain_movements)
    print(f"\nSignal: {signal2}")
    print(f"Confidence: {confidence2:.1f}%")
    print("(Should be WAIT because confidence < 70%)")
