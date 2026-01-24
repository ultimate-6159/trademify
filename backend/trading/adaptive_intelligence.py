"""
Adaptive Intelligence System for Layer 17-20

แนวคิดหลัก:
- Layer 1-16: เข้มงวดคงที่ (Gate Keepers)
- Layer 17-20: ปรับตัวตามสถานการณ์ (Adaptive)

การทำงาน:
1. นับจำนวน Layer 1-16 ที่ผ่าน
2. คำนวณ Adaptive Mode ตามผลลัพธ์
3. ปรับ Threshold ของ Layer 17-20 ตาม Mode
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveMode(Enum):
    """โหมดการปรับตัว"""
    VERY_STRICT = "very_strict"    # Layer 1-16 ผ่านน้อย → ต้องเข้มงวดสุด
    STRICT = "strict"              # Layer 1-16 ผ่านบางส่วน → เข้มงวด
    NORMAL = "normal"              # Layer 1-16 ผ่านปานกลาง → ปกติ
    RELAXED = "relaxed"            # Layer 1-16 ผ่านมาก → ผ่อนคลาย
    FLEXIBLE = "flexible"          # Layer 1-16 ผ่านเกือบหมด → ยืดหยุ่น


@dataclass
class LayerResult:
    """ผลลัพธ์ของแต่ละ Layer"""
    layer_name: str
    layer_number: int
    can_trade: bool
    score: float
    confidence: float
    message: str
    is_base_layer: bool = True  # True for Layer 1-16


@dataclass
class AdaptiveContext:
    """Context สำหรับการตัดสินใจแบบ Adaptive"""
    base_layer_results: List[LayerResult]  # ผลลัพธ์ Layer 1-16
    signal_direction: str  # BUY / SELL
    signal_confidence: float  # ความมั่นใจของ signal
    pattern_similarity: float  # DTW similarity
    market_volatility: float  # ความผันผวน (0-100)
    trend_strength: float  # ความแรงของ trend (0-100)
    recent_win_rate: Optional[float] = None  # Win rate ล่าสุด


@dataclass
class AdaptiveDecision:
    """ผลการตัดสินใจแบบ Adaptive"""
    mode: AdaptiveMode
    can_trade: bool
    adjusted_thresholds: Dict[str, float]  # threshold ที่ปรับแล้ว
    position_multiplier: float  # ตัวคูณ position size
    confidence_boost: float  # เพิ่มความมั่นใจ
    reasons: List[str]  # เหตุผล


class AdaptiveIntelligence:
    """
    Adaptive Intelligence System
    
    ปรับ Layer 17-20 thresholds ตาม Layer 1-16 results
    """
    
    # Layer 1-16 definitions (Base Layers - Gate Keepers)
    BASE_LAYERS = {
        1: "Data Lake",
        2: "Pattern Matcher (DTW/FAISS)",
        3: "Voting System",
        4: "Enhanced Analyzer",
        5: "Advanced Intelligence (MTF)",
        6: "Smart Brain",
        7: "Neural Brain",
        8: "Deep Intelligence",
        9: "Quantum Strategy",
        10: "Alpha Engine",
        11: "Omega Brain",
        12: "Titan Core",
        13: "Continuous Learning",
        14: "Pro Features",
        15: "Risk Guardian",
        16: "Sentiment Analyzer"
    }
    
    # Layer 17-20 definitions (Adaptive Layers)
    ADAPTIVE_LAYERS = {
        17: "Ultra Intelligence",
        18: "Supreme Intelligence",
        19: "Transcendent Intelligence",
        20: "Omniscient Intelligence"
    }
    
    # Default thresholds for Layer 17-20
    DEFAULT_THRESHOLDS = {
        "ultra": {
            "min_score": 70,
            "min_confidence": 65,
            "min_factors": 5
        },
        "supreme": {
            "min_score": 75,
            "min_confidence": 70,
            "min_factors": 6
        },
        "transcendent": {
            "min_score": 80,
            "min_confidence": 75,
            "min_factors": 7
        },
        "omniscient": {
            "min_score": 85,
            "min_confidence": 80,
            "min_factors": 8
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, context: AdaptiveContext) -> AdaptiveDecision:
        """
        วิเคราะห์และตัดสินใจแบบ Adaptive
        
        Args:
            context: ข้อมูล context สำหรับการตัดสินใจ
            
        Returns:
            AdaptiveDecision: ผลการตัดสินใจ
        """
        # 1. นับ Layer 1-16 ที่ผ่าน
        passed_count, total_count = self._count_passed_layers(context.base_layer_results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        # 2. คำนวณ Adaptive Mode
        mode = self._calculate_mode(pass_rate, context)
        
        # 3. คำนวณ adjusted thresholds
        adjusted_thresholds = self._calculate_thresholds(mode)
        
        # 4. คำนวณ position multiplier
        position_multiplier = self._calculate_position_multiplier(mode, context)
        
        # 5. คำนวณ confidence boost
        confidence_boost = self._calculate_confidence_boost(mode, pass_rate)
        
        # 6. สรุป reasons
        reasons = self._generate_reasons(mode, passed_count, total_count, context)
        
        # 7. ตัดสินใจ can_trade
        can_trade = self._should_allow_trade(mode, passed_count, context)
        
        return AdaptiveDecision(
            mode=mode,
            can_trade=can_trade,
            adjusted_thresholds=adjusted_thresholds,
            position_multiplier=position_multiplier,
            confidence_boost=confidence_boost,
            reasons=reasons
        )
    
    def _count_passed_layers(self, results: List[LayerResult]) -> tuple:
        """นับจำนวน Layer ที่ผ่าน"""
        total = len(results)
        passed = sum(1 for r in results if r.can_trade)
        return passed, total
    
    def _calculate_mode(self, pass_rate: float, context: AdaptiveContext) -> AdaptiveMode:
        """
        คำนวณ Adaptive Mode ตาม pass rate และ context
        
        Pass Rate Mapping:
        - 80%+ → FLEXIBLE
        - 60-79% → RELAXED  
        - 40-59% → NORMAL
        - 20-39% → STRICT
        - <20% → VERY_STRICT
        """
        # Base mode from pass rate
        if pass_rate >= 80:
            base_mode = AdaptiveMode.FLEXIBLE
        elif pass_rate >= 60:
            base_mode = AdaptiveMode.RELAXED
        elif pass_rate >= 40:
            base_mode = AdaptiveMode.NORMAL
        elif pass_rate >= 20:
            base_mode = AdaptiveMode.STRICT
        else:
            base_mode = AdaptiveMode.VERY_STRICT
        
        # Adjust based on additional factors
        mode = base_mode
        
        # Strong signal → +1 flexibility
        if context.signal_confidence >= 80 and context.pattern_similarity >= 85:
            mode = self._increase_flexibility(mode)
            
        # High volatility → -1 flexibility (more cautious)
        if context.market_volatility >= 80:
            mode = self._decrease_flexibility(mode)
            
        # Very strong trend → +1 flexibility
        if context.trend_strength >= 75:
            mode = self._increase_flexibility(mode)
            
        # Good win rate → +1 flexibility
        if context.recent_win_rate and context.recent_win_rate >= 60:
            mode = self._increase_flexibility(mode)
        
        return mode
    
    def _increase_flexibility(self, mode: AdaptiveMode) -> AdaptiveMode:
        """เพิ่ม flexibility 1 level"""
        order = [
            AdaptiveMode.VERY_STRICT,
            AdaptiveMode.STRICT,
            AdaptiveMode.NORMAL,
            AdaptiveMode.RELAXED,
            AdaptiveMode.FLEXIBLE
        ]
        idx = order.index(mode)
        return order[min(idx + 1, len(order) - 1)]
    
    def _decrease_flexibility(self, mode: AdaptiveMode) -> AdaptiveMode:
        """ลด flexibility 1 level"""
        order = [
            AdaptiveMode.VERY_STRICT,
            AdaptiveMode.STRICT,
            AdaptiveMode.NORMAL,
            AdaptiveMode.RELAXED,
            AdaptiveMode.FLEXIBLE
        ]
        idx = order.index(mode)
        return order[max(idx - 1, 0)]
    
    def _calculate_thresholds(self, mode: AdaptiveMode) -> Dict[str, float]:
        """
        คำนวณ thresholds ที่ปรับแล้วตาม mode
        
        Adjustment Factors:
        - FLEXIBLE: -30% (ผ่อนปรนมาก)
        - RELAXED: -20%
        - NORMAL: 0%
        - STRICT: +10%
        - VERY_STRICT: +20%
        """
        adjustments = {
            AdaptiveMode.FLEXIBLE: 0.70,      # -30%
            AdaptiveMode.RELAXED: 0.80,       # -20%
            AdaptiveMode.NORMAL: 1.00,        # 0%
            AdaptiveMode.STRICT: 1.10,        # +10%
            AdaptiveMode.VERY_STRICT: 1.20    # +20%
        }
        
        factor = adjustments.get(mode, 1.0)
        
        result = {}
        for layer_name, thresholds in self.DEFAULT_THRESHOLDS.items():
            result[layer_name] = {
                "min_score": thresholds["min_score"] * factor,
                "min_confidence": thresholds["min_confidence"] * factor,
                "min_factors": max(3, int(thresholds["min_factors"] * factor))
            }
        
        return result
    
    def _calculate_position_multiplier(self, mode: AdaptiveMode, context: AdaptiveContext) -> float:
        """
        คำนวณ position size multiplier
        
        - FLEXIBLE: 1.0x (full size)
        - RELAXED: 0.8x
        - NORMAL: 0.6x
        - STRICT: 0.4x
        - VERY_STRICT: 0.2x
        """
        base_multipliers = {
            AdaptiveMode.FLEXIBLE: 1.0,
            AdaptiveMode.RELAXED: 0.8,
            AdaptiveMode.NORMAL: 0.6,
            AdaptiveMode.STRICT: 0.4,
            AdaptiveMode.VERY_STRICT: 0.2
        }
        
        multiplier = base_multipliers.get(mode, 0.5)
        
        # Bonus for high confidence
        if context.signal_confidence >= 85:
            multiplier *= 1.1
        
        # Penalty for high volatility
        if context.market_volatility >= 70:
            multiplier *= 0.8
        
        # Cap at 1.0
        return min(multiplier, 1.0)
    
    def _calculate_confidence_boost(self, mode: AdaptiveMode, pass_rate: float) -> float:
        """
        คำนวณ confidence boost
        
        ถ้า Layer 1-16 ผ่านมาก → เพิ่มความมั่นใจให้ Layer 17-20
        """
        if mode == AdaptiveMode.FLEXIBLE:
            return 15.0
        elif mode == AdaptiveMode.RELAXED:
            return 10.0
        elif mode == AdaptiveMode.NORMAL:
            return 5.0
        elif mode == AdaptiveMode.STRICT:
            return 0.0
        else:
            return -5.0  # ลด confidence ถ้า base layers ไม่ผ่าน
    
    def _should_allow_trade(self, mode: AdaptiveMode, passed_count: int, context: AdaptiveContext) -> bool:
        """
        ตัดสินใจว่าควรอนุญาตให้เทรดหรือไม่
        
        Rules:
        1. ต้องผ่านอย่างน้อย 6 Layer จาก 16 (37.5%)
        2. Signal confidence ต้อง >= 50%
        3. Pattern similarity ต้อง >= 60%
        """
        # Minimum requirements
        MIN_PASSED_LAYERS = 6
        MIN_SIGNAL_CONFIDENCE = 50
        MIN_PATTERN_SIMILARITY = 60
        
        # Check requirements
        if passed_count < MIN_PASSED_LAYERS:
            self.logger.info(f"❌ Adaptive: Only {passed_count}/16 layers passed (need {MIN_PASSED_LAYERS})")
            return False
        
        if context.signal_confidence < MIN_SIGNAL_CONFIDENCE:
            self.logger.info(f"❌ Adaptive: Signal confidence {context.signal_confidence}% < {MIN_SIGNAL_CONFIDENCE}%")
            return False
        
        if context.pattern_similarity < MIN_PATTERN_SIMILARITY:
            self.logger.info(f"❌ Adaptive: Pattern similarity {context.pattern_similarity}% < {MIN_PATTERN_SIMILARITY}%")
            return False
        
        # All checks passed
        return True
    
    def _generate_reasons(self, mode: AdaptiveMode, passed: int, total: int, context: AdaptiveContext) -> List[str]:
        """สร้าง reasons สำหรับการตัดสินใจ"""
        reasons = []
        
        reasons.append(f"📊 Base Layers: {passed}/{total} ผ่าน ({passed/total*100:.0f}%)")
        reasons.append(f"🎯 Adaptive Mode: {mode.value}")
        
        if context.signal_confidence >= 80:
            reasons.append(f"💪 Strong Signal: {context.signal_confidence:.0f}%")
        
        if context.pattern_similarity >= 85:
            reasons.append(f"🔍 High Pattern Match: {context.pattern_similarity:.0f}%")
        
        if context.market_volatility >= 70:
            reasons.append(f"⚠️ High Volatility: {context.market_volatility:.0f}%")
        
        if context.trend_strength >= 70:
            reasons.append(f"📈 Strong Trend: {context.trend_strength:.0f}%")
        
        return reasons


# ======================
# Helper Functions
# ======================

def create_layer_result(
    layer_name: str,
    layer_number: int,
    can_trade: bool,
    score: float,
    confidence: float,
    message: str
) -> LayerResult:
    """สร้าง LayerResult"""
    return LayerResult(
        layer_name=layer_name,
        layer_number=layer_number,
        can_trade=can_trade,
        score=score,
        confidence=confidence,
        message=message,
        is_base_layer=(layer_number <= 16)
    )


def collect_base_layer_results(
    voting_result: Any = None,
    enhanced_result: Any = None,
    advanced_result: Any = None,
    smart_brain_result: Any = None,
    neural_result: Any = None,
    deep_result: Any = None,
    quantum_result: Any = None,
    alpha_result: Any = None,
    omega_result: Any = None,
    titan_result: Any = None,
    pro_result: Any = None,
    risk_result: Any = None,
    learning_result: Any = None,
    sentiment_result: Any = None,
    pattern_confidence: float = 0,
    data_lake_valid: bool = False
) -> List[LayerResult]:
    """
    รวบรวมผลลัพธ์จาก Layer 1-16
    
    แต่ละ layer ที่ส่งมาจะถูกแปลงเป็น LayerResult
    """
    results = []
    
    # Layer 1: Data Lake
    results.append(create_layer_result(
        "Data Lake", 1, data_lake_valid, 
        100 if data_lake_valid else 0, 
        100 if data_lake_valid else 0,
        "Data validated" if data_lake_valid else "No data"
    ))
    
    # Layer 2: Pattern Matcher
    pattern_passed = pattern_confidence >= 60
    results.append(create_layer_result(
        "Pattern Matcher", 2, pattern_passed,
        pattern_confidence, pattern_confidence,
        f"Pattern similarity: {pattern_confidence:.0f}%"
    ))
    
    # Layer 3: Voting System
    if voting_result:
        can_trade = getattr(voting_result, 'can_trade', False)
        score = getattr(voting_result, 'final_score', 0)
        results.append(create_layer_result(
            "Voting System", 3, can_trade, score, score,
            getattr(voting_result, 'message', '')
        ))
    else:
        results.append(create_layer_result("Voting System", 3, False, 0, 0, "No result"))
    
    # Layer 4: Enhanced Analyzer
    if enhanced_result:
        can_trade = getattr(enhanced_result, 'can_trade', False)
        score = getattr(enhanced_result, 'confidence', 0)
        results.append(create_layer_result(
            "Enhanced Analyzer", 4, can_trade, score, score,
            getattr(enhanced_result, 'message', '')
        ))
    else:
        results.append(create_layer_result("Enhanced Analyzer", 4, False, 0, 0, "No result"))
    
    # Layer 5: Advanced Intelligence
    if advanced_result:
        can_trade = getattr(advanced_result, 'can_trade', False)
        score = getattr(advanced_result, 'confidence', 0)
        results.append(create_layer_result(
            "Advanced Intelligence", 5, can_trade, score, score,
            getattr(advanced_result, 'message', '')
        ))
    else:
        results.append(create_layer_result("Advanced Intelligence", 5, False, 0, 0, "No result"))
    
    # Layer 6: Smart Brain
    if smart_brain_result:
        can_trade = getattr(smart_brain_result, 'should_trade', False)
        score = getattr(smart_brain_result, 'confidence', 0)
        results.append(create_layer_result(
            "Smart Brain", 6, can_trade, score, score,
            getattr(smart_brain_result, 'reasoning', '')
        ))
    else:
        results.append(create_layer_result("Smart Brain", 6, False, 0, 0, "No result"))
    
    # Layer 7: Neural Brain
    if neural_result:
        can_trade = getattr(neural_result, 'should_trade', False)
        score = getattr(neural_result, 'confidence', 0)
        results.append(create_layer_result(
            "Neural Brain", 7, can_trade, score, score,
            getattr(neural_result, 'reasoning', '')
        ))
    else:
        results.append(create_layer_result("Neural Brain", 7, False, 0, 0, "No result"))
    
    # Layer 8: Deep Intelligence
    if deep_result:
        can_trade = getattr(deep_result, 'should_trade', False)
        score = getattr(deep_result, 'confidence', 0)
        results.append(create_layer_result(
            "Deep Intelligence", 8, can_trade, score, score,
            str(getattr(deep_result, 'factors', {}))
        ))
    else:
        results.append(create_layer_result("Deep Intelligence", 8, False, 0, 0, "No result"))
    
    # Layer 9: Quantum Strategy
    if quantum_result:
        can_trade = getattr(quantum_result, 'should_trade', False)
        score = getattr(quantum_result, 'confidence', 0)
        results.append(create_layer_result(
            "Quantum Strategy", 9, can_trade, score, score,
            getattr(quantum_result, 'reasoning', '')
        ))
    else:
        results.append(create_layer_result("Quantum Strategy", 9, False, 0, 0, "No result"))
    
    # Layer 10: Alpha Engine
    if alpha_result:
        can_trade = getattr(alpha_result, 'should_trade', False)
        score = getattr(alpha_result, 'confidence', 0)
        results.append(create_layer_result(
            "Alpha Engine", 10, can_trade, score, score,
            getattr(alpha_result, 'reasoning', '')
        ))
    else:
        results.append(create_layer_result("Alpha Engine", 10, False, 0, 0, "No result"))
    
    # Layer 11: Omega Brain
    if omega_result:
        can_trade = getattr(omega_result, 'should_trade', False)
        score = getattr(omega_result, 'confidence', 0)
        results.append(create_layer_result(
            "Omega Brain", 11, can_trade, score, score,
            str(getattr(omega_result, 'factors', {}))
        ))
    else:
        results.append(create_layer_result("Omega Brain", 11, False, 0, 0, "No result"))
    
    # Layer 12: Titan Core
    if titan_result:
        can_trade = getattr(titan_result, 'should_trade', False)
        score = getattr(titan_result, 'confidence', 0)
        results.append(create_layer_result(
            "Titan Core", 12, can_trade, score, score,
            getattr(titan_result, 'reasoning', '')
        ))
    else:
        results.append(create_layer_result("Titan Core", 12, False, 0, 0, "No result"))
    
    # Layer 13: Continuous Learning
    if learning_result:
        can_trade = getattr(learning_result, 'should_trade', True)  # Learning usually doesn't block
        score = getattr(learning_result, 'confidence', 50)
        results.append(create_layer_result(
            "Continuous Learning", 13, can_trade, score, score,
            "Learning system active"
        ))
    else:
        results.append(create_layer_result("Continuous Learning", 13, True, 50, 50, "Default"))
    
    # Layer 14: Pro Features
    if pro_result:
        can_trade = getattr(pro_result, 'should_trade', False)
        score = getattr(pro_result, 'confidence', 0)
        results.append(create_layer_result(
            "Pro Features", 14, can_trade, score, score,
            getattr(pro_result, 'reasoning', '')
        ))
    else:
        results.append(create_layer_result("Pro Features", 14, False, 0, 0, "No result"))
    
    # Layer 15: Risk Guardian
    if risk_result:
        can_trade = getattr(risk_result, 'can_trade', False)
        score = getattr(risk_result, 'risk_score', 0)
        results.append(create_layer_result(
            "Risk Guardian", 15, can_trade, score, score,
            f"Risk level: {getattr(risk_result, 'risk_level', 'unknown')}"
        ))
    else:
        results.append(create_layer_result("Risk Guardian", 15, False, 0, 0, "No result"))
    
    # Layer 16: Sentiment Analyzer
    if sentiment_result:
        # Sentiment usually returns score -100 to +100
        sentiment_score = getattr(sentiment_result, 'score', 0)
        can_trade = abs(sentiment_score) >= 20  # มี sentiment ที่ชัดเจน
        normalized = (sentiment_score + 100) / 2  # Convert to 0-100
        results.append(create_layer_result(
            "Sentiment Analyzer", 16, can_trade, normalized, normalized,
            f"Sentiment: {sentiment_score}"
        ))
    else:
        results.append(create_layer_result("Sentiment Analyzer", 16, True, 50, 50, "Neutral"))
    
    return results


# ======================
# Singleton Instance
# ======================
_adaptive_intelligence = None

def get_adaptive_intelligence() -> AdaptiveIntelligence:
    """Get singleton instance"""
    global _adaptive_intelligence
    if _adaptive_intelligence is None:
        _adaptive_intelligence = AdaptiveIntelligence()
    return _adaptive_intelligence
