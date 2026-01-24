"""
Adaptive Threshold System for Layer 17-20
==========================================

แนวคิด:
- Layer 1-16: เข้มงวดตามหลักการ (Gate Keeper) - ไม่ปรับ
- Layer 17-20 (Ultra/Supreme/Transcendent/Omniscient): ปรับ dynamic

การปรับ threshold ขึ้นอยู่กับ:
1. Base Layer Agreement (กี่ layer จาก 1-16 ที่ผ่าน)
2. Market Condition (volatility, trend strength)
3. Signal Strength (STRONG vs normal)
4. Historical Performance (ถ้ามี)

สูตร:
- ถ้า Layer 1-16 ผ่าน >= 80% → Layer 17-20 ยืดหยุ่นมาก (threshold -30%)
- ถ้า Layer 1-16 ผ่าน >= 60% → Layer 17-20 ยืดหยุ่นปานกลาง (threshold -20%)
- ถ้า Layer 1-16 ผ่าน >= 40% → Layer 17-20 ยืดหยุ่นน้อย (threshold -10%)
- ถ้า Layer 1-16 ผ่าน < 40% → Layer 17-20 เข้มงวดเพิ่ม (threshold +10%)
"""
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AdaptiveMode(Enum):
    """โหมดการปรับ threshold"""
    STRICT = "strict"           # เข้มงวดมาก (+10% threshold)
    NORMAL = "normal"           # ปกติ (threshold ตามค่าเดิม)
    RELAXED = "relaxed"         # ยืดหยุ่นน้อย (-10% threshold)
    FLEXIBLE = "flexible"       # ยืดหยุ่นปานกลาง (-20% threshold)
    VERY_FLEXIBLE = "very_flexible"  # ยืดหยุ่นมาก (-30% threshold)


@dataclass
class AdaptiveContext:
    """Context สำหรับการคำนวณ adaptive threshold"""
    # Base layer results (Layer 1-16)
    layers_passed: int = 0
    layers_total: int = 16
    
    # Signal info
    signal_strength: str = "NORMAL"  # STRONG_BUY/STRONG_SELL vs BUY/SELL
    quality: str = "MEDIUM"
    
    # Market condition
    volatility_state: str = "NORMAL"  # LOW/NORMAL/HIGH/EXTREME
    trend_strength: float = 0.0  # -100 to +100
    market_regime: str = "RANGING"
    
    # Historical performance
    recent_win_rate: float = 0.5
    recent_trades: int = 0
    
    @property
    def agreement_ratio(self) -> float:
        """สัดส่วน layer ที่ผ่าน"""
        if self.layers_total == 0:
            return 0.5
        return self.layers_passed / self.layers_total
    
    @property
    def is_strong_signal(self) -> bool:
        """เป็น STRONG signal หรือไม่"""
        return self.signal_strength in ["STRONG_BUY", "STRONG_SELL"]


class AdaptiveThresholdCalculator:
    """
    คำนวณ threshold แบบ adaptive สำหรับ Layer 17-20
    
    หลักการ:
    - Base threshold คือค่าเริ่มต้นของแต่ละ layer
    - ปรับตาม context (agreement ratio, market condition, signal strength)
    - มี floor/ceiling เพื่อไม่ให้ต่ำ/สูงเกินไป
    """
    
    # Base thresholds (ค่าเดิมที่เข้มงวด)
    BASE_THRESHOLDS = {
        "ultra": {
            "min_confidence": 65,
            "min_size_multiplier": 0.25,
        },
        "supreme": {
            "min_confidence": 60,
            "min_win_probability": 45,
            "min_confluence": 40,
        },
        "transcendent": {
            "min_score": 50,
            "min_win_probability": 40,
        },
        "omniscient": {
            "min_score": 60,
            "min_win_probability": 50,
            "min_expected_value": 0.01,
        },
    }
    
    # Minimum thresholds (floor - ห้ามต่ำกว่านี้)
    MIN_THRESHOLDS = {
        "ultra": {
            "min_confidence": 40,
            "min_size_multiplier": 0.10,
        },
        "supreme": {
            "min_confidence": 35,
            "min_win_probability": 30,
            "min_confluence": 25,
        },
        "transcendent": {
            "min_score": 30,
            "min_win_probability": 25,
        },
        "omniscient": {
            "min_score": 35,
            "min_win_probability": 30,
            "min_expected_value": -0.01,
        },
    }
    
    def __init__(self):
        self._current_mode = AdaptiveMode.NORMAL
        self._last_context: Optional[AdaptiveContext] = None
    
    def calculate_mode(self, context: AdaptiveContext) -> AdaptiveMode:
        """คำนวณโหมดจาก context"""
        ratio = context.agreement_ratio
        
        # Base mode from agreement ratio
        if ratio >= 0.80:
            mode = AdaptiveMode.VERY_FLEXIBLE
        elif ratio >= 0.60:
            mode = AdaptiveMode.FLEXIBLE
        elif ratio >= 0.40:
            mode = AdaptiveMode.RELAXED
        else:
            mode = AdaptiveMode.STRICT
        
        # Adjust for strong signals
        if context.is_strong_signal and mode != AdaptiveMode.VERY_FLEXIBLE:
            # Move one level more flexible for strong signals
            mode_order = [AdaptiveMode.STRICT, AdaptiveMode.NORMAL, AdaptiveMode.RELAXED, 
                         AdaptiveMode.FLEXIBLE, AdaptiveMode.VERY_FLEXIBLE]
            current_idx = mode_order.index(mode)
            mode = mode_order[min(current_idx + 1, len(mode_order) - 1)]
        
        # Adjust for high volatility (be more cautious)
        if context.volatility_state == "EXTREME":
            # Move one level more strict
            mode_order = [AdaptiveMode.STRICT, AdaptiveMode.NORMAL, AdaptiveMode.RELAXED, 
                         AdaptiveMode.FLEXIBLE, AdaptiveMode.VERY_FLEXIBLE]
            current_idx = mode_order.index(mode)
            mode = mode_order[max(current_idx - 1, 0)]
        
        # Adjust for good historical performance
        if context.recent_trades >= 10 and context.recent_win_rate >= 0.6:
            # Good track record - can be more flexible
            mode_order = [AdaptiveMode.STRICT, AdaptiveMode.NORMAL, AdaptiveMode.RELAXED, 
                         AdaptiveMode.FLEXIBLE, AdaptiveMode.VERY_FLEXIBLE]
            current_idx = mode_order.index(mode)
            mode = mode_order[min(current_idx + 1, len(mode_order) - 1)]
        
        self._current_mode = mode
        self._last_context = context
        return mode
    
    def get_multiplier(self, mode: AdaptiveMode) -> float:
        """
        Get threshold multiplier
        - > 1.0 = more strict (increase threshold)
        - < 1.0 = more flexible (decrease threshold)
        """
        multipliers = {
            AdaptiveMode.STRICT: 1.10,        # +10%
            AdaptiveMode.NORMAL: 1.00,        # ปกติ
            AdaptiveMode.RELAXED: 0.90,       # -10%
            AdaptiveMode.FLEXIBLE: 0.80,      # -20%
            AdaptiveMode.VERY_FLEXIBLE: 0.70, # -30%
        }
        return multipliers.get(mode, 1.0)
    
    def calculate_thresholds(self, layer: str, context: AdaptiveContext) -> Dict[str, float]:
        """คำนวณ thresholds สำหรับ layer ที่ระบุ"""
        mode = self.calculate_mode(context)
        multiplier = self.get_multiplier(mode)
        
        base = self.BASE_THRESHOLDS.get(layer, {})
        minimum = self.MIN_THRESHOLDS.get(layer, {})
        
        result = {}
        for key, base_value in base.items():
            min_value = minimum.get(key, base_value * 0.5)
            
            # Apply multiplier
            adjusted = base_value * multiplier
            
            # Ensure within bounds
            adjusted = max(min_value, adjusted)
            
            result[key] = adjusted
        
        logger.info(f"🎛️ [Adaptive] {layer} mode={mode.value}, multiplier={multiplier:.2f}")
        logger.info(f"   Base: {base}")
        logger.info(f"   Adjusted: {result}")
        
        return result
    
    def should_allow_trade(
        self,
        layer: str,
        layer_result: Dict[str, Any],
        context: AdaptiveContext
    ) -> tuple[bool, str]:
        """
        ตัดสินใจว่า layer นี้ควรอนุญาตให้เทรดหรือไม่
        
        Returns:
            (can_trade, reason)
        """
        thresholds = self.calculate_thresholds(layer, context)
        
        if layer == "ultra":
            confidence = layer_result.get("confidence", 0)
            size_mult = layer_result.get("size_multiplier", 0)
            
            if confidence < thresholds["min_confidence"]:
                return False, f"Confidence {confidence:.1f} < adaptive threshold {thresholds['min_confidence']:.1f}"
            if size_mult < thresholds["min_size_multiplier"]:
                return False, f"Size multiplier {size_mult:.2f} < adaptive threshold {thresholds['min_size_multiplier']:.2f}"
            
            return True, f"Passed adaptive (confidence={confidence:.1f}, threshold={thresholds['min_confidence']:.1f})"
        
        elif layer == "supreme":
            confidence = layer_result.get("confidence", 0)
            win_prob = layer_result.get("win_probability", 0)
            confluence = layer_result.get("confluence_score", 0)
            
            if confidence < thresholds["min_confidence"]:
                return False, f"Confidence {confidence:.1f} < {thresholds['min_confidence']:.1f}"
            if win_prob < thresholds["min_win_probability"]:
                return False, f"Win probability {win_prob:.1f} < {thresholds['min_win_probability']:.1f}"
            if confluence < thresholds["min_confluence"]:
                return False, f"Confluence {confluence:.1f} < {thresholds['min_confluence']:.1f}"
            
            return True, "Passed adaptive supreme thresholds"
        
        elif layer == "transcendent":
            score = layer_result.get("transcendent_score", 0)
            win_prob = layer_result.get("win_probability", 0)
            
            if score < thresholds["min_score"]:
                return False, f"Score {score:.1f} < {thresholds['min_score']:.1f}"
            if win_prob < thresholds["min_win_probability"]:
                return False, f"Win probability {win_prob:.1f} < {thresholds['min_win_probability']:.1f}"
            
            return True, "Passed adaptive transcendent thresholds"
        
        elif layer == "omniscient":
            score = layer_result.get("omniscient_score", 0)
            win_prob = layer_result.get("win_probability", 0)
            ev = layer_result.get("expected_value", 0)
            
            if score < thresholds["min_score"]:
                return False, f"Score {score:.1f} < {thresholds['min_score']:.1f}"
            if win_prob < thresholds["min_win_probability"]:
                return False, f"Win probability {win_prob:.1f} < {thresholds['min_win_probability']:.1f}"
            if ev < thresholds["min_expected_value"]:
                return False, f"Expected value {ev:.4f} < {thresholds['min_expected_value']:.4f}"
            
            return True, "Passed adaptive omniscient thresholds"
        
        return True, "Unknown layer - allowing by default"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current adaptive state summary"""
        return {
            "current_mode": self._current_mode.value if self._current_mode else "unknown",
            "multiplier": self.get_multiplier(self._current_mode) if self._current_mode else 1.0,
            "context": {
                "layers_passed": self._last_context.layers_passed if self._last_context else 0,
                "layers_total": self._last_context.layers_total if self._last_context else 16,
                "agreement_ratio": self._last_context.agreement_ratio if self._last_context else 0,
                "is_strong_signal": self._last_context.is_strong_signal if self._last_context else False,
            } if self._last_context else None,
        }


# Singleton instance
_adaptive_calculator: Optional[AdaptiveThresholdCalculator] = None

def get_adaptive_calculator() -> AdaptiveThresholdCalculator:
    """Get singleton instance"""
    global _adaptive_calculator
    if _adaptive_calculator is None:
        _adaptive_calculator = AdaptiveThresholdCalculator()
    return _adaptive_calculator


def count_base_layers_passed(analysis_results: Dict[str, Any]) -> tuple[int, int]:
    """
    นับจำนวน base layers (1-16) ที่ผ่าน
    
    Base Layers:
    1. Data Lake - READY
    2. Pattern Matcher - matches > 0
    3. Voting - signal != WAIT
    4. Enhanced - quality >= MEDIUM
    5. Advanced - can_trade
    6. Smart Brain - can_trade
    7. Neural Brain - can_trade
    8. Deep Intelligence - should_trade
    9. Quantum Strategy - should_trade
    10. Alpha Engine - should_trade
    11. Omega Brain - should_trade
    12. Titan Core - should_trade
    13. Continuous Learning - active
    14. Pro Features - can_trade
    15. Risk Guardian - can_trade
    16. Sentiment - no_override
    """
    passed = 0
    total = 16
    
    layers = analysis_results.get("layers", {})
    
    # 1. Data Lake
    if layers.get("data_lake", {}).get("status") == "READY":
        passed += 1
    
    # 2. Pattern Matcher
    if layers.get("pattern_matcher", {}).get("matches", 0) > 0:
        passed += 1
    
    # 3. Voting
    if layers.get("voting", {}).get("signal") not in [None, "WAIT"]:
        passed += 1
    
    # 4. Enhanced
    quality = layers.get("enhanced", {}).get("quality", "SKIP")
    if quality in ["MEDIUM", "HIGH", "PREMIUM"]:
        passed += 1
    
    # 5. Advanced
    if layers.get("advanced", {}).get("can_trade", True):  # Default True ถ้าไม่มีข้อมูล
        passed += 1
    
    # 6. Smart Brain
    if layers.get("smart", {}).get("can_trade", True):
        passed += 1
    
    # 7. Neural Brain
    if layers.get("neural", {}).get("can_trade", True):
        passed += 1
    
    # 8. Deep Intelligence
    if layers.get("deep", {}).get("should_trade", True):
        passed += 1
    
    # 9. Quantum Strategy
    if layers.get("quantum", {}).get("should_trade", True):
        passed += 1
    
    # 10. Alpha Engine
    if layers.get("alpha", {}).get("should_trade", True):
        passed += 1
    
    # 11. Omega Brain
    if layers.get("omega", {}).get("should_trade", True):
        passed += 1
    
    # 12. Titan Core
    if layers.get("titan", {}).get("should_trade", True):
        passed += 1
    
    # 13. Continuous Learning
    if layers.get("learning", {}).get("cycles", 0) >= 0:  # มี learning อยู่
        passed += 1
    
    # 14. Pro Features
    if layers.get("pro", {}).get("can_trade", True):
        passed += 1
    
    # 15. Risk Guardian
    if layers.get("risk", {}).get("can_trade", True):
        passed += 1
    
    # 16. Sentiment
    if layers.get("sentiment", {}).get("override") != "BLOCK":
        passed += 1
    
    return passed, total
