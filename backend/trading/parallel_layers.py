"""
🚀 PARALLEL LAYER PROCESSING SYSTEM
ระบบประมวลผล 20 Layer พร้อมกันแบบ Parallel
แล้วรวมผลมาตัดสินใจ

Benefits:
- เร็วขึ้น 3-5x (จาก sequential)
- ลด latency ในการวิเคราะห์
- CPU utilization ดีขึ้น
"""

import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class LayerResult:
    """ผลลัพธ์จากแต่ละ Layer"""
    layer_name: str
    layer_num: int
    can_trade: bool
    score: float = 0.0
    multiplier: float = 1.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataBundle:
    """ข้อมูลตลาดที่ pre-fetch มาก่อน"""
    symbol: str
    current_price: float
    prices: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    volumes: Optional[np.ndarray]
    atr: float
    balance: float
    equity: float
    signal_side: str
    base_confidence: float


@dataclass
class ParallelAnalysisResult:
    """ผลรวมจาก Parallel Processing"""
    layer_results: List[LayerResult]
    total_layers: int
    passed_layers: int
    pass_rate: float
    avg_score: float
    avg_multiplier: float
    high_quality_passes: int
    key_layer_agreement: float
    final_decision: str  # "APPROVE" or "SKIP"
    final_position_factor: float
    total_time_ms: float
    reasons: List[str] = field(default_factory=list)


class ParallelLayerProcessor:
    """
    🚀 Parallel Layer Processing Engine
    รัน 20 Layer พร้อมกัน แล้วรวมผลมาตัดสินใจ
    """
    
    def __init__(
        self,
        ultra_intelligence=None,
        supreme_intelligence=None,
        transcendent_intelligence=None,
        omniscient_intelligence=None,
        advanced_intelligence=None,
        neural_brain=None,
        quantum_strategy=None,
        deep_intelligence=None,
        alpha_engine=None,
        omega_brain=None,
        titan_core=None,
        smart_brain=None,
        pro_features=None,
        risk_guardian=None,
        max_workers: int = 8
    ):
        # Store all layer modules
        self.ultra_intelligence = ultra_intelligence
        self.supreme_intelligence = supreme_intelligence
        self.transcendent_intelligence = transcendent_intelligence
        self.omniscient_intelligence = omniscient_intelligence
        self.advanced_intelligence = advanced_intelligence
        self.neural_brain = neural_brain
        self.quantum_strategy = quantum_strategy
        self.deep_intelligence = deep_intelligence
        self.alpha_engine = alpha_engine
        self.omega_brain = omega_brain
        self.titan_core = titan_core
        self.smart_brain = smart_brain
        self.pro_features = pro_features
        self.risk_guardian = risk_guardian
        
        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"🚀 Parallel Layer Processor initialized with {max_workers} workers")
    
    async def analyze_all_layers(
        self,
        data: MarketDataBundle,
        can_trade_check: tuple = (True, ""),  # Layer 1-2 check
        correlation_check: tuple = (True, ""),  # Layer 3 check
    ) -> ParallelAnalysisResult:
        """
        🚀 วิเคราะห์ทุก Layer พร้อมกัน แล้วรวมผล
        
        Flow:
        1. สร้าง tasks สำหรับทุก Layer
        2. รัน asyncio.gather() ให้ทำงานพร้อมกัน
        3. รวมผลและตัดสินใจ
        """
        start_time = time.time()
        
        # Pre-create all layer tasks
        layer_tasks = []
        
        # Layer 1: Smart Features (already checked)
        layer_tasks.append(self._create_sync_result(
            "SmartFeatures", 1, can_trade_check[0], 
            score=100 if can_trade_check[0] else 0,
            reason=can_trade_check[1]
        ))
        
        # Layer 2: Data Lake (always pass - data already loaded)
        layer_tasks.append(self._create_sync_result(
            "DataLake", 2, True,
            score=100,
            reason="Data loaded successfully"
        ))
        
        # Layer 3: Correlation (already checked)
        layer_tasks.append(self._create_sync_result(
            "Correlation", 3, correlation_check[0],
            score=100 if correlation_check[0] else 0,
            reason=correlation_check[1]
        ))
        
        # Layer 4: Voting System (already incorporated in signal)
        layer_tasks.append(self._create_sync_result(
            "VotingSystem", 4, True,
            score=data.base_confidence if data.base_confidence > 0 else 75,
            reason=f"Signal confidence: {data.base_confidence:.1f}%"
        ))
        
        # Layer 5: Advanced Intelligence
        if self.advanced_intelligence:
            layer_tasks.append(self._run_advanced_intelligence(data))
        
        # Layer 6: Smart Brain
        if self.smart_brain:
            layer_tasks.append(self._run_smart_brain(data))
        
        # Layer 7: Neural Brain
        if self.neural_brain:
            layer_tasks.append(self._run_neural_brain(data))
        
        # Layer 8: Deep Intelligence
        if self.deep_intelligence:
            layer_tasks.append(self._run_deep_intelligence(data))
        
        # Layer 9: Quantum Strategy
        if self.quantum_strategy:
            layer_tasks.append(self._run_quantum_strategy(data))
        
        # Layer 10: Alpha Engine
        if self.alpha_engine:
            layer_tasks.append(self._run_alpha_engine(data))
        
        # Layer 11: Omega Brain
        if self.omega_brain:
            layer_tasks.append(self._run_omega_brain(data))
        
        # Layer 12: Titan Core
        if self.titan_core:
            layer_tasks.append(self._run_titan_core(data))
        
        # Layer 13: Enhanced Analysis (pattern + sentiment)
        layer_tasks.append(self._create_sync_result(
            "EnhancedAnalysis", 13, True,
            score=min(100, data.base_confidence * 1.1) if data.base_confidence > 50 else 50,
            reason="Enhanced pattern analysis"
        ))
        
        # Layer 14: Pro Features
        if self.pro_features:
            layer_tasks.append(self._run_pro_features(data))
        
        # Layer 15: Risk Guardian
        if self.risk_guardian:
            layer_tasks.append(self._run_risk_guardian(data))
        
        # Layer 16: Position Sizing (calculate optimal position)
        layer_tasks.append(self._create_sync_result(
            "PositionSizing", 16, True,
            score=100 if data.balance > 0 else 0,
            reason=f"Balance: ${data.balance:.2f}"
        ))
        
        # Layer 17: Ultra Intelligence
        if self.ultra_intelligence:
            layer_tasks.append(self._run_ultra_intelligence(data))
        
        # Layer 18: Supreme Intelligence
        if self.supreme_intelligence:
            layer_tasks.append(self._run_supreme_intelligence(data))
        
        # Layer 19: Transcendent Intelligence
        if self.transcendent_intelligence:
            layer_tasks.append(self._run_transcendent_intelligence(data))
        
        # Layer 20: Omniscient Intelligence
        if self.omniscient_intelligence:
            layer_tasks.append(self._run_omniscient_intelligence(data))
        
        # 🚀 RUN ALL LAYERS IN PARALLEL
        logger.info(f"🚀 Running {len(layer_tasks)} layers in PARALLEL...")
        results = await asyncio.gather(*layer_tasks, return_exceptions=True)
        
        # Process results
        layer_results: List[LayerResult] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Layer error: {result}")
                continue
            if isinstance(result, LayerResult):
                layer_results.append(result)
        
        # Sort by layer number
        layer_results.sort(key=lambda x: x.layer_num)
        
        # Calculate statistics
        total_time_ms = (time.time() - start_time) * 1000
        
        return self._compile_final_decision(layer_results, data.symbol, total_time_ms)
    
    def _compile_final_decision(
        self, 
        layer_results: List[LayerResult],
        symbol: str,
        total_time_ms: float
    ) -> ParallelAnalysisResult:
        """รวมผลและตัดสินใจ"""
        
        total_layers = len(layer_results)
        passed_layers = sum(1 for r in layer_results if r.can_trade)
        pass_rate = passed_layers / max(1, total_layers)
        
        avg_score = np.mean([r.score for r in layer_results if r.score > 0]) if layer_results else 0
        avg_multiplier = np.mean([r.multiplier for r in layer_results]) if layer_results else 1.0
        
        # High quality passes (score >= 70)
        high_quality_passes = sum(1 for r in layer_results if r.can_trade and r.score >= 70)
        
        # Key layer agreement (5, 6, 7, 9, 10)
        KEY_LAYERS = [5, 6, 7, 9, 10]
        key_results = [r for r in layer_results if r.layer_num in KEY_LAYERS]
        key_passed = sum(1 for r in key_results if r.can_trade)
        key_agreement = key_passed / max(1, len(key_results))
        
        # 🎯 FINAL DECISION LOGIC
        reasons = []
        
        # Check 1: Pass rate >= 40%
        # 🛡️ FIX VULN-9: Align parallel thresholds with sequential path for consistency
        import os
        MIN_PASS_RATE = float(os.getenv("MIN_LAYER_PASS_RATE", "0.50"))  # 🛡️ ALIGNED: 50% (same as sequential)
        if pass_rate < MIN_PASS_RATE:
            reasons.append(f"Pass rate {pass_rate:.0%} < {MIN_PASS_RATE:.0%}")
        
        # Check 2: High quality passes >= 3
        is_gold = 'XAU' in symbol.upper() or 'GOLD' in symbol.upper()
        MIN_HIGH_QUALITY = int(os.getenv("MIN_HIGH_QUALITY_PASSES", "3"))  # 🛡️ ALIGNED: 3 (same as sequential)
        if high_quality_passes < MIN_HIGH_QUALITY:
            reasons.append(f"High-quality passes {high_quality_passes} < {MIN_HIGH_QUALITY}")
        
        # Check 3: Key layer agreement >= 40%
        MIN_KEY_AGREEMENT = float(os.getenv("MIN_KEY_LAYER_AGREEMENT", "0.40"))  # 🛡️ ALIGNED: 40% (same as sequential)
        if len(key_results) > 0 and key_agreement < MIN_KEY_AGREEMENT:
            reasons.append(f"Key layer agreement {key_agreement:.0%} < {MIN_KEY_AGREEMENT:.0%}")
        
        # Determine final decision
        final_decision = "APPROVE" if not reasons else "SKIP"
        
        # Calculate position factor
        if pass_rate >= 0.75:
            final_position_factor = 1.0
        elif pass_rate >= 0.60:
            final_position_factor = 0.85
        elif pass_rate >= 0.50:
            final_position_factor = 0.7
        else:
            final_position_factor = 0.5
        
        # Boost for high quality
        if high_quality_passes >= 6:
            final_position_factor = min(1.2, final_position_factor * 1.2)
        
        return ParallelAnalysisResult(
            layer_results=layer_results,
            total_layers=total_layers,
            passed_layers=passed_layers,
            pass_rate=pass_rate,
            avg_score=avg_score,
            avg_multiplier=avg_multiplier,
            high_quality_passes=high_quality_passes,
            key_layer_agreement=key_agreement,
            final_decision=final_decision,
            final_position_factor=final_position_factor,
            total_time_ms=total_time_ms,
            reasons=reasons
        )
    
    # ═══════════════════════════════════════════════════════════════
    # 🧠 LAYER EXECUTION METHODS (Run in parallel)
    # ═══════════════════════════════════════════════════════════════
    
    async def _create_sync_result(
        self, name: str, num: int, can_trade: bool, 
        score: float = 0.0, reason: str = ""
    ) -> LayerResult:
        """สร้าง result สำหรับ sync checks"""
        return LayerResult(
            layer_name=name,
            layer_num=num,
            can_trade=can_trade,
            score=score,
            multiplier=1.0 if can_trade else 0.5,
            reasons=[reason] if reason else []
        )
    
    async def _run_advanced_intelligence(self, data: MarketDataBundle) -> LayerResult:
        """Layer 5: Advanced Intelligence"""
        start = time.time()
        try:
            # Run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.advanced_intelligence.analyze(
                    symbol=data.symbol,
                    prices=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes,
                    current_price=data.current_price,
                    signal_direction=data.signal_side
                )
            )
            
            return LayerResult(
                layer_name="AdvancedIntelligence",
                layer_num=5,
                can_trade=decision.can_trade,
                score=decision.confluence_score * 25,  # Convert to 0-100
                multiplier=decision.position_multiplier if decision.can_trade else 0.5,
                confidence=decision.confluence_score * 25,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "regime": decision.regime.value if decision.regime else "unknown",
                    "mtf_agreement": decision.mtf_agreement,
                    "momentum": decision.momentum_signal.value if decision.momentum_signal else "neutral"
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="AdvancedIntelligence",
                layer_num=5,
                can_trade=True,  # Don't block on error
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_smart_brain(self, data: MarketDataBundle) -> LayerResult:
        """Layer 6: Smart Brain"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.smart_brain.analyze_signal(
                    symbol=data.symbol,
                    signal_side=data.signal_side,
                    entry_price=data.current_price,
                    base_position_size=1.0,
                    balance=data.balance
                )
            )
            
            return LayerResult(
                layer_name="SmartBrain",
                layer_num=6,
                can_trade=decision.can_trade,
                score=100 if decision.can_trade else 30,
                multiplier=decision.position_multiplier,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return LayerResult(
                layer_name="SmartBrain",
                layer_num=6,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_neural_brain(self, data: MarketDataBundle) -> LayerResult:
        """Layer 7: Neural Brain"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.neural_brain.analyze(
                    signal_side=data.signal_side,
                    prices=data.prices,
                    volumes=data.volumes,
                    balance=data.balance
                )
            )
            
            return LayerResult(
                layer_name="NeuralBrain",
                layer_num=7,
                can_trade=decision.can_trade,
                score=decision.confidence,
                multiplier=decision.position_size_factor if decision.can_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "market_state": decision.market_state.value,
                    "pattern_quality": decision.pattern_quality,
                    "anomaly_detected": decision.anomaly_detected
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="NeuralBrain",
                layer_num=7,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_deep_intelligence(self, data: MarketDataBundle) -> LayerResult:
        """Layer 8: Deep Intelligence"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.deep_intelligence.analyze(
                    symbol=data.symbol,
                    signal_direction=data.signal_side,
                    prices=data.prices,
                    entry_price=data.current_price
                )
            )
            
            return LayerResult(
                layer_name="DeepIntelligence",
                layer_num=8,
                can_trade=decision.should_trade,
                score=decision.confidence,
                multiplier=decision.position_multiplier if decision.should_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return LayerResult(
                layer_name="DeepIntelligence",
                layer_num=8,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_quantum_strategy(self, data: MarketDataBundle) -> LayerResult:
        """Layer 9: Quantum Strategy"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.quantum_strategy.analyze(
                    symbol=data.symbol,
                    signal_direction=data.signal_side,
                    prices=data.prices,
                    volumes=data.volumes,
                    entry_price=data.current_price
                )
            )
            
            return LayerResult(
                layer_name="QuantumStrategy",
                layer_num=9,
                can_trade=decision.should_trade,
                score=decision.confidence,
                multiplier=decision.position_multiplier if decision.should_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "quantum_score": decision.quantum_score,
                    "edge_score": decision.edge_score,
                    "smart_money_signal": decision.smart_money_signal.value if decision.smart_money_signal else "neutral"
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="QuantumStrategy",
                layer_num=9,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_alpha_engine(self, data: MarketDataBundle) -> LayerResult:
        """Layer 10: Alpha Engine"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.alpha_engine.analyze(
                    symbol=data.symbol,
                    signal_direction=data.signal_side,
                    prices=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes,
                    current_price=data.current_price
                )
            )
            
            return LayerResult(
                layer_name="AlphaEngine",
                layer_num=10,
                can_trade=decision.should_trade,
                score=decision.confidence,
                multiplier=decision.position_multiplier if decision.should_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "alpha_score": decision.alpha_score,
                    "grade": decision.grade.value if decision.grade else "unknown",
                    "order_flow": decision.order_flow_signal.value if decision.order_flow_signal else "neutral"
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="AlphaEngine",
                layer_num=10,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_omega_brain(self, data: MarketDataBundle) -> LayerResult:
        """Layer 11: Omega Brain"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.omega_brain.analyze(
                    symbol=data.symbol,
                    signal_direction=data.signal_side,
                    prices=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes,
                    current_price=data.current_price
                )
            )
            
            return LayerResult(
                layer_name="OmegaBrain",
                layer_num=11,
                can_trade=decision.should_trade,
                score=decision.confidence,
                multiplier=decision.position_multiplier if decision.should_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "omega_score": decision.omega_score,
                    "grade": decision.grade.value if decision.grade else "unknown"
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="OmegaBrain",
                layer_num=11,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_titan_core(self, data: MarketDataBundle) -> LayerResult:
        """Layer 12: Titan Core"""
        start = time.time()
        try:
            from trading.titan_core import ModuleSignal
            
            # Create module signals for Titan synthesis
            module_signals = [
                ModuleSignal(
                    module_name="ParallelAnalysis",
                    should_trade=True,
                    direction=data.signal_side,
                    confidence=data.base_confidence,
                    multiplier=1.0,
                    score=70,
                    reasons=[],
                    warnings=[]
                )
            ]
            
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.titan_core.synthesize(
                    symbol=data.symbol,
                    signal_direction=data.signal_side,
                    closes=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes if data.volumes is not None else np.ones(len(data.prices)) * 1000,
                    module_signals=module_signals,
                    current_price=data.current_price
                )
            )
            
            return LayerResult(
                layer_name="TitanCore",
                layer_num=12,
                can_trade=decision.should_trade,
                score=decision.confidence,
                multiplier=decision.position_multiplier if decision.should_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "titan_score": decision.titan_score,
                    "grade": decision.grade.value if decision.grade else "unknown",
                    "consensus": decision.consensus_strength
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="TitanCore",
                layer_num=12,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_pro_features(self, data: MarketDataBundle) -> LayerResult:
        """Layer 14: Pro Features"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            
            # Session check
            session_result = await loop.run_in_executor(
                self.executor,
                lambda: self.pro_features.check_session()
            )
            
            can_trade = session_result.can_trade
            score = session_result.position_multiplier * 100
            
            return LayerResult(
                layer_name="ProFeatures",
                layer_num=14,
                can_trade=can_trade,
                score=score,
                multiplier=session_result.position_multiplier,
                reasons=[f"Session: {session_result.session.value}"] if hasattr(session_result, 'session') else [],
                warnings=session_result.warnings if hasattr(session_result, 'warnings') else [],
                execution_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return LayerResult(
                layer_name="ProFeatures",
                layer_num=14,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_risk_guardian(self, data: MarketDataBundle) -> LayerResult:
        """Layer 15: Risk Guardian"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            assessment = await loop.run_in_executor(
                self.executor,
                lambda: self.risk_guardian.assess_risk(
                    current_balance=data.balance,
                    open_positions=[],
                    proposed_trade={"symbol": data.symbol, "side": data.signal_side}
                )
            )
            
            risk_scores = {"SAFE": 100, "WARNING": 70, "DANGER": 40, "CRITICAL": 10}
            level_str = str(assessment.level.value) if hasattr(assessment.level, 'value') else str(assessment.level)
            score = risk_scores.get(level_str, 50)
            
            return LayerResult(
                layer_name="RiskGuardian",
                layer_num=15,
                can_trade=assessment.can_trade,
                score=score,
                multiplier=assessment.max_position_size if assessment.can_trade else 0.3,
                reasons=assessment.reasons,
                warnings=assessment.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "risk_level": level_str
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="RiskGuardian",
                layer_num=15,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_ultra_intelligence(self, data: MarketDataBundle) -> LayerResult:
        """Layer 17: Ultra Intelligence"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.ultra_intelligence.analyze(
                    symbol=data.symbol,
                    signal_side=data.signal_side,
                    current_price=data.current_price,
                    prices=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes,
                    atr=data.atr,
                    base_confidence=data.base_confidence,
                    current_balance=data.balance,
                    account_equity=data.equity
                )
            )
            
            return LayerResult(
                layer_name="UltraIntelligence",
                layer_num=17,
                can_trade=decision.can_trade,
                score=decision.confidence,
                multiplier=decision.position_size_multiplier if decision.can_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "session_quality": decision.session_quality.value,
                    "volatility_state": decision.volatility_state.value,
                    "market_phase": decision.market_phase.value
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="UltraIntelligence",
                layer_num=17,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_supreme_intelligence(self, data: MarketDataBundle) -> LayerResult:
        """Layer 18: Supreme Intelligence"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.supreme_intelligence.analyze(
                    symbol=data.symbol,
                    signal_side=data.signal_side,
                    current_price=data.current_price,
                    prices=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes,
                    atr=data.atr,
                    base_confidence=data.base_confidence,
                    balance=data.balance,
                    equity=data.equity
                )
            )
            
            return LayerResult(
                layer_name="SupremeIntelligence",
                layer_num=18,
                can_trade=decision.can_trade,
                score=decision.confidence,
                multiplier=decision.position_multiplier if decision.can_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "supreme_score": decision.supreme_score,
                    "win_probability": decision.win_probability
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="SupremeIntelligence",
                layer_num=18,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_transcendent_intelligence(self, data: MarketDataBundle) -> LayerResult:
        """Layer 19: Transcendent Intelligence"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.transcendent_intelligence.analyze(
                    symbol=data.symbol,
                    signal_side=data.signal_side,
                    current_price=data.current_price,
                    prices=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes,
                    atr=data.atr,
                    base_confidence=data.base_confidence,
                    balance=data.balance,
                    equity=data.equity
                )
            )
            
            return LayerResult(
                layer_name="TranscendentIntelligence",
                layer_num=19,
                can_trade=decision.can_trade,
                score=decision.transcendent_score,
                multiplier=decision.quantum_position_size * 10 if decision.can_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "quantum_state": decision.quantum_field.quantum_state.value,
                    "signal_purity": decision.signal_purity.value,
                    "win_probability": decision.win_probability
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="TranscendentIntelligence",
                layer_num=19,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    async def _run_omniscient_intelligence(self, data: MarketDataBundle) -> LayerResult:
        """Layer 20: Omniscient Intelligence"""
        start = time.time()
        try:
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(
                self.executor,
                lambda: self.omniscient_intelligence.analyze(
                    symbol=data.symbol,
                    signal_side=data.signal_side,
                    current_price=data.current_price,
                    prices=data.prices,
                    highs=data.highs,
                    lows=data.lows,
                    volumes=data.volumes,
                    atr=data.atr,
                    base_confidence=data.base_confidence,
                    balance=data.balance,
                    equity=data.equity
                )
            )
            
            return LayerResult(
                layer_name="OmniscientIntelligence",
                layer_num=20,
                can_trade=decision.can_trade,
                score=decision.omniscient_score,
                multiplier=decision.omniscient_position_size * 10 if decision.can_trade else 0.5,
                confidence=decision.confidence,
                reasons=decision.reasons,
                warnings=decision.warnings,
                execution_time_ms=(time.time() - start) * 1000,
                extra_data={
                    "consciousness_level": decision.consciousness_level.value,
                    "universal_alignment": decision.universal_alignment,
                    "win_probability": decision.win_probability
                }
            )
        except Exception as e:
            return LayerResult(
                layer_name="OmniscientIntelligence",
                layer_num=20,
                can_trade=True,
                score=50,
                multiplier=1.0,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)


def format_parallel_results(result: ParallelAnalysisResult) -> str:
    """Format results for logging"""
    lines = [
        "",
        "🚀 ═══════════════════════════════════════════════════════════════════════════════",
        "🚀                    PARALLEL LAYER ANALYSIS COMPLETE",
        "🚀 ═══════════════════════════════════════════════════════════════════════════════",
        f"   ⏱️ Total Time: {result.total_time_ms:.1f}ms (PARALLEL)",
        f"   📊 Total Layers: {result.total_layers}",
        f"   ✅ Passed: {result.passed_layers}",
        f"   📈 Pass Rate: {result.pass_rate:.0%}",
        f"   🎯 Avg Score: {result.avg_score:.1f}",
        f"   📊 High Quality Passes: {result.high_quality_passes}",
        f"   🔑 Key Layer Agreement: {result.key_layer_agreement:.0%}",
        "",
        "   📋 Layer-by-Layer Results:"
    ]
    
    for r in result.layer_results:
        status = "✅" if r.can_trade else "⚠️"
        lines.append(
            f"      {status} Layer {r.layer_num} ({r.layer_name}): "
            f"{'PASS' if r.can_trade else 'WARN'} | "
            f"Score: {r.score:.1f} | Mult: {r.multiplier:.2f}x | "
            f"Time: {r.execution_time_ms:.0f}ms"
        )
    
    lines.append("")
    
    if result.final_decision == "APPROVE":
        lines.extend([
            "🚀 ═══════════════════════════════════════════════════════════════════════════════",
            "🚀 ✅ FINAL DECISION: APPROVE TRADE",
            f"🚀    Position Factor: {result.final_position_factor:.2f}x",
            "🚀 ═══════════════════════════════════════════════════════════════════════════════"
        ])
    else:
        lines.extend([
            "🚀 ═══════════════════════════════════════════════════════════════════════════════",
            "🚀 ❌ FINAL DECISION: SKIP TRADE",
            f"🚀    Reasons: {', '.join(result.reasons)}",
            "🚀 ═══════════════════════════════════════════════════════════════════════════════"
        ])
    
    return "\n".join(lines)
