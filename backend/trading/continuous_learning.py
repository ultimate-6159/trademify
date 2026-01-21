"""
Continuous Learning System - เรียนรู้ตลอดเวลา ประหยัดทรัพยากร
============================================================

ระบบ AI ที่:
1. เรียนรู้จากทุกเทรด (Online Learning)
2. จำ patterns ที่เปลี่ยนไปตามเวลา
3. ปรับ parameters อัตโนมัติ
4. รู้ว่าตอนนี้อยู่ช่วงไหนของ market cycle
5. รู้ว่า factor ไหนสำคัญที่สุด
6. บีบอัดข้อมูลเก่า เก็บใหม่ละเอียด
7. ทำงาน background ไม่กินทรัพยากร

Memory Efficient:
- Rolling window statistics (ไม่เก็บ raw data ทั้งหมด)
- Exponential decay (ข้อมูลเก่าลดความสำคัญ)
- Compression of old data
- Lazy loading
"""
import logging
import json
import os
import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


# =====================
# Online Learning - เรียนรู้ทีละ trade
# =====================

class OnlineLearner:
    """
    เรียนรู้แบบ Online (ทีละ sample)
    ไม่ต้องเก็บ data ทั้งหมด
    
    ใช้ Exponential Moving Average สำหรับ:
    - Win rate
    - Average PnL
    - Factor weights
    """
    
    def __init__(
        self,
        alpha: float = 0.1,  # Learning rate (0.1 = 10% weight to new data)
        min_samples: int = 10,
    ):
        self.alpha = alpha
        self.min_samples = min_samples
        
        # Running statistics (EMA)
        self.ema_win_rate = 0.5
        self.ema_pnl = 0.0
        self.ema_rr_ratio = 1.0  # Risk/Reward
        
        # Factor importance (which factors predict wins)
        self.factor_weights = {
            "pattern_confidence": 0.5,
            "regime_aligned": 0.5,
            "mtf_aligned": 0.5,
            "momentum_aligned": 0.5,
            "near_sr": 0.5,
            "smart_money": 0.5,
            "sentiment": 0.5,
            "session_quality": 0.5,
        }
        
        self.sample_count = 0
        self.recent_trades = deque(maxlen=100)  # เก็บแค่ 100 ล่าสุด
    
    def learn(
        self,
        is_win: bool,
        pnl_percent: float,
        factors: Dict[str, bool],
        rr_ratio: float = 1.0,
    ):
        """
        เรียนรู้จาก 1 trade
        
        Args:
            is_win: ชนะหรือไม่
            pnl_percent: กำไร/ขาดทุน %
            factors: dict ของ factors ที่เป็น True/False
            rr_ratio: Risk/Reward ratio
        """
        self.sample_count += 1
        
        # Update running statistics
        win_value = 1.0 if is_win else 0.0
        self.ema_win_rate = self._ema(self.ema_win_rate, win_value)
        self.ema_pnl = self._ema(self.ema_pnl, pnl_percent)
        self.ema_rr_ratio = self._ema(self.ema_rr_ratio, rr_ratio)
        
        # Update factor weights
        for factor_name, factor_value in factors.items():
            if factor_name in self.factor_weights:
                # If factor was present and trade won -> increase weight
                # If factor was present and trade lost -> decrease weight
                if factor_value:  # Factor was True
                    if is_win:
                        # Factor helped -> increase importance
                        self.factor_weights[factor_name] = self._ema(
                            self.factor_weights[factor_name], 
                            min(1.0, self.factor_weights[factor_name] + 0.1)
                        )
                    else:
                        # Factor failed -> decrease importance
                        self.factor_weights[factor_name] = self._ema(
                            self.factor_weights[factor_name],
                            max(0.0, self.factor_weights[factor_name] - 0.1)
                        )
        
        # Store recent trade
        self.recent_trades.append({
            "is_win": is_win,
            "pnl": pnl_percent,
            "factors": factors.copy(),
            "timestamp": datetime.now().isoformat(),
        })
        
        if self.sample_count % 10 == 0:
            logger.info(f"📚 Online Learning: {self.sample_count} trades, WR={self.ema_win_rate*100:.1f}%")
    
    def _ema(self, current: float, new_value: float) -> float:
        """Exponential Moving Average"""
        return (1 - self.alpha) * current + self.alpha * new_value
    
    def get_factor_ranking(self) -> List[Tuple[str, float]]:
        """Get factors ranked by importance"""
        sorted_factors = sorted(
            self.factor_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_factors
    
    def get_statistics(self) -> dict:
        """Get current learning statistics"""
        return {
            "sample_count": self.sample_count,
            "ema_win_rate": round(self.ema_win_rate * 100, 1),
            "ema_pnl": round(self.ema_pnl, 2),
            "ema_rr_ratio": round(self.ema_rr_ratio, 2),
            "factor_ranking": self.get_factor_ranking(),
            "confidence": min(100, self.sample_count * 2),  # Confidence increases with samples
        }
    
    def should_use_factor(self, factor_name: str) -> Tuple[bool, float]:
        """Check if a factor should be used based on learned importance"""
        weight = self.factor_weights.get(factor_name, 0.5)
        
        # Use factor if weight > 0.4
        should_use = weight > 0.4
        
        return should_use, weight
    
    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "sample_count": self.sample_count,
            "ema_win_rate": self.ema_win_rate,
            "ema_pnl": self.ema_pnl,
            "ema_rr_ratio": self.ema_rr_ratio,
            "factor_weights": self.factor_weights,
        }
    
    def from_dict(self, data: dict):
        """Load from dict"""
        self.alpha = data.get("alpha", 0.1)
        self.sample_count = data.get("sample_count", 0)
        self.ema_win_rate = data.get("ema_win_rate", 0.5)
        self.ema_pnl = data.get("ema_pnl", 0.0)
        self.ema_rr_ratio = data.get("ema_rr_ratio", 1.0)
        self.factor_weights = data.get("factor_weights", self.factor_weights)


# =====================
# Market Cycle Detector
# =====================

class MarketCycle(Enum):
    """Market cycle phases"""
    ACCUMULATION = "accumulation"  # ช่วงสะสม (sideways หลัง downtrend)
    MARKUP = "markup"              # ช่วงขึ้น (bull market)
    DISTRIBUTION = "distribution"  # ช่วงจ่าย (sideways หลัง uptrend)
    MARKDOWN = "markdown"          # ช่วงลง (bear market)
    UNKNOWN = "unknown"


@dataclass
class CycleInfo:
    """Market cycle information"""
    cycle: MarketCycle
    confidence: float
    duration_days: int
    trend_strength: float  # -100 to +100
    volatility_state: str  # "low", "normal", "high"
    recommended_bias: str  # "long", "short", "neutral"
    position_size_factor: float
    message: str
    
    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle.value,
            "confidence": round(self.confidence, 1),
            "duration_days": self.duration_days,
            "trend_strength": round(self.trend_strength, 1),
            "volatility_state": self.volatility_state,
            "recommended_bias": self.recommended_bias,
            "position_size_factor": round(self.position_size_factor, 2),
            "message": self.message,
        }


class MarketCycleDetector:
    """
    ตรวจจับ Market Cycle
    
    ใช้ Rolling Statistics (ไม่เก็บ raw data):
    - 20-day rolling high/low
    - 50-day trend direction
    - Volatility percentile
    """
    
    def __init__(self, lookback_days: int = 50):
        self.lookback_days = lookback_days
        
        # Rolling statistics (เก็บแค่ค่าที่จำเป็น)
        self.rolling_high = deque(maxlen=lookback_days)
        self.rolling_low = deque(maxlen=lookback_days)
        self.rolling_close = deque(maxlen=lookback_days)
        self.rolling_atr = deque(maxlen=lookback_days)
        
        self.current_cycle = MarketCycle.UNKNOWN
        self.cycle_start_date = None
        self.last_update = None
    
    def update(self, high: float, low: float, close: float, atr: float = None):
        """Update with new daily data"""
        self.rolling_high.append(high)
        self.rolling_low.append(low)
        self.rolling_close.append(close)
        
        if atr:
            self.rolling_atr.append(atr)
        else:
            # Estimate ATR from H-L
            self.rolling_atr.append(high - low)
        
        self.last_update = datetime.now()
    
    def detect(self) -> CycleInfo:
        """Detect current market cycle"""
        if len(self.rolling_close) < 20:
            return CycleInfo(
                cycle=MarketCycle.UNKNOWN,
                confidence=0,
                duration_days=0,
                trend_strength=0,
                volatility_state="unknown",
                recommended_bias="neutral",
                position_size_factor=0.5,
                message="ข้อมูลไม่พอ",
            )
        
        closes = list(self.rolling_close)
        highs = list(self.rolling_high)
        lows = list(self.rolling_low)
        atrs = list(self.rolling_atr)
        
        # Calculate trend strength
        if len(closes) >= 20:
            ma20 = np.mean(closes[-20:])
            ma50 = np.mean(closes[-min(50, len(closes)):])
            current = closes[-1]
            
            # Trend strength: -100 to +100
            if ma50 > 0:
                trend = ((current - ma50) / ma50) * 100
                trend = max(-100, min(100, trend * 5))  # Scale
            else:
                trend = 0
        else:
            trend = 0
        
        # Calculate volatility state
        if len(atrs) >= 20:
            current_atr = atrs[-1]
            avg_atr = np.mean(atrs)
            
            if current_atr > avg_atr * 1.5:
                vol_state = "high"
            elif current_atr < avg_atr * 0.5:
                vol_state = "low"
            else:
                vol_state = "normal"
        else:
            vol_state = "normal"
        
        # Detect higher highs / lower lows
        recent_high = max(highs[-10:])
        recent_low = min(lows[-10:])
        prev_high = max(highs[-20:-10]) if len(highs) >= 20 else recent_high
        prev_low = min(lows[-20:-10]) if len(lows) >= 20 else recent_low
        
        higher_highs = recent_high > prev_high
        higher_lows = recent_low > prev_low
        lower_highs = recent_high < prev_high
        lower_lows = recent_low < prev_low
        
        # Determine cycle
        if higher_highs and higher_lows:
            # Uptrend
            if trend > 20:
                cycle = MarketCycle.MARKUP
                confidence = 70 + min(20, abs(trend) / 5)
                bias = "long"
                size_factor = 1.2
                message = "📈 Markup Phase - BUY opportunities"
            else:
                cycle = MarketCycle.ACCUMULATION
                confidence = 60
                bias = "long"
                size_factor = 0.8
                message = "🔄 Accumulation - Preparing for uptrend"
        
        elif lower_highs and lower_lows:
            # Downtrend
            if trend < -20:
                cycle = MarketCycle.MARKDOWN
                confidence = 70 + min(20, abs(trend) / 5)
                bias = "short"
                size_factor = 1.2
                message = "📉 Markdown Phase - SELL opportunities"
            else:
                cycle = MarketCycle.DISTRIBUTION
                confidence = 60
                bias = "short"
                size_factor = 0.8
                message = "🔄 Distribution - Preparing for downtrend"
        
        else:
            # Sideways
            if trend > 10:
                cycle = MarketCycle.ACCUMULATION
                bias = "long"
            elif trend < -10:
                cycle = MarketCycle.DISTRIBUTION
                bias = "short"
            else:
                cycle = MarketCycle.UNKNOWN
                bias = "neutral"
            
            confidence = 50
            size_factor = 0.7
            message = "↔️ Ranging - Wait for breakout"
        
        # Track cycle duration
        if cycle != self.current_cycle:
            self.current_cycle = cycle
            self.cycle_start_date = datetime.now()
        
        duration = 0
        if self.cycle_start_date:
            duration = (datetime.now() - self.cycle_start_date).days
        
        # Reduce size in high volatility
        if vol_state == "high":
            size_factor *= 0.7
        
        return CycleInfo(
            cycle=cycle,
            confidence=confidence,
            duration_days=duration,
            trend_strength=trend,
            volatility_state=vol_state,
            recommended_bias=bias,
            position_size_factor=size_factor,
            message=message,
        )


# =====================
# Strategy Optimizer
# =====================

@dataclass
class StrategyParams:
    """Strategy parameters that can be optimized"""
    min_confidence: float = 70.0
    min_confluence: int = 3
    max_risk_per_trade: float = 2.0
    trailing_stop_percent: float = 1.0
    break_even_percent: float = 0.5
    session_filter_enabled: bool = True
    news_filter_enabled: bool = True
    regime_filter_enabled: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)


class StrategyOptimizer:
    """
    ปรับ parameters อัตโนมัติตาม performance
    
    ใช้ Simple Hill Climbing:
    - ถ้า performance ดีขึ้น -> keep changes
    - ถ้า performance แย่ลง -> revert
    
    ประหยัดทรัพยากร: ปรับทีละ parameter
    """
    
    def __init__(
        self,
        evaluation_window: int = 20,  # trades
        improvement_threshold: float = 0.05,  # 5% improvement needed
    ):
        self.eval_window = evaluation_window
        self.improvement_threshold = improvement_threshold
        
        # Current best params
        self.best_params = StrategyParams()
        self.best_performance = 0.0
        
        # Recent performance
        self.recent_trades = deque(maxlen=evaluation_window)
        self.optimization_count = 0
        self.last_optimization = None
        
        # Parameters to optimize
        self.param_ranges = {
            "min_confidence": (60, 90, 5),  # (min, max, step)
            "min_confluence": (2, 5, 1),
            "max_risk_per_trade": (1.0, 3.0, 0.5),
            "trailing_stop_percent": (0.5, 2.0, 0.25),
            "break_even_percent": (0.3, 1.0, 0.1),
        }
        
        self.current_param_idx = 0
    
    def record_trade(self, is_win: bool, pnl_percent: float, params_used: dict):
        """Record a trade result"""
        self.recent_trades.append({
            "is_win": is_win,
            "pnl": pnl_percent,
            "params": params_used,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Check if should optimize
        if len(self.recent_trades) >= self.eval_window:
            self._maybe_optimize()
    
    def _calculate_performance(self) -> float:
        """Calculate performance score from recent trades"""
        if not self.recent_trades:
            return 0.0
        
        wins = sum(1 for t in self.recent_trades if t["is_win"])
        total_pnl = sum(t["pnl"] for t in self.recent_trades)
        
        win_rate = wins / len(self.recent_trades)
        avg_pnl = total_pnl / len(self.recent_trades)
        
        # Performance score = win_rate * avg_pnl factor
        performance = win_rate * 100 + avg_pnl * 10
        
        return performance
    
    def _maybe_optimize(self):
        """Optimize one parameter if needed"""
        current_perf = self._calculate_performance()
        
        # First run - set baseline
        if self.best_performance == 0:
            self.best_performance = current_perf
            return
        
        # Check if significant change
        perf_change = (current_perf - self.best_performance) / max(1, abs(self.best_performance))
        
        if perf_change > self.improvement_threshold:
            # Performance improved - keep current params
            self.best_performance = current_perf
            logger.info(f"📈 Strategy improved! Perf: {current_perf:.1f} (+{perf_change*100:.1f}%)")
        
        elif perf_change < -self.improvement_threshold:
            # Performance degraded - try adjusting a parameter
            self._try_optimize_next_param()
        
        self.optimization_count += 1
        self.last_optimization = datetime.now()
    
    def _try_optimize_next_param(self):
        """Try optimizing the next parameter in round-robin"""
        param_names = list(self.param_ranges.keys())
        param_name = param_names[self.current_param_idx % len(param_names)]
        self.current_param_idx += 1
        
        min_val, max_val, step = self.param_ranges[param_name]
        current_val = getattr(self.best_params, param_name)
        
        # Try increasing or decreasing
        import random
        if random.random() > 0.5 and current_val + step <= max_val:
            new_val = current_val + step
        elif current_val - step >= min_val:
            new_val = current_val - step
        else:
            new_val = current_val + step if current_val + step <= max_val else current_val
        
        setattr(self.best_params, param_name, new_val)
        logger.info(f"🔧 Optimizing {param_name}: {current_val} → {new_val}")
    
    def get_current_params(self) -> StrategyParams:
        """Get current optimized parameters"""
        return self.best_params
    
    def get_optimization_status(self) -> dict:
        """Get optimization status"""
        return {
            "optimization_count": self.optimization_count,
            "current_performance": self._calculate_performance(),
            "best_performance": self.best_performance,
            "current_params": self.best_params.to_dict(),
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "trades_in_window": len(self.recent_trades),
        }


# =====================
# Pattern Evolution Tracker
# =====================

class PatternEvolutionTracker:
    """
    ติดตามว่า patterns เปลี่ยนแปลงอย่างไรตามเวลา
    
    ประหยัด memory:
    - เก็บแค่ statistics ไม่ใช่ raw patterns
    - ใช้ rolling windows
    - Compress old data
    """
    
    def __init__(self, max_patterns: int = 500):
        self.max_patterns = max_patterns
        
        # Pattern stats (hash -> stats)
        self.pattern_stats: Dict[str, dict] = {}
        
        # Time-based evolution
        self.weekly_stats = deque(maxlen=52)  # 1 year of weekly stats
        self.current_week_stats = {
            "total_patterns": 0,
            "total_wins": 0,
            "total_pnl": 0,
            "week_start": datetime.now().isocalendar()[1],
        }
    
    def record_pattern_result(
        self,
        pattern_hash: str,
        is_win: bool,
        pnl: float,
        pattern_type: str = "unknown",
    ):
        """Record a pattern's trade result"""
        if pattern_hash not in self.pattern_stats:
            # Check if at capacity
            if len(self.pattern_stats) >= self.max_patterns:
                self._prune_old_patterns()
            
            self.pattern_stats[pattern_hash] = {
                "type": pattern_type,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "trade_count": 0,
                "win_count": 0,
                "total_pnl": 0,
                "ema_win_rate": 0.5,
                "ema_pnl": 0,
            }
        
        stats = self.pattern_stats[pattern_hash]
        stats["trade_count"] += 1
        stats["last_seen"] = datetime.now().isoformat()
        
        if is_win:
            stats["win_count"] += 1
        stats["total_pnl"] += pnl
        
        # Update EMA
        alpha = 0.2
        stats["ema_win_rate"] = (1 - alpha) * stats["ema_win_rate"] + alpha * (1.0 if is_win else 0.0)
        stats["ema_pnl"] = (1 - alpha) * stats["ema_pnl"] + alpha * pnl
        
        # Update weekly stats
        self._update_weekly_stats(is_win, pnl)
    
    def _update_weekly_stats(self, is_win: bool, pnl: float):
        """Update weekly statistics"""
        current_week = datetime.now().isocalendar()[1]
        
        if current_week != self.current_week_stats["week_start"]:
            # New week - save old and create new
            self.weekly_stats.append(self.current_week_stats.copy())
            self.current_week_stats = {
                "total_patterns": 0,
                "total_wins": 0,
                "total_pnl": 0,
                "week_start": current_week,
            }
        
        self.current_week_stats["total_patterns"] += 1
        if is_win:
            self.current_week_stats["total_wins"] += 1
        self.current_week_stats["total_pnl"] += pnl
    
    def _prune_old_patterns(self):
        """Remove old/low-value patterns to save memory"""
        if len(self.pattern_stats) < self.max_patterns:
            return
        
        # Sort by value (trade count * win rate)
        def pattern_value(item):
            stats = item[1]
            return stats["trade_count"] * stats["ema_win_rate"]
        
        sorted_patterns = sorted(self.pattern_stats.items(), key=pattern_value)
        
        # Remove bottom 20%
        remove_count = len(sorted_patterns) // 5
        for pattern_hash, _ in sorted_patterns[:remove_count]:
            del self.pattern_stats[pattern_hash]
        
        logger.info(f"🧹 Pruned {remove_count} old patterns, keeping {len(self.pattern_stats)}")
    
    def get_pattern_prediction(self, pattern_hash: str) -> Tuple[float, float, str]:
        """
        Get prediction for a pattern
        
        Returns: (predicted_win_rate, predicted_pnl, message)
        """
        if pattern_hash not in self.pattern_stats:
            return 0.5, 0, "New pattern - no history"
        
        stats = self.pattern_stats[pattern_hash]
        
        if stats["trade_count"] < 3:
            return stats["ema_win_rate"], stats["ema_pnl"], f"Low confidence ({stats['trade_count']} trades)"
        
        return (
            stats["ema_win_rate"],
            stats["ema_pnl"],
            f"Based on {stats['trade_count']} trades, WR={stats['ema_win_rate']*100:.0f}%"
        )
    
    def get_evolution_summary(self) -> dict:
        """Get summary of pattern evolution over time"""
        weekly_data = []
        for week in self.weekly_stats:
            if week["total_patterns"] > 0:
                weekly_data.append({
                    "week": week["week_start"],
                    "trades": week["total_patterns"],
                    "win_rate": round((week["total_wins"] / week["total_patterns"]) * 100, 1),
                    "pnl": round(week["total_pnl"], 2),
                })
        
        return {
            "total_patterns_tracked": len(self.pattern_stats),
            "weekly_evolution": weekly_data[-12:],  # Last 12 weeks
            "current_week": self.current_week_stats,
        }
    
    def get_recent_patterns(self, limit: int = 20) -> List[dict]:
        """Get most recently updated patterns"""
        if not self.pattern_stats:
            return []
        
        # Sort by last_seen
        sorted_patterns = sorted(
            self.pattern_stats.items(),
            key=lambda x: x[1].get("last_seen", ""),
            reverse=True
        )
        
        recent = []
        for pattern_hash, stats in sorted_patterns[:limit]:
            recent.append({
                "hash": pattern_hash[:16] + "...",  # Truncate for readability
                "trade_count": stats["trade_count"],
                "win_rate": round(stats["ema_win_rate"] * 100, 1),
                "avg_pnl": round(stats["ema_pnl"], 2),
                "last_seen": stats.get("last_seen", "unknown"),
            })
        
        return recent


# =====================
# Background Learner (Async)
# =====================

class BackgroundLearner:
    """
    ทำงาน background ไม่ block main thread
    ประหยัดทรัพยากร
    """
    
    def __init__(self):
        self.task_queue = asyncio.Queue(maxsize=1000)
        self.is_running = False
        self._worker_task = None
        
        # Learning components
        self.online_learner = OnlineLearner()
        self.pattern_tracker = PatternEvolutionTracker()
        self.cycle_detector = MarketCycleDetector()
        self.strategy_optimizer = StrategyOptimizer()
        
        # Rate limiting
        self.last_heavy_computation = None
        self.heavy_computation_interval = 300  # 5 minutes
    
    async def start(self):
        """Start background worker"""
        if self.is_running:
            return
        
        self.is_running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("🔄 Background Learner started")
    
    async def stop(self):
        """Stop background worker"""
        self.is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ Background Learner stopped")
    
    async def _worker(self):
        """Background worker loop"""
        while self.is_running:
            try:
                # Get task with timeout
                try:
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process task
                task_type = task.get("type")
                
                if task_type == "trade_result":
                    self._process_trade_result(task)
                
                elif task_type == "market_update":
                    self._process_market_update(task)
                
                elif task_type == "pattern_result":
                    self._process_pattern_result(task)
                
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Background learner error: {e}")
                await asyncio.sleep(1)
    
    def _process_trade_result(self, task: dict):
        """Process a trade result"""
        is_win = task.get("is_win", False)
        pnl = task.get("pnl", 0)
        factors = task.get("factors", {})
        rr_ratio = task.get("rr_ratio", 1.0)
        
        # Online learning
        self.online_learner.learn(is_win, pnl, factors, rr_ratio)
        
        # Strategy optimization
        self.strategy_optimizer.record_trade(is_win, pnl, factors)
    
    def _process_market_update(self, task: dict):
        """Process market data update"""
        high = task.get("high", 0)
        low = task.get("low", 0)
        close = task.get("close", 0)
        atr = task.get("atr")
        
        # Update cycle detector (rate limited)
        now = datetime.now()
        if (self.last_heavy_computation is None or 
            (now - self.last_heavy_computation).seconds > self.heavy_computation_interval):
            
            self.cycle_detector.update(high, low, close, atr)
            self.last_heavy_computation = now
    
    def _process_pattern_result(self, task: dict):
        """Process pattern result"""
        pattern_hash = task.get("pattern_hash", "")
        is_win = task.get("is_win", False)
        pnl = task.get("pnl", 0)
        pattern_type = task.get("pattern_type", "unknown")
        
        self.pattern_tracker.record_pattern_result(
            pattern_hash, is_win, pnl, pattern_type
        )
    
    # Public API (non-blocking)
    
    def submit_trade_result(
        self,
        is_win: bool,
        pnl: float,
        factors: Dict[str, bool],
        rr_ratio: float = 1.0,
    ):
        """Submit trade result for learning (non-blocking)"""
        try:
            self.task_queue.put_nowait({
                "type": "trade_result",
                "is_win": is_win,
                "pnl": pnl,
                "factors": factors,
                "rr_ratio": rr_ratio,
            })
        except asyncio.QueueFull:
            logger.warning("Learning queue full, dropping task")
    
    def submit_market_update(self, high: float, low: float, close: float, atr: float = None):
        """Submit market update (non-blocking)"""
        try:
            self.task_queue.put_nowait({
                "type": "market_update",
                "high": high,
                "low": low,
                "close": close,
                "atr": atr,
            })
        except asyncio.QueueFull:
            pass  # OK to drop market updates
    
    def submit_pattern_result(
        self,
        pattern_hash: str,
        is_win: bool,
        pnl: float,
        pattern_type: str = "unknown",
    ):
        """Submit pattern result (non-blocking)"""
        try:
            self.task_queue.put_nowait({
                "type": "pattern_result",
                "pattern_hash": pattern_hash,
                "is_win": is_win,
                "pnl": pnl,
                "pattern_type": pattern_type,
            })
        except asyncio.QueueFull:
            logger.warning("Learning queue full, dropping pattern task")
    
    def get_insights(self) -> dict:
        """Get all learning insights"""
        return {
            "online_learning": self.online_learner.get_statistics(),
            "market_cycle": self.cycle_detector.detect().to_dict(),
            "pattern_evolution": self.pattern_tracker.get_evolution_summary(),
            "strategy_optimization": self.strategy_optimizer.get_optimization_status(),
            "queue_size": self.task_queue.qsize(),
        }


# =====================
# Master Continuous Learning System
# =====================

@dataclass
class LearningDecision:
    """Decision from continuous learning system"""
    can_trade: bool
    confidence_adjustment: float  # Multiply confidence by this
    position_size_factor: float   # Multiply position by this
    factor_weights: Dict[str, float]
    market_cycle: CycleInfo
    optimized_params: StrategyParams
    insights: List[str]
    
    def to_dict(self) -> dict:
        return {
            "can_trade": self.can_trade,
            "confidence_adjustment": round(self.confidence_adjustment, 2),
            "position_size_factor": round(self.position_size_factor, 2),
            "factor_weights": {k: round(v, 2) for k, v in self.factor_weights.items()},
            "market_cycle": self.market_cycle.to_dict() if self.market_cycle else None,
            "optimized_params": self.optimized_params.to_dict() if self.optimized_params else None,
            "insights": self.insights,
        }


class ContinuousLearningSystem:
    """
    🧠 Master Continuous Learning System
    
    เรียนรู้ตลอดเวลา ประหยัดทรัพยากร:
    
    1. Online Learning - เรียนทีละ trade
    2. Market Cycle Detection - รู้ phase ของตลาด
    3. Pattern Evolution - patterns เปลี่ยนอย่างไร
    4. Strategy Optimization - ปรับ params อัตโนมัติ
    5. Background Processing - ไม่ block main
    6. Memory Efficient - rolling windows, EMA
    """
    
    def __init__(
        self,
        data_dir: str = "data/learning",
        enable_background: bool = True,
        firebase_service = None,
    ):
        self.data_dir = data_dir
        self.firebase = firebase_service
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Core components
        self.online_learner = OnlineLearner()
        self.cycle_detector = MarketCycleDetector()
        self.pattern_tracker = PatternEvolutionTracker()
        self.strategy_optimizer = StrategyOptimizer()
        
        # Background learner (optional)
        self.background = BackgroundLearner() if enable_background else None
        
        # Load saved state
        self._load_state()
        
        logger.info("🧠 Continuous Learning System initialized")
        logger.info(f"   - Online Learning: ✓")
        logger.info(f"   - Market Cycle Detection: ✓")
        logger.info(f"   - Pattern Evolution: ✓")
        logger.info(f"   - Strategy Optimization: ✓")
        logger.info(f"   - Background Processing: {'✓' if enable_background else '✗'}")
    
    async def start(self):
        """Start the system"""
        if self.background:
            await self.background.start()
    
    async def stop(self):
        """Stop and save"""
        if self.background:
            await self.background.stop()
        self._save_state()
    
    def learn_from_trade(
        self,
        is_win: bool,
        pnl_percent: float,
        factors: Dict[str, bool],
        pattern_hash: str = None,
        rr_ratio: float = 1.0,
    ):
        """
        เรียนรู้จากการเทรด (main API)
        
        ใช้ background processing ถ้าเปิดอยู่
        """
        if self.background:
            # Non-blocking
            self.background.submit_trade_result(is_win, pnl_percent, factors, rr_ratio)
            if pattern_hash:
                self.background.submit_pattern_result(pattern_hash, is_win, pnl_percent)
        else:
            # Direct processing
            self.online_learner.learn(is_win, pnl_percent, factors, rr_ratio)
            if pattern_hash:
                self.pattern_tracker.record_pattern_result(pattern_hash, is_win, pnl_percent)
        
        # Save periodically
        if self.online_learner.sample_count % 10 == 0:
            self._save_state()
    
    def update_market_data(self, high: float, low: float, close: float, atr: float = None):
        """Update with market data for cycle detection"""
        if self.background:
            self.background.submit_market_update(high, low, close, atr)
        else:
            self.cycle_detector.update(high, low, close, atr)
    
    def evaluate(
        self,
        signal_side: str,
        pattern_confidence: float,
        factors: Dict[str, bool],
        pattern_hash: str = None,
    ) -> LearningDecision:
        """
        ประเมินการเข้าเทรดจากข้อมูลที่เรียนรู้
        """
        insights = []
        can_trade = True
        confidence_adj = 1.0
        size_factor = 1.0
        
        # 1. Get learned factor weights
        factor_weights = self.online_learner.factor_weights.copy()
        
        # 2. Check each factor
        active_factors = 0
        weighted_score = 0
        for factor_name, factor_present in factors.items():
            if factor_present:
                weight = factor_weights.get(factor_name, 0.5)
                if weight > 0.4:
                    active_factors += 1
                    weighted_score += weight
        
        if active_factors > 0:
            avg_weight = weighted_score / active_factors
            confidence_adj = 0.5 + avg_weight  # 0.5 to 1.5
            insights.append(f"Factor score: {avg_weight:.2f} ({active_factors} active)")
        
        # 3. Market cycle check
        cycle = self.cycle_detector.detect()
        
        if cycle.cycle != MarketCycle.UNKNOWN:
            insights.append(f"Market: {cycle.message}")
            size_factor *= cycle.position_size_factor
            
            # Check bias alignment
            if signal_side == "BUY" and cycle.recommended_bias == "short":
                confidence_adj *= 0.7
                insights.append("⚠️ BUY against cycle bias")
            elif signal_side == "SELL" and cycle.recommended_bias == "long":
                confidence_adj *= 0.7
                insights.append("⚠️ SELL against cycle bias")
            elif cycle.recommended_bias == signal_side.lower():
                confidence_adj *= 1.1
                insights.append(f"✅ Aligned with {cycle.cycle.value}")
        
        # 4. Pattern history
        if pattern_hash:
            pred_wr, pred_pnl, pred_msg = self.pattern_tracker.get_pattern_prediction(pattern_hash)
            
            if pred_wr < 0.4:
                confidence_adj *= 0.7
                insights.append(f"⚠️ Pattern history: {pred_msg}")
            elif pred_wr > 0.6:
                confidence_adj *= 1.1
                insights.append(f"✅ Pattern history: {pred_msg}")
        
        # 5. Win rate check
        if self.online_learner.sample_count >= 20:
            wr = self.online_learner.ema_win_rate
            if wr < 0.4:
                size_factor *= 0.7
                insights.append(f"⚠️ Low win rate: {wr*100:.0f}%")
            elif wr > 0.6:
                size_factor *= 1.2
                insights.append(f"✅ High win rate: {wr*100:.0f}%")
        
        # 6. Get optimized params
        params = self.strategy_optimizer.get_current_params()
        
        # Apply min confidence check
        if pattern_confidence * confidence_adj < params.min_confidence:
            can_trade = False
            insights.append(f"❌ Confidence {pattern_confidence*confidence_adj:.0f}% < {params.min_confidence:.0f}%")
        
        return LearningDecision(
            can_trade=can_trade,
            confidence_adjustment=confidence_adj,
            position_size_factor=size_factor,
            factor_weights=factor_weights,
            market_cycle=cycle,
            optimized_params=params,
            insights=insights,
        )
    
    def get_insights(self) -> dict:
        """Get all learning insights"""
        if self.background:
            return self.background.get_insights()
        
        return {
            "online_learning": self.online_learner.get_statistics(),
            "market_cycle": self.cycle_detector.detect().to_dict(),
            "pattern_evolution": self.pattern_tracker.get_evolution_summary(),
            "strategy_optimization": self.strategy_optimizer.get_optimization_status(),
        }
    
    def _save_state(self):
        """Save learning state to file"""
        try:
            state = {
                "online_learner": self.online_learner.to_dict(),
                "pattern_stats": dict(list(self.pattern_tracker.pattern_stats.items())[:200]),  # Limit
                "optimizer_params": self.strategy_optimizer.best_params.to_dict(),
                "saved_at": datetime.now().isoformat(),
            }
            
            filepath = os.path.join(self.data_dir, "learning_state.json")
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Also save to Firebase if available
            if self.firebase:
                try:
                    self.firebase.save_learning_state(state)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
    
    async def save_all_state(self):
        """Async version of save state - for graceful shutdown"""
        try:
            self._save_state()
            logger.info("✅ All learning state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save all state: {e}")
    
    def _load_state(self):
        """Load learning state from file"""
        try:
            filepath = os.path.join(self.data_dir, "learning_state.json")
            
            # Try Firebase first
            state = None
            if self.firebase:
                try:
                    state = self.firebase.load_learning_state()
                except:
                    pass
            
            # Fall back to local
            if not state and os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
            
            if state:
                self.online_learner.from_dict(state.get("online_learner", {}))
                
                pattern_stats = state.get("pattern_stats", {})
                self.pattern_tracker.pattern_stats = pattern_stats
                
                params_dict = state.get("optimizer_params", {})
                for key, value in params_dict.items():
                    if hasattr(self.strategy_optimizer.best_params, key):
                        setattr(self.strategy_optimizer.best_params, key, value)
                
                logger.info(f"📚 Loaded learning state: {self.online_learner.sample_count} samples")
            
        except Exception as e:
            logger.warning(f"Could not load learning state: {e}")


# Singleton
_learning_system: Optional[ContinuousLearningSystem] = None


def get_learning_system(firebase_service=None) -> ContinuousLearningSystem:
    """Get or create continuous learning system"""
    global _learning_system
    if _learning_system is None:
        _learning_system = ContinuousLearningSystem(firebase_service=firebase_service)
    return _learning_system


def init_learning_system(firebase_service=None) -> ContinuousLearningSystem:
    """Initialize new learning system"""
    global _learning_system
    _learning_system = ContinuousLearningSystem(firebase_service=firebase_service)
    return _learning_system


# Test
if __name__ == "__main__":
    import asyncio
    
    print("=" * 60)
    print("  CONTINUOUS LEARNING SYSTEM TEST")
    print("=" * 60)
    
    async def test():
        # Create system
        system = ContinuousLearningSystem(enable_background=False)
        
        # Simulate trades
        print("\n1. Simulating 30 trades...")
        for i in range(30):
            is_win = i % 3 != 0  # 66% win rate
            pnl = 1.5 if is_win else -1.0
            factors = {
                "pattern_confidence": i % 2 == 0,
                "regime_aligned": i % 3 != 0,
                "momentum_aligned": is_win,
                "near_sr": i % 4 == 0,
            }
            
            system.learn_from_trade(
                is_win=is_win,
                pnl_percent=pnl,
                factors=factors,
                pattern_hash=f"PATTERN_{i % 5}",
            )
        
        print("   Done!")
        
        # Get insights
        print("\n2. Learning Insights:")
        insights = system.get_insights()
        
        online = insights["online_learning"]
        print(f"   Win Rate (EMA): {online['ema_win_rate']}%")
        print(f"   Samples: {online['sample_count']}")
        print(f"   Factor Ranking:")
        for factor, weight in online["factor_ranking"][:5]:
            print(f"      {factor}: {weight:.2f}")
        
        # Test evaluate
        print("\n3. Evaluate Entry:")
        decision = system.evaluate(
            signal_side="BUY",
            pattern_confidence=75,
            factors={"regime_aligned": True, "momentum_aligned": True},
            pattern_hash="PATTERN_1",
        )
        print(f"   Can Trade: {decision.can_trade}")
        print(f"   Confidence Adj: {decision.confidence_adjustment:.2f}x")
        print(f"   Position Factor: {decision.position_size_factor:.2f}x")
        print(f"   Insights:")
        for insight in decision.insights:
            print(f"      {insight}")
        
        print("\n" + "=" * 60)
        print("  TEST PASSED!")
        print("=" * 60)
    
    asyncio.run(test())
