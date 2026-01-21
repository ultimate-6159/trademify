"""
Smart Brain - AI Self-Learning Trading System
==============================================

ระบบฉลาดที่เรียนรู้จากการเทรดของตัวเอง:

1. **Trade Journal** - บันทึกทุกเทรดพร้อมบริบท
2. **Pattern Memory** - จำว่า pattern ไหนได้/เสีย
3. **Adaptive Risk** - ปรับ size ตาม performance
4. **Time Analysis** - รู้ว่าช่วงไหนเทรดดี/แย่
5. **Symbol Performance** - รู้ว่า symbol ไหนเก่ง
6. **Entry Optimization** - รอ pullback เข้าราคาดีกว่า
7. **Partial Take Profit** - ปิดบางส่วน ล็อคกำไร
8. **Stale Trade Exit** - ปิดเทรดที่ไม่ไปไหน
9. **Self Improvement** - ปรับ settings อัตโนมัติ
"""
import json
import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# TRADE JOURNAL - บันทึกทุกเทรดพร้อมบริบท
# =============================================================================

@dataclass
class TradeRecord:
    """บันทึกเทรดพร้อมบริบททั้งหมด"""
    trade_id: str
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: float = 0
    take_profit: float = 0
    quantity: float = 0
    
    # Context when opened
    entry_time: str = ""
    session: str = ""  # LONDON, NY, OVERLAP, etc.
    signal_quality: str = ""  # PREMIUM, HIGH, MEDIUM, LOW
    pattern_confidence: float = 0
    market_regime: str = ""  # TRENDING, RANGING, VOLATILE
    
    # Result
    exit_time: Optional[str] = None
    pnl: float = 0
    pnl_percent: float = 0
    exit_reason: str = ""  # TP, SL, TRAILING, MANUAL, TIME
    holding_time_hours: float = 0
    
    # Pattern info
    pattern_id: Optional[str] = None
    similar_patterns_count: int = 0
    
    def is_win(self) -> bool:
        return self.pnl > 0
    
    def to_dict(self) -> dict:
        return asdict(self)


class TradeJournal:
    """บันทึกและวิเคราะห์ประวัติเทรด - รองรับทั้ง Local และ Firebase"""
    
    def __init__(self, data_dir: str = "data/journal", firebase_service=None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.trades: List[TradeRecord] = []
        self.firebase = firebase_service
        self._load_trades()
    
    def set_firebase(self, firebase_service):
        """Set Firebase service for cloud sync"""
        self.firebase = firebase_service
        if firebase_service:
            logger.info("📱 TradeJournal: Firebase connected")
    
    def _load_trades(self):
        """Load trades from Firebase first, then local file"""
        # Try Firebase first
        if self.firebase:
            try:
                firebase_trades = self.firebase.load_trade_journal()
                if firebase_trades:
                    if isinstance(firebase_trades, dict):
                        firebase_trades = list(firebase_trades.values())
                    self.trades = [TradeRecord(**t) for t in firebase_trades if t]
                    logger.info(f"☁️ Loaded {len(self.trades)} trades from Firebase")
                    return
            except Exception as e:
                logger.warning(f"Firebase load failed: {e}")
        
        # Fallback to local file
        filepath = os.path.join(self.data_dir, "trades.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data]
                logger.info(f"💾 Loaded {len(self.trades)} trades from local file")
            except Exception as e:
                logger.warning(f"Failed to load journal: {e}")
                self.trades = []
    
    def _save_trades(self):
        """Save trades to both Firebase and local file"""
        # Save to Firebase
        if self.firebase:
            try:
                self.firebase.save_trade_journal([t.to_dict() for t in self.trades])
            except Exception as e:
                logger.warning(f"Firebase save failed: {e}")
        
        # Always save local backup
        filepath = os.path.join(self.data_dir, "trades.json")
        try:
            with open(filepath, 'w') as f:
                json.dump([t.to_dict() for t in self.trades], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save journal: {e}")
    
    def record_entry(self, trade: TradeRecord):
        """Record trade entry"""
        self.trades.append(trade)
        self._save_trades()
        
        # Also add to Firebase trade history
        if self.firebase:
            self.firebase.add_trade(trade.to_dict())
        
        logger.info(f"📝 Journal: Recorded entry {trade.trade_id}")
    
    def record_exit(self, trade_id: str, exit_price: float, exit_reason: str):
        """Record trade exit"""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                trade.exit_price = exit_price
                trade.exit_time = datetime.now().isoformat()
                trade.exit_reason = exit_reason
                
                # Calculate PnL
                if trade.side == "BUY":
                    trade.pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                else:
                    trade.pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100
                
                trade.pnl = trade.pnl_percent * trade.quantity
                
                # Calculate holding time
                if trade.entry_time:
                    entry_dt = datetime.fromisoformat(trade.entry_time)
                    trade.holding_time_hours = (datetime.now() - entry_dt).total_seconds() / 3600
                
                self._save_trades()
                logger.info(f"📝 Journal: Recorded exit {trade_id} - {exit_reason} - PnL: {trade.pnl_percent:.2f}%")
                return
    
    def get_recent_trades(self, days: int = 30) -> List[TradeRecord]:
        """Get trades from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            t for t in self.trades 
            if t.entry_time and datetime.fromisoformat(t.entry_time) > cutoff
        ]
    
    def get_stats(self, days: int = 30) -> dict:
        """Get trading statistics"""
        recent = self.get_recent_trades(days)
        closed = [t for t in recent if t.exit_price is not None]
        
        if not closed:
            return {"total": 0, "win_rate": 0, "avg_pnl": 0}
        
        wins = [t for t in closed if t.is_win()]
        
        return {
            "total": len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0,
            "avg_pnl": sum(t.pnl_percent for t in closed) / len(closed),
            "total_pnl": sum(t.pnl_percent for t in closed),
            "avg_holding_hours": sum(t.holding_time_hours for t in closed) / len(closed),
        }


# =============================================================================
# PATTERN MEMORY - จำว่า pattern ไหนได้/เสีย
# =============================================================================

@dataclass 
class PatternMemory:
    """ความจำของ pattern"""
    pattern_hash: str
    times_seen: int = 0
    times_traded: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0
    last_seen: str = ""
    
    @property
    def win_rate(self) -> float:
        if self.times_traded == 0:
            return 0
        return self.wins / self.times_traded * 100
    
    @property
    def avg_pnl(self) -> float:
        if self.times_traded == 0:
            return 0
        return self.total_pnl / self.times_traded
    
    def should_trade(self) -> Tuple[bool, str]:
        """ควรเทรด pattern นี้มั้ย"""
        if self.times_traded < 3:
            return True, "Not enough data"
        
        if self.win_rate < 40:
            return False, f"Poor win rate: {self.win_rate:.0f}%"
        
        if self.avg_pnl < -1.0:
            return False, f"Negative avg PnL: {self.avg_pnl:.2f}%"
        
        return True, f"Good pattern: {self.win_rate:.0f}% win, {self.avg_pnl:.2f}% avg"


class PatternMemoryBank:
    """ธนาคารความจำ patterns - รองรับทั้ง Local และ Firebase"""
    
    def __init__(self, data_dir: str = "data/patterns", firebase_service=None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.patterns: Dict[str, PatternMemory] = {}
        self.firebase = firebase_service
        self._load()
    
    def set_firebase(self, firebase_service):
        """Set Firebase service for cloud sync"""
        self.firebase = firebase_service
        if firebase_service:
            logger.info("📱 PatternMemoryBank: Firebase connected")
    
    def _load(self):
        # Try Firebase first
        if self.firebase:
            try:
                firebase_patterns = self.firebase.load_pattern_memory()
                if firebase_patterns:
                    self.patterns = {k: PatternMemory(**v) for k, v in firebase_patterns.items() if v}
                    logger.info(f"☁️ Loaded {len(self.patterns)} pattern memories from Firebase")
                    return
            except Exception as e:
                logger.warning(f"Firebase pattern load failed: {e}")
        
        # Fallback to local
        filepath = os.path.join(self.data_dir, "pattern_memory.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.patterns = {k: PatternMemory(**v) for k, v in data.items()}
                logger.info(f"💾 Loaded {len(self.patterns)} pattern memories from local")
            except:
                pass
    
    def _save(self):
        # Save to Firebase
        if self.firebase:
            try:
                self.firebase.save_pattern_memory({k: asdict(v) for k, v in self.patterns.items()})
            except Exception as e:
                logger.warning(f"Firebase pattern save failed: {e}")
        
        # Always save local backup
        filepath = os.path.join(self.data_dir, "pattern_memory.json")
        try:
            with open(filepath, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.patterns.items()}, f)
        except:
            pass
    
    def record_pattern(self, pattern_hash: str):
        """บันทึกว่าเห็น pattern นี้"""
        if pattern_hash not in self.patterns:
            self.patterns[pattern_hash] = PatternMemory(pattern_hash=pattern_hash)
        
        self.patterns[pattern_hash].times_seen += 1
        self.patterns[pattern_hash].last_seen = datetime.now().isoformat()
        self._save()
    
    def record_trade_result(self, pattern_hash: str, is_win: bool, pnl_percent: float):
        """บันทึกผลเทรดของ pattern"""
        if pattern_hash not in self.patterns:
            self.patterns[pattern_hash] = PatternMemory(pattern_hash=pattern_hash)
        
        mem = self.patterns[pattern_hash]
        mem.times_traded += 1
        mem.total_pnl += pnl_percent
        
        if is_win:
            mem.wins += 1
        else:
            mem.losses += 1
        
        self._save()
        logger.info(f"📊 Pattern {pattern_hash[:8]}: Win rate {mem.win_rate:.0f}% ({mem.times_traded} trades)")
    
    def should_trade(self, pattern_hash: str) -> Tuple[bool, str]:
        """ควรเทรด pattern นี้มั้ย"""
        if pattern_hash not in self.patterns:
            return True, "New pattern"
        
        return self.patterns[pattern_hash].should_trade()


# =============================================================================
# ADAPTIVE RISK - ปรับ size ตาม performance
# =============================================================================

class AdaptiveRisk:
    """
    ปรับ Risk แบบ Dynamic ตาม performance
    
    หลักการ:
    - Winning streak → ค่อยๆ เพิ่ม size (max 1.5x)
    - Losing streak → ลด size (min 0.25x)
    - Based on recent performance, not just streak
    """
    
    def __init__(
        self,
        base_risk: float = 2.0,
        min_multiplier: float = 0.25,
        max_multiplier: float = 1.5,
        lookback_trades: int = 10,
    ):
        self.base_risk = base_risk
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.lookback_trades = lookback_trades
        
        self.recent_results: List[bool] = []  # True = win
        self.recent_pnl: List[float] = []
    
    def record_result(self, is_win: bool, pnl_percent: float):
        """บันทึกผลเทรด"""
        self.recent_results.append(is_win)
        self.recent_pnl.append(pnl_percent)
        
        # Keep only lookback trades
        if len(self.recent_results) > self.lookback_trades:
            self.recent_results = self.recent_results[-self.lookback_trades:]
            self.recent_pnl = self.recent_pnl[-self.lookback_trades:]
    
    def get_risk_multiplier(self) -> Tuple[float, str]:
        """
        คำนวณ multiplier ตาม performance
        
        Returns:
            (multiplier, reason)
        """
        if len(self.recent_results) < 3:
            return 1.0, "Not enough data"
        
        # Calculate recent win rate
        win_rate = sum(self.recent_results) / len(self.recent_results)
        
        # Calculate recent PnL
        avg_pnl = sum(self.recent_pnl) / len(self.recent_pnl)
        
        # Check for streaks
        streak = 0
        for r in reversed(self.recent_results):
            if r == self.recent_results[-1]:
                streak += 1
            else:
                break
        
        is_winning_streak = self.recent_results[-1] if self.recent_results else False
        
        # Calculate multiplier
        if is_winning_streak and streak >= 3:
            # Winning streak - increase cautiously
            bonus = min(0.1 * (streak - 2), 0.5)  # Max +50%
            mult = min(1.0 + bonus, self.max_multiplier)
            reason = f"🔥 Winning streak ({streak}): +{bonus*100:.0f}%"
        
        elif not is_winning_streak and streak >= 2:
            # Losing streak - decrease
            penalty = min(0.25 * (streak - 1), 0.75)  # Max -75%
            mult = max(1.0 - penalty, self.min_multiplier)
            reason = f"❄️ Losing streak ({streak}): -{penalty*100:.0f}%"
        
        elif win_rate >= 0.7 and avg_pnl > 0.5:
            # Excellent performance
            mult = min(1.25, self.max_multiplier)
            reason = f"✨ Excellent: {win_rate*100:.0f}% win, {avg_pnl:.1f}% avg"
        
        elif win_rate <= 0.4 or avg_pnl < -0.5:
            # Poor performance
            mult = max(0.5, self.min_multiplier)
            reason = f"⚠️ Poor: {win_rate*100:.0f}% win, {avg_pnl:.1f}% avg"
        
        else:
            mult = 1.0
            reason = f"Normal: {win_rate*100:.0f}% win"
        
        return mult, reason


# =============================================================================
# TIME ANALYSIS - รู้ว่าช่วงไหนเทรดดี/แย่
# =============================================================================

class TimeAnalyzer:
    """วิเคราะห์ว่าช่วงเวลาไหนเทรดได้ดี"""
    
    def __init__(self):
        # Track performance by hour (UTC)
        self.hourly_stats: Dict[int, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )
        
        # Track by day of week
        self.daily_stats: Dict[int, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )
    
    def record_trade(self, entry_time: datetime, is_win: bool, pnl: float):
        """บันทึกเทรด"""
        hour = entry_time.hour
        day = entry_time.weekday()
        
        self.hourly_stats[hour]["trades"] += 1
        self.hourly_stats[hour]["pnl"] += pnl
        if is_win:
            self.hourly_stats[hour]["wins"] += 1
        
        self.daily_stats[day]["trades"] += 1
        self.daily_stats[day]["pnl"] += pnl
        if is_win:
            self.daily_stats[day]["wins"] += 1
    
    def should_trade_now(self) -> Tuple[bool, float, str]:
        """
        ควรเทรดตอนนี้มั้ย
        
        Returns:
            (should_trade, multiplier, reason)
        """
        now = datetime.utcnow()
        hour = now.hour
        day = now.weekday()
        
        hour_data = self.hourly_stats[hour]
        day_data = self.daily_stats[day]
        
        # Not enough data
        if hour_data["trades"] < 5 and day_data["trades"] < 5:
            return True, 1.0, "Not enough data for this time"
        
        # Check hour performance
        hour_win_rate = hour_data["wins"] / hour_data["trades"] if hour_data["trades"] > 0 else 0.5
        hour_avg_pnl = hour_data["pnl"] / hour_data["trades"] if hour_data["trades"] > 0 else 0
        
        # Check day performance
        day_win_rate = day_data["wins"] / day_data["trades"] if day_data["trades"] > 0 else 0.5
        
        # Bad hour
        if hour_data["trades"] >= 5 and (hour_win_rate < 0.35 or hour_avg_pnl < -1.0):
            return False, 0, f"❌ Hour {hour}:00 UTC has {hour_win_rate*100:.0f}% win rate"
        
        # Bad day  
        if day_data["trades"] >= 5 and day_win_rate < 0.35:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            return False, 0, f"❌ {day_names[day]} has {day_win_rate*100:.0f}% win rate"
        
        # Good hour
        if hour_data["trades"] >= 5 and hour_win_rate > 0.65:
            return True, 1.2, f"✨ Hour {hour}:00 UTC has {hour_win_rate*100:.0f}% win rate"
        
        return True, 1.0, "Normal time"
    
    def get_best_hours(self) -> List[int]:
        """Get best trading hours"""
        good_hours = []
        for hour, stats in self.hourly_stats.items():
            if stats["trades"] >= 5:
                win_rate = stats["wins"] / stats["trades"]
                if win_rate >= 0.6:
                    good_hours.append(hour)
        return sorted(good_hours)


# =============================================================================
# SYMBOL PERFORMANCE - รู้ว่า symbol ไหนเก่ง
# =============================================================================

class SymbolAnalyzer:
    """วิเคราะห์ performance แต่ละ symbol"""
    
    def __init__(self):
        self.stats: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "avg_holding": 0.0}
        )
    
    def record_trade(self, symbol: str, is_win: bool, pnl: float, holding_hours: float):
        """บันทึกเทรด"""
        self.stats[symbol]["trades"] += 1
        self.stats[symbol]["pnl"] += pnl
        self.stats[symbol]["avg_holding"] = (
            (self.stats[symbol]["avg_holding"] * (self.stats[symbol]["trades"] - 1) + holding_hours) 
            / self.stats[symbol]["trades"]
        )
        if is_win:
            self.stats[symbol]["wins"] += 1
    
    def should_trade(self, symbol: str) -> Tuple[bool, float, str]:
        """
        ควรเทรด symbol นี้มั้ย
        
        Returns:
            (should_trade, multiplier, reason)
        """
        if symbol not in self.stats or self.stats[symbol]["trades"] < 5:
            return True, 1.0, "Not enough data"
        
        s = self.stats[symbol]
        win_rate = s["wins"] / s["trades"]
        avg_pnl = s["pnl"] / s["trades"]
        
        # Bad symbol
        if win_rate < 0.35 or avg_pnl < -1.0:
            return False, 0, f"❌ {symbol}: {win_rate*100:.0f}% win, {avg_pnl:.1f}% avg"
        
        # Great symbol
        if win_rate > 0.65 and avg_pnl > 0.5:
            return True, 1.3, f"✨ {symbol}: {win_rate*100:.0f}% win, {avg_pnl:.1f}% avg"
        
        # Good symbol
        if win_rate > 0.55:
            return True, 1.1, f"👍 {symbol}: {win_rate*100:.0f}% win"
        
        return True, 1.0, f"{symbol}: {win_rate*100:.0f}% win"
    
    def get_best_symbols(self) -> List[str]:
        """Get best performing symbols"""
        good = []
        for symbol, stats in self.stats.items():
            if stats["trades"] >= 5:
                win_rate = stats["wins"] / stats["trades"]
                if win_rate >= 0.55:
                    good.append(symbol)
        return good


# =============================================================================
# ENTRY OPTIMIZATION - รอ pullback เข้าราคาดีกว่า
# =============================================================================

@dataclass
class PendingEntry:
    """Entry ที่รอ pullback"""
    symbol: str
    side: str
    target_price: float  # ราคาที่ต้องการ
    original_signal_price: float
    max_wait_hours: float
    created_at: datetime
    analysis: dict  # เก็บ analysis ไว้
    
    def is_expired(self) -> bool:
        elapsed = (datetime.now() - self.created_at).total_seconds() / 3600
        return elapsed > self.max_wait_hours


class EntryOptimizer:
    """
    รอ pullback ก่อนเข้า
    
    หลักการ:
    - BUY signal → รอราคาลงมาหน่อยค่อยเข้า
    - SELL signal → รอราคาขึ้นมาหน่อยค่อยเข้า
    - ได้ entry ที่ดีกว่า = R:R ดีกว่า
    """
    
    def __init__(
        self,
        pullback_percent: float = 0.2,  # รอ pullback 0.2%
        max_wait_hours: float = 4,      # รอไม่เกิน 4 ชม.
    ):
        self.pullback_percent = pullback_percent
        self.max_wait_hours = max_wait_hours
        self.pending_entries: Dict[str, PendingEntry] = {}
    
    def create_pending_entry(
        self,
        symbol: str,
        side: str,
        current_price: float,
        analysis: dict,
    ) -> PendingEntry:
        """สร้าง pending entry รอ pullback"""
        if side == "BUY":
            target = current_price * (1 - self.pullback_percent / 100)
        else:
            target = current_price * (1 + self.pullback_percent / 100)
        
        entry = PendingEntry(
            symbol=symbol,
            side=side,
            target_price=target,
            original_signal_price=current_price,
            max_wait_hours=self.max_wait_hours,
            created_at=datetime.now(),
            analysis=analysis,
        )
        
        self.pending_entries[symbol] = entry
        logger.info(f"📋 Pending entry: {symbol} {side} @ {target:.5f} (waiting for pullback)")
        
        return entry
    
    def check_entry(self, symbol: str, current_price: float) -> Tuple[bool, Optional[dict]]:
        """
        ตรวจสอบว่าถึงราคา entry หรือยัง
        
        Returns:
            (should_enter, analysis)
        """
        if symbol not in self.pending_entries:
            return False, None
        
        entry = self.pending_entries[symbol]
        
        # Check expiry
        if entry.is_expired():
            logger.info(f"⏰ Pending entry expired: {symbol}")
            del self.pending_entries[symbol]
            return False, None
        
        # Check price
        if entry.side == "BUY" and current_price <= entry.target_price:
            logger.info(f"✅ Pullback entry triggered: {symbol} BUY @ {current_price:.5f}")
            analysis = entry.analysis
            del self.pending_entries[symbol]
            return True, analysis
        
        if entry.side == "SELL" and current_price >= entry.target_price:
            logger.info(f"✅ Pullback entry triggered: {symbol} SELL @ {current_price:.5f}")
            analysis = entry.analysis
            del self.pending_entries[symbol]
            return True, analysis
        
        return False, None
    
    def cancel_pending(self, symbol: str):
        """ยกเลิก pending entry"""
        if symbol in self.pending_entries:
            del self.pending_entries[symbol]


# =============================================================================
# PARTIAL TAKE PROFIT - ปิดบางส่วน ล็อคกำไร
# =============================================================================

@dataclass
class PartialTPPlan:
    """แผนการ partial TP"""
    position_id: str
    original_quantity: float
    remaining_quantity: float
    tp1_price: float
    tp1_percent: float = 50  # ปิด 50% ที่ TP1
    tp1_hit: bool = False
    tp2_price: Optional[float] = None


class PartialTPManager:
    """
    จัดการ Partial Take Profit
    
    หลักการ:
    - TP1: ปิด 50% ที่ 1R (เท่ากับระยะ SL)
    - TP2: Trail ที่เหลือ
    - ล็อคกำไรบางส่วน + ให้โอกาสวิ่งต่อ
    """
    
    def __init__(self, tp1_percent: float = 50):
        self.tp1_percent = tp1_percent
        self.plans: Dict[str, PartialTPPlan] = {}
    
    def create_plan(
        self,
        position_id: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        quantity: float,
    ) -> PartialTPPlan:
        """สร้างแผน partial TP"""
        # TP1 = 1R (เท่ากับระยะไป SL)
        sl_distance = abs(entry_price - stop_loss)
        
        if side == "BUY":
            tp1 = entry_price + sl_distance
            tp2 = entry_price + (sl_distance * 2)  # 2R
        else:
            tp1 = entry_price - sl_distance
            tp2 = entry_price - (sl_distance * 2)
        
        plan = PartialTPPlan(
            position_id=position_id,
            original_quantity=quantity,
            remaining_quantity=quantity,
            tp1_price=tp1,
            tp1_percent=self.tp1_percent,
            tp2_price=tp2,
        )
        
        self.plans[position_id] = plan
        logger.info(f"📊 Partial TP plan: TP1={tp1:.5f} ({self.tp1_percent}%), TP2={tp2:.5f} (trail)")
        
        return plan
    
    def check_tp1(
        self,
        position_id: str,
        side: str,
        current_price: float,
    ) -> Tuple[bool, float]:
        """
        ตรวจสอบว่าถึง TP1 หรือยัง
        
        Returns:
            (should_close_partial, quantity_to_close)
        """
        if position_id not in self.plans:
            return False, 0
        
        plan = self.plans[position_id]
        
        if plan.tp1_hit:
            return False, 0
        
        # Check TP1
        if side == "BUY" and current_price >= plan.tp1_price:
            close_qty = plan.original_quantity * (plan.tp1_percent / 100)
            plan.tp1_hit = True
            plan.remaining_quantity = plan.original_quantity - close_qty
            logger.info(f"✨ TP1 hit! Closing {close_qty:.2f} lots ({plan.tp1_percent}%)")
            return True, close_qty
        
        if side == "SELL" and current_price <= plan.tp1_price:
            close_qty = plan.original_quantity * (plan.tp1_percent / 100)
            plan.tp1_hit = True
            plan.remaining_quantity = plan.original_quantity - close_qty
            return True, close_qty
        
        return False, 0
    
    def clear_plan(self, position_id: str):
        """Clear plan when position closed"""
        if position_id in self.plans:
            del self.plans[position_id]


# =============================================================================
# STALE TRADE EXIT - ปิดเทรดที่ไม่ไปไหน
# =============================================================================

class StaleTradeMonitor:
    """
    ตรวจจับและปิดเทรดที่ไม่ไปไหน
    
    หลักการ:
    - ถ้าเข้ามา X ชม. แล้วไม่ถึง TP หรือ SL
    - และไม่มี profit → ปิด
    - ดีกว่าถือต่อไปเสียเวลา
    """
    
    def __init__(
        self,
        max_holding_hours: float = 24,  # ถือไม่เกิน 24 ชม.
        min_profit_to_stay: float = 0.3,  # ต้อง profit > 0.3% ถึงจะถือต่อ
    ):
        self.max_holding_hours = max_holding_hours
        self.min_profit_to_stay = min_profit_to_stay
        
        self.open_times: Dict[str, datetime] = {}
    
    def record_open(self, position_id: str):
        """บันทึกเวลาเปิด"""
        self.open_times[position_id] = datetime.now()
    
    def should_close_stale(
        self,
        position_id: str,
        side: str,
        entry_price: float,
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        ควรปิดเพราะ stale หรือไม่
        
        Returns:
            (should_close, reason)
        """
        if position_id not in self.open_times:
            return False, ""
        
        # Calculate holding time
        holding_hours = (datetime.now() - self.open_times[position_id]).total_seconds() / 3600
        
        if holding_hours < self.max_holding_hours / 2:
            return False, ""  # ยังไม่ถึงครึ่งเวลา
        
        # Calculate profit
        if side == "BUY":
            profit_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_percent = ((entry_price - current_price) / entry_price) * 100
        
        # Stale conditions
        if holding_hours >= self.max_holding_hours and profit_percent < self.min_profit_to_stay:
            return True, f"Stale: {holding_hours:.1f}h, only {profit_percent:.2f}% profit"
        
        # Loss + long holding
        if holding_hours >= self.max_holding_hours * 0.75 and profit_percent < 0:
            return True, f"Stale loss: {holding_hours:.1f}h, {profit_percent:.2f}%"
        
        return False, ""
    
    def clear_position(self, position_id: str):
        """Clear when closed"""
        if position_id in self.open_times:
            del self.open_times[position_id]


# =============================================================================
# SMART BRAIN - รวมทุกอย่าง
# =============================================================================

@dataclass
class SmartDecision:
    """ผลการตัดสินใจจาก Smart Brain"""
    can_trade: bool
    risk_multiplier: float
    should_wait_pullback: bool
    reasons: List[str]
    insights: List[str]
    
    def to_dict(self) -> dict:
        return {
            "can_trade": self.can_trade,
            "risk_multiplier": self.risk_multiplier,
            "should_wait_pullback": self.should_wait_pullback,
            "reasons": self.reasons,
            "insights": self.insights,
        }


class SmartBrain:
    """
    สมองกลอัจฉริยะ - เรียนรู้และปรับตัวเอง
    
    รองรับ:
    - Firebase Cloud Storage (ข้อมูลไม่หาย)
    - Local Backup (offline fallback)
    
    รวม:
    - Trade Journal (บันทึกทุกเทรด)
    - Pattern Memory (จำว่า pattern ไหนได้/เสีย)
    - Adaptive Risk (ปรับ size ตาม performance)
    - Time Analysis (รู้ว่าช่วงไหนเทรดดี)
    - Symbol Analysis (รู้ว่า symbol ไหนเก่ง)
    - Entry Optimizer (รอ pullback)
    - Partial TP (ปิดบางส่วน)
    - Stale Trade Exit (ปิดเทรดค้าง)
    """
    
    def __init__(
        self,
        enable_pullback_entry: bool = True,
        enable_partial_tp: bool = True,
        enable_stale_exit: bool = True,
        enable_adaptive_risk: bool = True,
        firebase_service = None,
    ):
        self.enable_pullback_entry = enable_pullback_entry
        self.enable_partial_tp = enable_partial_tp
        self.enable_stale_exit = enable_stale_exit
        self.enable_adaptive_risk = enable_adaptive_risk
        self.firebase = firebase_service
        
        # Components - Initialize with Firebase if available
        self.journal = TradeJournal(firebase_service=firebase_service)
        self.pattern_memory = PatternMemoryBank(firebase_service=firebase_service)
        self.adaptive_risk = AdaptiveRisk()
        self.time_analyzer = TimeAnalyzer()
        self.symbol_analyzer = SymbolAnalyzer()
        self.entry_optimizer = EntryOptimizer()
        self.partial_tp = PartialTPManager()
        self.stale_monitor = StaleTradeMonitor()
        
        if firebase_service:
            logger.info("🧠 Smart Brain initialized with Firebase ☁️")
        else:
            logger.info("🧠 Smart Brain initialized (local storage)")
        logger.info(f"   - Pullback Entry: {enable_pullback_entry}")
        logger.info(f"   - Partial TP: {enable_partial_tp}")
        logger.info(f"   - Stale Exit: {enable_stale_exit}")
        logger.info(f"   - Adaptive Risk: {enable_adaptive_risk}")
    
    def evaluate_entry(
        self,
        symbol: str,
        side: str,
        pattern_hash: Optional[str] = None,
    ) -> SmartDecision:
        """
        ประเมินว่าควรเข้าเทรดมั้ย
        
        รวมการตัดสินใจจากทุก component
        """
        reasons = []
        insights = []
        can_trade = True
        risk_mult = 1.0
        wait_pullback = False
        
        # 1. Pattern Memory Check
        if pattern_hash:
            should_trade, msg = self.pattern_memory.should_trade(pattern_hash)
            if not should_trade:
                can_trade = False
                reasons.append(msg)
            else:
                insights.append(f"Pattern: {msg}")
        
        # 2. Time Analysis
        time_ok, time_mult, time_msg = self.time_analyzer.should_trade_now()
        if not time_ok:
            can_trade = False
            reasons.append(time_msg)
        else:
            risk_mult *= time_mult
            if time_mult != 1.0:
                insights.append(time_msg)
        
        # 3. Symbol Analysis
        sym_ok, sym_mult, sym_msg = self.symbol_analyzer.should_trade(symbol)
        if not sym_ok:
            can_trade = False
            reasons.append(sym_msg)
        else:
            risk_mult *= sym_mult
            if sym_mult != 1.0:
                insights.append(sym_msg)
        
        # 4. Adaptive Risk
        if self.enable_adaptive_risk:
            adapt_mult, adapt_msg = self.adaptive_risk.get_risk_multiplier()
            risk_mult *= adapt_mult
            if adapt_mult != 1.0:
                insights.append(adapt_msg)
        
        # 5. Entry Optimization suggestion
        if self.enable_pullback_entry and can_trade:
            # Check if it's a good time for pullback
            journal_stats = self.journal.get_stats(30)
            if journal_stats.get("win_rate", 50) >= 50:
                wait_pullback = True
                insights.append("💡 Consider waiting for pullback entry")
        
        # Final multiplier bounds
        risk_mult = max(0.25, min(1.5, risk_mult))
        
        return SmartDecision(
            can_trade=can_trade,
            risk_multiplier=risk_mult,
            should_wait_pullback=wait_pullback,
            reasons=reasons,
            insights=insights,
        )
    
    def record_trade_open(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        quantity: float,
        signal_quality: str,
        pattern_confidence: float,
        session: str,
        market_regime: str,
        pattern_hash: Optional[str] = None,
    ):
        """บันทึกการเปิดเทรด"""
        # Journal
        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            quantity=quantity,
            entry_time=datetime.now().isoformat(),
            session=session,
            signal_quality=signal_quality,
            pattern_confidence=pattern_confidence,
            market_regime=market_regime,
            pattern_id=pattern_hash,
        )
        self.journal.record_entry(record)
        
        # Stale monitor
        self.stale_monitor.record_open(trade_id)
        
        # Partial TP plan
        if self.enable_partial_tp:
            self.partial_tp.create_plan(
                position_id=trade_id,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                quantity=quantity,
            )
    
    def record_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pattern_hash: Optional[str] = None,
    ):
        """บันทึกการปิดเทรด"""
        # Journal
        self.journal.record_exit(trade_id, exit_price, exit_reason)
        
        # Find the trade
        trade = None
        for t in self.journal.trades:
            if t.trade_id == trade_id:
                trade = t
                break
        
        if trade:
            is_win = trade.is_win()
            pnl = trade.pnl_percent
            
            # Pattern memory
            if pattern_hash:
                self.pattern_memory.record_trade_result(pattern_hash, is_win, pnl)
            
            # Adaptive risk
            self.adaptive_risk.record_result(is_win, pnl)
            
            # Time analyzer
            if trade.entry_time:
                entry_dt = datetime.fromisoformat(trade.entry_time)
                self.time_analyzer.record_trade(entry_dt, is_win, pnl)
            
            # Symbol analyzer
            self.symbol_analyzer.record_trade(
                trade.symbol, is_win, pnl, trade.holding_time_hours
            )
        
        # Cleanup
        self.stale_monitor.clear_position(trade_id)
        self.partial_tp.clear_plan(trade_id)
    
    def get_insights(self) -> dict:
        """Get trading insights from learned data"""
        journal_stats = self.journal.get_stats(30)
        
        return {
            "performance_30d": journal_stats,
            "best_hours": self.time_analyzer.get_best_hours(),
            "best_symbols": self.symbol_analyzer.get_best_symbols(),
            "adaptive_risk_mult": self.adaptive_risk.get_risk_multiplier()[0],
            "patterns_in_memory": len(self.pattern_memory.patterns),
        }


# Singleton
_smart_brain: Optional[SmartBrain] = None

def get_smart_brain(firebase_service=None) -> SmartBrain:
    """Get Smart Brain singleton with optional Firebase support"""
    global _smart_brain
    if _smart_brain is None:
        _smart_brain = SmartBrain(firebase_service=firebase_service)
    elif firebase_service and not _smart_brain.firebase:
        # Connect Firebase if not already connected
        _smart_brain.firebase = firebase_service
        _smart_brain.journal.set_firebase(firebase_service)
        _smart_brain.pattern_memory.set_firebase(firebase_service)
        logger.info("🧠 Smart Brain connected to Firebase ☁️")
    return _smart_brain


def init_smart_brain(firebase_service=None) -> SmartBrain:
    """Initialize a new Smart Brain instance (replaces existing)"""
    global _smart_brain
    _smart_brain = SmartBrain(firebase_service=firebase_service)
    return _smart_brain


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  SMART BRAIN TEST")
    print("=" * 60)
    
    brain = SmartBrain()
    
    # Test entry evaluation
    print("\n1. Entry Evaluation:")
    decision = brain.evaluate_entry("EURUSDm", "BUY")
    print(f"   Can Trade: {decision.can_trade}")
    print(f"   Risk Mult: {decision.risk_multiplier}x")
    print(f"   Wait Pullback: {decision.should_wait_pullback}")
    
    # Test adaptive risk
    print("\n2. Adaptive Risk:")
    brain.adaptive_risk.record_result(True, 0.8)
    brain.adaptive_risk.record_result(True, 1.2)
    brain.adaptive_risk.record_result(True, 0.5)
    mult, msg = brain.adaptive_risk.get_risk_multiplier()
    print(f"   After 3 wins: {mult}x - {msg}")
    
    # Test pattern memory
    print("\n3. Pattern Memory:")
    brain.pattern_memory.record_pattern("ABC123")
    brain.pattern_memory.record_trade_result("ABC123", True, 1.5)
    brain.pattern_memory.record_trade_result("ABC123", True, 0.8)
    brain.pattern_memory.record_trade_result("ABC123", False, -0.5)
    should, msg = brain.pattern_memory.should_trade("ABC123")
    print(f"   Pattern ABC123: {should} - {msg}")
    
    print("\n" + "=" * 60)
    print("  Smart Brain Ready!")
    print("=" * 60)
