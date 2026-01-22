"""
Risk Guardian - ระบบป้องกันการล้างพอร์ต
==========================================

หน้าที่หลัก:
1. Max Daily Loss Cutoff - หยุดเทรดทันทีเมื่อขาดทุนถึงลิมิต
2. Position Size Calculator - คำนวณ lot ตามความเสี่ยงที่ยอมรับได้
3. Stop Loss Guardian - บังคับให้มี SL เสมอ
4. Max Drawdown Monitor - ตรวจสอบ Drawdown ตลอดเวลา
5. Correlation Risk - ไม่เปิด Position เหมือนกันหลายตัว

🔒 RULE: ไม่มี SL = ไม่เทรด!
"""
import asyncio
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """ระดับความเสี่ยง"""
    SAFE = "SAFE"           # ปกติ - เทรดได้
    WARNING = "WARNING"     # เตือน - ลด position size
    DANGER = "DANGER"       # อันตราย - ห้ามเปิด position ใหม่
    CRITICAL = "CRITICAL"   # วิกฤต - ปิดทุก position ทันที!


@dataclass
class DailyStats:
    """สถิติรายวัน"""
    date: date = field(default_factory=date.today)
    starting_balance: float = 0.0
    current_balance: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    
    @property
    def daily_pnl_percent(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return ((self.current_balance - self.starting_balance) / self.starting_balance) * 100
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "starting_balance": round(self.starting_balance, 2),
            "current_balance": round(self.current_balance, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "daily_pnl_percent": round(self.daily_pnl_percent, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 1),
            "max_drawdown": round(self.max_drawdown, 2),
        }


@dataclass
class RiskAssessment:
    """ผลการประเมินความเสี่ยง"""
    level: RiskLevel
    can_trade: bool
    max_position_size: float  # Multiplier (0.0 - 1.5)
    reasons: List[str]
    warnings: List[str]
    daily_stats: Optional[DailyStats] = None
    
    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "can_trade": self.can_trade,
            "max_position_size": round(self.max_position_size, 2),
            "reasons": self.reasons,
            "warnings": self.warnings,
            "daily_stats": self.daily_stats.to_dict() if self.daily_stats else None,
        }


class RiskGuardian:
    """
    ผู้พิทักษ์ความเสี่ยง - ป้องกันการล้างพอร์ต
    
    กฎหลัก:
    1. ขาดทุนเกิน Max Daily Loss → หยุดเทรดทั้งวัน
    2. Drawdown เกิน Max → ปิด Position + หยุดเทรด
    3. Position ใหม่ต้องมี SL เสมอ
    4. Risk per trade ไม่เกิน Max Risk %
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 2.0,    # % of balance
        max_daily_loss: float = 5.0,        # % of starting balance
        max_drawdown: float = 10.0,         # % of peak
        max_positions: int = 5,
        max_correlated_positions: int = 2,  # Max positions in same direction
        min_stop_loss_percent: float = 0.1,  # Minimum SL distance (0.1% for forex)
        max_stop_loss_percent: float = 10.0,  # Maximum SL distance
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_positions = max_positions
        self.max_correlated_positions = max_correlated_positions
        self.min_stop_loss_percent = min_stop_loss_percent
        self.max_stop_loss_percent = max_stop_loss_percent
        
        # Daily tracking
        self.daily_stats = DailyStats()
        self.trading_locked = False
        self.lock_reason = ""
        
        # Position tracking
        self.open_positions: Dict[str, dict] = {}  # symbol -> position info
        
        logger.info(f"🛡️ Risk Guardian initialized:")
        logger.info(f"   Max Risk/Trade: {max_risk_per_trade}%")
        logger.info(f"   Max Daily Loss: {max_daily_loss}%")
        logger.info(f"   Max Drawdown: {max_drawdown}%")
    
    def reset_daily(self, current_balance: float) -> None:
        """Reset สถิติรายวัน"""
        today = date.today()
        
        if self.daily_stats.date != today:
            logger.info(f"📅 New trading day: {today}")
            self.daily_stats = DailyStats(
                date=today,
                starting_balance=current_balance,
                current_balance=current_balance,
                peak_balance=current_balance,
            )
            self.trading_locked = False
            self.lock_reason = ""
    
    def assess_risk(
        self,
        current_balance: float,
        open_positions: List[dict],
        proposed_trade: Optional[dict] = None,
    ) -> RiskAssessment:
        """
        ประเมินความเสี่ยงปัจจุบัน
        
        Returns:
            RiskAssessment with trading permission
        """
        reasons = []
        warnings = []
        
        # Reset if new day
        self.reset_daily(current_balance)
        
        # Update stats
        self.daily_stats.current_balance = current_balance
        
        # Track peak and drawdown
        if current_balance > self.daily_stats.peak_balance:
            self.daily_stats.peak_balance = current_balance
        
        current_drawdown = 0
        if self.daily_stats.peak_balance > 0:
            current_drawdown = ((self.daily_stats.peak_balance - current_balance) 
                               / self.daily_stats.peak_balance) * 100
            self.daily_stats.max_drawdown = max(self.daily_stats.max_drawdown, current_drawdown)
        
        # Calculate unrealized PnL
        unrealized_pnl = sum(p.get("pnl", 0) for p in open_positions)
        self.daily_stats.unrealized_pnl = unrealized_pnl
        
        # ========== RISK CHECKS ==========
        
        # 1. Check if trading is locked
        if self.trading_locked:
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                can_trade=False,
                max_position_size=0.0,
                reasons=[f"🔒 Trading locked: {self.lock_reason}"],
                warnings=[],
                daily_stats=self.daily_stats,
            )
        
        # 2. Check Daily Loss Limit
        daily_loss_percent = -self.daily_stats.daily_pnl_percent
        if daily_loss_percent >= self.max_daily_loss:
            self.trading_locked = True
            self.lock_reason = f"Daily loss limit reached ({daily_loss_percent:.1f}% >= {self.max_daily_loss}%)"
            
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                can_trade=False,
                max_position_size=0.0,
                reasons=[f"🚨 DAILY LOSS LIMIT: Lost {daily_loss_percent:.1f}% today (max: {self.max_daily_loss}%)"],
                warnings=["Trading disabled until tomorrow"],
                daily_stats=self.daily_stats,
            )
        
        # 3. Check Drawdown
        if current_drawdown >= self.max_drawdown:
            self.trading_locked = True
            self.lock_reason = f"Max drawdown reached ({current_drawdown:.1f}%)"
            
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                can_trade=False,
                max_position_size=0.0,
                reasons=[f"🚨 MAX DRAWDOWN: {current_drawdown:.1f}% (max: {self.max_drawdown}%)"],
                warnings=["Close all positions recommended"],
                daily_stats=self.daily_stats,
            )
        
        # 4. Check Max Positions
        if len(open_positions) >= self.max_positions:
            return RiskAssessment(
                level=RiskLevel.DANGER,
                can_trade=False,
                max_position_size=0.0,
                reasons=[f"Max positions reached ({len(open_positions)}/{self.max_positions})"],
                warnings=[],
                daily_stats=self.daily_stats,
            )
        
        # 5. Calculate risk level and position size
        risk_level = RiskLevel.SAFE
        max_position_size = 1.0
        
        # Reduce size if approaching daily loss limit
        remaining_loss_room = self.max_daily_loss - daily_loss_percent
        if remaining_loss_room < self.max_daily_loss * 0.5:
            max_position_size = 0.5
            risk_level = RiskLevel.WARNING
            warnings.append(f"Only {remaining_loss_room:.1f}% loss room remaining - reduced position size")
        
        # Reduce size if approaching drawdown limit
        remaining_dd_room = self.max_drawdown - current_drawdown
        if remaining_dd_room < self.max_drawdown * 0.3:
            max_position_size = min(max_position_size, 0.3)
            risk_level = RiskLevel.WARNING
            warnings.append(f"Only {remaining_dd_room:.1f}% drawdown room remaining")
        
        # 6. Check correlated positions (same direction trades)
        if proposed_trade:
            proposed_side = proposed_trade.get("side", "")
            same_direction_count = sum(
                1 for p in open_positions
                if p.get("side", "").upper() == proposed_side.upper()
            )
            if same_direction_count >= self.max_correlated_positions:
                return RiskAssessment(
                    level=RiskLevel.DANGER,
                    can_trade=False,
                    max_position_size=0.0,
                    reasons=[f"Too many {proposed_side} positions ({same_direction_count})"],
                    warnings=["Reduce directional exposure"],
                    daily_stats=self.daily_stats,
                )
        
        # Determine final risk level
        if daily_loss_percent > self.max_daily_loss * 0.7:
            risk_level = RiskLevel.DANGER
            max_position_size = 0.25
            warnings.append("Approaching daily loss limit - minimal position size")
        
        return RiskAssessment(
            level=risk_level,
            can_trade=True,
            max_position_size=max_position_size,
            reasons=reasons,
            warnings=warnings,
            daily_stats=self.daily_stats,
        )
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        risk_multiplier: float = 1.0,  # From quality assessment
    ) -> Tuple[float, dict]:
        """
        คำนวณ Position Size ตามความเสี่ยงที่ยอมรับได้
        
        Formula: Position Size = (Balance * Risk%) / Stop Distance
        
        Returns:
            (lot_size, calculation_details)
        """
        if stop_loss <= 0 or entry_price <= 0:
            return 0.0, {"error": "Invalid prices"}
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_percent = (stop_distance / entry_price) * 100
        
        # Validate stop distance
        if stop_distance_percent < self.min_stop_loss_percent:
            return 0.0, {
                "error": f"Stop loss too tight: {stop_distance_percent:.2f}% (min: {self.min_stop_loss_percent}%)"
            }
        
        if stop_distance_percent > self.max_stop_loss_percent:
            return 0.0, {
                "error": f"Stop loss too wide: {stop_distance_percent:.2f}% (max: {self.max_stop_loss_percent}%)"
            }
        
        # Calculate risk amount
        effective_risk = self.max_risk_per_trade * risk_multiplier
        risk_amount = balance * (effective_risk / 100)
        
        # Calculate position size
        # For forex: lot_size = risk_amount / (stop_pips * pip_value)
        # Simplified: lot_size = risk_amount / stop_distance
        lot_size = risk_amount / stop_distance
        
        # Round to standard lot sizes (0.01 step for forex)
        lot_size = max(0.01, round(lot_size, 2))
        
        # Cap at reasonable maximum - but ensure minimum 0.01 lot for micro accounts
        # Note: entry_price comparison only valid for forex pairs (1.0-2.0 range)
        # For Gold/Indices, use different formula
        if entry_price < 100:  # Forex pairs like EURUSD (1.0-2.0)
            max_lot = min(10.0, balance / entry_price * 0.1)
        else:  # Gold, Indices, etc (high prices like 4000-5000)
            # For small accounts, allow minimum lot regardless
            max_lot = max(0.01, min(10.0, balance * 0.1))  # 10% of balance as max
        
        lot_size = min(lot_size, max_lot)
        
        # Ensure minimum 0.01 for any trade (micro lot)
        if lot_size < 0.01 and balance >= 1.0:  # Allow trading if at least $1 balance
            lot_size = 0.01
            logger.warning(f"⚠️ Position size capped at minimum 0.01 lot for small account")
        
        calculation = {
            "balance": balance,
            "risk_percent": effective_risk,
            "risk_amount": round(risk_amount, 2),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "stop_distance": round(stop_distance, 5),
            "stop_distance_percent": round(stop_distance_percent, 2),
            "lot_size": lot_size,
            "max_loss_amount": round(lot_size * stop_distance, 2),
        }
        
        logger.info(f"📊 Position Size Calculation:")
        logger.info(f"   Balance: ${balance:.2f} | Risk: {effective_risk:.1f}%")
        logger.info(f"   Stop Distance: {stop_distance_percent:.2f}%")
        logger.info(f"   → Lot Size: {lot_size}")
        
        return lot_size, calculation
    
    def validate_stop_loss(
        self,
        side: str,
        entry_price: float,
        stop_loss: Optional[float],
        atr: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Validate และ fix Stop Loss
        
        Returns:
            (validated_stop_loss, message)
        """
        side = side.upper()
        
        # If no SL provided, calculate from ATR or default %
        if not stop_loss or stop_loss <= 0:
            if atr and atr > 0:
                # ATR-based SL (2x ATR)
                sl_distance = atr * 2
            else:
                # Default 2%
                sl_distance = entry_price * 0.02
            
            if side == "BUY":
                stop_loss = entry_price - sl_distance
            else:
                stop_loss = entry_price + sl_distance
            
            return stop_loss, f"Auto-calculated SL: {stop_loss:.5f} (2x ATR or 2%)"
        
        # Validate direction
        if side == "BUY" and stop_loss >= entry_price:
            new_sl = entry_price - (entry_price * 0.02)
            return new_sl, f"Invalid SL for BUY. Fixed: {new_sl:.5f}"
        
        if side == "SELL" and stop_loss <= entry_price:
            new_sl = entry_price + (entry_price * 0.02)
            return new_sl, f"Invalid SL for SELL. Fixed: {new_sl:.5f}"
        
        # Validate distance
        distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        
        if distance_pct < self.min_stop_loss_percent:
            if side == "BUY":
                stop_loss = entry_price * (1 - self.min_stop_loss_percent / 100)
            else:
                stop_loss = entry_price * (1 + self.min_stop_loss_percent / 100)
            return stop_loss, f"SL too tight. Adjusted to {self.min_stop_loss_percent}%"
        
        if distance_pct > self.max_stop_loss_percent:
            if side == "BUY":
                stop_loss = entry_price * (1 - self.max_stop_loss_percent / 100)
            else:
                stop_loss = entry_price * (1 + self.max_stop_loss_percent / 100)
            return stop_loss, f"SL too wide. Capped at {self.max_stop_loss_percent}%"
        
        return stop_loss, "SL validated OK"
    
    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """บันทึกผลเทรด"""
        self.daily_stats.total_trades += 1
        self.daily_stats.realized_pnl += pnl
        
        if is_win:
            self.daily_stats.winning_trades += 1
        else:
            self.daily_stats.losing_trades += 1
        
        logger.info(f"📝 Trade recorded: {'WIN' if is_win else 'LOSS'} ${pnl:.2f}")
        logger.info(f"   Daily PnL: ${self.daily_stats.realized_pnl:.2f} ({self.daily_stats.daily_pnl_percent:.1f}%)")
    
    def unlock_trading(self, reason: str = "Manual unlock") -> bool:
        """ปลดล็อคการเทรด (ใช้ด้วยความระวัง!)"""
        if self.trading_locked:
            logger.warning(f"⚠️ Trading unlocked: {reason}")
            self.trading_locked = False
            self.lock_reason = ""
            return True
        return False
    
    def get_status(self) -> dict:
        """Get current risk status"""
        return {
            "trading_locked": self.trading_locked,
            "lock_reason": self.lock_reason,
            "daily_stats": self.daily_stats.to_dict(),
            "settings": {
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_daily_loss": self.max_daily_loss,
                "max_drawdown": self.max_drawdown,
                "max_positions": self.max_positions,
            }
        }


# Singleton
_risk_guardian: Optional[RiskGuardian] = None

def get_risk_guardian() -> RiskGuardian:
    global _risk_guardian
    if _risk_guardian is None:
        _risk_guardian = RiskGuardian()
    return _risk_guardian


def create_risk_guardian(
    max_risk_per_trade: float = 2.0,
    max_daily_loss: float = 5.0,
    max_drawdown: float = 10.0,
) -> RiskGuardian:
    """Create new Risk Guardian with custom settings"""
    global _risk_guardian
    _risk_guardian = RiskGuardian(
        max_risk_per_trade=max_risk_per_trade,
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
    )
    return _risk_guardian
