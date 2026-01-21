"""
Pro Trading Features - สิ่งที่ Professional Trader ทำ
======================================================

1. Session Filter - เทรดเฉพาะช่วงเวลาดี
2. News Filter - หยุดเทรดช่วงข่าวสำคัญ
3. Trailing Stop - ล็อค profit อัตโนมัติ
4. Break-Even - ย้าย SL ไปจุดเข้าเมื่อ profit
5. Partial Take Profit - ปิดบางส่วนที่ TP1
6. Losing Streak Stop - หยุดเมื่อแพ้ติดต่อกัน
7. Correlation Filter - ไม่เปิดคู่ที่ correlate กัน
8. Volatility Filter - ไม่เทรดเมื่อ volatility ผิดปกติ
"""
import asyncio
import logging
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# 1. SESSION FILTER - เทรดเฉพาะช่วงเวลาดี
# =============================================================================

class TradingSession(str, Enum):
    """ช่วงเวลาเทรด"""
    SYDNEY = "SYDNEY"       # 22:00 - 07:00 UTC
    TOKYO = "TOKYO"         # 00:00 - 09:00 UTC
    LONDON = "LONDON"       # 08:00 - 17:00 UTC
    NEW_YORK = "NEW_YORK"   # 13:00 - 22:00 UTC
    OVERLAP_LN = "OVERLAP_LONDON_NY"  # 13:00 - 17:00 UTC (BEST!)
    OFF_HOURS = "OFF_HOURS"


@dataclass
class SessionInfo:
    """ข้อมูล Session ปัจจุบัน"""
    current_session: TradingSession
    is_optimal: bool  # ช่วงเวลาที่ดีที่สุด
    sessions_active: List[TradingSession]
    quality_score: int  # 0-100
    recommendation: str
    
    def to_dict(self) -> dict:
        return {
            "current_session": self.current_session.value,
            "is_optimal": self.is_optimal,
            "sessions_active": [s.value for s in self.sessions_active],
            "quality_score": self.quality_score,
            "recommendation": self.recommendation,
        }


class SessionFilter:
    """
    กรองช่วงเวลาเทรด
    
    Best: London-NY Overlap (13:00-17:00 UTC)
    Good: London (08:00-17:00 UTC), NY (13:00-22:00 UTC)
    OK: Tokyo (00:00-09:00 UTC)
    Avoid: Sydney, Off-hours
    """
    
    # Session times (UTC)
    SESSIONS = {
        TradingSession.SYDNEY: (time(22, 0), time(7, 0)),    # crosses midnight
        TradingSession.TOKYO: (time(0, 0), time(9, 0)),
        TradingSession.LONDON: (time(8, 0), time(17, 0)),
        TradingSession.NEW_YORK: (time(13, 0), time(22, 0)),
        TradingSession.OVERLAP_LN: (time(13, 0), time(17, 0)),
    }
    
    # Quality scores
    QUALITY = {
        TradingSession.OVERLAP_LN: 100,  # Best!
        TradingSession.LONDON: 85,
        TradingSession.NEW_YORK: 80,
        TradingSession.TOKYO: 60,
        TradingSession.SYDNEY: 40,
        TradingSession.OFF_HOURS: 20,
    }
    
    def get_session_info(self, current_time: Optional[datetime] = None) -> SessionInfo:
        """Get current session information"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        current_t = current_time.time()
        active_sessions = []
        
        # Check each session
        for session, (start, end) in self.SESSIONS.items():
            if self._is_in_session(current_t, start, end):
                active_sessions.append(session)
        
        # Determine primary session
        if TradingSession.OVERLAP_LN in active_sessions:
            primary = TradingSession.OVERLAP_LN
        elif TradingSession.LONDON in active_sessions:
            primary = TradingSession.LONDON
        elif TradingSession.NEW_YORK in active_sessions:
            primary = TradingSession.NEW_YORK
        elif TradingSession.TOKYO in active_sessions:
            primary = TradingSession.TOKYO
        elif TradingSession.SYDNEY in active_sessions:
            primary = TradingSession.SYDNEY
        else:
            primary = TradingSession.OFF_HOURS
            active_sessions = [TradingSession.OFF_HOURS]
        
        quality = self.QUALITY.get(primary, 20)
        is_optimal = primary == TradingSession.OVERLAP_LN
        
        # Recommendation
        if quality >= 80:
            rec = "✅ Optimal trading time - Full position size"
        elif quality >= 60:
            rec = "⚠️ Good trading time - Normal position size"
        elif quality >= 40:
            rec = "⚠️ Moderate - Reduce position size by 50%"
        else:
            rec = "❌ Poor trading time - Avoid new trades"
        
        return SessionInfo(
            current_session=primary,
            is_optimal=is_optimal,
            sessions_active=active_sessions,
            quality_score=quality,
            recommendation=rec,
        )
    
    def _is_in_session(self, current: time, start: time, end: time) -> bool:
        """Check if current time is within session"""
        if start <= end:
            return start <= current <= end
        else:
            # Session crosses midnight
            return current >= start or current <= end
    
    def should_trade(self, symbol: str, min_quality: int = 40) -> Tuple[bool, str]:
        """
        ควรเทรดหรือไม่
        
        Returns:
            (should_trade, reason)
        """
        info = self.get_session_info()
        
        # Gold (XAU) trades well in all sessions
        if "XAU" in symbol.upper() and info.quality_score >= 40:
            return True, f"Gold OK during {info.current_session.value}"
        
        # Forex needs better sessions
        if info.quality_score >= min_quality:
            return True, f"Session: {info.current_session.value} ({info.quality_score}%)"
        
        return False, f"Poor session: {info.current_session.value} ({info.quality_score}%)"


# =============================================================================
# 2. NEWS FILTER - หยุดเทรดช่วงข่าวสำคัญ
# =============================================================================

@dataclass
class EconomicEvent:
    """ข่าวเศรษฐกิจ"""
    time: datetime
    currency: str
    event: str
    impact: str  # HIGH, MEDIUM, LOW
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


class NewsFilter:
    """
    กรองข่าวเศรษฐกิจสำคัญ
    
    ข่าว HIGH IMPACT ที่ต้องหลีกเลี่ยง:
    - NFP (Non-Farm Payrolls)
    - FOMC (Fed Meeting)
    - CPI (Inflation)
    - ECB/BOE/BOJ Meetings
    - GDP
    """
    
    # High impact events
    HIGH_IMPACT_EVENTS = [
        "Non-Farm Payrolls",
        "NFP",
        "FOMC",
        "Federal Funds Rate",
        "Interest Rate Decision",
        "CPI",
        "Consumer Price Index",
        "GDP",
        "ECB Press Conference",
        "BOE Interest Rate",
        "BOJ Interest Rate",
        "Employment Change",
        "Unemployment Rate",
        "Retail Sales",
    ]
    
    # Buffer before/after news (minutes)
    BEFORE_NEWS_BUFFER = 30  # หยุด 30 นาทีก่อนข่าว
    AFTER_NEWS_BUFFER = 15   # รอ 15 นาทีหลังข่าว
    
    def __init__(self):
        self.events_cache: List[EconomicEvent] = []
        self.last_fetch: Optional[datetime] = None
    
    async def fetch_events(self) -> List[EconomicEvent]:
        """
        Fetch economic calendar (จาก Free API)
        
        Note: In production, use paid API like ForexFactory, Investing.com
        """
        # For now, return hardcoded major events
        # In production, integrate with economic calendar API
        
        now = datetime.utcnow()
        
        # Hardcoded high-impact events (example)
        # These would be fetched from API in production
        events = [
            # Example: Next NFP is first Friday of month at 13:30 UTC
            EconomicEvent(
                time=self._get_next_nfp(),
                currency="USD",
                event="Non-Farm Payrolls",
                impact="HIGH",
            ),
        ]
        
        self.events_cache = events
        self.last_fetch = now
        
        return events
    
    def _get_next_nfp(self) -> datetime:
        """Get next NFP date (first Friday of month, 13:30 UTC)"""
        now = datetime.utcnow()
        
        # Find first Friday of current/next month
        if now.day <= 7:
            year, month = now.year, now.month
        else:
            if now.month == 12:
                year, month = now.year + 1, 1
            else:
                year, month = now.year, now.month + 1
        
        # Find first Friday
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        nfp_date = first_day + timedelta(days=days_until_friday)
        
        return nfp_date.replace(hour=13, minute=30)
    
    def is_news_blackout(self, symbol: str) -> Tuple[bool, Optional[EconomicEvent]]:
        """
        Check if we're in news blackout period
        
        Returns:
            (is_blackout, event) - True if should not trade
        """
        now = datetime.utcnow()
        
        # Get currency from symbol
        currencies = self._get_currencies(symbol)
        
        for event in self.events_cache:
            if event.impact != "HIGH":
                continue
            
            if event.currency not in currencies:
                continue
            
            # Check time window
            before_start = event.time - timedelta(minutes=self.BEFORE_NEWS_BUFFER)
            after_end = event.time + timedelta(minutes=self.AFTER_NEWS_BUFFER)
            
            if before_start <= now <= after_end:
                return True, event
        
        return False, None
    
    def _get_currencies(self, symbol: str) -> List[str]:
        """Extract currencies from symbol"""
        symbol = symbol.upper().replace("M", "")  # Remove 'm' suffix
        
        if "EUR" in symbol:
            return ["EUR", "USD"]
        elif "GBP" in symbol:
            return ["GBP", "USD"]
        elif "XAU" in symbol:
            return ["USD", "XAU"]
        elif "JPY" in symbol:
            return ["JPY", "USD"]
        
        return ["USD"]
    
    def should_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        ควรเทรดหรือไม่
        
        Returns:
            (should_trade, reason)
        """
        is_blackout, event = self.is_news_blackout(symbol)
        
        if is_blackout and event:
            return False, f"⚠️ News Blackout: {event.event} at {event.time.strftime('%H:%M')} UTC"
        
        return True, "No high-impact news nearby"


# =============================================================================
# 3. TRAILING STOP - ล็อค profit อัตโนมัติ
# =============================================================================

class TrailingStopManager:
    """
    จัดการ Trailing Stop
    
    หลักการ:
    - เมื่อ profit ถึง activation level → เริ่ม trailing
    - SL จะตามราคาไปเรื่อยๆ (ไม่ถอยหลัง)
    - ล็อค profit ไว้ไม่ให้กลับมาติดลบ
    """
    
    def __init__(
        self,
        activation_percent: float = 1.0,  # เริ่ม trail เมื่อ profit 1%
        trailing_percent: float = 0.5,     # ระยะห่าง trail 0.5%
    ):
        self.activation_percent = activation_percent
        self.trailing_percent = trailing_percent
        
        # Track highest profit per position
        self.highest_prices: Dict[str, float] = {}
    
    def calculate_trailing_stop(
        self,
        position_id: str,
        side: str,  # "BUY" or "SELL"
        entry_price: float,
        current_price: float,
        current_sl: Optional[float],
    ) -> Tuple[Optional[float], str]:
        """
        Calculate new trailing stop level
        
        Returns:
            (new_sl, message) - None if no change needed
        """
        side = side.upper()
        
        # Calculate current profit %
        if side == "BUY":
            profit_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_percent = ((entry_price - current_price) / entry_price) * 100
        
        # Check activation
        if profit_percent < self.activation_percent:
            return None, f"Not activated yet ({profit_percent:.2f}% < {self.activation_percent}%)"
        
        # Track highest price
        if position_id not in self.highest_prices:
            if side == "BUY":
                self.highest_prices[position_id] = current_price
            else:
                self.highest_prices[position_id] = current_price
        
        # Update highest
        if side == "BUY":
            if current_price > self.highest_prices[position_id]:
                self.highest_prices[position_id] = current_price
            highest = self.highest_prices[position_id]
            
            # Calculate new SL
            new_sl = highest * (1 - self.trailing_percent / 100)
            
            # Only move SL up
            if current_sl and new_sl <= current_sl:
                return None, "SL already better"
            
        else:  # SELL
            if current_price < self.highest_prices[position_id]:
                self.highest_prices[position_id] = current_price
            lowest = self.highest_prices[position_id]
            
            new_sl = lowest * (1 + self.trailing_percent / 100)
            
            if current_sl and new_sl >= current_sl:
                return None, "SL already better"
        
        return new_sl, f"Trailing SL moved to {new_sl:.5f}"
    
    def clear_position(self, position_id: str):
        """Clear position from tracking"""
        if position_id in self.highest_prices:
            del self.highest_prices[position_id]


# =============================================================================
# 4. BREAK-EVEN MANAGER - ย้าย SL ไปจุดเข้า
# =============================================================================

class BreakEvenManager:
    """
    จัดการ Break-Even
    
    หลักการ:
    - เมื่อ profit ถึง activation level
    - ย้าย SL ไปจุดเข้า + buffer
    - Profit = 0 minimum (ไม่ขาดทุน)
    """
    
    def __init__(
        self,
        activation_percent: float = 0.5,  # Activate when 0.5% profit
        buffer_percent: float = 0.05,      # Move SL to entry + 0.05%
    ):
        self.activation_percent = activation_percent
        self.buffer_percent = buffer_percent
        self.positions_at_be: set = set()  # Track positions already at BE
    
    def should_move_to_breakeven(
        self,
        position_id: str,
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: Optional[float],
    ) -> Tuple[Optional[float], str]:
        """
        Check if should move to break-even
        
        Returns:
            (new_sl, message)
        """
        if position_id in self.positions_at_be:
            return None, "Already at break-even"
        
        side = side.upper()
        
        # Calculate profit
        if side == "BUY":
            profit_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_percent = ((entry_price - current_price) / entry_price) * 100
        
        if profit_percent < self.activation_percent:
            return None, f"Not enough profit ({profit_percent:.2f}% < {self.activation_percent}%)"
        
        # Calculate BE level
        if side == "BUY":
            be_level = entry_price * (1 + self.buffer_percent / 100)
            
            # Only if better than current SL
            if current_sl and be_level <= current_sl:
                return None, "Current SL already better than BE"
            
        else:  # SELL
            be_level = entry_price * (1 - self.buffer_percent / 100)
            
            if current_sl and be_level >= current_sl:
                return None, "Current SL already better than BE"
        
        self.positions_at_be.add(position_id)
        return be_level, f"✅ Moved to Break-Even: {be_level:.5f}"
    
    def clear_position(self, position_id: str):
        """Clear position from tracking"""
        self.positions_at_be.discard(position_id)


# =============================================================================
# 5. LOSING STREAK STOP - หยุดเมื่อแพ้ติดต่อกัน
# =============================================================================

class LosingStreakMonitor:
    """
    Monitor และหยุดเมื่อแพ้ติดต่อกัน
    
    หลักการ:
    - แพ้ 3 ครั้งติด → ลด position size 50%
    - แพ้ 5 ครั้งติด → หยุดเทรดทั้งวัน
    """
    
    def __init__(
        self,
        reduce_at_streak: int = 3,    # ลด size เมื่อแพ้ 3 ติด
        stop_at_streak: int = 5,      # หยุดเมื่อแพ้ 5 ติด
    ):
        self.reduce_at_streak = reduce_at_streak
        self.stop_at_streak = stop_at_streak
        
        self.current_streak = 0  # Negative = losing, Positive = winning
        self.trades_today: List[bool] = []  # True = win, False = loss
    
    def record_trade(self, is_win: bool):
        """Record trade result"""
        self.trades_today.append(is_win)
        
        if is_win:
            if self.current_streak < 0:
                self.current_streak = 1  # Reset to 1 win
            else:
                self.current_streak += 1
        else:
            if self.current_streak > 0:
                self.current_streak = -1  # Reset to 1 loss
            else:
                self.current_streak -= 1
        
        logger.info(f"Trade recorded: {'WIN' if is_win else 'LOSS'}, Streak: {self.current_streak}")
    
    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on streak
        
        Returns:
            0.0 = don't trade
            0.5 = half position
            1.0 = normal
        """
        if self.current_streak <= -self.stop_at_streak:
            return 0.0  # Stop trading
        elif self.current_streak <= -self.reduce_at_streak:
            return 0.5  # Reduce position
        else:
            return 1.0  # Normal
    
    def should_trade(self) -> Tuple[bool, str]:
        """
        Check if should trade based on streak
        
        Returns:
            (should_trade, reason)
        """
        if self.current_streak <= -self.stop_at_streak:
            return False, f"🛑 Losing streak: {-self.current_streak} losses - STOP TRADING"
        
        if self.current_streak <= -self.reduce_at_streak:
            return True, f"⚠️ Losing streak: {-self.current_streak} - Reduce position 50%"
        
        if self.current_streak >= 3:
            return True, f"🔥 Winning streak: {self.current_streak} - Stay focused!"
        
        return True, "OK"
    
    def reset_daily(self):
        """Reset for new trading day"""
        self.trades_today = []
        # Keep streak but cap at -2 (give fresh start but remember)
        if self.current_streak < -2:
            self.current_streak = -2
        logger.info("Daily reset - Streak capped at -2 for fresh start")


# =============================================================================
# 6. CORRELATION FILTER - ไม่เปิดคู่ที่ correlate กัน
# =============================================================================

class CorrelationFilter:
    """
    ป้องกันการเปิด position ในคู่เงินที่ correlate กัน
    
    หลักการ:
    - EURUSD และ GBPUSD correlate สูง (+0.85)
    - ถ้า Long EURUSD แล้ว → ไม่ควร Long GBPUSD
    - ลด exposure ในทิศทางเดียวกัน
    """
    
    # Correlation matrix (simplified)
    CORRELATIONS = {
        ("EURUSD", "GBPUSD"): 0.85,   # สูง - ระวัง
        ("EURUSD", "USDCHF"): -0.90,  # สูง inverse
        ("GBPUSD", "EURGBP"): -0.50,  # ปานกลาง inverse
        ("XAUUSD", "EURUSD"): 0.60,   # ปานกลาง
        ("XAUUSD", "USDJPY"): -0.40,  # ต่ำ inverse
    }
    
    CORRELATION_THRESHOLD = 0.70  # ถือว่า correlate สูงเมื่อ > 0.70
    
    def __init__(self, max_correlated_positions: int = 1):
        self.max_correlated_positions = max_correlated_positions
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Remove suffix like 'm', '.pro'"""
        return symbol.upper().replace("M", "").replace(".PRO", "")
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        s1 = self._normalize_symbol(symbol1)
        s2 = self._normalize_symbol(symbol2)
        
        if s1 == s2:
            return 1.0
        
        # Check both orders
        corr = self.CORRELATIONS.get((s1, s2))
        if corr is not None:
            return corr
        
        corr = self.CORRELATIONS.get((s2, s1))
        if corr is not None:
            return corr
        
        return 0.0  # Unknown = assume no correlation
    
    def check_correlation_risk(
        self,
        new_symbol: str,
        new_side: str,
        existing_positions: List[dict],
    ) -> Tuple[bool, str]:
        """
        Check if new position would create correlation risk
        
        Returns:
            (is_safe, message)
        """
        new_side = new_side.upper()
        correlated_count = 0
        
        for pos in existing_positions:
            pos_symbol = pos.get("symbol", "")
            pos_side = pos.get("side", "").upper()
            
            corr = self.get_correlation(new_symbol, pos_symbol)
            
            # High positive correlation + same direction = risk
            if corr >= self.CORRELATION_THRESHOLD and pos_side == new_side:
                correlated_count += 1
                if correlated_count >= self.max_correlated_positions:
                    return False, f"❌ Correlated: {new_symbol} & {pos_symbol} ({corr:.0%}), both {new_side}"
            
            # High negative correlation + opposite direction = also risk
            if corr <= -self.CORRELATION_THRESHOLD and pos_side != new_side:
                correlated_count += 1
                if correlated_count >= self.max_correlated_positions:
                    return False, f"❌ Inverse correlated: {new_symbol} & {pos_symbol} ({corr:.0%})"
        
        return True, "OK - No correlation risk"


# =============================================================================
# MASTER PRO FEATURES CLASS
# =============================================================================

@dataclass
class ProTradingDecision:
    """ผลการตัดสินใจจาก Pro Features"""
    can_trade: bool
    position_multiplier: float  # 0.0 - 1.0
    reasons: List[str]
    warnings: List[str]
    session_info: Optional[SessionInfo] = None
    
    def to_dict(self) -> dict:
        return {
            "can_trade": self.can_trade,
            "position_multiplier": self.position_multiplier,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "session_info": self.session_info.to_dict() if self.session_info else None,
        }


class ProTradingFeatures:
    """
    รวม Pro Trading Features ทั้งหมด
    
    ใช้ตรวจสอบก่อนเทรดทุกครั้ง
    """
    
    def __init__(
        self,
        enable_session_filter: bool = True,
        enable_news_filter: bool = True,
        enable_correlation_filter: bool = True,
        enable_losing_streak_stop: bool = True,
        min_session_quality: int = 40,
    ):
        self.enable_session_filter = enable_session_filter
        self.enable_news_filter = enable_news_filter
        self.enable_correlation_filter = enable_correlation_filter
        self.enable_losing_streak_stop = enable_losing_streak_stop
        self.min_session_quality = min_session_quality
        
        # Initialize components
        self.session_filter = SessionFilter()
        self.news_filter = NewsFilter()
        self.correlation_filter = CorrelationFilter()
        self.losing_streak = LosingStreakMonitor()
        self.trailing_stop = TrailingStopManager()
        self.break_even = BreakEvenManager()
        
        logger.info("🏆 Pro Trading Features initialized")
    
    def check_entry(
        self,
        symbol: str,
        side: str,
        existing_positions: List[dict],
    ) -> ProTradingDecision:
        """
        ตรวจสอบว่าควรเข้าเทรดหรือไม่
        
        Returns:
            ProTradingDecision with all checks
        """
        reasons = []
        warnings = []
        can_trade = True
        position_multiplier = 1.0
        
        # 1. Session Filter
        session_info = None
        if self.enable_session_filter:
            session_info = self.session_filter.get_session_info()
            
            if session_info.quality_score < self.min_session_quality:
                can_trade = False
                reasons.append(f"❌ Poor session: {session_info.current_session.value}")
            elif session_info.quality_score < 60:
                position_multiplier *= 0.5
                warnings.append(f"⚠️ Moderate session - 50% position")
            elif session_info.is_optimal:
                warnings.append(f"✅ Optimal: {session_info.current_session.value}")
        
        # 2. News Filter
        if self.enable_news_filter and can_trade:
            should_trade, news_msg = self.news_filter.should_trade(symbol)
            if not should_trade:
                can_trade = False
                reasons.append(news_msg)
            else:
                warnings.append(news_msg)
        
        # 3. Correlation Filter
        if self.enable_correlation_filter and can_trade:
            is_safe, corr_msg = self.correlation_filter.check_correlation_risk(
                symbol, side, existing_positions
            )
            if not is_safe:
                can_trade = False
                reasons.append(corr_msg)
        
        # 4. Losing Streak
        if self.enable_losing_streak_stop and can_trade:
            should_trade, streak_msg = self.losing_streak.should_trade()
            if not should_trade:
                can_trade = False
                reasons.append(streak_msg)
            else:
                streak_mult = self.losing_streak.get_position_multiplier()
                position_multiplier *= streak_mult
                if streak_mult < 1.0:
                    warnings.append(streak_msg)
        
        return ProTradingDecision(
            can_trade=can_trade,
            position_multiplier=position_multiplier,
            reasons=reasons,
            warnings=warnings,
            session_info=session_info,
        )
    
    def record_trade_result(self, is_win: bool):
        """Record trade result for streak tracking"""
        self.losing_streak.record_trade(is_win)
    
    def manage_position(
        self,
        position_id: str,
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: Optional[float],
    ) -> Tuple[Optional[float], List[str]]:
        """
        Manage existing position (Trailing Stop, Break-Even)
        
        Returns:
            (new_sl, messages)
        """
        messages = []
        new_sl = None
        
        # 1. Break-Even first
        be_sl, be_msg = self.break_even.should_move_to_breakeven(
            position_id, side, entry_price, current_price, current_sl
        )
        if be_sl:
            new_sl = be_sl
            messages.append(be_msg)
            current_sl = new_sl  # Update for trailing calculation
        
        # 2. Trailing Stop
        trail_sl, trail_msg = self.trailing_stop.calculate_trailing_stop(
            position_id, side, entry_price, current_price, current_sl
        )
        if trail_sl:
            new_sl = trail_sl
            messages.append(trail_msg)
        
        return new_sl, messages
    
    def clear_position(self, position_id: str):
        """Clear position from tracking"""
        self.trailing_stop.clear_position(position_id)
        self.break_even.clear_position(position_id)
    
    def reset_daily(self):
        """Reset for new trading day"""
        self.losing_streak.reset_daily()


# Singleton
_pro_features: Optional[ProTradingFeatures] = None

def get_pro_features() -> ProTradingFeatures:
    global _pro_features
    if _pro_features is None:
        _pro_features = ProTradingFeatures()
    return _pro_features


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  PRO TRADING FEATURES TEST")
    print("=" * 60)
    
    pro = ProTradingFeatures()
    
    # Test session
    print("\n1. Session Filter:")
    session = pro.session_filter.get_session_info()
    print(f"   Current: {session.current_session.value}")
    print(f"   Quality: {session.quality_score}%")
    print(f"   Optimal: {session.is_optimal}")
    
    # Test entry check
    print("\n2. Entry Check (EURUSDm BUY):")
    decision = pro.check_entry("EURUSDm", "BUY", [])
    print(f"   Can Trade: {decision.can_trade}")
    print(f"   Position Mult: {decision.position_multiplier}x")
    for w in decision.warnings:
        print(f"   {w}")
    
    # Test correlation
    print("\n3. Correlation Check:")
    existing = [{"symbol": "EURUSDm", "side": "BUY"}]
    decision2 = pro.check_entry("GBPUSDm", "BUY", existing)
    print(f"   GBPUSD BUY with existing EURUSD BUY:")
    print(f"   Can Trade: {decision2.can_trade}")
    for r in decision2.reasons:
        print(f"   {r}")
    
    # Test losing streak
    print("\n4. Losing Streak:")
    pro.losing_streak.record_trade(False)  # Loss
    pro.losing_streak.record_trade(False)  # Loss
    pro.losing_streak.record_trade(False)  # Loss
    mult = pro.losing_streak.get_position_multiplier()
    print(f"   After 3 losses: Position Mult = {mult}x")
    
    print("\n✅ Pro Features Ready!")
