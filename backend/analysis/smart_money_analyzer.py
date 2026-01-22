"""
Smart Money Analyzer - Trade Like Institutions
==============================================
รวมหลักการ Smart Money Concept:

1. Contrarian Sentiment (Trade สวนทาง Retail)
2. Order Flow Analysis (วิเคราะห์ Volume imbalance)  
3. Liquidity Hunt Detection (หา Stop Hunt zones)
4. Market Structure (BOS, CHoCH, FVG)
5. Supply/Demand Zones (Institutional Order Blocks)

หลักการสำคัญ:
- คนซื้อมากๆ ราคาจะลด (Market Makers ขาย)
- คนขายมากๆ ราคาจะขึ้น (Market Makers ซื้อ)
- Smart Money ล่า Stop Loss ก่อนเคลื่อนตัว
- Price ต้องการ Liquidity เพื่อเคลื่อนที่
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from .sentiment_analyzer import (
    SentimentAnalyzer, 
    AggregatedSentiment, 
    SentimentSignal,
    get_sentiment_analyzer
)

logger = logging.getLogger(__name__)


class SmartMoneySignal(Enum):
    """Smart Money Signal Types"""
    STRONG_BUY = "STRONG_BUY"    # High probability long
    BUY = "BUY"                   # Moderate probability long
    NEUTRAL = "NEUTRAL"          # No clear edge
    SELL = "SELL"                # Moderate probability short
    STRONG_SELL = "STRONG_SELL"  # High probability short


class MarketStructure(Enum):
    """Market Structure Types"""
    BULLISH_BOS = "BULLISH_BOS"     # Break of Structure bullish
    BEARISH_BOS = "BEARISH_BOS"     # Break of Structure bearish
    BULLISH_CHOCH = "BULLISH_CHOCH" # Change of Character bullish
    BEARISH_CHOCH = "BEARISH_CHOCH" # Change of Character bearish
    RANGING = "RANGING"             # No clear structure


@dataclass
class LiquidityZone:
    """Liquidity/Stop Hunt Zone"""
    price: float
    type: str  # "BUYSIDE" (above price) or "SELLSIDE" (below price)
    strength: float  # 0-100
    untested: bool
    description: str


@dataclass 
class OrderBlock:
    """Institutional Order Block (Supply/Demand Zone)"""
    top: float
    bottom: float
    type: str  # "DEMAND" or "SUPPLY"
    strength: float  # 0-100
    touches: int  # Number of times tested
    valid: bool


@dataclass
class FairValueGap:
    """Fair Value Gap (FVG) / Imbalance"""
    top: float
    bottom: float
    type: str  # "BULLISH" or "BEARISH"
    filled_percent: float  # 0-100
    age_candles: int


@dataclass
class SmartMoneyAnalysis:
    """Complete Smart Money Analysis Result"""
    symbol: str
    timestamp: datetime
    
    # Final signal
    signal: SmartMoneySignal
    confidence: float  # 0-100
    
    # Component scores (0-100)
    sentiment_score: float      # Contrarian sentiment
    order_flow_score: float     # Volume/Order flow
    structure_score: float      # Market structure
    liquidity_score: float      # Liquidity hunt probability
    
    # Detailed analysis
    sentiment: Optional[AggregatedSentiment] = None
    market_structure: MarketStructure = MarketStructure.RANGING
    liquidity_zones: List[LiquidityZone] = field(default_factory=list)
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FairValueGap] = field(default_factory=list)
    
    # Trade setup
    entry_zone: Optional[Tuple[float, float]] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    
    # Explanation
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal.value,
            "confidence": round(self.confidence, 1),
            "scores": {
                "sentiment": round(self.sentiment_score, 1),
                "order_flow": round(self.order_flow_score, 1),
                "structure": round(self.structure_score, 1),
                "liquidity": round(self.liquidity_score, 1),
            },
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "market_structure": self.market_structure.value,
            "liquidity_zones": [
                {"price": z.price, "type": z.type, "strength": z.strength}
                for z in self.liquidity_zones[:3]
            ],
            "order_blocks": [
                {"top": ob.top, "bottom": ob.bottom, "type": ob.type, "strength": ob.strength}
                for ob in self.order_blocks[:3]
            ],
            "setup": {
                "entry_zone": self.entry_zone,
                "stop_loss": self.stop_loss,
                "take_profit_1": self.take_profit_1,
                "take_profit_2": self.take_profit_2,
            },
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


class SmartMoneyAnalyzer:
    """
    Smart Money Concept Analyzer
    
    วิเคราะห์ตลาดแบบ Institutional:
    1. Sentiment Analysis (Contrarian)
    2. Market Structure (BOS, CHoCH)
    3. Liquidity Hunt Zones
    4. Order Blocks (Supply/Demand)
    5. Fair Value Gaps
    
    🔥 STRICT MODE: Enhanced contrarian weight
    """
    
    # 🔥 STRICT Weight for each factor - Sentiment is KING
    WEIGHTS = {
        "sentiment": 0.40,     # Contrarian sentiment (เพิ่มจาก 0.30!)
        "order_flow": 0.25,    # Volume/Order flow
        "structure": 0.20,     # Market structure (ลดลง)
        "liquidity": 0.15,     # Liquidity zones (ลดลง)
    }
    
    def __init__(self):
        self.sentiment_analyzer = get_sentiment_analyzer()
    
    async def analyze(
        self,
        symbol: str,
        ohlcv: Dict[str, np.ndarray],
        current_price: float,
        htf_ohlcv: Optional[Dict[str, np.ndarray]] = None,
    ) -> SmartMoneyAnalysis:
        """
        Perform complete Smart Money analysis
        
        Args:
            symbol: Trading symbol (e.g., EURUSDm)
            ohlcv: Dict with open, high, low, close, volume arrays
            current_price: Current market price
            htf_ohlcv: Higher timeframe OHLCV data
            
        Returns:
            SmartMoneyAnalysis with signal and detailed breakdown
        """
        reasons = []
        warnings = []
        
        opens = ohlcv.get("open", np.array([]))
        highs = ohlcv.get("high", np.array([]))
        lows = ohlcv.get("low", np.array([]))
        closes = ohlcv.get("close", np.array([]))
        volumes = ohlcv.get("volume", np.array([1.0] * len(closes)))
        
        logger.info(f"=" * 60)
        logger.info(f"🧠 [SMART MONEY] Analyzing {symbol} @ {current_price:.5f}")
        logger.info(f"=" * 60)
        
        # 1. SENTIMENT ANALYSIS (Contrarian) - 40% (เพิ่มจาก 30%!)
        # นี่คือจุดสำคัญที่สุด!
        try:
            sentiment = await self.sentiment_analyzer.get_sentiment(symbol)
            logger.info(f"📊 [SENTIMENT] Real data from {sentiment.sources_count} sources")
        except Exception as e:
            logger.warning(f"Sentiment fetch failed, using mock: {e}")
            sentiment = self.sentiment_analyzer.get_mock_sentiment(symbol)
            logger.info(f"📊 [SENTIMENT] Using MOCK data")
        
        sentiment_score = self._calculate_sentiment_score(sentiment)
        
        # 🔥 STRICT LOGGING: Show exactly what sentiment says
        logger.info(f"📊 [SENTIMENT] Retail Long: {sentiment.avg_long_percent:.1f}% | Short: {sentiment.avg_short_percent:.1f}%")
        logger.info(f"📊 [SENTIMENT] Signal: {sentiment.signal.value} | Confidence: {sentiment.confidence:.1f}%")
        logger.info(f"📊 [SENTIMENT] Score Weight: {sentiment_score:.1f} × {self.WEIGHTS['sentiment']:.2f} = {sentiment_score * self.WEIGHTS['sentiment']:.1f}")
        
        # Add sentiment reasoning
        if sentiment.signal == SentimentSignal.STRONG_BUY:
            reasons.append(f"🟢 Retail {sentiment.avg_short_percent:.0f}% Short - Extreme bearish = BUY")
            logger.info(f"✅ [CONTRARIAN] Retail very SHORT → We should BUY!")
        elif sentiment.signal == SentimentSignal.STRONG_SELL:
            reasons.append(f"🔴 Retail {sentiment.avg_long_percent:.0f}% Long - Extreme bullish = SELL")
            logger.info(f"✅ [CONTRARIAN] Retail very LONG → We should SELL!")
        elif sentiment.signal == SentimentSignal.BUY:
            reasons.append(f"🟢 Retail leaning short ({sentiment.avg_short_percent:.0f}%) - Contrarian BUY")
            logger.info(f"✅ [CONTRARIAN] Retail leaning short → BUY edge")
        elif sentiment.signal == SentimentSignal.SELL:
            reasons.append(f"🔴 Retail leaning long ({sentiment.avg_long_percent:.0f}%) - Contrarian SELL")
            logger.info(f"✅ [CONTRARIAN] Retail leaning long → SELL edge")
        else:
            warnings.append("Sentiment balanced - No clear contrarian edge")
            logger.info(f"⚪ [CONTRARIAN] Sentiment balanced - No clear edge")
        
        # 2. ORDER FLOW ANALYSIS - 25%
        order_flow_score, volume_signal = self._analyze_order_flow(
            closes, volumes, current_price
        )
        logger.info(f"📈 [ORDER FLOW] Volume Signal: {volume_signal} | Score: {order_flow_score:.1f}")
        
        if volume_signal == "BULLISH":
            reasons.append("📈 Volume shows buying pressure accumulation")
        elif volume_signal == "BEARISH":
            reasons.append("📉 Volume shows selling pressure distribution")
        
        # 3. MARKET STRUCTURE - 20%
        market_structure, structure_score = self._analyze_market_structure(
            highs, lows, closes
        )
        logger.info(f"📊 [STRUCTURE] {market_structure.value} | Score: {structure_score:.1f}")
        
        if market_structure == MarketStructure.BULLISH_BOS:
            reasons.append("📊 Bullish Break of Structure confirmed")
        elif market_structure == MarketStructure.BEARISH_BOS:
            reasons.append("📊 Bearish Break of Structure confirmed")
        elif market_structure == MarketStructure.BULLISH_CHOCH:
            reasons.append("🔄 Bullish Change of Character - Potential reversal up")
        elif market_structure == MarketStructure.BEARISH_CHOCH:
            reasons.append("🔄 Bearish Change of Character - Potential reversal down")
        
        # 4. LIQUIDITY ZONES - 15%
        liquidity_zones, liquidity_score = self._find_liquidity_zones(
            highs, lows, closes, current_price
        )
        logger.info(f"💧 [LIQUIDITY] Found {len(liquidity_zones)} zones | Score: {liquidity_score:.1f}")
        
        # Check if price near liquidity
        for zone in liquidity_zones:
            if zone.untested and zone.strength > 70:
                distance_pct = abs(zone.price - current_price) / current_price * 100
                if distance_pct < 1:  # Within 1%
                    if zone.type == "BUYSIDE":
                        warnings.append(f"⚠️ Price near buyside liquidity at {zone.price:.5f}")
                    else:
                        warnings.append(f"⚠️ Price near sellside liquidity at {zone.price:.5f}")
        
        # 5. ORDER BLOCKS
        order_blocks = self._find_order_blocks(opens, highs, lows, closes, volumes)
        
        for ob in order_blocks:
            if ob.valid and ob.strength > 60:
                if ob.bottom <= current_price <= ob.top:
                    if ob.type == "DEMAND":
                        reasons.append(f"🟦 Price in demand zone ({ob.bottom:.5f}-{ob.top:.5f})")
                    else:
                        reasons.append(f"🟥 Price in supply zone ({ob.bottom:.5f}-{ob.top:.5f})")
        
        # 6. FAIR VALUE GAPS
        fvgs = self._find_fvgs(highs, lows, closes)
        
        unfilled_fvgs = [f for f in fvgs if f.filled_percent < 50]
        if unfilled_fvgs:
            nearest = min(unfilled_fvgs, key=lambda f: abs((f.top + f.bottom) / 2 - current_price))
            mid = (nearest.top + nearest.bottom) / 2
            if abs(mid - current_price) / current_price < 0.02:  # Within 2%
                if nearest.type == "BULLISH":
                    reasons.append(f"📊 Unfilled bullish FVG nearby ({nearest.bottom:.5f}-{nearest.top:.5f})")
                else:
                    reasons.append(f"📊 Unfilled bearish FVG nearby ({nearest.bottom:.5f}-{nearest.top:.5f})")
        
        # CALCULATE FINAL SIGNAL
        weighted_score = (
            sentiment_score * self.WEIGHTS["sentiment"] +
            order_flow_score * self.WEIGHTS["order_flow"] +
            structure_score * self.WEIGHTS["structure"] +
            liquidity_score * self.WEIGHTS["liquidity"]
        )
        
        logger.info(f"-" * 50)
        logger.info(f"📊 [SMART MONEY] Final Score Calculation:")
        logger.info(f"   Sentiment: {sentiment_score:.1f} × {self.WEIGHTS['sentiment']:.2f} = {sentiment_score * self.WEIGHTS['sentiment']:.1f}")
        logger.info(f"   Order Flow: {order_flow_score:.1f} × {self.WEIGHTS['order_flow']:.2f} = {order_flow_score * self.WEIGHTS['order_flow']:.1f}")
        logger.info(f"   Structure: {structure_score:.1f} × {self.WEIGHTS['structure']:.2f} = {structure_score * self.WEIGHTS['structure']:.1f}")
        logger.info(f"   Liquidity: {liquidity_score:.1f} × {self.WEIGHTS['liquidity']:.2f} = {liquidity_score * self.WEIGHTS['liquidity']:.1f}")
        logger.info(f"   🎯 TOTAL WEIGHTED: {weighted_score:.1f}")
        
        # Determine signal direction based on sentiment (primary) + confluence
        signal, confidence = self._determine_signal(
            sentiment, 
            market_structure,
            order_flow_score,
            weighted_score
        )
        
        logger.info(f"-" * 50)
        logger.info(f"🧠 [SMART MONEY DECISION]")
        logger.info(f"   Signal: {signal.value}")
        logger.info(f"   Confidence: {confidence:.1f}%")
        logger.info(f"   Reasons: {len(reasons)} | Warnings: {len(warnings)}")
        logger.info(f"=" * 60)
        
        # Calculate trade setup
        entry_zone, stop_loss, tp1, tp2 = self._calculate_trade_setup(
            signal, current_price, highs, lows, liquidity_zones, order_blocks
        )
        
        return SmartMoneyAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=signal,
            confidence=confidence,
            sentiment_score=sentiment_score,
            order_flow_score=order_flow_score,
            structure_score=structure_score,
            liquidity_score=liquidity_score,
            sentiment=sentiment,
            market_structure=market_structure,
            liquidity_zones=liquidity_zones,
            order_blocks=order_blocks,
            fvgs=fvgs,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            reasons=reasons,
            warnings=warnings,
        )
    
    def _calculate_sentiment_score(self, sentiment: AggregatedSentiment) -> float:
        """
        Convert sentiment to score based on contrarian logic
        
        Extreme retail positioning = High score (good for contrarian)
        """
        if sentiment.signal == SentimentSignal.STRONG_BUY:
            return 90  # Retail very short = Strong buy signal
        elif sentiment.signal == SentimentSignal.STRONG_SELL:
            return 90  # Retail very long = Strong sell signal
        elif sentiment.signal == SentimentSignal.BUY:
            return 70
        elif sentiment.signal == SentimentSignal.SELL:
            return 70
        else:
            return 50  # Neutral
    
    def _analyze_order_flow(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        current_price: float
    ) -> Tuple[float, str]:
        """
        Analyze order flow using volume
        
        Returns:
            (score, signal) - score 0-100, signal BULLISH/BEARISH/NEUTRAL
        """
        if len(closes) < 20 or len(volumes) < 20:
            return 50.0, "NEUTRAL"
        
        # Calculate volume-weighted price change
        recent_closes = closes[-20:]
        recent_volumes = volumes[-20:]
        
        # Up volume vs Down volume
        price_changes = np.diff(recent_closes)
        up_volume = sum(recent_volumes[i+1] for i in range(len(price_changes)) if price_changes[i] > 0)
        down_volume = sum(recent_volumes[i+1] for i in range(len(price_changes)) if price_changes[i] < 0)
        
        total_volume = up_volume + down_volume
        if total_volume == 0:
            return 50.0, "NEUTRAL"
        
        up_ratio = up_volume / total_volume
        
        if up_ratio > 0.65:
            return 75 + (up_ratio - 0.65) * 100, "BULLISH"
        elif up_ratio < 0.35:
            return 75 + (0.35 - up_ratio) * 100, "BEARISH"
        else:
            return 50.0, "NEUTRAL"
    
    def _analyze_market_structure(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Tuple[MarketStructure, float]:
        """
        Analyze market structure (BOS, CHoCH)
        
        BOS = Break of Structure (continuation)
        CHoCH = Change of Character (reversal)
        """
        if len(highs) < 30:
            return MarketStructure.RANGING, 50.0
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(highs, "high")
        swing_lows = self._find_swing_points(lows, "low")
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketStructure.RANGING, 50.0
        
        # Check structure
        last_high = swing_highs[-1]
        prev_high = swing_highs[-2]
        last_low = swing_lows[-1]
        prev_low = swing_lows[-2]
        
        current_close = closes[-1]
        
        # Higher highs and higher lows = Bullish
        # Lower highs and lower lows = Bearish
        
        hh = last_high > prev_high  # Higher high
        hl = last_low > prev_low    # Higher low
        lh = last_high < prev_high  # Lower high
        ll = last_low < prev_low    # Lower low
        
        if hh and hl:
            # Bullish structure
            if current_close > last_high:
                return MarketStructure.BULLISH_BOS, 80.0
            return MarketStructure.BULLISH_BOS, 65.0
        elif lh and ll:
            # Bearish structure
            if current_close < last_low:
                return MarketStructure.BEARISH_BOS, 80.0
            return MarketStructure.BEARISH_BOS, 65.0
        elif hh and ll:
            # Change of character - potential bullish reversal
            return MarketStructure.BULLISH_CHOCH, 70.0
        elif lh and hl:
            # Change of character - potential bearish reversal
            return MarketStructure.BEARISH_CHOCH, 70.0
        
        return MarketStructure.RANGING, 50.0
    
    def _find_swing_points(
        self,
        data: np.ndarray,
        point_type: str,
        lookback: int = 5
    ) -> List[float]:
        """Find swing high/low points"""
        swings = []
        
        for i in range(lookback, len(data) - lookback):
            if point_type == "high":
                if all(data[i] >= data[i-j] for j in range(1, lookback+1)) and \
                   all(data[i] >= data[i+j] for j in range(1, lookback+1)):
                    swings.append(data[i])
            else:
                if all(data[i] <= data[i-j] for j in range(1, lookback+1)) and \
                   all(data[i] <= data[i+j] for j in range(1, lookback+1)):
                    swings.append(data[i])
        
        return swings
    
    def _find_liquidity_zones(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        current_price: float
    ) -> Tuple[List[LiquidityZone], float]:
        """
        Find liquidity zones (Stop Hunt areas)
        
        Liquidity = Clusters of stop losses
        - Above swing highs = Buyside liquidity (buy stops)
        - Below swing lows = Sellside liquidity (sell stops)
        """
        zones = []
        
        if len(highs) < 20:
            return zones, 50.0
        
        # Find swing highs (buyside liquidity above)
        swing_highs = self._find_swing_points(highs, "high")
        for sh in swing_highs[-5:]:  # Last 5 swing highs
            # Check if untested (not broken yet)
            untested = all(h < sh * 1.001 for h in highs[-10:])
            zones.append(LiquidityZone(
                price=sh,
                type="BUYSIDE",
                strength=70 if untested else 40,
                untested=untested,
                description=f"Buy stops above swing high at {sh:.5f}"
            ))
        
        # Find swing lows (sellside liquidity below)
        swing_lows = self._find_swing_points(lows, "low")
        for sl in swing_lows[-5:]:  # Last 5 swing lows
            untested = all(l > sl * 0.999 for l in lows[-10:])
            zones.append(LiquidityZone(
                price=sl,
                type="SELLSIDE",
                strength=70 if untested else 40,
                untested=untested,
                description=f"Sell stops below swing low at {sl:.5f}"
            ))
        
        # Calculate score based on nearby untested liquidity
        score = 50.0
        for zone in zones:
            if zone.untested:
                distance_pct = abs(zone.price - current_price) / current_price * 100
                if distance_pct < 2:  # Within 2%
                    score = max(score, 70)
                elif distance_pct < 5:
                    score = max(score, 60)
        
        return zones, score
    
    def _find_order_blocks(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> List[OrderBlock]:
        """
        Find order blocks (institutional supply/demand zones)
        
        Order Block = Last bullish/bearish candle before a strong move
        """
        order_blocks = []
        
        if len(closes) < 10:
            return order_blocks
        
        # Look for significant moves
        for i in range(5, len(closes) - 3):
            # Calculate move strength
            move = (closes[i+2] - closes[i]) / closes[i] * 100
            
            # Strong bullish move (> 1%) with bullish continuation
            if move > 1:
                # The order block is the last bearish candle before the move
                for j in range(i, max(i-5, 0), -1):
                    if closes[j] < opens[j]:  # Bearish candle
                        ob = OrderBlock(
                            top=highs[j],
                            bottom=lows[j],
                            type="DEMAND",
                            strength=min(move * 20, 100),
                            touches=0,
                            valid=True
                        )
                        # Check if still valid (not broken)
                        ob.valid = all(lows[k] > ob.bottom * 0.99 for k in range(j+1, len(lows)))
                        if ob.valid:
                            order_blocks.append(ob)
                        break
            
            # Strong bearish move (< -1%)
            elif move < -1:
                for j in range(i, max(i-5, 0), -1):
                    if closes[j] > opens[j]:  # Bullish candle
                        ob = OrderBlock(
                            top=highs[j],
                            bottom=lows[j],
                            type="SUPPLY",
                            strength=min(abs(move) * 20, 100),
                            touches=0,
                            valid=True
                        )
                        ob.valid = all(highs[k] < ob.top * 1.01 for k in range(j+1, len(highs)))
                        if ob.valid:
                            order_blocks.append(ob)
                        break
        
        return order_blocks[-5:]  # Return last 5
    
    def _find_fvgs(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> List[FairValueGap]:
        """
        Find Fair Value Gaps (imbalances)
        
        FVG = Gap between candle wicks (unfilled area)
        """
        fvgs = []
        
        if len(highs) < 3:
            return fvgs
        
        for i in range(1, len(highs) - 1):
            # Bullish FVG: Low of candle 3 > High of candle 1
            if lows[i+1] > highs[i-1]:
                gap_size = lows[i+1] - highs[i-1]
                gap_pct = gap_size / closes[i] * 100
                
                if gap_pct > 0.1:  # Minimum 0.1% gap
                    # Check fill level
                    filled = 0
                    for j in range(i+2, len(lows)):
                        if lows[j] <= highs[i-1]:
                            filled = 100
                            break
                        elif lows[j] < lows[i+1]:
                            filled = max(filled, (lows[i+1] - lows[j]) / gap_size * 100)
                    
                    fvgs.append(FairValueGap(
                        top=lows[i+1],
                        bottom=highs[i-1],
                        type="BULLISH",
                        filled_percent=filled,
                        age_candles=len(highs) - i - 1
                    ))
            
            # Bearish FVG: High of candle 3 < Low of candle 1
            if highs[i+1] < lows[i-1]:
                gap_size = lows[i-1] - highs[i+1]
                gap_pct = gap_size / closes[i] * 100
                
                if gap_pct > 0.1:
                    filled = 0
                    for j in range(i+2, len(highs)):
                        if highs[j] >= lows[i-1]:
                            filled = 100
                            break
                        elif highs[j] > highs[i+1]:
                            filled = max(filled, (highs[j] - highs[i+1]) / gap_size * 100)
                    
                    fvgs.append(FairValueGap(
                        top=lows[i-1],
                        bottom=highs[i+1],
                        type="BEARISH",
                        filled_percent=filled,
                        age_candles=len(highs) - i - 1
                    ))
        
        return fvgs[-10:]  # Return last 10
    
    def _determine_signal(
        self,
        sentiment: AggregatedSentiment,
        structure: MarketStructure,
        order_flow_score: float,
        weighted_score: float
    ) -> Tuple[SmartMoneySignal, float]:
        """
        Determine final signal based on all factors
        
        SENTIMENT IS PRIMARY! (Contrarian)
        """
        # Start with sentiment signal
        if sentiment.signal == SentimentSignal.STRONG_BUY:
            base_signal = SmartMoneySignal.STRONG_BUY
            confidence = sentiment.confidence
        elif sentiment.signal == SentimentSignal.STRONG_SELL:
            base_signal = SmartMoneySignal.STRONG_SELL
            confidence = sentiment.confidence
        elif sentiment.signal == SentimentSignal.BUY:
            base_signal = SmartMoneySignal.BUY
            confidence = sentiment.confidence
        elif sentiment.signal == SentimentSignal.SELL:
            base_signal = SmartMoneySignal.SELL
            confidence = sentiment.confidence
        else:
            base_signal = SmartMoneySignal.NEUTRAL
            confidence = 50
        
        # Confluence from structure
        if base_signal in [SmartMoneySignal.BUY, SmartMoneySignal.STRONG_BUY]:
            if structure in [MarketStructure.BULLISH_BOS, MarketStructure.BULLISH_CHOCH]:
                confidence = min(confidence + 10, 95)
            elif structure in [MarketStructure.BEARISH_BOS]:
                confidence = max(confidence - 15, 30)
                base_signal = SmartMoneySignal.BUY  # Downgrade
        elif base_signal in [SmartMoneySignal.SELL, SmartMoneySignal.STRONG_SELL]:
            if structure in [MarketStructure.BEARISH_BOS, MarketStructure.BEARISH_CHOCH]:
                confidence = min(confidence + 10, 95)
            elif structure in [MarketStructure.BULLISH_BOS]:
                confidence = max(confidence - 15, 30)
                base_signal = SmartMoneySignal.SELL  # Downgrade
        
        # Adjust for order flow
        if order_flow_score > 70:
            if base_signal in [SmartMoneySignal.BUY, SmartMoneySignal.STRONG_BUY]:
                confidence = min(confidence + 5, 95)
            else:
                confidence = max(confidence - 5, 30)
        elif order_flow_score < 30:
            if base_signal in [SmartMoneySignal.SELL, SmartMoneySignal.STRONG_SELL]:
                confidence = min(confidence + 5, 95)
            else:
                confidence = max(confidence - 5, 30)
        
        return base_signal, confidence
    
    def _calculate_trade_setup(
        self,
        signal: SmartMoneySignal,
        current_price: float,
        highs: np.ndarray,
        lows: np.ndarray,
        liquidity_zones: List[LiquidityZone],
        order_blocks: List[OrderBlock]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[float], Optional[float]]:
        """Calculate entry, SL, and TP levels"""
        
        if signal == SmartMoneySignal.NEUTRAL:
            return None, None, None, None
        
        is_long = signal in [SmartMoneySignal.BUY, SmartMoneySignal.STRONG_BUY]
        
        if len(highs) < 10:
            # Simple calculation
            atr_estimate = np.mean(highs[-10:] - lows[-10:]) if len(highs) >= 10 else current_price * 0.01
            
            if is_long:
                entry = (current_price * 0.998, current_price)
                sl = current_price - atr_estimate * 2
                tp1 = current_price + atr_estimate * 2
                tp2 = current_price + atr_estimate * 4
            else:
                entry = (current_price, current_price * 1.002)
                sl = current_price + atr_estimate * 2
                tp1 = current_price - atr_estimate * 2
                tp2 = current_price - atr_estimate * 4
            
            return entry, sl, tp1, tp2
        
        atr = np.mean(highs[-14:] - lows[-14:])
        
        if is_long:
            # Entry zone: Near demand/order block or current
            entry = (current_price - atr * 0.5, current_price)
            
            # Stop loss: Below nearest sellside liquidity
            sellside = [z for z in liquidity_zones if z.type == "SELLSIDE" and z.price < current_price]
            if sellside:
                sl = min(z.price for z in sellside) - atr * 0.5
            else:
                sl = current_price - atr * 2
            
            # Take profits: At buyside liquidity
            buyside = [z for z in liquidity_zones if z.type == "BUYSIDE" and z.price > current_price]
            if buyside:
                sorted_buyside = sorted(buyside, key=lambda z: z.price)
                tp1 = sorted_buyside[0].price if len(sorted_buyside) > 0 else current_price + atr * 2
                tp2 = sorted_buyside[1].price if len(sorted_buyside) > 1 else current_price + atr * 4
            else:
                tp1 = current_price + atr * 2
                tp2 = current_price + atr * 4
        else:
            entry = (current_price, current_price + atr * 0.5)
            
            buyside = [z for z in liquidity_zones if z.type == "BUYSIDE" and z.price > current_price]
            if buyside:
                sl = max(z.price for z in buyside) + atr * 0.5
            else:
                sl = current_price + atr * 2
            
            sellside = [z for z in liquidity_zones if z.type == "SELLSIDE" and z.price < current_price]
            if sellside:
                sorted_sellside = sorted(sellside, key=lambda z: z.price, reverse=True)
                tp1 = sorted_sellside[0].price if len(sorted_sellside) > 0 else current_price - atr * 2
                tp2 = sorted_sellside[1].price if len(sorted_sellside) > 1 else current_price - atr * 4
            else:
                tp1 = current_price - atr * 2
                tp2 = current_price - atr * 4
        
        return entry, sl, tp1, tp2


# Singleton
_smart_money_analyzer: Optional[SmartMoneyAnalyzer] = None

def get_smart_money_analyzer() -> SmartMoneyAnalyzer:
    global _smart_money_analyzer
    if _smart_money_analyzer is None:
        _smart_money_analyzer = SmartMoneyAnalyzer()
    return _smart_money_analyzer


# CLI Test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        analyzer = SmartMoneyAnalyzer()
        
        # Mock OHLCV data
        np.random.seed(42)
        n = 100
        closes = 1.0850 + np.cumsum(np.random.randn(n) * 0.001)
        highs = closes + np.random.rand(n) * 0.002
        lows = closes - np.random.rand(n) * 0.002
        opens = closes - np.random.randn(n) * 0.001
        volumes = np.random.rand(n) * 1000 + 500
        
        ohlcv = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        }
        
        print("=" * 60)
        print("  SMART MONEY ANALYSIS")
        print("=" * 60)
        print()
        
        for symbol in ["EURUSDm", "XAUUSDm"]:
            result = await analyzer.analyze(
                symbol=symbol,
                ohlcv=ohlcv,
                current_price=closes[-1]
            )
            
            print(f"📊 {symbol}")
            print(f"   Signal: {result.signal.value}")
            print(f"   Confidence: {result.confidence:.0f}%")
            print(f"   Scores:")
            print(f"     - Sentiment: {result.sentiment_score:.0f}")
            print(f"     - Order Flow: {result.order_flow_score:.0f}")
            print(f"     - Structure: {result.structure_score:.0f}")
            print(f"     - Liquidity: {result.liquidity_score:.0f}")
            print(f"   Market Structure: {result.market_structure.value}")
            print(f"   Reasons:")
            for r in result.reasons:
                print(f"     {r}")
            if result.warnings:
                print(f"   Warnings:")
                for w in result.warnings:
                    print(f"     {w}")
            if result.entry_zone:
                print(f"   Trade Setup:")
                print(f"     Entry: {result.entry_zone[0]:.5f} - {result.entry_zone[1]:.5f}")
                print(f"     SL: {result.stop_loss:.5f}")
                print(f"     TP1: {result.take_profit_1:.5f}")
                print(f"     TP2: {result.take_profit_2:.5f}")
            print()
    
    asyncio.run(test())
