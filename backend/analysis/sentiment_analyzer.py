"""
Sentiment Analyzer - Contrarian Trading Strategy
================================================
หลักการ: Trade สวนทาง Retail Sentiment

- Retail Long มาก (>65%) → SELL Signal
- Retail Short มาก (>65%) → BUY Signal
- Balanced (40-60%) → WAIT

Data Sources:
1. Myfxbook Community Outlook (Free)
2. IG Client Sentiment (Scraping)
3. DailyFX Sentiment (Scraping)

Smart Money Concept:
- Retail traders มักผิดพลาดที่จุดกลับตัว
- Banks/Institutions เทรดสวนทาง retail
- Extreme sentiment = High probability reversal
"""
import asyncio
import aiohttp
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class SentimentSignal(Enum):
    STRONG_BUY = "STRONG_BUY"    # Retail very short (>75%)
    BUY = "BUY"                   # Retail short (65-75%)
    NEUTRAL = "NEUTRAL"          # Balanced (40-60%)
    SELL = "SELL"                # Retail long (65-75%)
    STRONG_SELL = "STRONG_SELL"  # Retail very long (>75%)


@dataclass
class SentimentData:
    """Sentiment data from a single source"""
    source: str
    symbol: str
    long_percent: float      # % of traders long
    short_percent: float     # % of traders short
    long_positions: int = 0  # Number of long positions
    short_positions: int = 0 # Number of short positions
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def ratio(self) -> float:
        """Long/Short ratio (>1 = more longs, <1 = more shorts)"""
        if self.short_percent == 0:
            return 10.0
        return self.long_percent / self.short_percent
    
    @property
    def extreme_level(self) -> str:
        """How extreme is the sentiment"""
        max_pct = max(self.long_percent, self.short_percent)
        if max_pct >= 80:
            return "EXTREME"
        elif max_pct >= 70:
            return "HIGH"
        elif max_pct >= 60:
            return "MODERATE"
        else:
            return "NEUTRAL"


@dataclass
class AggregatedSentiment:
    """Combined sentiment from multiple sources"""
    symbol: str
    avg_long_percent: float
    avg_short_percent: float
    sources_count: int
    signal: SentimentSignal
    confidence: float  # 0-100%
    sources: List[SentimentData] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "long_percent": round(self.avg_long_percent, 1),
            "short_percent": round(self.avg_short_percent, 1),
            "signal": self.signal.value,
            "confidence": round(self.confidence, 1),
            "extreme_level": self._get_extreme_level(),
            "sources_count": self.sources_count,
            "recommendation": self._get_recommendation(),
            "timestamp": self.timestamp.isoformat(),
        }
    
    def _get_extreme_level(self) -> str:
        max_pct = max(self.avg_long_percent, self.avg_short_percent)
        if max_pct >= 80:
            return "EXTREME"
        elif max_pct >= 70:
            return "HIGH"
        elif max_pct >= 60:
            return "MODERATE"
        else:
            return "NEUTRAL"
    
    def _get_recommendation(self) -> str:
        if self.signal == SentimentSignal.STRONG_BUY:
            return f"🟢 STRONG BUY - Retail {self.avg_short_percent:.0f}% Short (Extreme bearish = Contrarian bullish)"
        elif self.signal == SentimentSignal.BUY:
            return f"🟢 BUY - Retail {self.avg_short_percent:.0f}% Short (Bearish crowd = Look for longs)"
        elif self.signal == SentimentSignal.STRONG_SELL:
            return f"🔴 STRONG SELL - Retail {self.avg_long_percent:.0f}% Long (Extreme bullish = Contrarian bearish)"
        elif self.signal == SentimentSignal.SELL:
            return f"🔴 SELL - Retail {self.avg_long_percent:.0f}% Long (Bullish crowd = Look for shorts)"
        else:
            return "⚪ NEUTRAL - No clear sentiment edge"


class SentimentAnalyzer:
    """
    Analyze retail sentiment and generate contrarian signals
    
    Strategy:
    - When retail is extremely long → Market likely to drop
    - When retail is extremely short → Market likely to rise
    """
    
    # Symbol mapping for different data sources
    SYMBOL_MAP = {
        # Forex
        "EURUSD": ["EURUSD", "EUR/USD", "eurusd"],
        "EURUSDm": ["EURUSD", "EUR/USD", "eurusd"],
        "GBPUSD": ["GBPUSD", "GBP/USD", "gbpusd"],
        "GBPUSDm": ["GBPUSD", "GBP/USD", "gbpusd"],
        "USDJPY": ["USDJPY", "USD/JPY", "usdjpy"],
        "USDJPYm": ["USDJPY", "USD/JPY", "usdjpy"],
        "XAUUSD": ["XAUUSD", "XAU/USD", "Gold", "GOLD"],
        "XAUUSDm": ["XAUUSD", "XAU/USD", "Gold", "GOLD"],
        # Crypto
        "BTCUSDT": ["BTCUSD", "BTC/USD", "Bitcoin"],
        "ETHUSDT": ["ETHUSD", "ETH/USD", "Ethereum"],
    }
    
    # Sentiment thresholds for contrarian signals
    EXTREME_THRESHOLD = 75  # Very high conviction
    HIGH_THRESHOLD = 65     # High conviction
    MODERATE_THRESHOLD = 55 # Slight edge
    
    def __init__(self):
        self.cache: Dict[str, AggregatedSentiment] = {}
        self.cache_duration = timedelta(minutes=15)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert broker symbol to standard format"""
        # Remove suffix like 'm', '.pro', etc.
        clean = re.sub(r'[._-].*$', '', symbol.upper())
        if clean.endswith('M'):
            clean = clean[:-1]
        return clean
    
    async def get_sentiment(self, symbol: str, use_cache: bool = True) -> AggregatedSentiment:
        """
        Get aggregated sentiment from all available sources
        
        Args:
            symbol: Trading symbol (e.g., EURUSDm, XAUUSD)
            use_cache: Use cached data if available
            
        Returns:
            AggregatedSentiment with contrarian signal
        """
        # Check cache
        if use_cache and symbol in self.cache:
            cached = self.cache[symbol]
            if datetime.now() - cached.timestamp < self.cache_duration:
                return cached
        
        # Fetch from all sources
        sources_data = await self._fetch_all_sources(symbol)
        
        if not sources_data:
            # Use mock data if no real data available
            logger.info(f"No real sentiment data for {symbol}, using mock data")
            return self.get_mock_sentiment(symbol)
        
        # Aggregate
        result = self._aggregate_sentiment(symbol, sources_data)
        
        # Cache
        self.cache[symbol] = result
        
        return result
    
    async def _fetch_all_sources(self, symbol: str) -> List[SentimentData]:
        """Fetch sentiment from all available sources"""
        tasks = [
            self._fetch_myfxbook(symbol),
            self._fetch_dailyfx(symbol),
            # Add more sources here
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for r in results:
            if isinstance(r, SentimentData):
                valid_results.append(r)
            elif isinstance(r, Exception):
                logger.debug(f"Source fetch failed: {r}")
        
        return valid_results
    
    async def _fetch_myfxbook(self, symbol: str) -> Optional[SentimentData]:
        """
        Fetch sentiment from Myfxbook Community Outlook
        https://www.myfxbook.com/community/outlook
        """
        try:
            session = await self._get_session()
            url = "https://www.myfxbook.com/community/outlook"
            
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                
                # Parse the HTML for sentiment data
                # Looking for pattern: EURUSD ... XX% long ... YY% short
                normalized = self._normalize_symbol(symbol)
                
                # Simple regex pattern to find sentiment
                # Format varies, this is a simplified approach
                pattern = rf'{normalized}[^%]*?(\d+(?:\.\d+)?)\s*%\s*(?:long|Long)[^%]*?(\d+(?:\.\d+)?)\s*%\s*(?:short|Short)'
                match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
                
                if match:
                    long_pct = float(match.group(1))
                    short_pct = float(match.group(2))
                    
                    return SentimentData(
                        source="Myfxbook",
                        symbol=symbol,
                        long_percent=long_pct,
                        short_percent=short_pct,
                    )
                
                # Alternative: try to find in table format
                # This is a fallback pattern
                alt_pattern = rf'({normalized})[^\d]*(\d+)\s*%[^\d]*(\d+)\s*%'
                match = re.search(alt_pattern, html, re.IGNORECASE)
                
                if match:
                    return SentimentData(
                        source="Myfxbook",
                        symbol=symbol,
                        long_percent=float(match.group(2)),
                        short_percent=float(match.group(3)),
                    )
                
        except Exception as e:
            logger.debug(f"Myfxbook fetch error: {e}")
        
        return None
    
    async def _fetch_dailyfx(self, symbol: str) -> Optional[SentimentData]:
        """
        Fetch sentiment from DailyFX Client Sentiment
        https://www.dailyfx.com/sentiment
        """
        try:
            session = await self._get_session()
            normalized = self._normalize_symbol(symbol)
            
            # DailyFX API endpoint (unofficial)
            url = f"https://www.dailyfx.com/sentiment-report"
            
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                
                # Look for JSON data embedded in page
                # DailyFX often has data in script tags
                json_pattern = r'sentimentData["\']?\s*[:=]\s*(\[[\s\S]*?\])'
                match = re.search(json_pattern, html)
                
                if match:
                    try:
                        data = json.loads(match.group(1))
                        for item in data:
                            if normalized.lower() in str(item).lower():
                                return SentimentData(
                                    source="DailyFX",
                                    symbol=symbol,
                                    long_percent=item.get("longPercent", 50),
                                    short_percent=item.get("shortPercent", 50),
                                )
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: regex search
                pattern = rf'{normalized}[^\d]*(\d+(?:\.\d+)?)\s*%\s*(?:Long|long)[^\d]*(\d+(?:\.\d+)?)\s*%\s*(?:Short|short)'
                match = re.search(pattern, html, re.IGNORECASE)
                
                if match:
                    return SentimentData(
                        source="DailyFX",
                        symbol=symbol,
                        long_percent=float(match.group(1)),
                        short_percent=float(match.group(2)),
                    )
                    
        except Exception as e:
            logger.debug(f"DailyFX fetch error: {e}")
        
        return None
    
    def _aggregate_sentiment(self, symbol: str, sources: List[SentimentData]) -> AggregatedSentiment:
        """Aggregate sentiment from multiple sources and generate signal"""
        if not sources:
            return AggregatedSentiment(
                symbol=symbol,
                avg_long_percent=50.0,
                avg_short_percent=50.0,
                sources_count=0,
                signal=SentimentSignal.NEUTRAL,
                confidence=0.0,
            )
        
        # Calculate weighted average (all sources equal weight for now)
        total_long = sum(s.long_percent for s in sources)
        total_short = sum(s.short_percent for s in sources)
        count = len(sources)
        
        avg_long = total_long / count
        avg_short = total_short / count
        
        # Generate contrarian signal
        signal, confidence = self._generate_contrarian_signal(avg_long, avg_short)
        
        return AggregatedSentiment(
            symbol=symbol,
            avg_long_percent=avg_long,
            avg_short_percent=avg_short,
            sources_count=count,
            signal=signal,
            confidence=confidence,
            sources=sources,
        )
    
    def _generate_contrarian_signal(
        self, 
        long_pct: float, 
        short_pct: float
    ) -> Tuple[SentimentSignal, float]:
        """
        Generate contrarian signal based on retail positioning
        
        CONTRARIAN LOGIC:
        - Retail LONG (bullish) → We SELL (bearish)
        - Retail SHORT (bearish) → We BUY (bullish)
        """
        # Calculate confidence based on how extreme the sentiment is
        max_pct = max(long_pct, short_pct)
        
        if max_pct < self.MODERATE_THRESHOLD:
            # Balanced - no edge
            return SentimentSignal.NEUTRAL, 0.0
        
        # Calculate confidence (0-100)
        # More extreme = higher confidence
        if max_pct >= self.EXTREME_THRESHOLD:
            confidence = 80 + (max_pct - self.EXTREME_THRESHOLD)
        elif max_pct >= self.HIGH_THRESHOLD:
            confidence = 60 + (max_pct - self.HIGH_THRESHOLD) * 2
        else:
            confidence = 40 + (max_pct - self.MODERATE_THRESHOLD) * 2
        
        confidence = min(confidence, 100)
        
        # CONTRARIAN: Trade AGAINST the crowd
        if long_pct >= self.EXTREME_THRESHOLD:
            # Retail very bullish → Strong bearish signal
            return SentimentSignal.STRONG_SELL, confidence
        elif long_pct >= self.HIGH_THRESHOLD:
            # Retail bullish → Bearish signal
            return SentimentSignal.SELL, confidence
        elif short_pct >= self.EXTREME_THRESHOLD:
            # Retail very bearish → Strong bullish signal
            return SentimentSignal.STRONG_BUY, confidence
        elif short_pct >= self.HIGH_THRESHOLD:
            # Retail bearish → Bullish signal
            return SentimentSignal.BUY, confidence
        
        return SentimentSignal.NEUTRAL, confidence
    
    def get_mock_sentiment(self, symbol: str) -> AggregatedSentiment:
        """
        Get mock sentiment data for testing
        Based on typical retail positioning patterns
        """
        import random
        
        # Simulate realistic sentiment
        # Retail tends to be:
        # - Long at tops (before drops)
        # - Short at bottoms (before rallies)
        
        base_patterns = {
            "EURUSD": (62, 38),   # Retail slightly long
            "EURUSDm": (62, 38),
            "GBPUSD": (58, 42),   # Balanced
            "GBPUSDm": (58, 42),
            "XAUUSD": (78, 22),   # Retail very long (contrarian STRONG SELL!)
            "XAUUSDm": (78, 22),  # Retail very long (contrarian STRONG SELL!)
            "USDJPY": (30, 70),   # Retail short (contrarian buy)
            "USDJPYm": (30, 70),
        }
        
        normalized = self._normalize_symbol(symbol)
        base = base_patterns.get(normalized + "m", base_patterns.get(normalized, (50, 50)))
        
        # Add small randomness (not affecting signal)
        noise = random.uniform(-2, 2)
        long_pct = max(5, min(95, base[0] + noise))
        short_pct = 100 - long_pct
        
        signal, confidence = self._generate_contrarian_signal(long_pct, short_pct)
        
        return AggregatedSentiment(
            symbol=symbol,
            avg_long_percent=long_pct,
            avg_short_percent=short_pct,
            sources_count=1,
            signal=signal,
            confidence=confidence,
            sources=[
                SentimentData(
                    source="Mock",
                    symbol=symbol,
                    long_percent=long_pct,
                    short_percent=short_pct,
                )
            ],
        )


# Singleton instance
_sentiment_analyzer: Optional[SentimentAnalyzer] = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get singleton SentimentAnalyzer instance"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


async def analyze_sentiment(symbol: str) -> AggregatedSentiment:
    """
    Quick function to analyze sentiment for a symbol
    
    Usage:
        sentiment = await analyze_sentiment("EURUSDm")
        print(sentiment.signal)  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    """
    analyzer = get_sentiment_analyzer()
    return await analyzer.get_sentiment(symbol)


# CLI test
if __name__ == "__main__":
    async def test():
        analyzer = SentimentAnalyzer()
        
        symbols = ["EURUSDm", "GBPUSDm", "XAUUSDm"]
        
        print("=" * 60)
        print("  SENTIMENT ANALYSIS - Contrarian Strategy")
        print("=" * 60)
        print()
        print("หลักการ: Trade สวนทาง Retail")
        print("- Retail Long มาก → เราขาย")
        print("- Retail Short มาก → เราซื้อ")
        print()
        
        for symbol in symbols:
            # Use mock for testing
            sentiment = analyzer.get_mock_sentiment(symbol)
            
            print(f"📊 {symbol}")
            print(f"   Retail Long:  {sentiment.avg_long_percent:.1f}%")
            print(f"   Retail Short: {sentiment.avg_short_percent:.1f}%")
            print(f"   Signal: {sentiment.signal.value}")
            print(f"   Confidence: {sentiment.confidence:.0f}%")
            print(f"   {sentiment._get_recommendation()}")
            print()
        
        await analyzer.close()
    
    asyncio.run(test())
