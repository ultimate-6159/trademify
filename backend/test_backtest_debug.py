"""Debug script for backtest"""
import asyncio
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_signals():
    """Test how many technical signals are generated"""
    from backtesting.data_loader import HistoricalDataLoader
    
    loader = HistoricalDataLoader()
    data = await loader.load_data('EURUSDm', 'H1', 1)
    print(f"Data loaded: {len(data)} bars")
    
    # Check each condition individually
    signals_found = 0
    condition_counts = {
        'has_uptrend': 0,
        'has_downtrend': 0,
        'in_pullback_zone': 0,
        'rsi_ok_buy': 0,
        'rsi_ok_sell': 0,
        'macd_bullish': 0,
        'macd_bearish': 0,
        'volatility_ok': 0,
        'active_session': 0,
        'bullish_candle': 0,
        'bearish_candle': 0,
        'bounce_up': 0,
        'bounce_down': 0,
    }
    
    for i in range(100, min(len(data), 500)):  # Check bars 100-500
        df = data.iloc[i-100:i]
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values
        o = df['open'].values
        
        current_price = c[-1]
        current_time = data.index[i]
        
        # Moving Averages
        sma_10 = np.mean(c[-10:])
        sma_20 = np.mean(c[-20:])
        sma_50 = np.mean(c[-50:])
        
        # Trend
        strong_uptrend = (sma_20 > sma_50) and (current_price > sma_20)
        moderate_uptrend = (sma_10 > sma_20) and (current_price > sma_10)
        has_uptrend = strong_uptrend or moderate_uptrend
        
        strong_downtrend = (sma_20 < sma_50) and (current_price < sma_20)
        moderate_downtrend = (sma_10 < sma_20) and (current_price < sma_10)
        has_downtrend = strong_downtrend or moderate_downtrend
        
        if has_uptrend:
            condition_counts['has_uptrend'] += 1
        if has_downtrend:
            condition_counts['has_downtrend'] += 1
        
        # Pullback zone
        distance = abs(current_price - sma_20) / sma_20 * 100
        in_pullback_zone = distance <= 0.5
        if in_pullback_zone:
            condition_counts['in_pullback_zone'] += 1
        
        # RSI
        delta = np.diff(c)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 0.0001)))
        
        if 30 <= rsi <= 60:
            condition_counts['rsi_ok_buy'] += 1
        if 40 <= rsi <= 70:
            condition_counts['rsi_ok_sell'] += 1
        
        # MACD
        def ema(data, period):
            if len(data) < period:
                return np.mean(data)
            mult = 2 / (period + 1)
            e = data[0]
            for p in data[1:]:
                e = (p * mult) + (e * (1 - mult))
            return e
        
        ema_12 = ema(c, 12)
        ema_26 = ema(c, 26)
        macd = ema_12 - ema_26
        
        if macd > 0:
            condition_counts['macd_bullish'] += 1
        if macd < 0:
            condition_counts['macd_bearish'] += 1
        
        # ATR volatility
        tr1 = h[-14:] - l[-14:]
        atr = np.mean(tr1)
        atr_pct = (atr / current_price) * 100
        if atr_pct < 2.0:
            condition_counts['volatility_ok'] += 1
        
        # Session
        hour = current_time.hour
        if 4 <= hour <= 22:
            condition_counts['active_session'] += 1
        
        # Candle
        is_bullish = c[-1] > o[-1]
        is_bearish = c[-1] < o[-1]
        body_ratio = abs(c[-1] - o[-1]) / max(h[-1] - l[-1], 0.00001)
        
        if is_bullish and body_ratio > 0.4:
            condition_counts['bullish_candle'] += 1
        if is_bearish and body_ratio > 0.4:
            condition_counts['bearish_candle'] += 1
        
        # Bounce
        if l[-1] <= sma_20 * 1.002 and c[-1] > sma_20:
            condition_counts['bounce_up'] += 1
        if h[-1] >= sma_20 * 0.998 and c[-1] < sma_20:
            condition_counts['bounce_down'] += 1
        
        # Check full signal
        buy_conditions = [
            has_uptrend,
            in_pullback_zone,
            30 <= rsi <= 60,
            macd > 0,
            atr_pct < 2.0,
            4 <= hour <= 22,
            is_bullish and body_ratio > 0.4,
            l[-1] <= sma_20 * 1.002 and c[-1] > sma_20,
        ]
        
        sell_conditions = [
            has_downtrend,
            in_pullback_zone,
            40 <= rsi <= 70,
            macd < 0,
            atr_pct < 2.0,
            4 <= hour <= 22,
            is_bearish and body_ratio > 0.4,
            h[-1] >= sma_20 * 0.998 and c[-1] < sma_20,
        ]
        
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        if buy_score >= 5 or sell_score >= 5:
            signals_found += 1
    
    print(f"\n?? Condition counts (out of 400 bars):")
    for cond, count in condition_counts.items():
        pct = count / 400 * 100
        print(f"   {cond}: {count} ({pct:.1f}%)")
    
    print(f"\n?? Total signals found: {signals_found}")


if __name__ == "__main__":
    asyncio.run(test_signals())
