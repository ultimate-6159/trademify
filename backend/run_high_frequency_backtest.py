"""
?? High Frequency Backtest Script
????? High Frequency Trading Config ???????????????

????????: 10-15 orders/day ?????????????????????
"""
import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.real_backtest_engine import RealIntelligenceBacktest, RealBacktestConfig
from config.high_frequency_trading import (
    HIGH_FREQUENCY_CONFIG,
    BALANCED_CONFIG,
    ACTIVE_CONFIG,
    get_config_for_mode
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)


async def run_high_frequency_backtest(
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    years: int = 1,
    mode: str = "high_frequency"
):
    """
    ??? Backtest ???? High Frequency Config
    
    Args:
        symbol: Trading symbol (EURUSD, GBPUSD, XAUUSD)
        timeframe: H1, M15, M30
        years: ???????????????
        mode: conservative, balanced, active, high_frequency, aggressive
    """
    
    # Get trading config based on mode
    trading_config = get_config_for_mode(mode)
    
    print("")
    print("?? ???????????????????????????????????????????????????????????????")
    print("??       HIGH FREQUENCY BACKTEST")
    print(f"??       Mode: {mode.upper()} ({trading_config.max_daily_trades} trades/day target)")
    print("?? ???????????????????????????????????????????????????????????????")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Years: {years}")
    print(f"   Min Quality: {trading_config.min_quality}")
    print(f"   Min Confidence: {trading_config.min_confidence}%")
    print(f"   Max Daily Trades: {trading_config.max_daily_trades}")
    print(f"   Layer Pass Rate: {trading_config.min_layer_pass_rate:.0%}")
    print("?? ???????????????????????????????????????????????????????????????")
    print("")
    
    # Create backtest config from trading config
    config = RealBacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        years=years,
        min_quality=trading_config.min_quality,
        min_layer_pass_rate=trading_config.min_layer_pass_rate,
        
        # Account settings - ??????? conservative ????????????????????????
        initial_balance=10000.0,
        max_risk_per_trade=trading_config.max_risk_per_trade,  # 2% ??? config
        max_daily_loss=trading_config.max_daily_loss,  # 5% ??? config
        max_drawdown=15.0,  # ????? 25% ????????????????????????
        
        # Signal settings
        min_confidence=trading_config.min_confidence,
        
        # Execution settings
        slippage_pips=1.0,
        spread_pips=1.5,
    )
    
    # Run backtest
    engine = RealIntelligenceBacktest(config)
    result = await engine.run()
    
    # Print additional analysis
    print("")
    print("?? ???????????????????????????????????????????????????????????????")
    print("??       HIGH FREQUENCY ANALYSIS")
    print("?? ???????????????????????????????????????????????????????????????")
    
    if result.get('total_trades', 0) > 0:
        trading_days = result.get('trading_days', 252)
        trades_per_day = result.get('total_trades', 0) / max(trading_days, 1)
        
        print(f"   ?? Trades per Day: {trades_per_day:.1f}")
        print(f"   ?? Target: {trading_config.max_daily_trades} trades/day")
        
        if trades_per_day >= 10:
            print(f"   ? TARGET ACHIEVED! ({trades_per_day:.1f} >= 10)")
        elif trades_per_day >= 5:
            print(f"   ?? Good but need more trades ({trades_per_day:.1f})")
        else:
            print(f"   ? Too few trades ({trades_per_day:.1f})")
        
        # Win rate analysis
        win_rate = result.get('win_rate', 0)
        if win_rate >= 65:
            print(f"   ? WIN RATE: Excellent ({win_rate:.1f}%)")
        elif win_rate >= 55:
            print(f"   ?? WIN RATE: Good ({win_rate:.1f}%)")
        else:
            print(f"   ? WIN RATE: Needs improvement ({win_rate:.1f}%)")
        
        # Profit factor analysis
        pf = result.get('profit_factor', 0)
        if pf >= 1.5:
            print(f"   ? PROFIT FACTOR: Excellent ({pf:.2f})")
        elif pf >= 1.2:
            print(f"   ?? PROFIT FACTOR: Good ({pf:.2f})")
        elif pf >= 1.0:
            print(f"   ?? PROFIT FACTOR: Break-even ({pf:.2f})")
        else:
            print(f"   ? PROFIT FACTOR: Losing ({pf:.2f})")
    
    print("?? ???????????????????????????????????????????????????????????????")
    print("")
    
    return result


async def compare_modes(symbol: str = "EURUSD", years: int = 1):
    """
    ?????????????? Trading Modes
    """
    modes = ["conservative", "balanced", "active", "high_frequency"]
    results = {}
    
    print("\n" + "=" * 70)
    print("   COMPARING ALL TRADING MODES")
    print("=" * 70 + "\n")
    
    for mode in modes:
        print(f"\n?? Testing {mode.upper()} mode...")
        result = await run_high_frequency_backtest(
            symbol=symbol,
            timeframe="H1",
            years=years,
            mode=mode
        )
        results[mode] = result
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("   COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Mode':<15} {'Trades':<10} {'Win%':<10} {'PF':<10} {'Return%':<12} {'MaxDD%':<10}")
    print("-" * 70)
    
    for mode, result in results.items():
        trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)
        pf = result.get('profit_factor', 0)
        ret = result.get('total_return_pct', 0)
        dd = result.get('max_drawdown_pct', 0)
        
        print(f"{mode:<15} {trades:<10} {win_rate:<10.1f} {pf:<10.2f} {ret:<12.2f} {dd:<10.1f}")
    
    print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    # Parse command line args
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Compare all modes
        symbol = sys.argv[2] if len(sys.argv) > 2 else "EURUSD"
        years = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        asyncio.run(compare_modes(symbol, years))
    else:
        # Single mode backtest
        symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
        timeframe = sys.argv[2] if len(sys.argv) > 2 else "H1"
        years = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        mode = sys.argv[4] if len(sys.argv) > 4 else "high_frequency"
        
        asyncio.run(run_high_frequency_backtest(symbol, timeframe, years, mode))
