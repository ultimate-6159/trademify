"""
Run Real Intelligence Backtest
ใช้ 20-Layer System เหมือน Live Trading 100%
"""
import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.real_backtest_engine import RealIntelligenceBacktest, RealBacktestConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)


async def main():
    """Run the real intelligence backtest"""
    
    # Parse command line args
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "H1"
    years = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    min_quality = sys.argv[4] if len(sys.argv) > 4 else "MEDIUM"
    min_pass_rate = float(sys.argv[5]) if len(sys.argv) > 5 else 0.40
    
    print("")
    print("🧠 ═══════════════════════════════════════════════════════════════")
    print("🧠       TRADEMIFY REAL INTELLIGENCE BACKTEST")
    print("🧠       ใช้ 20-Layer System เหมือน Live Trading 100%")
    print("🧠 ═══════════════════════════════════════════════════════════════")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Years: {years}")
    print(f"   Min Quality: {min_quality}")
    print(f"   Min Pass Rate: {min_pass_rate:.0%}")
    print("🧠 ═══════════════════════════════════════════════════════════════")
    print("")
    
    # Create config
    config = RealBacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        years=years,
        min_quality=min_quality,
        min_layer_pass_rate=min_pass_rate,
        
        # Account settings
        initial_balance=10000.0,
        max_risk_per_trade=1.0,
        max_daily_loss=3.0,
        max_drawdown=25.0,
        
        # Signal settings
        min_confidence=65.0,
        
        # Execution settings
        slippage_pips=1.0,
        spread_pips=1.5,
    )
    
    # Run backtest
    engine = RealIntelligenceBacktest(config)
    result = await engine.run()
    
    print("")
    print("✅ Backtest complete! Check data/backtest_results/ for detailed reports.")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
