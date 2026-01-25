"""
?? Gold Backtest with Custom Balance
????? Backtest ????????????????????????

Usage:
    python run_gold_backtest.py [balance] [years] [mode]
    
Examples:
    python run_gold_backtest.py 100 1 balanced      # $100, 1 year, balanced mode
    python run_gold_backtest.py 500 1 high_frequency # $500, 1 year, high_frequency mode
    python run_gold_backtest.py 1000 2 active       # $1000, 2 years, active mode
"""
import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.real_backtest_engine import RealIntelligenceBacktest, RealBacktestConfig
from config.high_frequency_trading import get_config_for_mode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)


async def run_gold_backtest(
    initial_balance: float = 100.0,
    years: int = 1,
    mode: str = "balanced"
):
    """
    ??? Backtest ?????? Gold (XAUUSDm) ???????????????????
    
    Args:
        initial_balance: ??????? (???? $100, $500, $1000)
        years: ???????????????
        mode: conservative, balanced, active, high_frequency, aggressive
    """
    
    # Get trading config based on mode
    trading_config = get_config_for_mode(mode)
    
    print("")
    print("?? ???????????????????????????????????????????????????????????????")
    print("??       GOLD (XAUUSDm) BACKTEST")
    print(f"??       Mode: {mode.upper()}")
    print("?? ???????????????????????????????????????????????????????????????")
    print(f"   ?? Initial Balance: ${initial_balance:,.2f}")
    print(f"   ?? Period: {years} year(s)")
    print(f"   ?? Min Quality: {trading_config.min_quality}")
    print(f"   ?? Min Confidence: {trading_config.min_confidence}%")
    print(f"   ?? Max Daily Trades: {trading_config.max_daily_trades}")
    print(f"   ??? Risk per Trade: {trading_config.max_risk_per_trade}%")
    print(f"   ?? Max Drawdown: 15%")
    print("?? ???????????????????????????????????????????????????????????????")
    print("")
    
    # Calculate appropriate lot size for small accounts
    # For $100 account, we need to use micro lots (0.01)
    # Risk calculation: $100 * 2% = $2 risk per trade
    
    # Create backtest config - MATCHED WITH LIVE TRADING
    config = RealBacktestConfig(
        symbol="XAUUSD",  # Gold
        timeframe="H1",
        years=years,
        min_quality=trading_config.min_quality,
        min_layer_pass_rate=trading_config.min_layer_pass_rate,
        
        # Account settings - SAME AS LIVE
        initial_balance=initial_balance,
        max_risk_per_trade=trading_config.max_risk_per_trade,  # 2%
        max_daily_loss=trading_config.max_daily_loss,  # 5%
        max_drawdown=15.0,  # 15% max drawdown
        max_positions=5,  # ? SAME AS LIVE (5 positions)
        
        # Signal settings
        min_confidence=trading_config.min_confidence,
        
        # Execution settings - REALISTIC COSTS
        slippage_pips=2.0,   # ? Realistic slippage
        spread_pips=2.5,     # ? Gold has wider spread
        commission_per_lot=7.0,  # ? $7/lot commission
    )
    
    # Run backtest
    engine = RealIntelligenceBacktest(config)
    result = await engine.run()
    
    # Print detailed analysis
    print("")
    print("?? ???????????????????????????????????????????????????????????????")
    print("??       DETAILED ANALYSIS")
    print("?? ???????????????????????????????????????????????????????????????")
    
    if result.get('total_trades', 0) > 0:
        initial = initial_balance
        final = result.get('final_balance', initial_balance)
        total_return = result.get('total_return_pct', 0)
        profit = final - initial
        
        print(f"   ?? Starting Capital: ${initial:,.2f}")
        print(f"   ?? Final Balance: ${final:,.2f}")
        print(f"   ?? Total Profit/Loss: ${profit:,.2f}")
        print(f"   ?? Total Return: {total_return:,.2f}%")
        print("")
        
        # Trading statistics
        total_trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)
        pf = result.get('profit_factor', 0)
        max_dd = result.get('max_drawdown_pct', 0)
        
        trading_days = result.get('trading_days', 252)
        trades_per_day = total_trades / max(trading_days, 1)
        
        print(f"   ?? Total Trades: {total_trades}")
        print(f"   ?? Trades per Day: {trades_per_day:.1f}")
        print(f"   ? Win Rate: {win_rate:.1f}%")
        print(f"   ?? Profit Factor: {pf:.2f}")
        print(f"   ?? Max Drawdown: {max_dd:.1f}%")
        print("")
        
        # Performance rating
        print("   ?? PERFORMANCE RATING:")
        
        if win_rate >= 80 and pf >= 2.0:
            print("   ????? EXCELLENT - Ready for live trading!")
        elif win_rate >= 70 and pf >= 1.5:
            print("   ???? VERY GOOD - Consider paper trading first")
        elif win_rate >= 60 and pf >= 1.2:
            print("   ??? GOOD - Needs more testing")
        elif win_rate >= 50 and pf >= 1.0:
            print("   ?? AVERAGE - Needs optimization")
        else:
            print("   ? POOR - Do not use for live trading")
        
        # Risk analysis for small account
        print("")
        print(f"   ??? RISK ANALYSIS (${initial:.0f} account):")
        risk_per_trade_dollar = initial * (trading_config.max_risk_per_trade / 100)
        max_daily_loss_dollar = initial * (trading_config.max_daily_loss / 100)
        max_dd_dollar = initial * (15 / 100)
        
        print(f"   - Risk per Trade: ${risk_per_trade_dollar:.2f}")
        print(f"   - Max Daily Loss: ${max_daily_loss_dollar:.2f}")
        print(f"   - Max Drawdown: ${max_dd_dollar:.2f}")
        
        # Lot size recommendation
        print("")
        print("   ?? LOT SIZE RECOMMENDATION:")
        if initial <= 100:
            print("   - Use 0.01 lot (micro lot)")
            print("   - Suitable for cent accounts")
        elif initial <= 500:
            print("   - Use 0.01-0.02 lots")
        elif initial <= 1000:
            print("   - Use 0.02-0.05 lots")
        else:
            print("   - Use standard position sizing (2% risk)")
    
    print("?? ???????????????????????????????????????????????????????????????")
    print("")
    
    return result


if __name__ == "__main__":
    # Parse command line args
    balance = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mode = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    
    asyncio.run(run_gold_backtest(balance, years, mode))
