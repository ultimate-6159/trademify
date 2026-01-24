"""
Trademify Backtest Runner
สคริปต์สำหรับรัน Backtest ย้อนหลัง 10 ปี

Usage:
    python run_backtest.py --symbol EURUSDm --timeframe H1 --years 10
    python run_backtest.py --symbol BTCUSDT --timeframe H4 --years 5 --balance 50000
"""
import asyncio
import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting import BacktestEngine, BacktestConfig, BacktestReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="🧪 Trademify Backtest Engine - Test your strategy with 10 years of data"
    )
    
    # Required arguments
    parser.add_argument(
        "--symbol", "-s",
        default="EURUSDm",
        help="Symbol to backtest (e.g., EURUSDm, BTCUSDT)"
    )
    
    parser.add_argument(
        "--timeframe", "-t",
        default="H1",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        help="Timeframe for backtest"
    )
    
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=10,
        help="Number of years to backtest (default: 10)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--balance", "-b",
        type=float,
        default=10000.0,
        help="Initial balance in USD (default: 10000)"
    )
    
    parser.add_argument(
        "--quality", "-q",
        default="MEDIUM",
        choices=["PREMIUM", "HIGH", "MEDIUM", "LOW"],
        help="Minimum signal quality (default: MEDIUM)"
    )
    
    parser.add_argument(
        "--signal-mode",
        default="technical",
        choices=["technical", "pattern"],
        help="Signal generation mode: technical (indicators) or pattern (FAISS)"
    )
    
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.40,
        help="Minimum layer pass rate (0.0-1.0, default: 0.40)"
    )
    
    parser.add_argument(
        "--max-risk",
        type=float,
        default=2.0,
        help="Max risk per trade %% (default: 2.0)"
    )
    
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=20.0,
        help="Max drawdown %% before stopping (default: 20.0)"
    )
    
    parser.add_argument(
        "--no-intelligence",
        action="store_true",
        help="Skip intelligence layer analysis (faster but less accurate)"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report with charts"
    )
    
    parser.add_argument(
        "--excel-report",
        action="store_true",
        help="Generate Excel report"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data/backtest_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n")
    print("=" * 70)
    print("🧪 TRADEMIFY BACKTEST ENGINE")
    print("=" * 70)
    print(f"   Symbol:           {args.symbol}")
    print(f"   Timeframe:        {args.timeframe}")
    print(f"   Period:           {args.years} years")
    print(f"   Initial Balance:  ${args.balance:,.2f}")
    print(f"   Min Quality:      {args.quality}")
    print(f"   Signal Mode:      {args.signal_mode}")
    print(f"   Min Pass Rate:    {args.min_pass_rate:.0%}")
    print(f"   Max Risk/Trade:   {args.max_risk}%")
    print(f"   Max Drawdown:     {args.max_drawdown}%")
    print("=" * 70)
    print("\n")
    
    # Create config
    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        years=args.years,
        initial_balance=args.balance,
        min_quality=args.quality,
        min_layer_pass_rate=args.min_pass_rate,
        max_risk_per_trade=args.max_risk,
        max_drawdown=args.max_drawdown,
        use_full_intelligence=not args.no_intelligence,
        signal_mode=args.signal_mode,
        output_dir=args.output_dir
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    result = await engine.run()
    
    # Generate reports
    if args.html_report or args.excel_report:
        reporter = BacktestReporter(args.output_dir)
        
        if args.html_report:
            html_path = reporter.generate_html_report(result)
            print(f"\n📄 HTML Report: {html_path}")
        
        if args.excel_report:
            excel_path = reporter.generate_excel_report(result)
            print(f"\n📊 Excel Report: {excel_path}")
    
    # Print final summary
    print("\n")
    print("=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"   Total Trades:       {result.total_trades}")
    print(f"   Win Rate:           {result.win_rate:.1f}%")
    print(f"   Total Return:       {result.total_return:+.2f}%")
    print(f"   Profit Factor:      {result.profit_factor:.2f}")
    print(f"   Max Drawdown:       {result.max_drawdown:.2f}%")
    print(f"   Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print("=" * 70)
    print("\n")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
