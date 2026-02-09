"""Debug: trace _simulate to find why only 10 signals"""
import asyncio, logging, sys
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING, format='%(message)s')

from backtesting import BacktestEngine, BacktestConfig
from backtesting.backtest_engine import TradeStatus

config = BacktestConfig(
    symbol='XAUUSDm', timeframe='H1', years=10,
    initial_balance=500, min_quality='LOW', min_confidence=40.0,
    max_risk_per_trade=1.0, max_drawdown=30.0, max_daily_loss=3.0,
    use_full_intelligence=False
)

engine = BacktestEngine(config)

async def test():
    await engine.initialize()
    await engine.load_data()
    
    ws = 60
    total_bars = len(engine.data)
    
    daily_skip_count = 0
    dd_break = False
    signal_count = 0
    execute_count = 0
    
    last_date = None
    
    for i in range(ws, min(ws + 2000, total_bars)):
        current_time = engine.data.index[i]
        current_bar = engine.data.iloc[i]
        
        # 1. Check and close existing trades
        await engine._check_open_trades(current_time, current_bar)
        
        # 2. Check daily risk limits
        if engine.daily_pnl <= -config.max_daily_loss * engine.balance / 100:
            daily_skip_count += 1
            # Still update equity curve logic
            open_pnl = engine._calculate_open_pnl(current_bar['close'])
            engine.equity = engine.balance + open_pnl
            if engine.equity > engine.peak_equity:
                engine.peak_equity = engine.equity
            drawdown = (engine.peak_equity - engine.equity) / engine.peak_equity * 100
            if drawdown >= config.max_drawdown:
                dd_break = True
                print(f"DRAWDOWN BREAK at {current_time}: {drawdown:.2f}%, balance={engine.balance:.2f}")
                break
            if i > ws and current_time.date() != engine.data.index[i-1].date():
                engine.daily_pnl = 0.0
            continue
        
        # 3. Get window data
        window_data = engine.data.iloc[i-ws+1:i+1].copy()
        
        # 4. Analyze
        signal = await engine._analyze_bar(window_data, current_time, current_bar)
        
        if signal:
            signal_count += 1
            if engine._should_execute(signal):
                await engine._execute_signal(signal, current_time, current_bar)
                execute_count += 1
                if execute_count <= 5:
                    print(f"TRADE #{execute_count} at {current_time}: {signal['signal']} conf={signal['confidence']} qual={signal['quality']} balance={engine.balance:.2f}")
        
        # 6. Update equity
        open_pnl = engine._calculate_open_pnl(current_bar['close'])
        engine.equity = engine.balance + open_pnl
        if engine.equity > engine.peak_equity:
            engine.peak_equity = engine.equity
        
        # 7. Drawdown check
        drawdown = (engine.peak_equity - engine.equity) / engine.peak_equity * 100
        if drawdown >= config.max_drawdown:
            dd_break = True
            print(f"DRAWDOWN BREAK at {current_time}: {drawdown:.2f}%, balance={engine.balance:.2f}, equity={engine.equity:.2f}")
            break
        
        # 8. Reset daily PnL
        if i > ws and current_time.date() != engine.data.index[i-1].date():
            engine.daily_pnl = 0.0
    
    # Close all trades
    await engine._close_all_trades(engine.data.index[-1], engine.data.iloc[-1])
    
    print(f"\n--- SUMMARY (first 2000 bars) ---")
    print(f"Signals generated: {signal_count}")
    print(f"Trades executed: {execute_count}")
    print(f"Daily skip bars: {daily_skip_count}")
    print(f"Drawdown break: {dd_break}")
    print(f"Balance: ${engine.balance:.2f}")
    print(f"Peak equity: ${engine.peak_equity:.2f}")
    print(f"Open trades: {len([t for t in engine.trades if t.status == TradeStatus.OPEN])}")
    print(f"Closed trades: {len([t for t in engine.trades if t.status != TradeStatus.OPEN])}")
    
    # Show trade results
    for t in engine.trades[:10]:
        print(f"  {t.id}: {t.side} entry={t.entry_price:.2f} SL={t.stop_loss:.2f} TP={t.take_profit:.2f} exit={t.exit_price} pnl={t.pnl:.2f} status={t.status.value}")

asyncio.run(test())
