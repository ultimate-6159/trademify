# ?? Gold M15 High Win Rate Strategy Configuration

## ?? ????????????????????? Backtest

| Metric | M15 (1 Year) | H1 (1 Year) |
|--------|-------------|-------------|
| **Win Rate** | **87.6%** ? | **79.4%** ? |
| **Total Trades** | 1,075 (~4.3/day) | 63 (~0.25/day) |
| **Total Return** | **+99.44%** ?? | +13.75% |
| **Profit Factor** | **1.51** | 1.71 |
| **Max Drawdown** | 6.96% | 3.88% |
| **Expectancy** | **$0.92** | $2.18 |
| **Sharpe Ratio** | **2.24** | 3.84 |
| **Calmar Ratio** | **14.50** ?? | 3.58 |
| **Max Consec. Wins** | **69** | 12 |
| **Max Consec. Losses** | 5 | 2 |
| **Best Hours** | 2, 16, 14 UTC | 18, 14, 7 UTC |
| **Best Days** | Wed, Tue, Thu | Wed, Thu, Mon |

---

## ?? BEST Configuration (99%+ Return)

### M15 Strategy - MAXIMUM PROFIT

```python
from backtesting.backtest_engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    # Symbol & Timeframe
    symbol='XAUUSDm',
    timeframe='M15',
    years=1,
    
    # Account
    initial_balance=1000,
    max_risk_per_trade=1.0,  # 1% risk per trade
    
    # Signal Filters (Relaxed for more signals)
    min_quality='LOW',        # Allow LOW quality signals
    min_confidence=60.0,      # 60% confidence minimum
    min_layer_pass_rate=0.20, # 20% layer pass rate
    
    # Intelligence (OFF for speed)
    use_full_intelligence=False,
    use_live_trading_logic=False,
    
    # Execution
    spread_pips=3.0,          # Gold spread ~30 cents
    slippage_pips=1.0,
    
    # ?? TRAILING STOP SETTINGS (KEY TO HIGH PROFIT)
    use_trailing_stop=True,
    trailing_activation_pct=0.20,  # Activate at 20% of TP (early)
    trailing_distance_pct=0.35,    # Trail at 35% of profit (tight)
)
```

### H1 Strategy (Lower Frequency + Stable Returns)

```python
config = BacktestConfig(
    symbol='XAUUSDm',
    timeframe='H1',
    years=1,
    initial_balance=1000,
    min_quality='MEDIUM',     # Higher quality filter
    min_confidence=65.0,
    min_layer_pass_rate=0.20,
    use_full_intelligence=False,
    use_live_trading_logic=False,
    spread_pips=3.0,
    use_trailing_stop=True,
)
```

---

## ?? Strategy Logic

### Entry Conditions (Need 3/10 for M15, 4/10 for H1)

1. **Trend**: EMA alignment (Fast > Mid > Slow > Trend)
2. **Crossover**: EMA Fast crosses EMA Mid
3. **RSI Range**: 30-70 (not extreme)
4. **RSI Momentum**: Rising for BUY, Falling for SELL
5. **Session**: London (7-16 UTC) or NY (13-21 UTC)
6. **Candle**: Bullish/Bearish body > 25%
7. **Entry Zone**: Within 3 ATR of EMA Slow
8. **Volatility**: ATR% < 4%

### SL/TP Settings (M15 - OPTIMAL)

- **SL** = 2.0 × ATR (moderate room)
- **TP** = 0.6 × SL (good profit target)
- **R:R** = 0.6:1 (needs 63% to break even)
- **Min SL** = $3, **Max SL** = $10
- **Breakeven** = Move SL to entry+10% when profit reaches 50% of TP

### Trailing Stop Logic (KEY TO 99% RETURN)

1. **Activation**: When profit reaches **20%** of TP distance (early activation)
2. **Trail Distance**: **35%** of current profit (tight trailing)
3. **Breakeven Stop**: SL moves to entry+10% when 50% of TP reached
4. **Direction**: Only moves in favorable direction (never back)

---

## ?? Risk Management

| Setting | Value | Description |
|---------|-------|-------------|
| Max Risk/Trade | 1% | Maximum risk per single trade |
| Max Daily Loss | 3% | Stop trading if daily loss exceeds |
| Max Drawdown | 30% | Stop bot if total drawdown exceeds |
| Max Open Trades | 5 | Maximum simultaneous open positions |

---

## ?? Best Trading Times (UTC)

### M15
- **Best Hours**: 14:00, 16:00, 02:00
- **Best Days**: Wednesday, Tuesday, Thursday
- **Avoid**: Asian session (22:00-06:00), Friday after 19:00

### H1
- **Best Hours**: 14:00, 18:00, 07:00
- **Best Days**: Wednesday, Thursday, Monday
- **Best Session**: London-NY Overlap (13:00-16:00)

---

## ?? How to Run Backtest

```bash
cd C:\trademify\backend

# M15 Strategy - MAXIMUM PROFIT (99%+ return)
python -c "import asyncio; from backtesting.backtest_engine import BacktestEngine, BacktestConfig; config = BacktestConfig(symbol='XAUUSDm', timeframe='M15', years=1, initial_balance=1000, min_quality='LOW', min_confidence=60.0, use_full_intelligence=False, use_live_trading_logic=False, min_layer_pass_rate=0.20, spread_pips=3.0, use_trailing_stop=True, trailing_activation_pct=0.20, trailing_distance_pct=0.35); engine = BacktestEngine(config); asyncio.run(engine.run())"

# H1 Strategy (Stable + Lower Risk)
python -c "import asyncio; from backtesting.backtest_engine import BacktestEngine, BacktestConfig; config = BacktestConfig(symbol='XAUUSDm', timeframe='H1', years=1, initial_balance=1000, min_quality='MEDIUM', min_confidence=65.0, use_full_intelligence=False, use_live_trading_logic=False, min_layer_pass_rate=0.20, spread_pips=3.0, use_trailing_stop=True); engine = BacktestEngine(config); asyncio.run(engine.run())"
```

---

## ?? Tuning Tips

### To Increase Win Rate (decrease trades)
- Increase `min_quality` to 'MEDIUM' or 'HIGH'
- Increase `min_confidence` to 70+
- Decrease SL/TP ratio (e.g., TP = 0.3 × SL)

### To Increase Trades (may decrease win rate)
- Decrease `min_quality` to 'LOW'
- Decrease `min_confidence` to 55
- Decrease `min_conditions` in strategy

### To Increase Profit
- Enable `use_trailing_stop`
- Increase `trailing_activation_pct` to 0.4
- Use H1 timeframe for better R:R

---

## ? Live Trading Notes

For live trading, ensure:
1. MT5 terminal is running
2. Symbol 'XAUUSDm' is available (Exness micro)
3. Sufficient margin for lot sizes
4. Stable VPS connection
5. Consider using `use_full_intelligence=True` for better filtering

---

*Strategy optimized for XAUUSDm (Gold) on Exness micro accounts*
*Last updated: 2025-01-25*
