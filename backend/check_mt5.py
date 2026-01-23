import MetaTrader5 as mt5
import sys

print("=" * 50)
print(" MT5 Connection Diagnostic")
print("=" * 50)

# Try to initialize
if not mt5.initialize():
    error = mt5.last_error()
    print(f"ERROR: MT5 Initialize failed!")
    print(f"Error code: {error}")
    print()
    print("Possible causes:")
    print("1. MT5 Terminal is not running")
    print("2. MT5 Terminal is not logged in")
    print("3. AutoTrading is disabled")
    print()
    print("SOLUTION:")
    print("1. Open MetaTrader 5 on VPS")
    print("2. Login to account 267643655")
    print("3. Enable AutoTrading (Ctrl+E)")
    sys.exit(1)

# Get account info
account = mt5.account_info()
if account is None:
    print("ERROR: Cannot get account info")
    print("Make sure you are logged in to MT5")
    mt5.shutdown()
    sys.exit(1)

print(f"Account: {account.login}")
print(f"Server: {account.server}")
print(f"Balance: {account.balance}")
print(f"Leverage: 1:{account.leverage}")
print(f"Trade Mode: {account.trade_mode}")
print()

# Check symbols
print("Checking symbols...")
for symbol in ['EURUSDm', 'GBPUSDm', 'XAUUSDm']:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"  {symbol}: NOT FOUND!")
    else:
        if not info.visible:
            mt5.symbol_select(symbol, True)
            print(f"  {symbol}: Enabled")
        else:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"  {symbol}: OK (Bid: {tick.bid})")
            else:
                print(f"  {symbol}: No tick data")

# Check candle data
print()
print("Checking H1 candle data...")
for symbol in ['EURUSDm', 'GBPUSDm', 'XAUUSDm']:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates is not None and len(rates) > 0:
        print(f"  {symbol}: {len(rates)} candles OK")
    else:
        print(f"  {symbol}: NO DATA - {mt5.last_error()}")

mt5.shutdown()
print()
print("MT5 Check Complete!")
