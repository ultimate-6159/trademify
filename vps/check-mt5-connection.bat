@echo off
title MT5 Connection Check
color 0E

echo.
echo ============================================
echo      MT5 CONNECTION DIAGNOSTIC
echo ============================================
echo.

cd /d C:\trademify\backend

python -c "
import MetaTrader5 as mt5
import sys

print()
print('Checking MT5 connection...')
print()

if not mt5.initialize():
    error = mt5.last_error()
    print('=' * 50)
    print(' ERROR: MT5 NOT CONNECTED!')
    print('=' * 50)
    print()
    print(f'Error: {error}')
    print()
    print('TO FIX THIS:')
    print('1. Open MetaTrader 5 application')
    print('2. Login to account: 415146568')
    print('3. Server: Exness-MT5Trial14')
    print('4. Enable AutoTrading (Ctrl+E)')
    print()
    sys.exit(1)

account = mt5.account_info()
if account:
    print('=' * 50)
    print(' MT5 CONNECTED SUCCESSFULLY!')
    print('=' * 50)
    print()
    print(f'Account: {account.login}')
    print(f'Server: {account.server}')
    print(f'Balance: ${account.balance:,.2f}')
    print(f'Equity: ${account.equity:,.2f}')
    print()
    
    # Check symbols
    for symbol in ['EURUSDm', 'GBPUSDm', 'XAUUSDm']:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
        if rates is not None and len(rates) >= 60:
            print(f'{symbol}: OK ({len(rates)} candles)')
        else:
            print(f'{symbol}: NEED MORE DATA')
            mt5.symbol_select(symbol, True)
    
    print()
    print('System ready for trading!')

mt5.shutdown()
"

echo.
pause
