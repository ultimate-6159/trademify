@echo off
title Trademify Gold Trading Bot
cd /d %~dp0backend

echo.
echo ============================================================
echo    ?? TRADEMIFY GOLD TRADING BOT ??
echo    Symbol: XAUUSDm (Gold) ONLY
echo    Target: 10-15 trades/day
echo    Expected Win Rate: 85%+
echo ============================================================
echo.

REM Activate virtual environment
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

REM Default parameters - GOLD ONLY
set BROKER=MT5
set SYMBOL=XAUUSDm
set TIMEFRAME=%1
set QUALITY=%2
set WINDOW=%3

if "%TIMEFRAME%"=="" set TIMEFRAME=H1
if "%QUALITY%"=="" set QUALITY=MEDIUM
if "%WINDOW%"=="" set WINDOW=60

echo Broker: %BROKER%
echo Symbol: %SYMBOL% (Gold Only)
echo Timeframe: %TIMEFRAME%
echo Quality: %QUALITY%
echo Window: %WINDOW%
echo.

python ai_trading_bot.py --broker %BROKER% --symbols %SYMBOL% --timeframe %TIMEFRAME% --quality %QUALITY% --window %WINDOW%

pause
