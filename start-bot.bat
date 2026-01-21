@echo off
REM ============================================
REM Trademify AI Trading Bot - Quick Start
REM ระบบเทรดอัตโนมัติด้วย AI เพียงหนึ่งเดียว
REM ============================================

echo.
echo =============================================
echo    TRADEMIFY AI TRADING BOT
echo    Expert Pattern Recognition System
echo =============================================
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    echo Run: python -m venv venv
    pause
    exit /b 1
)

REM Get parameters or use defaults
set BROKER=%1
set SYMBOLS=%2
set TIMEFRAME=%3
set QUALITY=%4
set INTERVAL=%5

if "%BROKER%"=="" set BROKER=MT5
if "%SYMBOLS%"=="" set SYMBOLS=EURUSD,GBPUSD,XAUUSD
if "%TIMEFRAME%"=="" set TIMEFRAME=H1
if "%QUALITY%"=="" set QUALITY=HIGH
if "%INTERVAL%"=="" set INTERVAL=60

REM Check if MT5 is running (for MT5 broker)
if "%BROKER%"=="MT5" (
    tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
    if errorlevel 1 (
        echo [WARNING] MT5 Terminal not running!
        echo Starting MT5...
        
        if exist "C:\Program Files\MetaTrader 5\terminal64.exe" (
            start "" "C:\Program Files\MetaTrader 5\terminal64.exe"
        ) else (
            echo [ERROR] MT5 not found at default location
            echo Please start MT5 manually
        )
        
        echo Waiting 30 seconds for MT5 to start...
        timeout /t 30 /nobreak
    )
)

echo.
echo Configuration:
echo   Broker: %BROKER%
echo   Symbols: %SYMBOLS%
echo   Timeframe: %TIMEFRAME%
echo   Quality: %QUALITY% (PREMIUM/HIGH/MEDIUM/LOW)
echo   Interval: %INTERVAL% seconds
echo   Mode: PAPER TRADING (safe)
echo.
echo To use LIVE TRADING, add --real flag
echo.

cd backend

REM Start the AI Trading Bot
echo Starting AI Trading Bot...
echo.
python ai_trading_bot.py --broker %BROKER% --symbols %SYMBOLS% --timeframe %TIMEFRAME% --quality %QUALITY% --interval %INTERVAL%

pause
