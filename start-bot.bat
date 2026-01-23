@echo off
:: ============================================
:: Trademify AI Trading Bot - Quick Start
:: สำหรับรันเฉพาะ Bot โดยตรง
:: ============================================

title Trademify AI Trading Bot
color 0B

echo.
echo =============================================
echo    TRADEMIFY AI TRADING BOT
echo    Expert Pattern Recognition System
echo =============================================
echo.

:: Set path
cd /d C:\trademify

:: Check venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Run: vps\setup-vps-complete.ps1
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Get parameters or use defaults
set BROKER=%1
set SYMBOLS=%2
set TIMEFRAME=%3
set QUALITY=%4
set INTERVAL=%5

if "%BROKER%"=="" set BROKER=MT5
if "%SYMBOLS%"=="" set SYMBOLS=EURUSDm,GBPUSDm,XAUUSDm
if "%TIMEFRAME%"=="" set TIMEFRAME=H1
if "%QUALITY%"=="" set QUALITY=MEDIUM
if "%INTERVAL%"=="" set INTERVAL=60

:: Check MT5
if "%BROKER%"=="MT5" (
    tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I "terminal64.exe">NUL
    if errorlevel 1 (
        echo [WARNING] MT5 Terminal not running!
        echo Please start MT5 and login first.
        echo.
        pause
    )
)

echo.
echo Configuration:
echo   Broker: %BROKER%
echo   Symbols: %SYMBOLS%
echo   Timeframe: %TIMEFRAME%
echo   Quality: %QUALITY%
echo   Interval: %INTERVAL% seconds
echo.

cd backend

echo Starting AI Trading Bot...
echo.
python ai_trading_bot.py --broker %BROKER% --symbols %SYMBOLS% --timeframe %TIMEFRAME% --quality %QUALITY% --interval %INTERVAL%

pause
