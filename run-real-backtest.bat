@echo off
title Trademify Real Intelligence Backtest
cd /d %~dp0backend

echo.
echo ============================================================
echo    TRADEMIFY REAL INTELLIGENCE BACKTEST
echo    Using 20-Layer System Same as Live Trading 100%%
echo ============================================================
echo.

REM Activate virtual environment
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

REM Default parameters
set SYMBOL=%1
set TIMEFRAME=%2
set YEARS=%3
set QUALITY=%4
set PASS_RATE=%5

if "%SYMBOL%"=="" set SYMBOL=EURUSD
if "%TIMEFRAME%"=="" set TIMEFRAME=H1
if "%YEARS%"=="" set YEARS=2
if "%QUALITY%"=="" set QUALITY=MEDIUM
if "%PASS_RATE%"=="" set PASS_RATE=0.40

echo Symbol: %SYMBOL%
echo Timeframe: %TIMEFRAME%
echo Years: %YEARS%
echo Min Quality: %QUALITY%
echo Min Pass Rate: %PASS_RATE%
echo.

python run_real_backtest.py %SYMBOL% %TIMEFRAME% %YEARS% %QUALITY% %PASS_RATE%

echo.
echo ============================================================
echo Results saved in: backend\data\backtest_results\
echo ============================================================
pause
