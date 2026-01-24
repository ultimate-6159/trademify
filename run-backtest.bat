@echo off
REM ====================================================
REM Trademify Backtest Runner
REM Test your strategy with 10 years of historical data
REM ====================================================

echo.
echo ========================================================
echo     TRADEMIFY BACKTEST ENGINE
echo     Test your strategy with 10 years of data
echo ========================================================
echo.

cd /d %~dp0backend

REM Activate virtual environment
call ..\venv\Scripts\activate.bat

REM Default parameters
set SYMBOL=%1
set TIMEFRAME=%2
set YEARS=%3
set BALANCE=%4
set QUALITY=%5

if "%SYMBOL%"=="" set SYMBOL=XAUUSDm
if "%TIMEFRAME%"=="" set TIMEFRAME=H1
if "%YEARS%"=="" set YEARS=10
if "%BALANCE%"=="" set BALANCE=10000
if "%QUALITY%"=="" set QUALITY=MEDIUM

echo.
echo Running backtest:
echo   Symbol: %SYMBOL%
echo   Timeframe: %TIMEFRAME%
echo   Years: %YEARS%
echo   Balance: $%BALANCE%
echo   Quality: %QUALITY%
echo.

python run_backtest.py --symbol %SYMBOL% --timeframe %TIMEFRAME% --years %YEARS% --balance %BALANCE% --quality %QUALITY% --html-report

echo.
echo ========================================================
echo Backtest complete! Check data/backtest_results for reports
echo ========================================================
pause
