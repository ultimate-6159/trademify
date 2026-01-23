@echo off
title Fix MT5 Setup
color 0E

echo.
echo ============================================
echo      FIX MT5 SETUP FOR TRADEMIFY
echo ============================================
echo.

:: Stop all Python processes
echo [1/5] Stopping all Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

:: Check Python version
echo [2/5] Checking Python...
python --version
where python

:: Install MetaTrader5 package
echo [3/5] Installing MetaTrader5 package...
pip install MetaTrader5 --upgrade

:: Install all requirements
echo [4/5] Installing all requirements...
cd /d C:\trademify\backend
pip install -r requirements.txt

:: Test MT5 connection
echo [5/5] Testing MT5 connection...
python -c "import MetaTrader5 as mt5; print('Init:', mt5.initialize()); info=mt5.account_info(); print('Account:', info.login if info else 'NONE', 'Balance:', info.balance if info else 0); mt5.shutdown()"

echo.
echo ============================================
echo      SETUP COMPLETE!
echo ============================================
echo.
echo If MT5 test shows account info, run:
echo   cd /d C:\trademify\backend
echo   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
echo.
pause
