@echo off
title Trademify VPS Update and Restart
color 0A

echo.
echo ============================================
echo      TRADEMIFY VPS UPDATE AND RESTART
echo ============================================
echo.

cd /d D:\projectx\trademify

echo [1/5] Stopping current services...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/5] Pulling latest code from GitHub...
git pull origin main
if errorlevel 1 (
    echo ERROR: Git pull failed!
    pause
    exit /b 1
)

echo [3/5] Installing any new dependencies...
cd backend
pip install -r requirements.txt --quiet

echo [4/5] Starting API server...
start "Trademify API" cmd /k "cd /d D:\projectx\trademify\backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
timeout /t 5 /nobreak >nul

echo [5/5] Starting AI Trading Bot...
start "Trademify Bot" cmd /k "cd /d D:\projectx\trademify\backend && python ai_trading_bot.py --broker MT5 --symbols EURUSDm,GBPUSDm,XAUUSDm --quality MEDIUM"

echo.
echo ============================================
echo      UPDATE COMPLETE!
echo ============================================
echo.
echo API Server: http://66.42.50.149:8000
echo API Docs:   http://66.42.50.149:8000/docs
echo.
echo Check intelligence layers at:
echo http://66.42.50.149:8000/api/v1/intelligence/layers
echo.
pause
