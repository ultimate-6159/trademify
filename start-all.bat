@echo off
title Trademify - One Click Start
color 0A
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║           🚀 TRADEMIFY ONE-CLICK START                    ║
echo  ║                                                            ║
echo  ║    API Server + Trading Bot = ALL IN ONE                  ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.
echo [%TIME%] Starting Trademify...
echo.

cd /d %~dp0

:: Check if MT5 is running
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo [WARNING] MetaTrader 5 is NOT running!
    echo [WARNING] Please start MT5 first for trading to work.
    echo.
    pause
)

:: Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo [ERROR] Please run setup-vps-complete.ps1 first
    pause
    exit /b 1
)

:: Check venv
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo [ERROR] Please run setup-vps-complete.ps1 first
    pause
    exit /b 1
)

:: Check .env
if not exist "backend\.env" (
    echo [ERROR] backend\.env not found!
    echo [ERROR] Please copy .env.example to .env and configure
    pause
    exit /b 1
)

echo [OK] All checks passed!
echo.
echo ════════════════════════════════════════════════════════════
echo  Starting API Server + Bot (Auto-Start Enabled)
echo  Bot will start automatically with saved settings
echo ════════════════════════════════════════════════════════════
echo.
echo  Press Ctrl+C to stop
echo.

:: Start API (which auto-starts bot)
cd backend
..\venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

pause
