@echo off
title Trademify Status
color 0B
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║           📊 TRADEMIFY STATUS CHECK                       ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

:: Check API Server
echo [Checking] API Server (Port 8000)...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ API Server: RUNNING
    curl -s http://localhost:8000/health
    echo.
) else (
    echo   ❌ API Server: NOT RUNNING
)
echo.

:: Check MT5
echo [Checking] MetaTrader 5...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I "terminal64.exe" >NUL
if %errorlevel% equ 0 (
    echo   ✅ MT5: RUNNING
) else (
    echo   ❌ MT5: NOT RUNNING
)
echo.

:: Check Bot Status via API
echo [Checking] Trading Bot...
curl -s http://localhost:8000/api/v1/bot/status 2>nul
echo.
echo.

pause
