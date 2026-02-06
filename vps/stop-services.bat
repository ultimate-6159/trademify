@echo off
title Trademify - Stopping Services
color 0C

echo.
echo ============================================
echo    Trademify - Stopping All Services
echo ============================================
echo.

:: Stop Windows Service
echo [1/4] Stopping Windows Service...
net stop TrademifyAPI 2>nul
if not errorlevel 1 echo       [OK] TrademifyAPI service stopped

:: Stop Task Scheduler tasks
echo [2/4] Stopping Scheduled Tasks...
schtasks /End /TN "TrademifyBot" 2>nul
schtasks /End /TN "TrademifyWatchdog" 2>nul
if not errorlevel 1 echo       [OK] Scheduled tasks stopped

:: Kill Trademify windows (manual mode)
echo [3/4] Stopping manual processes...
taskkill /FI "WINDOWTITLE eq Trademify Backend*" /F 2>nul
taskkill /FI "WINDOWTITLE eq Trademify Frontend*" /F 2>nul
taskkill /FI "WINDOWTITLE eq Trademify Trading*" /F 2>nul

:: Also kill any lingering processes
echo [4/4] Cleaning up...
taskkill /F /IM "uvicorn.exe" 2>nul
taskkill /F /FI "WINDOWTITLE eq npm*" 2>nul

:: Wait a moment
timeout /t 2 /nobreak > nul

echo.
echo ============================================
echo    All Services Stopped!
echo ============================================
echo.
echo Stopped:
echo   - Windows Service (TrademifyAPI)
echo   - Scheduled Tasks (TrademifyBot, Watchdog)
echo   - Manual processes
echo.
pause
