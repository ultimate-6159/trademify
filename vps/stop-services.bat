@echo off
title Trademify - Stopping Services
color 0C

echo.
echo ============================================
echo    Trademify - Stopping All Services
echo ============================================
echo.

:: Kill Trademify windows
echo Stopping Backend...
taskkill /FI "WINDOWTITLE eq Trademify Backend*" /F 2>nul

echo Stopping Frontend...
taskkill /FI "WINDOWTITLE eq Trademify Frontend*" /F 2>nul

echo Stopping Trading Bot...
taskkill /FI "WINDOWTITLE eq Trademify Trading*" /F 2>nul

:: Also kill any lingering processes
taskkill /F /IM "uvicorn.exe" 2>nul
taskkill /F /FI "WINDOWTITLE eq npm*" 2>nul

:: Wait a moment
timeout /t 2 /nobreak > nul

echo.
echo ============================================
echo    All Services Stopped!
echo ============================================
echo.
pause
