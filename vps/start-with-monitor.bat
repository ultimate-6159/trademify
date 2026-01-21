@echo off
title Trademify - Service Monitor
color 0B

echo.
echo ============================================
echo    Trademify Service Monitor
echo ============================================
echo.
echo This will monitor Backend and Frontend
echo and auto-restart them if they crash.
echo.
echo Press Ctrl+C to stop monitoring.
echo.

cd /d C:\trademify\vps
powershell -ExecutionPolicy Bypass -File "service-monitor.ps1" -StartServices

pause
