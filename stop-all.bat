@echo off
title Trademify - Stop All
color 0C
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║           🛑 TRADEMIFY STOP ALL                           ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.
echo [%TIME%] Stopping all Trademify processes...
echo.

:: Kill Python processes (API + Bot)
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Trademify*" 2>nul
taskkill /F /IM python.exe /FI "MEMUSAGE gt 50000" 2>nul

:: Also try to kill uvicorn
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000"') do (
    taskkill /F /PID %%a 2>nul
)

echo.
echo [OK] All Trademify processes stopped!
echo.
pause
