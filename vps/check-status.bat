@echo off
title Trademify - Status Check
color 0E

echo.
echo ============================================
echo    Trademify Service Status
echo ============================================
echo.

:: Get IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set "IP=%%a"
    goto :gotip
)
:gotip
set IP=%IP: =%

echo VPS IP: %IP%
echo.

:: Check Backend using PowerShell (more reliable than curl)
echo Checking Backend (port 8000)...
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 5 -UseBasicParsing; Write-Host '  [OK] Backend is running' -ForegroundColor Green } catch { Write-Host '  [DOWN] Backend is not responding' -ForegroundColor Red }"
echo.

:: Check Frontend
echo Checking Frontend (port 5173)...
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:5173' -TimeoutSec 5 -UseBasicParsing; Write-Host '  [OK] Frontend is running' -ForegroundColor Green } catch { Write-Host '  [DOWN] Frontend is not responding' -ForegroundColor Red }"
echo.

:: Check MT5
echo Checking MT5 Terminal...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I "terminal64.exe" >NUL
if errorlevel 1 (
    echo   [DOWN] MT5 Terminal is not running
) else (
    echo   [OK] MT5 Terminal is running
)
echo.

:: Check ports
echo Checking Ports...
netstat -an | findstr "8000 5173" | findstr "LISTENING"
echo.

:: Show URLs
echo ============================================
echo Access URLs:
echo   Frontend:  http://%IP%:5173
echo   API Docs:  http://%IP%:8000/docs
echo ============================================
echo.

pause
