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

echo ============================================
echo    Automation Status
echo ============================================

:: Check Windows Service
echo.
echo [Windows Service]
sc query TrademifyAPI 2>nul | findstr "STATE" >nul
if errorlevel 1 (
    echo   TrademifyAPI: NOT INSTALLED
) else (
    for /f "tokens=4" %%a in ('sc query TrademifyAPI ^| findstr "STATE"') do (
        echo   TrademifyAPI: %%a
    )
)

:: Check Task Scheduler
echo.
echo [Task Scheduler]
schtasks /Query /TN "TrademifyBot" 2>nul | findstr "Ready Running" >nul
if errorlevel 1 (
    echo   TrademifyBot: NOT INSTALLED
) else (
    for /f "tokens=3" %%a in ('schtasks /Query /TN "TrademifyBot" /FO LIST ^| findstr "Status"') do (
        echo   TrademifyBot: %%a
    )
)
schtasks /Query /TN "TrademifyWatchdog" 2>nul | findstr "Ready Running" >nul
if errorlevel 1 (
    echo   TrademifyWatchdog: NOT INSTALLED
) else (
    for /f "tokens=3" %%a in ('schtasks /Query /TN "TrademifyWatchdog" /FO LIST ^| findstr "Status"') do (
        echo   TrademifyWatchdog: %%a
    )
)

echo.
echo ============================================
echo    Application Status
echo ============================================
echo.

:: Check Backend using PowerShell
echo Checking Backend API (port 8000)...
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 5 -UseBasicParsing; $json = $r.Content | ConvertFrom-Json; Write-Host ('  [OK] Backend is running - ' + $json.status) -ForegroundColor Green } catch { Write-Host '  [DOWN] Backend is not responding' -ForegroundColor Red }"

:: Check Bot Status
echo.
echo Checking Trading Bot...
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/api/v1/bot/status' -TimeoutSec 5 -UseBasicParsing; $json = $r.Content | ConvertFrom-Json; Write-Host ('  [OK] Bot Status: ' + $json.status + ' | Uptime: ' + $json.uptime) -ForegroundColor Green } catch { Write-Host '  [DOWN] Bot API not responding' -ForegroundColor Red }"

:: Check Frontend
echo.
echo Checking Frontend (port 5173)...
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:5173' -TimeoutSec 5 -UseBasicParsing; Write-Host '  [OK] Frontend is running' -ForegroundColor Green } catch { Write-Host '  [DOWN] Frontend is not responding' -ForegroundColor Red }"

:: Check MT5
echo.
echo Checking MT5 Terminal...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I "terminal64.exe" >NUL
if errorlevel 1 (
    echo   [DOWN] MT5 Terminal is not running
) else (
    echo   [OK] MT5 Terminal is running
)

:: Check ports
echo.
echo ============================================
echo    Network Ports
echo ============================================
netstat -an | findstr "8000 5173" | findstr "LISTENING"

:: Show URLs
echo.
echo ============================================
echo    Access URLs
echo ============================================
echo   Frontend:  http://%IP%:5173
echo   API Docs:  http://%IP%:8000/docs
echo   Bot API:   http://%IP%:8000/api/v1/bot/status
echo ============================================
echo.

pause
