@echo off
title Trademify - Starting Services
color 0A

echo.
echo ============================================
echo    Trademify - Starting All Services
echo ============================================
echo.
echo Choose startup method:
echo   1. Windows Service (NSSM) - Recommended
echo   2. Task Scheduler
echo   3. Manual (legacy)
echo.
set /p METHOD="Enter choice (1/2/3): "

:: Refresh PATH
set "PATH=%PATH%;C:\Program Files\nodejs;C:\Python311;C:\Python311\Scripts"

:: Get VPS IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set "IP=%%a"
    goto :gotip
)
:gotip
set IP=%IP: =%

echo.
echo Your VPS IP: %IP%
echo.

:: Create logs directory
if not exist "C:\trademify\logs" mkdir "C:\trademify\logs"

if "%METHOD%"=="1" goto :service
if "%METHOD%"=="2" goto :scheduler
if "%METHOD%"=="3" goto :manual
goto :manual

:service
echo [Windows Service] Starting TrademifyAPI...
net start TrademifyAPI 2>nul
if errorlevel 1 (
    echo   Service not installed. Installing now...
    powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\install-service.ps1"
) else (
    echo   [OK] TrademifyAPI service started!
)
goto :checkhealth

:scheduler
echo [Task Scheduler] Starting TrademifyBot...
schtasks /Run /TN "TrademifyBot" 2>nul
if errorlevel 1 (
    echo   Task not found. Installing now...
    powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\setup-task-scheduler.ps1"
) else (
    echo   [OK] TrademifyBot task started!
)
goto :checkhealth

:manual
echo [Manual] Starting services in windows...

:: Kill any existing processes first
echo Stopping any existing services...
taskkill /F /FI "WINDOWTITLE eq Trademify*" 2>nul
timeout /t 2 /nobreak > nul

:: Start Backend API
echo [1/2] Starting Backend API on port 8000...
start "Trademify Backend" /MIN cmd /c "cd /d C:\trademify && call venv\Scripts\activate.bat && cd backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: Wait for backend
echo       Waiting for Backend to start...
timeout /t 10 /nobreak > nul

:: Verify backend is running
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 5 -UseBasicParsing; Write-Host '       [OK] Backend is running' -ForegroundColor Green } catch { Write-Host '       [WARN] Backend may still be starting' -ForegroundColor Yellow }"

:: Start Frontend
echo [2/2] Starting Frontend on port 5173...
start "Trademify Frontend" /MIN cmd /c "cd /d C:\trademify\frontend && npm run dev -- --host 0.0.0.0"

:: Wait for frontend
echo       Waiting for Frontend to start...
timeout /t 8 /nobreak > nul
goto :done

:checkhealth
echo.
echo Waiting for services to start...
timeout /t 10 /nobreak > nul

:: Check API health
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 10 -UseBasicParsing; Write-Host '  [OK] API is healthy!' -ForegroundColor Green } catch { Write-Host '  [WARN] API still starting...' -ForegroundColor Yellow }"

:done
echo.
echo ============================================
echo    All Services Started Successfully!
echo ============================================
echo.
echo Access URLs:
echo   Frontend:  http://%IP%:5173
echo   API Docs:  http://%IP%:8000/docs
echo   Bot API:   http://%IP%:8000/api/v1/bot/status
echo.
echo Mode Options:
echo   1 = Windows Service (auto-start on boot)
echo   2 = Task Scheduler (auto-start + watchdog)
echo   3 = Manual (runs in windows)
echo.
echo Use stop-services.bat to stop all services.
echo Use check-status.bat to view detailed status.
echo.

pause
exit /b 0
