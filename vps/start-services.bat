@echo off
title Trademify - Starting Services
color 0A

echo.
echo ============================================
echo    Trademify - Starting All Services
echo ============================================
echo.

:: Refresh PATH to ensure node/npm are available
set "PATH=%PATH%;C:\Program Files\nodejs;C:\Python311;C:\Python311\Scripts"

:: Get VPS IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set "IP=%%a"
    goto :gotip
)
:gotip
set IP=%IP: =%

echo Your VPS IP: %IP%
echo.

:: Create logs directory
if not exist "C:\trademify\logs" mkdir "C:\trademify\logs"

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
echo       [OK] Frontend started

echo.
echo ============================================
echo    All Services Started Successfully!
echo ============================================
echo.
echo Access URLs:
echo   Frontend:  http://%IP%:5173
echo   API Docs:  http://%IP%:8000/docs
echo.
echo Services are running in minimized windows.
echo Use stop-services.bat to stop all services.
echo.

:: Open browser
start http://localhost:5173

exit /b 0
