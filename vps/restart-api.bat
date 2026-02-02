@echo off
title Trademify - Restart API Only
color 0E

echo.
echo ============================================
echo    Trademify - Restart API Server
echo ============================================
echo.

:: Refresh PATH
set "PATH=%PATH%;C:\Python311;C:\Python311\Scripts"

:: Kill existing API process
echo [1/3] Stopping existing API server...
taskkill /F /FI "WINDOWTITLE eq Trademify Backend*" 2>nul
timeout /t 3 /nobreak > nul

:: Start Backend API
echo [2/3] Starting Backend API on port 8000...
start "Trademify Backend" /MIN cmd /c "cd /d C:\trademify && call venv\Scripts\activate.bat && cd backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: Wait for backend
echo [3/3] Waiting for Backend to start...
timeout /t 10 /nobreak > nul

:: Verify backend is running
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 5 -UseBasicParsing; Write-Host '[OK] API Server restarted successfully!' -ForegroundColor Green } catch { Write-Host '[WARN] API may still be starting...' -ForegroundColor Yellow }"

echo.
echo ============================================
echo    API Restart Complete!
echo ============================================
echo.
pause
