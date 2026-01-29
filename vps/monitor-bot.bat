@echo off
REM ============================================
REM   TRADEMIFY 24/7 MONITOR - Keep Bot Alive!
REM ============================================
REM   Script ??????????????? restart bot ????????????
REM   ?????? 5 ???????? Task Scheduler
REM ============================================

echo [%date% %time%] Checking Trademify status...

cd /d C:\trademify

REM Check if API is responding
curl -s -o nul -w "%%{http_code}" http://localhost:8000/health > temp_status.txt 2>&1
set /p STATUS=<temp_status.txt
del temp_status.txt

if "%STATUS%"=="200" (
    echo [%date% %time%] ? API is healthy
    
    REM Check bot status
    curl -s http://localhost:8000/api/v1/unified/status > temp_bot.txt 2>&1
    findstr /C:"\"running\": true" temp_bot.txt >nul
    if %ERRORLEVEL%==0 (
        echo [%date% %time%] ? Bot is running
    ) else (
        echo [%date% %time%] ?? Bot is stopped - Restarting...
        curl -X POST "http://localhost:8000/api/v1/unified/start" -H "Content-Type: application/json" -d "{\"mode\":\"auto\",\"symbols\":\"XAUUSDm\",\"timeframe\":\"H1\",\"signal_mode\":\"technical\",\"quality\":\"MEDIUM\",\"interval\":60}"
    )
    del temp_bot.txt
) else (
    echo [%date% %time%] ? API not responding (Status: %STATUS%) - Restarting...
    
    REM Kill existing processes
    taskkill /F /IM python.exe 2>nul
    
    REM Wait a bit
    timeout /t 5 /nobreak >nul
    
    REM Start API
    cd backend
    start /B cmd /c "call ..\venv\Scripts\activate.bat && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
    
    echo [%date% %time%] ?? API restart initiated
)

echo [%date% %time%] Check complete.
