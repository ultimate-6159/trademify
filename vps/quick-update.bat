@echo off
title Trademify - Quick Update
color 0B

echo.
echo ============================================
echo    Trademify - Quick Update (No Stop)
echo ============================================
echo.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

echo Project Directory: %PROJECT_DIR%
echo.

:: Pull latest code
echo Pulling latest changes from GitHub...
cd /d %PROJECT_DIR%
git fetch origin
git pull origin main

if %errorlevel% neq 0 (
    echo.
    echo ❌ Git pull failed!
    pause
    exit /b 1
)

echo.
echo ✅ Code updated successfully!
echo.
echo ============================================
echo    IMPORTANT: Restart Required
echo ============================================
echo.
echo To apply changes, restart the API server:
echo   1. Close any running python/uvicorn
echo   2. Run: start-services.bat
echo.
echo Or restart Windows Service if using as service.
echo.

pause
