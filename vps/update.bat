@echo off
title Trademify - Update from GitHub
color 0B

echo.
echo ============================================
echo    Trademify - Update from GitHub
echo ============================================
echo.

:: Stop services first
echo Stopping services...
call C:\trademify\vps\stop-services.bat

echo.
echo Pulling latest changes from GitHub...
cd /d C:\trademify
git fetch origin
git reset --hard origin/main

echo.
echo Updating Backend dependencies...
call venv\Scripts\activate.bat
cd backend
pip install -r requirements.txt --quiet

echo.
echo Updating Frontend dependencies...
cd ..\frontend
call npm install --silent 2>nul

echo.
echo ============================================
echo    Update Complete!
echo ============================================
echo.
echo Start services with: start-services.bat
echo.

pause
