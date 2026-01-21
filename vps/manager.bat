@echo off
title Trademify Manager
color 0B

echo.
echo ╔══════════════════════════════════════════════╗
echo ║     Trademify - Service Manager              ║
echo ║     Windows VPS Control Panel                ║
echo ╚══════════════════════════════════════════════╝
echo.

:menu
echo  [1] Start Services
echo  [2] Stop Services  
echo  [3] Restart Services
echo  [4] Check Status
echo  [5] View Logs
echo  [6] Install Auto-Start
echo  [7] Update Code
echo  [8] Open Dashboard
echo  [9] Exit
echo.

set /p choice="Select option (1-9): "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto restart
if "%choice%"=="4" goto status
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto install
if "%choice%"=="7" goto update
if "%choice%"=="8" goto dashboard
if "%choice%"=="9" goto end

echo Invalid option!
goto menu

:start
powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\trademify.ps1" start
pause
goto menu

:stop
powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\trademify.ps1" stop
pause
goto menu

:restart
powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\trademify.ps1" restart
pause
goto menu

:status
powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\trademify.ps1" status
pause
goto menu

:logs
powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\trademify.ps1" logs
pause
goto menu

:install
powershell -ExecutionPolicy Bypass -File "C:\trademify\vps\trademify.ps1" install
pause
goto menu

:update
echo.
echo Updating Trademify...
cd /d C:\trademify
git pull origin main
echo.
echo Update complete! Please restart services.
pause
goto menu

:dashboard
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set "IP=%%a"
    goto :gotip
)
:gotip
set IP=%IP: =%
start http://%IP%:5173/enhanced
goto menu

:end
exit
