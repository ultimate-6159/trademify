@echo off
:: ============================================================
:: Trademify - Kill All Services (Double-click to run)
:: ============================================================
:: This will stop all Trademify services, Python processes, etc.
:: ============================================================

echo.
echo ============================================================
echo        TRADEMIFY - KILL ALL SERVICES
echo ============================================================
echo.

:: Run PowerShell script as Administrator
powershell -ExecutionPolicy Bypass -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0kill-all-services.ps1\"' -Verb RunAs"

exit
