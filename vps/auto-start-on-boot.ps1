# ============================================
# ?? AUTO-START ON BOOT - Trademify Trading Bot
# ============================================
# Script ??????????????? bot ???????????????????????? Windows boot
#
# ???????:
#   1. Run PowerShell as Administrator
#   2. cd C:\trademify
#   3. .\vps\auto-start-on-boot.ps1
# ============================================

$ErrorActionPreference = "Stop"

Write-Host "?? Setting up Auto-Start on Boot for Trademify..." -ForegroundColor Cyan
Write-Host ""

# Config
$TRADEMIFY_DIR = "C:\trademify"
$TASK_NAME = "TrademifyAutoStart"
$STARTUP_SCRIPT = "$TRADEMIFY_DIR\vps\startup-bot.bat"

# 1. Create startup script
Write-Host "?? Creating startup script..." -ForegroundColor Yellow

$startupContent = @"
@echo off
echo ============================================
echo   TRADEMIFY AUTO-START
echo   %date% %time%
echo ============================================

cd /d C:\trademify

REM Wait for network
echo Waiting for network...
timeout /t 10 /nobreak >nul

REM Start MT5 first (if not running)
echo Starting MT5...
start "" "C:\Program Files\Exness MT5 Terminal\terminal64.exe"
timeout /t 15 /nobreak >nul

REM Activate venv and start API
echo Starting Trademify API + Bot...
call venv\Scripts\activate.bat
cd backend
start /B python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

echo ============================================
echo   TRADEMIFY STARTED!
echo   API: http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo ============================================
"@

Set-Content -Path $STARTUP_SCRIPT -Value $startupContent -Encoding ASCII
Write-Host "? Created: $STARTUP_SCRIPT" -ForegroundColor Green

# 2. Create scheduled task for system startup
Write-Host ""
Write-Host "?? Creating Windows Scheduled Task..." -ForegroundColor Yellow

# Remove existing task if exists
$existingTask = Get-ScheduledTask -TaskName $TASK_NAME -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "   Removing existing task..." -ForegroundColor Gray
    Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false
}

# Create new task
$action = New-ScheduledTaskAction -Execute $STARTUP_SCRIPT -WorkingDirectory $TRADEMIFY_DIR
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TASK_NAME -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Auto-start Trademify Trading Bot on system boot"

Write-Host "? Scheduled Task created: $TASK_NAME" -ForegroundColor Green

# 3. Create startup shortcut in Startup folder (backup method)
Write-Host ""
Write-Host "?? Creating Startup folder shortcut..." -ForegroundColor Yellow

$startupFolder = [Environment]::GetFolderPath('Startup')
$shortcutPath = "$startupFolder\Trademify.lnk"

$WScriptShell = New-Object -ComObject WScript.Shell
$shortcut = $WScriptShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $STARTUP_SCRIPT
$shortcut.WorkingDirectory = $TRADEMIFY_DIR
$shortcut.Description = "Start Trademify Trading Bot"
$shortcut.Save()

Write-Host "? Startup shortcut created: $shortcutPath" -ForegroundColor Green

# 4. Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ?? AUTO-START CONFIGURED!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  ? Startup script: $STARTUP_SCRIPT"
Write-Host "  ? Scheduled Task: $TASK_NAME"
Write-Host "  ? Startup shortcut: $shortcutPath"
Write-Host ""
Write-Host "  ?? Bot will auto-start when Windows boots!" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To test: Restart your VPS/computer" -ForegroundColor Gray
Write-Host "  To disable: Run 'Unregister-ScheduledTask -TaskName $TASK_NAME'" -ForegroundColor Gray
Write-Host ""
