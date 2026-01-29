# ============================================
# ?? SETUP 24/7 MONITORING - Trademify Trading Bot
# ============================================
# Script ???????????? Task Scheduler ?????????? bot ??? 5 ????
# ??? restart ???????????? bot ?????????
#
# ???????:
#   1. Run PowerShell as Administrator
#   2. cd C:\trademify
#   3. .\vps\setup-monitoring.ps1
# ============================================

$ErrorActionPreference = "Stop"

Write-Host "?? Setting up 24/7 Monitoring for Trademify..." -ForegroundColor Cyan
Write-Host ""

# Config
$TRADEMIFY_DIR = "C:\trademify"
$MONITOR_TASK = "TrademifyMonitor"
$MONITOR_SCRIPT = "$TRADEMIFY_DIR\vps\monitor-bot.bat"

# 1. Create scheduled task for monitoring (every 5 minutes)
Write-Host "?? Creating Monitor Task (runs every 5 minutes)..." -ForegroundColor Yellow

# Remove existing task if exists
$existingTask = Get-ScheduledTask -TaskName $MONITOR_TASK -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "   Removing existing task..." -ForegroundColor Gray
    Unregister-ScheduledTask -TaskName $MONITOR_TASK -Confirm:$false
}

# Create trigger for every 5 minutes
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)

$action = New-ScheduledTaskAction -Execute $MONITOR_SCRIPT -WorkingDirectory $TRADEMIFY_DIR
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 2)

Register-ScheduledTask -TaskName $MONITOR_TASK -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Monitor Trademify Bot and restart if needed (every 5 minutes)"

Write-Host "? Monitor Task created: $MONITOR_TASK" -ForegroundColor Green

# 2. Run initial check
Write-Host ""
Write-Host "?? Running initial health check..." -ForegroundColor Yellow
& $MONITOR_SCRIPT

# 3. Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ?? 24/7 MONITORING CONFIGURED!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  ? Monitor script: $MONITOR_SCRIPT"
Write-Host "  ? Monitor Task: $MONITOR_TASK (every 5 minutes)"
Write-Host ""
Write-Host "  ?? Bot will be auto-restarted if it stops!" -ForegroundColor Yellow
Write-Host ""
Write-Host "  View logs: Get-EventLog -LogName Application -Source 'Task Scheduler'" -ForegroundColor Gray
Write-Host "  Disable: Unregister-ScheduledTask -TaskName $MONITOR_TASK" -ForegroundColor Gray
Write-Host ""
