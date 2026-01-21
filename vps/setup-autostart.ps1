# ============================================
# Trademify - Windows Task Scheduler Setup
# สร้าง Scheduled Task สำหรับ Auto-Start
# ============================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Setting up Auto-Start on Boot" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Task name
$TaskName = "Trademify Auto Start"

# Remove existing task
Write-Host "Removing existing task if any..." -ForegroundColor Yellow
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Create action - run the monitor script
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"C:\trademify\vps\service-monitor.ps1`" -StartServices" `
    -WorkingDirectory "C:\trademify"

# Trigger - at startup with delay
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Trigger.Delay = "PT60S"  # 60 second delay after boot

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -RestartCount 3 `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)  # No time limit

# Principal - run as SYSTEM with highest privileges
$Principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

# Register the task
Write-Host "Creating scheduled task..." -ForegroundColor Yellow
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Starts Trademify Backend and Frontend on system boot with auto-restart monitoring"

Write-Host ""
Write-Host "OK Task '$TaskName' created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "The system will now:" -ForegroundColor Cyan
Write-Host "  - Start Trademify services 60 seconds after boot" -ForegroundColor White
Write-Host "  - Monitor services and auto-restart if they crash" -ForegroundColor White
Write-Host "  - Log all activity to C:\trademify\logs\" -ForegroundColor White
Write-Host ""

# Verify
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "Task Status: $($task.State)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "To test, you can run the task manually:" -ForegroundColor Yellow
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
Write-Host ""
