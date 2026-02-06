# =====================================================
# ?? TRADEMIFY TASK SCHEDULER SETUP
# =====================================================
# Alternative to Windows Service - Simpler setup!
# 
# Features:
# - Start on Windows boot (no login required)
# - Auto-restart on crash (every 1 minute check)
# - Run as SYSTEM account
# - Daily restart for stability
#
# Usage:
#   .\setup-task-scheduler.ps1           # Install
#   .\setup-task-scheduler.ps1 -Remove   # Uninstall
# =====================================================

param(
    [switch]$Remove
)

# Configuration
$TaskName = "TrademifyBot"
$TaskDescription = "Trademify AI Trading Bot - 10 Year Autonomous Operation"
$ProjectPath = "C:\trademify"
$PythonPath = "$ProjectPath\venv\Scripts\python.exe"
$ApiScript = "$ProjectPath\backend\api\main.py"
$LogPath = "$ProjectPath\logs"

# Ensure running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "? Please run as Administrator!" -ForegroundColor Red
    exit 1
}

# Create logs directory
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
    Write-Host "?? Created logs directory: $LogPath" -ForegroundColor Green
}

# Remove existing task if requested or before recreating
if ($Remove -or (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
    Write-Host "??? Removing existing task: $TaskName" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    
    if ($Remove) {
        Write-Host "? Task removed successfully!" -ForegroundColor Green
        exit 0
    }
}

Write-Host "
??????????????????????????????????????????????????????????????
?     ?? TRADEMIFY TASK SCHEDULER SETUP                      ?
?     10-50 Year Autonomous Operation                        ?
??????????????????????????????????????????????????????????????
" -ForegroundColor Cyan

# =====================================================
# Create startup script
# =====================================================
$StartupScript = @"
@echo off
cd /d $ProjectPath\backend
call $ProjectPath\venv\Scripts\activate.bat
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 >> "$LogPath\trademify_%date:~-4,4%%date:~-7,2%%date:~-10,2%.log" 2>&1
"@

$StartupScriptPath = "$ProjectPath\vps\start-trademify.bat"
$StartupScript | Out-File -FilePath $StartupScriptPath -Encoding ASCII
Write-Host "?? Created startup script: $StartupScriptPath" -ForegroundColor Green

# =====================================================
# Create Task Scheduler Task
# =====================================================
Write-Host "`n?? Creating scheduled task..." -ForegroundColor Cyan

# Task action - run the startup script
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$StartupScriptPath`"" `
    -WorkingDirectory "$ProjectPath\backend"

# Trigger 1: At system startup
$TriggerStartup = New-ScheduledTaskTrigger -AtStartup

# Trigger 2: Daily restart at 3 AM for stability
$TriggerDaily = New-ScheduledTaskTrigger -Daily -At "03:00"

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365) `
    -MultipleInstances IgnoreNew

# Principal - Run as SYSTEM with highest privileges
$Principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $Action `
        -Trigger $TriggerStartup, $TriggerDaily `
        -Settings $Settings `
        -Principal $Principal `
        -Force | Out-Null
    
    Write-Host "? Task created successfully!" -ForegroundColor Green
}
catch {
    Write-Host "? Failed to create task: $_" -ForegroundColor Red
    exit 1
}

# =====================================================
# Create Watchdog Task (checks every 5 minutes)
# =====================================================
$WatchdogTaskName = "TrademifyWatchdog"
$WatchdogScript = @"
`$ErrorActionPreference = 'SilentlyContinue'
`$response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 10 -UseBasicParsing
if (`$response.StatusCode -ne 200) {
    # Restart the main task
    Stop-ScheduledTask -TaskName "$TaskName" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
    Start-ScheduledTask -TaskName "$TaskName"
    Add-Content -Path "$LogPath\watchdog.log" -Value "`$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Restarted bot (health check failed)"
}
"@

$WatchdogScriptPath = "$ProjectPath\vps\watchdog.ps1"
$WatchdogScript | Out-File -FilePath $WatchdogScriptPath -Encoding UTF8
Write-Host "?? Created watchdog script: $WatchdogScriptPath" -ForegroundColor Green

# Remove existing watchdog task
Unregister-ScheduledTask -TaskName $WatchdogTaskName -Confirm:$false -ErrorAction SilentlyContinue

# Create watchdog task
$WatchdogAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$WatchdogScriptPath`""

$WatchdogTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)  # 10 years!

$WatchdogSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask `
    -TaskName $WatchdogTaskName `
    -Description "Trademify Watchdog - Monitors and restarts bot if needed" `
    -Action $WatchdogAction `
    -Trigger $WatchdogTrigger `
    -Settings $WatchdogSettings `
    -Principal $Principal `
    -Force | Out-Null

Write-Host "? Watchdog task created!" -ForegroundColor Green

# =====================================================
# Create Log Rotation Task (daily at 4 AM)
# =====================================================
$LogRotationTaskName = "TrademifyLogRotation"
$LogRotationScript = @"
`$LogPath = "$LogPath"
`$MaxAgeDays = 30
`$MaxSizeMB = 100

# Delete logs older than 30 days
Get-ChildItem -Path `$LogPath -Filter "*.log" | Where-Object {
    `$_.LastWriteTime -lt (Get-Date).AddDays(-`$MaxAgeDays)
} | Remove-Item -Force

# Compress logs larger than 100MB
Get-ChildItem -Path `$LogPath -Filter "*.log" | Where-Object {
    (`$_.Length / 1MB) -gt `$MaxSizeMB
} | ForEach-Object {
    `$zipPath = `$_.FullName -replace '\.log$', '.zip'
    Compress-Archive -Path `$_.FullName -DestinationPath `$zipPath -Force
    Remove-Item `$_.FullName -Force
}

Add-Content -Path "`$LogPath\maintenance.log" -Value "`$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Log rotation completed"
"@

$LogRotationScriptPath = "$ProjectPath\vps\log-rotation.ps1"
$LogRotationScript | Out-File -FilePath $LogRotationScriptPath -Encoding UTF8
Write-Host "?? Created log rotation script: $LogRotationScriptPath" -ForegroundColor Green

# Remove existing log rotation task
Unregister-ScheduledTask -TaskName $LogRotationTaskName -Confirm:$false -ErrorAction SilentlyContinue

$LogRotationAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$LogRotationScriptPath`""

$LogRotationTrigger = New-ScheduledTaskTrigger -Daily -At "04:00"

Register-ScheduledTask `
    -TaskName $LogRotationTaskName `
    -Description "Trademify Log Rotation - Cleans old logs daily" `
    -Action $LogRotationAction `
    -Trigger $LogRotationTrigger `
    -Settings $WatchdogSettings `
    -Principal $Principal `
    -Force | Out-Null

Write-Host "? Log rotation task created!" -ForegroundColor Green

# =====================================================
# Start the main task now
# =====================================================
Write-Host "`n?? Starting Trademify..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $TaskName

# Wait and verify
Start-Sleep -Seconds 10
$task = Get-ScheduledTask -TaskName $TaskName
if ($task.State -eq "Running") {
    Write-Host "? Trademify is running!" -ForegroundColor Green
} else {
    Write-Host "?? Task state: $($task.State)" -ForegroundColor Yellow
}

# =====================================================
# Summary
# =====================================================
Write-Host "
??????????????????????????????????????????????????????????????
?     ? SETUP COMPLETE!                                     ?
??????????????????????????????????????????????????????????????
?                                                            ?
?  ?? Tasks Created:                                         ?
?     • $TaskName - Main bot (runs at startup)               ?
?     • $WatchdogTaskName - Health monitor (every 5 min)     ?
?     • $LogRotationTaskName - Log cleanup (daily 4 AM)      ?
?                                                            ?
?  ?? Features:                                              ?
?     • Auto-start on Windows boot                           ?
?     • Auto-restart on crash (1 min retry)                  ?
?     • Daily restart at 3 AM for stability                  ?
?     • Log rotation (30 days retention)                     ?
?     • Health monitoring every 5 minutes                    ?
?                                                            ?
?  ?? Logs: $LogPath                                         ?
?                                                            ?
?  ?? API: http://localhost:8000                             ?
?  ?? Docs: http://localhost:8000/docs                       ?
?                                                            ?
??????????????????????????????????????????????????????????????
?  Commands:                                                 ?
?    Start:  Start-ScheduledTask -TaskName $TaskName         ?
?    Stop:   Stop-ScheduledTask -TaskName $TaskName          ?
?    Status: Get-ScheduledTask -TaskName $TaskName           ?
?    Remove: .\setup-task-scheduler.ps1 -Remove              ?
??????????????????????????????????????????????????????????????
" -ForegroundColor Cyan

# Test health endpoint
Write-Host "`n?? Testing health endpoint..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 30 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "? API is healthy!" -ForegroundColor Green
    }
}
catch {
    Write-Host "? API is starting up... (check in a few seconds)" -ForegroundColor Yellow
}

Write-Host "`n?? Trademify will now run automatically for 10-50 years!" -ForegroundColor Green
