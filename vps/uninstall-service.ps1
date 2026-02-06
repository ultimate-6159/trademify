# =====================================================
# ??? TRADEMIFY UNINSTALL SCRIPT
# =====================================================
# Complete cleanup of all Trademify automation
# 
# This removes:
# - NSSM Windows Service (if installed)
# - Task Scheduler tasks
# - Watchdog and log rotation tasks
# - Generated scripts (optional)
#
# Usage:
#   .\uninstall-service.ps1           # Uninstall all
#   .\uninstall-service.ps1 -KeepLogs # Keep log files
#   .\uninstall-service.ps1 -Force    # No confirmation
# =====================================================

param(
    [switch]$KeepLogs,
    [switch]$Force
)

# Configuration
$ProjectPath = "C:\trademify"
$ServiceName = "TrademifyAPI"
$TaskNames = @("TrademifyBot", "TrademifyWatchdog", "TrademifyLogRotation")
$LogPath = "$ProjectPath\logs"
$NssmPath = "$ProjectPath\nssm\nssm.exe"

# Ensure running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "? Please run as Administrator!" -ForegroundColor Red
    exit 1
}

Write-Host "
??????????????????????????????????????????????????????????????
?     ??? TRADEMIFY UNINSTALL                                 ?
?     Complete Cleanup                                       ?
??????????????????????????????????????????????????????????????
" -ForegroundColor Yellow

# Confirmation
if (-not $Force) {
    Write-Host "?? This will remove all Trademify automation!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The following will be removed:" -ForegroundColor Cyan
    Write-Host "  • Windows Service: $ServiceName" -ForegroundColor White
    Write-Host "  • Scheduled Tasks: $($TaskNames -join ', ')" -ForegroundColor White
    if (-not $KeepLogs) {
        Write-Host "  • Log files in: $LogPath" -ForegroundColor White
    }
    Write-Host ""
    
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "? Uninstall cancelled." -ForegroundColor Red
        exit 0
    }
}

Write-Host "`n?? Starting uninstall..." -ForegroundColor Cyan

# =====================================================
# 1. Stop and Remove NSSM Windows Service
# =====================================================
Write-Host "`n?? Checking Windows Service..." -ForegroundColor Cyan

$service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue

if ($service) {
    Write-Host "  Found service: $ServiceName (Status: $($service.Status))" -ForegroundColor White
    
    # Stop service if running
    if ($service.Status -eq "Running") {
        Write-Host "  ?? Stopping service..." -ForegroundColor Yellow
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }
    
    # Remove using NSSM if available
    if (Test-Path $NssmPath) {
        Write-Host "  ??? Removing service with NSSM..." -ForegroundColor Yellow
        & $NssmPath remove $ServiceName confirm 2>&1 | Out-Null
    } else {
        # Fallback to sc.exe
        Write-Host "  ??? Removing service with sc.exe..." -ForegroundColor Yellow
        sc.exe delete $ServiceName 2>&1 | Out-Null
    }
    
    # Verify removal
    Start-Sleep -Seconds 2
    $serviceCheck = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $serviceCheck) {
        Write-Host "  ? Service removed successfully!" -ForegroundColor Green
    } else {
        Write-Host "  ?? Service may require reboot to fully remove" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ?? Service not found (not installed)" -ForegroundColor Gray
}

# =====================================================
# 2. Remove Task Scheduler Tasks
# =====================================================
Write-Host "`n?? Removing Scheduled Tasks..." -ForegroundColor Cyan

foreach ($taskName in $TaskNames) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    
    if ($task) {
        Write-Host "  Found task: $taskName (State: $($task.State))" -ForegroundColor White
        
        # Stop if running
        if ($task.State -eq "Running") {
            Write-Host "  ?? Stopping task..." -ForegroundColor Yellow
            Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        }
        
        # Remove task
        Write-Host "  ??? Removing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        
        # Verify
        $taskCheck = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if (-not $taskCheck) {
            Write-Host "  ? Task '$taskName' removed!" -ForegroundColor Green
        } else {
            Write-Host "  ?? Failed to remove task '$taskName'" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ?? Task '$taskName' not found" -ForegroundColor Gray
    }
}

# =====================================================
# 3. Remove Generated Scripts
# =====================================================
Write-Host "`n?? Removing generated scripts..." -ForegroundColor Cyan

$scriptsToRemove = @(
    "$ProjectPath\vps\start-trademify.bat",
    "$ProjectPath\vps\watchdog.ps1",
    "$ProjectPath\vps\log-rotation.ps1"
)

foreach ($script in $scriptsToRemove) {
    if (Test-Path $script) {
        Remove-Item $script -Force -ErrorAction SilentlyContinue
        Write-Host "  ? Removed: $script" -ForegroundColor Green
    }
}

# =====================================================
# 4. Clean up logs (optional)
# =====================================================
if (-not $KeepLogs) {
    Write-Host "`n?? Cleaning up logs..." -ForegroundColor Cyan
    
    if (Test-Path $LogPath) {
        $logFiles = Get-ChildItem -Path $LogPath -File -ErrorAction SilentlyContinue
        $logCount = $logFiles.Count
        $logSize = ($logFiles | Measure-Object -Property Length -Sum).Sum / 1MB
        
        if ($logCount -gt 0) {
            Write-Host "  Found $logCount log files ($('{0:N2}' -f $logSize) MB)" -ForegroundColor White
            
            # Remove all log files
            Remove-Item "$LogPath\*" -Force -Recurse -ErrorAction SilentlyContinue
            Write-Host "  ? Log files removed!" -ForegroundColor Green
        } else {
            Write-Host "  ?? No log files found" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "`n?? Keeping logs (-KeepLogs specified)" -ForegroundColor Gray
}

# =====================================================
# 5. Clean up bot state file
# =====================================================
Write-Host "`n?? Cleaning up state files..." -ForegroundColor Cyan

$stateFiles = @(
    "$ProjectPath\backend\bot_state.json",
    "$ProjectPath\backend\api\bot_state.json"
)

foreach ($stateFile in $stateFiles) {
    if (Test-Path $stateFile) {
        Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
        Write-Host "  ? Removed: $stateFile" -ForegroundColor Green
    }
}

# =====================================================
# 6. Kill any running Python processes (optional)
# =====================================================
Write-Host "`n?? Checking for running Python processes..." -ForegroundColor Cyan

$pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*trademify*"
}

if ($pythonProcesses) {
    Write-Host "  Found $($pythonProcesses.Count) Trademify Python process(es)" -ForegroundColor White
    
    foreach ($proc in $pythonProcesses) {
        Write-Host "  ?? Stopping PID: $($proc.Id)..." -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "  ? Python processes stopped!" -ForegroundColor Green
} else {
    Write-Host "  ?? No Trademify Python processes running" -ForegroundColor Gray
}

# =====================================================
# Summary
# =====================================================
Write-Host "
??????????????????????????????????????????????????????????????
?     ? UNINSTALL COMPLETE!                                 ?
??????????????????????????????????????????????????????????????
?                                                            ?
?  ??? Removed:                                               ?
?     • Windows Service: $ServiceName                        ?
?     • Scheduled Tasks: TrademifyBot, Watchdog, LogRotation ?
?     • Generated scripts                                    ?
$(if (-not $KeepLogs) { "?     • Log files                                            ?" } else { "?     • Logs: KEPT (use -KeepLogs)                           ?" })
?     • State files                                          ?
?                                                            ?
??????????????????????????????????????????????????????????????
?  ?? Note:                                                  ?
?     • Project files are NOT deleted                        ?
?     • Virtual environment is NOT deleted                   ?
?     • You can reinstall anytime using:                     ?
?       - .\install-service.ps1 (Windows Service)            ?
?       - .\setup-task-scheduler.ps1 (Task Scheduler)        ?
??????????????????????????????????????????????????????????????
" -ForegroundColor Cyan

Write-Host "?? Trademify automation has been completely removed!" -ForegroundColor Green

# Check if reboot needed
$pendingReboot = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager" -Name PendingFileRenameOperations -ErrorAction SilentlyContinue)
if ($pendingReboot) {
    Write-Host "`n?? A system reboot may be required to complete cleanup." -ForegroundColor Yellow
}
