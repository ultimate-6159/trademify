# ============================================
# Trademify - Service Monitor with Auto-Restart
# รัน Script นี้เพื่อ Monitor และ Auto-Restart
# ปรับปรุงสำหรับ Windows Server 2016+
# ============================================

param(
    [int]$CheckInterval = 30,  # Check every 30 seconds
    [switch]$StartServices     # Start services on launch
)

$ErrorActionPreference = "SilentlyContinue"

# Configuration
$BackendUrl = "http://localhost:8000/health"
$FrontendUrl = "http://localhost:5173"
$LogPath = "C:\trademify\logs"
$MaxRestarts = 10
$RestartCooldown = 60  # seconds between restarts
$MT5Path = "C:\Program Files\MetaTrader 5\terminal64.exe"

# State tracking
$BackendRestarts = 0
$FrontendRestarts = 0
$LastBackendRestart = [DateTime]::MinValue
$LastFrontendRestart = [DateTime]::MinValue

# Create logs directory
if (!(Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
}

$MonitorLog = Join-Path $LogPath "monitor.log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    
    # Console output with colors
    switch ($Level) {
        "ERROR" { Write-Host $LogMessage -ForegroundColor Red }
        "WARN"  { Write-Host $LogMessage -ForegroundColor Yellow }
        "OK"    { Write-Host $LogMessage -ForegroundColor Green }
        default { Write-Host $LogMessage }
    }
    
    # File output
    Add-Content -Path $MonitorLog -Value $LogMessage
}

function Test-Backend {
    try {
        $response = Invoke-WebRequest -Uri $BackendUrl -TimeoutSec 5 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-Frontend {
    try {
        $response = Invoke-WebRequest -Uri $FrontendUrl -TimeoutSec 5 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-Backend {
    Write-Log "Starting Backend..." "WARN"
    
    # Kill any existing python processes using the port
    $existingPID = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess | Select-Object -First 1
    if ($existingPID) {
        Stop-Process -Id $existingPID -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    
    $process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c cd /d C:\trademify && call venv\Scripts\activate.bat && cd backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 2>&1 >> C:\trademify\logs\backend.log" `
        -WindowStyle Hidden `
        -PassThru
    
    Start-Sleep -Seconds 10
    
    if (Test-Backend) {
        Write-Log "Backend started successfully (PID: $($process.Id))" "OK"
        return $true
    } else {
        Write-Log "Backend failed to start" "ERROR"
        return $false
    }
}

function Start-Frontend {
    Write-Log "Starting Frontend..." "WARN"
    
    # Kill any existing node processes using the port
    $existingPID = (Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue).OwningProcess | Select-Object -First 1
    if ($existingPID) {
        Stop-Process -Id $existingPID -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    
    $process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c cd /d C:\trademify\frontend && npm run dev -- --host 0.0.0.0 2>&1 >> C:\trademify\logs\frontend.log" `
        -WindowStyle Hidden `
        -PassThru
    
    Start-Sleep -Seconds 8
    
    if (Test-Frontend) {
        Write-Log "Frontend started successfully (PID: $($process.Id))" "OK"
        return $true
    } else {
        Write-Log "Frontend may still be starting..." "WARN"
        return $true
    }
}

function Start-MT5 {
    $mt5Process = Get-Process "terminal64" -ErrorAction SilentlyContinue
    if ($mt5Process) {
        Write-Log "MT5 already running (PID: $($mt5Process.Id))" "OK"
        return $true
    }
    
    if (Test-Path $MT5Path) {
        Write-Log "Starting MT5 Terminal..." "WARN"
        Start-Process -FilePath $MT5Path -ArgumentList "/portable"
        Start-Sleep -Seconds 5
        
        $mt5Process = Get-Process "terminal64" -ErrorAction SilentlyContinue
        if ($mt5Process) {
            Write-Log "MT5 started (PID: $($mt5Process.Id))" "OK"
            return $true
        }
    } else {
        Write-Log "MT5 not found at: $MT5Path" "WARN"
    }
    return $false
}
        Write-Log "Frontend may still be starting..." "WARN"
        return $true
    }
}

# Header
Clear-Host
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Trademify Service Monitor" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop monitoring" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Service Monitor started"
Write-Log "Check interval: ${CheckInterval}s"

# Start services if requested
if ($StartServices) {
    Write-Log "Starting services..."
    Start-MT5
    Start-Backend
    Start-Frontend
}

# Main monitoring loop
while ($true) {
    $now = Get-Date
    
    # Check Backend
    if (!(Test-Backend)) {
        $timeSinceRestart = ($now - $LastBackendRestart).TotalSeconds
        
        if ($BackendRestarts -lt $MaxRestarts -and $timeSinceRestart -gt $RestartCooldown) {
            Write-Log "Backend is DOWN! Restarting... (attempt $($BackendRestarts + 1)/$MaxRestarts)" "ERROR"
            
            # Kill any zombie processes
            Get-Process -Name "python" -ErrorAction SilentlyContinue | 
                Where-Object { $_.MainWindowTitle -like "*uvicorn*" } | 
                Stop-Process -Force
            
            Start-Sleep -Seconds 2
            
            if (Start-Backend) {
                $BackendRestarts++
                $LastBackendRestart = $now
            }
        } elseif ($BackendRestarts -ge $MaxRestarts) {
            Write-Log "Backend max restarts reached! Manual intervention required." "ERROR"
        }
    } else {
        # Reset restart counter after 5 minutes of stability
        if ($BackendRestarts -gt 0 -and ($now - $LastBackendRestart).TotalMinutes -gt 5) {
            $BackendRestarts = 0
            Write-Log "Backend stable, reset restart counter" "OK"
        }
    }
    
    # Check Frontend
    if (!(Test-Frontend)) {
        $timeSinceRestart = ($now - $LastFrontendRestart).TotalSeconds
        
        if ($FrontendRestarts -lt $MaxRestarts -and $timeSinceRestart -gt $RestartCooldown) {
            Write-Log "Frontend is DOWN! Restarting... (attempt $($FrontendRestarts + 1)/$MaxRestarts)" "ERROR"
            
            # Kill any zombie processes
            Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
            
            Start-Sleep -Seconds 2
            
            if (Start-Frontend) {
                $FrontendRestarts++
                $LastFrontendRestart = $now
            }
        } elseif ($FrontendRestarts -ge $MaxRestarts) {
            Write-Log "Frontend max restarts reached! Manual intervention required." "ERROR"
        }
    } else {
        # Reset restart counter after 5 minutes of stability
        if ($FrontendRestarts -gt 0 -and ($now - $LastFrontendRestart).TotalMinutes -gt 5) {
            $FrontendRestarts = 0
            Write-Log "Frontend stable, reset restart counter" "OK"
        }
    }
    
    # Status update every 5 checks
    if ((Get-Date).Second -lt $CheckInterval) {
        $backendStatus = if (Test-Backend) { "UP" } else { "DOWN" }
        $frontendStatus = if (Test-Frontend) { "UP" } else { "DOWN" }
        Write-Log "Status - Backend: $backendStatus, Frontend: $frontendStatus"
    }
    
    Start-Sleep -Seconds $CheckInterval
}
