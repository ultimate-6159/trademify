# ============================================
# Trademify - Ultimate Service Manager
# จัดการ Services แบบครบวงจร + Health Check
# สำหรับ Windows Server 2016+
# ============================================

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "install")]
    [string]$Action = "status"
)

$ErrorActionPreference = "SilentlyContinue"

# Configuration
$Config = @{
    RootPath = "C:\trademify"
    BackendPort = 8000
    FrontendPort = 5173
    HealthUrl = "http://localhost:8000/health"
    LogPath = "C:\trademify\logs"
    MT5Path = "C:\Program Files\MetaTrader 5\terminal64.exe"
}

# Colors
function Write-Success { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "❌ $msg" -ForegroundColor Red }
function Write-Warn { param($msg) Write-Host "⚠️ $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "ℹ️ $msg" -ForegroundColor Cyan }

# ============================================
# Functions
# ============================================

function Test-ServiceHealth {
    param([string]$Url, [int]$Timeout = 5)
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $Timeout -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Get-ProcessByPort {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($connection) {
        return Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    }
    return $null
}

function Stop-ServiceByPort {
    param([int]$Port, [string]$Name)
    $process = Get-ProcessByPort -Port $Port
    if ($process) {
        Write-Warn "Stopping $Name (PID: $($process.Id))..."
        Stop-Process -Id $process.Id -Force
        Start-Sleep -Seconds 2
        return $true
    }
    return $false
}

function Start-Backend {
    Write-Info "Starting Backend on port $($Config.BackendPort)..."
    
    # Check if already running
    if (Test-ServiceHealth -Url $Config.HealthUrl) {
        Write-Success "Backend already running"
        return $true
    }
    
    # Kill any zombie process
    Stop-ServiceByPort -Port $Config.BackendPort -Name "Backend"
    
    # Start new process
    $startInfo = @{
        FilePath = "cmd.exe"
        ArgumentList = "/c cd /d $($Config.RootPath) && call venv\Scripts\activate.bat && cd backend && python -m uvicorn api.main:app --host 0.0.0.0 --port $($Config.BackendPort) 2>&1 >> $($Config.LogPath)\backend.log"
        WindowStyle = "Hidden"
        PassThru = $true
    }
    
    $process = Start-Process @startInfo
    
    # Wait and verify
    Write-Info "Waiting for Backend to start..."
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 2
        if (Test-ServiceHealth -Url $Config.HealthUrl) {
            Write-Success "Backend started (PID: $($process.Id))"
            return $true
        }
    }
    
    Write-Error "Backend failed to start - check logs"
    return $false
}

function Start-Frontend {
    Write-Info "Starting Frontend on port $($Config.FrontendPort)..."
    
    # Check if already running
    $frontendProcess = Get-ProcessByPort -Port $Config.FrontendPort
    if ($frontendProcess) {
        Write-Success "Frontend already running (PID: $($frontendProcess.Id))"
        return $true
    }
    
    # Start new process
    $startInfo = @{
        FilePath = "cmd.exe"
        ArgumentList = "/c cd /d $($Config.RootPath)\frontend && npm run dev -- --host 0.0.0.0 2>&1 >> $($Config.LogPath)\frontend.log"
        WindowStyle = "Hidden"
        PassThru = $true
    }
    
    $process = Start-Process @startInfo
    
    # Wait
    Start-Sleep -Seconds 5
    
    $frontendProcess = Get-ProcessByPort -Port $Config.FrontendPort
    if ($frontendProcess) {
        Write-Success "Frontend started (PID: $($frontendProcess.Id))"
        return $true
    }
    
    Write-Error "Frontend failed to start - check logs"
    return $false
}

function Start-MT5 {
    Write-Info "Checking MT5 Terminal..."
    
    $mt5Process = Get-Process "terminal64" -ErrorAction SilentlyContinue
    if ($mt5Process) {
        Write-Success "MT5 already running (PID: $($mt5Process.Id))"
        return $true
    }
    
    if (Test-Path $Config.MT5Path) {
        Write-Warn "Starting MT5 Terminal..."
        Start-Process -FilePath $Config.MT5Path -ArgumentList "/portable"
        Start-Sleep -Seconds 5
        
        $mt5Process = Get-Process "terminal64" -ErrorAction SilentlyContinue
        if ($mt5Process) {
            Write-Success "MT5 started (PID: $($mt5Process.Id))"
            return $true
        }
    }
    
    Write-Error "MT5 not found at: $($Config.MT5Path)"
    return $false
}

function Stop-AllServices {
    Write-Info "Stopping all services..."
    
    Stop-ServiceByPort -Port $Config.BackendPort -Name "Backend"
    Stop-ServiceByPort -Port $Config.FrontendPort -Name "Frontend"
    
    # Also kill by window title
    taskkill /F /FI "WINDOWTITLE eq Trademify*" 2>$null
    
    Write-Success "All services stopped"
}

function Show-Status {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║       Trademify Service Status               ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    
    # Backend
    $backendOk = Test-ServiceHealth -Url $Config.HealthUrl
    $backendProcess = Get-ProcessByPort -Port $Config.BackendPort
    if ($backendOk) {
        Write-Host "  Backend API    : " -NoNewline
        Write-Host "🟢 Running" -ForegroundColor Green -NoNewline
        if ($backendProcess) { Write-Host " (PID: $($backendProcess.Id))" -ForegroundColor Gray }
        else { Write-Host "" }
    } else {
        Write-Host "  Backend API    : " -NoNewline
        Write-Host "🔴 Stopped" -ForegroundColor Red
    }
    
    # Frontend
    $frontendProcess = Get-ProcessByPort -Port $Config.FrontendPort
    if ($frontendProcess) {
        Write-Host "  Frontend       : " -NoNewline
        Write-Host "🟢 Running" -ForegroundColor Green -NoNewline
        Write-Host " (PID: $($frontendProcess.Id))" -ForegroundColor Gray
    } else {
        Write-Host "  Frontend       : " -NoNewline
        Write-Host "🔴 Stopped" -ForegroundColor Red
    }
    
    # MT5
    $mt5Process = Get-Process "terminal64" -ErrorAction SilentlyContinue
    if ($mt5Process) {
        Write-Host "  MT5 Terminal   : " -NoNewline
        Write-Host "🟢 Running" -ForegroundColor Green -NoNewline
        Write-Host " (PID: $($mt5Process.Id))" -ForegroundColor Gray
    } else {
        Write-Host "  MT5 Terminal   : " -NoNewline
        Write-Host "🔴 Stopped" -ForegroundColor Red
    }
    
    Write-Host ""
    
    # Get VPS IP
    $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" } | Select-Object -First 1).IPAddress
    
    Write-Host "  URLs:" -ForegroundColor Yellow
    Write-Host "    Dashboard : http://${ip}:$($Config.FrontendPort)" -ForegroundColor Gray
    Write-Host "    API       : http://${ip}:$($Config.BackendPort)" -ForegroundColor Gray
    Write-Host "    Docs      : http://${ip}:$($Config.BackendPort)/docs" -ForegroundColor Gray
    Write-Host ""
}

function Show-Logs {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Recent Logs (last 30 lines)" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
    
    $tradingLog = "$($Config.RootPath)\backend\logs\trading_bot.log"
    if (Test-Path $tradingLog) {
        Write-Host ""
        Write-Host "[Trading Bot Log]" -ForegroundColor Yellow
        Get-Content $tradingLog -Tail 30
    }
    
    $monitorLog = "$($Config.LogPath)\monitor.log"
    if (Test-Path $monitorLog) {
        Write-Host ""
        Write-Host "[Monitor Log]" -ForegroundColor Yellow
        Get-Content $monitorLog -Tail 20
    }
}

function Install-AutoStart {
    Write-Info "Installing Auto-Start Task..."
    
    $TaskName = "Trademify Auto Start"
    
    # Remove existing
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    
    # Create action
    $Action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$($Config.RootPath)\vps\service-monitor.ps1`" -StartServices" `
        -WorkingDirectory $Config.RootPath
    
    # Trigger at startup
    $Trigger = New-ScheduledTaskTrigger -AtStartup
    $Trigger.Delay = "PT60S"
    
    # Settings
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -RestartCount 3 `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0)
    
    # Principal
    $Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    
    # Register
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal
    
    Write-Success "Auto-Start Task installed"
    Write-Info "Trademify will start automatically 60 seconds after Windows boot"
}

# ============================================
# Main
# ============================================

# Ensure logs directory exists
if (!(Test-Path $Config.LogPath)) {
    New-Item -ItemType Directory -Path $Config.LogPath -Force | Out-Null
}

switch ($Action) {
    "start" {
        Write-Host ""
        Write-Host "🚀 Starting Trademify Services..." -ForegroundColor Cyan
        Write-Host ""
        
        Start-MT5
        Start-Backend
        Start-Frontend
        
        Write-Host ""
        Show-Status
    }
    
    "stop" {
        Write-Host ""
        Write-Host "🛑 Stopping Trademify Services..." -ForegroundColor Cyan
        Write-Host ""
        
        Stop-AllServices
    }
    
    "restart" {
        Write-Host ""
        Write-Host "🔄 Restarting Trademify Services..." -ForegroundColor Cyan
        Write-Host ""
        
        Stop-AllServices
        Start-Sleep -Seconds 3
        Start-MT5
        Start-Backend
        Start-Frontend
        
        Write-Host ""
        Show-Status
    }
    
    "status" {
        Show-Status
    }
    
    "logs" {
        Show-Logs
    }
    
    "install" {
        Install-AutoStart
    }
}
