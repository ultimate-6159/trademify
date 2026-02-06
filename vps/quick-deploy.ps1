# =====================================================
# ?? TRADEMIFY QUICK DEPLOY
# =====================================================
# One-click production deployment script
# 
# This script:
# 1. Validates environment
# 2. Installs automation (Service or Scheduler)
# 3. Verifies everything is running
# 4. Opens browser to check
#
# Usage:
#   .\quick-deploy.ps1                    # Interactive
#   .\quick-deploy.ps1 -Mode Service      # Windows Service
#   .\quick-deploy.ps1 -Mode Scheduler    # Task Scheduler
# =====================================================

param(
    [ValidateSet("Service", "Scheduler", "")]
    [string]$Mode = ""
)

$ErrorActionPreference = "Stop"
$ProjectPath = "C:\trademify"

# Ensure running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "? Please run as Administrator!" -ForegroundColor Red
    Write-Host "   Right-click PowerShell ? Run as Administrator" -ForegroundColor Yellow
    exit 1
}

Write-Host "
??????????????????????????????????????????????????????????????
?     ?? TRADEMIFY QUICK DEPLOY                              ?
?     Production Ready in Minutes                            ?
??????????????????????????????????????????????????????????????
" -ForegroundColor Cyan

# =====================================================
# 1. Environment Validation
# =====================================================
Write-Host "?? Validating environment..." -ForegroundColor Yellow
Write-Host ""

$checks = @()

# Check Python
$pythonPath = "$ProjectPath\venv\Scripts\python.exe"
if (Test-Path $pythonPath) {
    $pythonVersion = & $pythonPath --version 2>&1
    $checks += @{ Name = "Python (venv)"; Status = "OK"; Detail = $pythonVersion }
} else {
    $checks += @{ Name = "Python (venv)"; Status = "FAIL"; Detail = "Not found at $pythonPath" }
}

# Check backend
$backendPath = "$ProjectPath\backend\api\main.py"
if (Test-Path $backendPath) {
    $checks += @{ Name = "Backend API"; Status = "OK"; Detail = "main.py found" }
} else {
    $checks += @{ Name = "Backend API"; Status = "FAIL"; Detail = "main.py not found" }
}

# Check .env
$envPath = "$ProjectPath\backend\.env"
if (Test-Path $envPath) {
    $checks += @{ Name = "Environment (.env)"; Status = "OK"; Detail = ".env configured" }
} else {
    $checks += @{ Name = "Environment (.env)"; Status = "WARN"; Detail = ".env not found - using defaults" }
}

# Check MT5
$mt5Running = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
if ($mt5Running) {
    $checks += @{ Name = "MT5 Terminal"; Status = "OK"; Detail = "Running" }
} else {
    $checks += @{ Name = "MT5 Terminal"; Status = "WARN"; Detail = "Not running - start for live trading" }
}

# Display results
foreach ($check in $checks) {
    $color = switch ($check.Status) {
        "OK" { "Green" }
        "WARN" { "Yellow" }
        "FAIL" { "Red" }
    }
    $icon = switch ($check.Status) {
        "OK" { "?" }
        "WARN" { "??" }
        "FAIL" { "?" }
    }
    Write-Host "  $icon $($check.Name): $($check.Detail)" -ForegroundColor $color
}

# Check for critical failures
$failures = $checks | Where-Object { $_.Status -eq "FAIL" }
if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "? Critical checks failed. Please fix issues above." -ForegroundColor Red
    exit 1
}

Write-Host ""

# =====================================================
# 2. Choose Deployment Mode
# =====================================================
if ([string]::IsNullOrEmpty($Mode)) {
    Write-Host "?? Choose deployment mode:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  1. Windows Service (NSSM)" -ForegroundColor White
    Write-Host "     - Best for production VPS" -ForegroundColor Gray
    Write-Host "     - Starts before user login" -ForegroundColor Gray
    Write-Host "     - More robust and faster" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Task Scheduler" -ForegroundColor White
    Write-Host "     - Simpler setup" -ForegroundColor Gray
    Write-Host "     - Built-in watchdog" -ForegroundColor Gray
    Write-Host "     - Easier to manage" -ForegroundColor Gray
    Write-Host ""
    
    $choice = Read-Host "Enter choice (1 or 2)"
    $Mode = if ($choice -eq "1") { "Service" } else { "Scheduler" }
}

Write-Host ""
Write-Host "?? Deploying with: $Mode mode" -ForegroundColor Cyan
Write-Host ""

# =====================================================
# 3. Stop Existing Services
# =====================================================
Write-Host "?? Stopping existing services..." -ForegroundColor Yellow

# Stop Windows Service
net stop TrademifyAPI 2>$null | Out-Null

# Stop Scheduled Tasks
schtasks /End /TN "TrademifyBot" 2>$null | Out-Null
schtasks /End /TN "TrademifyWatchdog" 2>$null | Out-Null

# Kill manual processes
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*trademify*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2
Write-Host "  Done" -ForegroundColor Green

# =====================================================
# 4. Run Installation Script
# =====================================================
Write-Host ""
Write-Host "?? Installing $Mode..." -ForegroundColor Yellow
Write-Host ""

try {
    if ($Mode -eq "Service") {
        & "$ProjectPath\vps\install-service.ps1"
    } else {
        & "$ProjectPath\vps\setup-task-scheduler.ps1"
    }
}
catch {
    Write-Host "? Installation failed: $_" -ForegroundColor Red
    exit 1
}

# =====================================================
# 5. Verify Deployment
# =====================================================
Write-Host ""
Write-Host "?? Verifying deployment..." -ForegroundColor Yellow
Write-Host ""

# Wait for startup
Write-Host "  Waiting for API to start (30 seconds)..." -ForegroundColor Gray
$maxAttempts = 6
$attempt = 0
$healthy = $false

while ($attempt -lt $maxAttempts -and -not $healthy) {
    Start-Sleep -Seconds 5
    $attempt++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $healthy = $true
        }
    }
    catch {
        Write-Host "  Attempt $attempt/$maxAttempts - Still starting..." -ForegroundColor Gray
    }
}

if ($healthy) {
    Write-Host "  ? API is healthy!" -ForegroundColor Green
    
    # Get bot status
    try {
        $botResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/bot/status" -TimeoutSec 5 -UseBasicParsing
        $botStatus = $botResponse.Content | ConvertFrom-Json
        Write-Host "  ? Bot Status: $($botStatus.status)" -ForegroundColor Green
    }
    catch {
        Write-Host "  ?? Could not get bot status" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ?? API not responding yet - may need more time" -ForegroundColor Yellow
}

# =====================================================
# 6. Summary
# =====================================================
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" } | Select-Object -First 1).IPAddress

Write-Host "
??????????????????????????????????????????????????????????????
?     ? DEPLOYMENT COMPLETE!                                ?
??????????????????????????????????????????????????????????????
?                                                            ?
?  ?? Access URLs:                                           ?
?     API Docs:  http://localhost:8000/docs                  ?
?     Health:    http://localhost:8000/health                ?
$(if ($ip) { "?     Remote:   http://${ip}:8000/docs                      ?" })
?                                                            ?
?  ?? Monitoring:                                            ?
?     .\check-status.bat     - View status                   ?
?     .\stop-services.bat    - Stop all                      ?
?     .\uninstall-service.ps1 - Remove all                   ?
?                                                            ?
?  ?? Logs: $ProjectPath\logs                                ?
?                                                            ?
??????????????????????????????????????????????????????????????
?  ?? Trademify is now running 24/7!                         ?
?     Auto-restart on crash ?                                ?
?     Auto-start on boot ?                                   ?
$(if ($Mode -eq "Scheduler") { "?     Watchdog every 5 min ?                                 ?" })
??????????????????????????????????????????????????????????????
" -ForegroundColor Cyan

# Open browser
$openBrowser = Read-Host "Open API docs in browser? (y/n)"
if ($openBrowser -eq "y") {
    Start-Process "http://localhost:8000/docs"
}

Write-Host ""
Write-Host "?? Happy Trading!" -ForegroundColor Green
