# ============================================================
# Trademify - Kill All Services Script
# ============================================================
# Usage: Right-click > Run with PowerShell (as Administrator)
# Or: powershell -ExecutionPolicy Bypass -File kill-all-services.ps1
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Red
Write-Host "        TRADEMIFY - KILL ALL SERVICES" -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Red
Write-Host ""

# Run as Administrator check
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[WARNING] Not running as Administrator. Some services may not stop." -ForegroundColor Yellow
}

# ============================================================
# 1. STOP WINDOWS SERVICES (NSSM)
# ============================================================
Write-Host "[1/5] Stopping Windows Services..." -ForegroundColor Cyan

$services = @("TrademifyBot", "TrademifyAPI", "TrademifyWatchdog")
foreach ($svc in $services) {
    $service = Get-Service -Name $svc -ErrorAction SilentlyContinue
    if ($service) {
        Write-Host "   Stopping $svc..." -ForegroundColor Yellow
        Stop-Service -Name $svc -Force -ErrorAction SilentlyContinue
        # Also try NSSM stop
        & nssm stop $svc 2>$null
        Write-Host "   [OK] $svc stopped" -ForegroundColor Green
    } else {
        Write-Host "   [SKIP] $svc not found" -ForegroundColor Gray
    }
}

# ============================================================
# 2. KILL PYTHON PROCESSES (Trademify related)
# ============================================================
Write-Host ""
Write-Host "[2/5] Killing Python processes..." -ForegroundColor Cyan

# Get all Python processes
$pythonProcesses = Get-Process -Name "python*" -ErrorAction SilentlyContinue

if ($pythonProcesses) {
    foreach ($proc in $pythonProcesses) {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)" -ErrorAction SilentlyContinue).CommandLine
            
            # Check if it's Trademify related
            if ($cmdLine -match "trademify|ai_trading_bot|uvicorn|main:app") {
                Write-Host "   Killing PID $($proc.Id): $($proc.ProcessName)" -ForegroundColor Yellow
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                Write-Host "   [OK] Killed" -ForegroundColor Green
            }
        } catch {
            # If can't get cmdline, kill anyway if user confirms
        }
    }
} else {
    Write-Host "   [SKIP] No Python processes found" -ForegroundColor Gray
}

# ============================================================
# 3. KILL ALL PYTHON (Force - use if above didn't work)
# ============================================================
Write-Host ""
Write-Host "[3/5] Force killing remaining Python..." -ForegroundColor Cyan

taskkill /F /IM python.exe 2>$null
taskkill /F /IM pythonw.exe 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "   [OK] All Python processes killed" -ForegroundColor Green
} else {
    Write-Host "   [SKIP] No Python processes to kill" -ForegroundColor Gray
}

# ============================================================
# 4. KILL PROCESSES ON SPECIFIC PORTS
# ============================================================
Write-Host ""
Write-Host "[4/5] Killing processes on ports 8000, 5173..." -ForegroundColor Cyan

$ports = @(8000, 5173)
foreach ($port in $ports) {
    $netstat = netstat -ano | Select-String ":$port " | Select-String "LISTENING"
    if ($netstat) {
        $pid = ($netstat -split '\s+')[-1]
        if ($pid -match '^\d+$') {
            Write-Host "   Killing process on port $port (PID: $pid)..." -ForegroundColor Yellow
            taskkill /F /PID $pid 2>$null
            Write-Host "   [OK] Port $port freed" -ForegroundColor Green
        }
    } else {
        Write-Host "   [SKIP] Port $port not in use" -ForegroundColor Gray
    }
}

# ============================================================
# 5. OPTIONAL: KILL MT5 (Uncomment if needed)
# ============================================================
Write-Host ""
Write-Host "[5/5] MT5 Terminal..." -ForegroundColor Cyan

# Uncomment below lines to also kill MT5
# taskkill /F /IM terminal64.exe 2>$null
# Write-Host "   [OK] MT5 killed" -ForegroundColor Green

Write-Host "   [SKIP] MT5 kept running (uncomment in script to kill)" -ForegroundColor Gray

# ============================================================
# SUMMARY
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "        ALL TRADEMIFY SERVICES STOPPED!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start bot manually:" -ForegroundColor Cyan
Write-Host "   cd C:\trademify\backend" -ForegroundColor White
Write-Host "   python ai_trading_bot.py MT5 XAUUSDm H1 MEDIUM 60" -ForegroundColor White
Write-Host ""
Write-Host "Or use:" -ForegroundColor Cyan
Write-Host "   C:\trademify\start-bot.bat" -ForegroundColor White
Write-Host ""

# Keep window open
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
