# ============================================
# ??? Trademify Windows Service Installer
# ============================================
# Uses NSSM (Non-Sucking Service Manager) to run as Windows Service
# 
# Features:
# - Auto-start on Windows boot
# - Auto-restart on crash
# - Runs without user login
# - 10+ year autonomous operation
#
# Usage:
#   .\install-service.ps1           # Install with defaults
#   .\install-service.ps1 -Uninstall # Uninstall service
# ============================================

param(
    [switch]$Uninstall,
    [string]$InstallPath = "C:\trademify",
    [string]$NssmPath = "C:\nssm\nssm.exe"
)

$ErrorActionPreference = "Stop"

# Service names
$API_SERVICE = "TrademifyAPI"
$BOT_SERVICE = "TrademifyBot"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Trademify Windows Service Installer" -ForegroundColor Cyan
Write-Host "   ?? 10+ Year Autonomous Operation" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "? ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "   Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# ============================================
# UNINSTALL MODE
# ============================================
if ($Uninstall) {
    Write-Host "??? Uninstalling Trademify services..." -ForegroundColor Yellow
    
    # Stop and remove API service
    if (Get-Service -Name $API_SERVICE -ErrorAction SilentlyContinue) {
        Write-Host "   Stopping $API_SERVICE..."
        Stop-Service -Name $API_SERVICE -Force -ErrorAction SilentlyContinue
        & $NssmPath remove $API_SERVICE confirm 2>$null
        Write-Host "   ? $API_SERVICE removed" -ForegroundColor Green
    }
    
    # Stop and remove Bot service (if separate)
    if (Get-Service -Name $BOT_SERVICE -ErrorAction SilentlyContinue) {
        Write-Host "   Stopping $BOT_SERVICE..."
        Stop-Service -Name $BOT_SERVICE -Force -ErrorAction SilentlyContinue
        & $NssmPath remove $BOT_SERVICE confirm 2>$null
        Write-Host "   ? $BOT_SERVICE removed" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "? Uninstall complete!" -ForegroundColor Green
    exit 0
}

# ============================================
# CHECK PREREQUISITES
# ============================================
Write-Host "?? Checking prerequisites..." -ForegroundColor Cyan

# Check NSSM
if (-not (Test-Path $NssmPath)) {
    Write-Host "? NSSM not found at $NssmPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "?? Installing NSSM..." -ForegroundColor Yellow
    
    # Create nssm directory
    $nssmDir = Split-Path $NssmPath -Parent
    if (-not (Test-Path $nssmDir)) {
        New-Item -ItemType Directory -Path $nssmDir -Force | Out-Null
    }
    
    # ?? Force TLS 1.2 for secure download
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    
    # Download NSSM from GitHub mirror (more reliable)
    $nssmUrl = "https://github.com/kirillkovalenko/nssm/releases/download/2.24.101-rc/nssm-2.24.101-rc.zip"
    $nssmZipPath = "$env:TEMP\nssm.zip"
    $nssmExtractPath = "$env:TEMP\nssm_extract"
    
    try {
        Write-Host "   Downloading from GitHub mirror..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $nssmUrl -OutFile $nssmZipPath -UseBasicParsing -TimeoutSec 60
        
        # Extract
        if (Test-Path $nssmExtractPath) { Remove-Item $nssmExtractPath -Recurse -Force }
        Expand-Archive -Path $nssmZipPath -DestinationPath $nssmExtractPath -Force
        
        # Find and copy nssm.exe
        $nssmExe = Get-ChildItem -Path $nssmExtractPath -Recurse -Filter "nssm.exe" | Where-Object { $_.Directory.Name -eq "win64" } | Select-Object -First 1
        if (-not $nssmExe) {
            $nssmExe = Get-ChildItem -Path $nssmExtractPath -Recurse -Filter "nssm.exe" | Select-Object -First 1
        }
        
        if ($nssmExe) {
            Copy-Item $nssmExe.FullName -Destination $NssmPath -Force
            Write-Host "   ? NSSM installed to $NssmPath" -ForegroundColor Green
        } else {
            throw "nssm.exe not found in archive"
        }
        
        # Cleanup
        Remove-Item $nssmZipPath -Force -ErrorAction SilentlyContinue
        Remove-Item $nssmExtractPath -Recurse -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Host "   ? Failed to download NSSM: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "   ?? Please download manually:" -ForegroundColor Yellow
        Write-Host "      1. Go to: https://nssm.cc/download" -ForegroundColor White
        Write-Host "      2. Download nssm-2.24.zip" -ForegroundColor White
        Write-Host "      3. Extract win64\nssm.exe to C:\nssm\" -ForegroundColor White
        Write-Host ""
        
        # Try alternate download from nssm.cc with TLS fix
        Write-Host "   ?? Trying alternate download..." -ForegroundColor Yellow
        try {
            $altUrl = "https://nssm.cc/release/nssm-2.24.zip"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls -bor [Net.SecurityProtocolType]::Tls11 -bor [Net.SecurityProtocolType]::Tls12
            $webClient = New-Object System.Net.WebClient
            $webClient.DownloadFile($altUrl, $nssmZipPath)
            
            Expand-Archive -Path $nssmZipPath -DestinationPath $nssmExtractPath -Force
            Copy-Item "$nssmExtractPath\nssm-2.24\win64\nssm.exe" -Destination $NssmPath -Force
            Remove-Item $nssmZipPath -Force -ErrorAction SilentlyContinue
            Remove-Item $nssmExtractPath -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "   ? NSSM installed successfully!" -ForegroundColor Green
        } catch {
            Write-Host "   ? Alternate download also failed" -ForegroundColor Red
            Write-Host "   Please download manually from https://nssm.cc/download" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Check Python
$pythonPath = "$InstallPath\venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "? Python venv not found at $pythonPath" -ForegroundColor Red
    Write-Host "   Run: python -m venv $InstallPath\venv" -ForegroundColor Yellow
    exit 1
}
Write-Host "   ? Python venv found" -ForegroundColor Green

# Check backend
$backendPath = "$InstallPath\backend"
if (-not (Test-Path $backendPath)) {
    Write-Host "? Backend not found at $backendPath" -ForegroundColor Red
    exit 1
}
Write-Host "   ? Backend found" -ForegroundColor Green

# Create logs directory
$logsPath = "$InstallPath\logs"
if (-not (Test-Path $logsPath)) {
    New-Item -ItemType Directory -Path $logsPath -Force | Out-Null
}
Write-Host "   ? Logs directory ready" -ForegroundColor Green

# ============================================
# CREATE WRAPPER SCRIPT
# ============================================
Write-Host ""
Write-Host "?? Creating service wrapper script..." -ForegroundColor Cyan

$wrapperScript = @"
@echo off
REM Trademify API Service Wrapper
REM This script is called by NSSM to start the API

cd /d $InstallPath
call venv\Scripts\activate.bat
cd backend

REM Set environment variables
set PYTHONUNBUFFERED=1
set PYTHONPATH=$InstallPath\backend

REM Start uvicorn
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
"@

$wrapperPath = "$InstallPath\vps\run-api-service.bat"
$wrapperScript | Out-File -FilePath $wrapperPath -Encoding ASCII
Write-Host "   ? Wrapper script created" -ForegroundColor Green

# ============================================
# INSTALL API SERVICE
# ============================================
Write-Host ""
Write-Host "?? Installing $API_SERVICE..." -ForegroundColor Cyan

# Remove existing service if exists
if (Get-Service -Name $API_SERVICE -ErrorAction SilentlyContinue) {
    Write-Host "   Removing existing service..."
    Stop-Service -Name $API_SERVICE -Force -ErrorAction SilentlyContinue
    & $NssmPath remove $API_SERVICE confirm 2>$null
    Start-Sleep -Seconds 2
}

# Install new service
& $NssmPath install $API_SERVICE $wrapperPath

# Configure service parameters
& $NssmPath set $API_SERVICE DisplayName "Trademify Trading API"
& $NssmPath set $API_SERVICE Description "AI Trading Bot API Server - 10 Year Autonomous Operation"
& $NssmPath set $API_SERVICE AppDirectory $backendPath
& $NssmPath set $API_SERVICE Start SERVICE_AUTO_START

# Configure restart behavior (CRITICAL for 10 year operation!)
& $NssmPath set $API_SERVICE AppExit Default Restart
& $NssmPath set $API_SERVICE AppRestartDelay 5000  # 5 seconds between restarts
& $NssmPath set $API_SERVICE AppThrottle 10000     # 10 seconds throttle

# Configure logging
& $NssmPath set $API_SERVICE AppStdout "$logsPath\api-stdout.log"
& $NssmPath set $API_SERVICE AppStderr "$logsPath\api-stderr.log"
& $NssmPath set $API_SERVICE AppStdoutCreationDisposition 4  # Append
& $NssmPath set $API_SERVICE AppStderrCreationDisposition 4  # Append
& $NssmPath set $API_SERVICE AppRotateFiles 1
& $NssmPath set $API_SERVICE AppRotateBytes 10485760  # 10MB per file
& $NssmPath set $API_SERVICE AppRotateOnline 1

# Configure shutdown
& $NssmPath set $API_SERVICE AppStopMethodSkip 0
& $NssmPath set $API_SERVICE AppStopMethodConsole 3000
& $NssmPath set $API_SERVICE AppStopMethodWindow 3000
& $NssmPath set $API_SERVICE AppStopMethodThreads 1000

Write-Host "   ? $API_SERVICE installed" -ForegroundColor Green

# ============================================
# START SERVICE
# ============================================
Write-Host ""
Write-Host "?? Starting service..." -ForegroundColor Cyan

Start-Service -Name $API_SERVICE
Start-Sleep -Seconds 5

# Check if running
$service = Get-Service -Name $API_SERVICE
if ($service.Status -eq "Running") {
    Write-Host "   ? $API_SERVICE is running!" -ForegroundColor Green
} else {
    Write-Host "   ?? Service may still be starting..." -ForegroundColor Yellow
    Write-Host "   Check status with: Get-Service $API_SERVICE" -ForegroundColor Yellow
}

# ============================================
# VERIFY API IS RESPONDING
# ============================================
Write-Host ""
Write-Host "?? Verifying API..." -ForegroundColor Cyan

$maxRetries = 6
$retryCount = 0
$apiReady = $false

while ($retryCount -lt $maxRetries -and -not $apiReady) {
    Start-Sleep -Seconds 5
    $retryCount++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $apiReady = $true
            Write-Host "   ? API is responding!" -ForegroundColor Green
        }
    } catch {
        Write-Host "   ? Waiting for API... ($retryCount/$maxRetries)" -ForegroundColor Yellow
    }
}

if (-not $apiReady) {
    Write-Host "   ?? API not responding yet - check logs at $logsPath" -ForegroundColor Yellow
}

# ============================================
# SUMMARY
# ============================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "   ? Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "?? Service Status:" -ForegroundColor Cyan
Get-Service -Name $API_SERVICE | Format-Table Name, Status, StartType -AutoSize
Write-Host ""
Write-Host "?? Important Paths:" -ForegroundColor Cyan
Write-Host "   Logs:    $logsPath" -ForegroundColor White
Write-Host "   Config:  $backendPath\.env" -ForegroundColor White
Write-Host "   API:     http://localhost:8000" -ForegroundColor White
Write-Host "   Docs:    http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "?? Management Commands:" -ForegroundColor Cyan
Write-Host "   Start:   Start-Service $API_SERVICE" -ForegroundColor White
Write-Host "   Stop:    Stop-Service $API_SERVICE" -ForegroundColor White
Write-Host "   Status:  Get-Service $API_SERVICE" -ForegroundColor White
Write-Host "   Logs:    Get-Content $logsPath\api-stdout.log -Tail 50" -ForegroundColor White
Write-Host ""
Write-Host "?? Auto-Restart: ENABLED (restarts on crash)" -ForegroundColor Green
Write-Host "?? Auto-Start:   ENABLED (starts on Windows boot)" -ForegroundColor Green
Write-Host "?? Log Rotation: ENABLED (10MB per file)" -ForegroundColor Green
Write-Host ""
Write-Host "?? Your bot will now run automatically 24/7!" -ForegroundColor Green
Write-Host ""
