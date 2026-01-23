# ============================================
# Trademify - Complete VPS Setup Script
# รันครั้งเดียว ติดตั้งทุกอย่างอัตโนมัติ
# สำหรับ Windows Server / Windows 10/11
# ============================================
# Usage: 
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   irm https://raw.githubusercontent.com/ultimate-6159/trademify/main/vps/setup-vps-complete.ps1 | iex
# หรือ:
#   .\setup-vps-complete.ps1
# ============================================

param(
    [string]$InstallPath = "C:\trademify",
    [switch]$SkipMT5 = $false
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# Enable TLS 1.2/1.3
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

function Write-Step { param($step, $msg) Write-Host "[$step] $msg" -ForegroundColor Yellow }
function Write-OK { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Err { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  $msg" -ForegroundColor Gray }

Clear-Host
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                           ║" -ForegroundColor Cyan
Write-Host "║   🤖 TRADEMIFY AI TRADING BOT - VPS SETUP                ║" -ForegroundColor Cyan
Write-Host "║   One-Click Installation for Windows                      ║" -ForegroundColor Cyan
Write-Host "║                                                           ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$TotalSteps = 7
$TempDir = "$env:TEMP\trademify-setup"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# ============================================
# Step 1: Check Admin Rights
# ============================================
Write-Step "1/$TotalSteps" "Checking administrator rights..."

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Err "Please run as Administrator!"
    Write-Info "Right-click PowerShell -> Run as Administrator"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-OK "Running as Administrator"

# ============================================
# Step 2: Install Python 3.11+
# ============================================
Write-Step "2/$TotalSteps" "Checking Python..."

$python = Get-Command python -ErrorAction SilentlyContinue
$needPython = $true

if ($python) {
    $ver = python --version 2>&1
    if ($ver -match "3\.(1[1-9]|[2-9]\d)") {
        Write-OK "Python already installed: $ver"
        $needPython = $false
    }
}

if ($needPython) {
    Write-Info "Downloading Python 3.11..."
    $pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $pythonExe = "$TempDir\python-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonExe -UseBasicParsing
        Write-Info "Installing Python (this may take a minute)..."
        Start-Process -FilePath $pythonExe -ArgumentList '/quiet', 'InstallAllUsers=1', 'PrependPath=1', 'Include_pip=1', 'Include_test=0' -Wait
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-OK "Python installed successfully"
    } catch {
        Write-Err "Failed to install Python: $_"
        Write-Info "Please install Python 3.11+ manually from https://python.org"
    }
}

# ============================================
# Step 3: Install Git
# ============================================
Write-Step "3/$TotalSteps" "Checking Git..."

$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-OK "Git already installed"
} else {
    Write-Info "Downloading Git..."
    $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe"
    $gitExe = "$TempDir\git-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $gitUrl -OutFile $gitExe -UseBasicParsing
        Write-Info "Installing Git..."
        Start-Process -FilePath $gitExe -ArgumentList '/VERYSILENT', '/NORESTART', '/NOCANCEL' -Wait
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-OK "Git installed successfully"
    } catch {
        Write-Err "Failed to install Git: $_"
    }
}

# ============================================
# Step 4: Clone/Update Repository
# ============================================
Write-Step "4/$TotalSteps" "Setting up Trademify repository..."

if (Test-Path "$InstallPath\.git") {
    Write-Info "Updating existing installation..."
    Set-Location $InstallPath
    git pull origin main 2>&1 | Out-Null
    Write-OK "Repository updated"
} else {
    if (Test-Path $InstallPath) {
        Write-Info "Removing old installation..."
        Remove-Item -Recurse -Force $InstallPath -ErrorAction SilentlyContinue
    }
    
    Write-Info "Cloning repository..."
    git clone https://github.com/ultimate-6159/trademify.git $InstallPath 2>&1 | Out-Null
    Write-OK "Repository cloned to $InstallPath"
}

Set-Location $InstallPath

# ============================================
# Step 5: Setup Python Environment
# ============================================
Write-Step "5/$TotalSteps" "Setting up Python environment (2-5 minutes)..."

# Create venv
if (-not (Test-Path "$InstallPath\venv")) {
    Write-Info "Creating virtual environment..."
    python -m venv venv 2>&1 | Out-Null
}

# Activate and install
Write-Info "Installing Python packages..."
& "$InstallPath\venv\Scripts\python.exe" -m pip install --upgrade pip -q 2>&1 | Out-Null
& "$InstallPath\venv\Scripts\pip.exe" install -r "$InstallPath\backend\requirements.txt" -q 2>&1 | Out-Null
& "$InstallPath\venv\Scripts\pip.exe" install MetaTrader5 -q 2>&1 | Out-Null

Write-OK "Python environment ready"

# ============================================
# Step 6: Configure Environment
# ============================================
Write-Step "6/$TotalSteps" "Configuring environment..."

# Create logs directory
New-Item -ItemType Directory -Path "$InstallPath\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "$InstallPath\backend\logs" -Force | Out-Null

# Create .env if not exists
$envFile = "$InstallPath\backend\.env"
if (-not (Test-Path $envFile)) {
    Write-Info "Creating .env configuration file..."
    Copy-Item "$InstallPath\backend\.env.example" $envFile -Force
}

# Configure Firewall
Write-Info "Configuring firewall..."
netsh advfirewall firewall delete rule name="Trademify API" 2>&1 | Out-Null
netsh advfirewall firewall delete rule name="Trademify Frontend" 2>&1 | Out-Null
netsh advfirewall firewall add rule name="Trademify API" dir=in action=allow protocol=tcp localport=8000 | Out-Null
netsh advfirewall firewall add rule name="Trademify Frontend" dir=in action=allow protocol=tcp localport=5173 | Out-Null

Write-OK "Environment configured"

# ============================================
# Step 7: Create Desktop Shortcuts
# ============================================
Write-Step "7/$TotalSteps" "Creating shortcuts..."

$Desktop = [Environment]::GetFolderPath("Desktop")
$WshShell = New-Object -ComObject WScript.Shell

# Start shortcut
$shortcut = $WshShell.CreateShortcut("$Desktop\Start Trademify.lnk")
$shortcut.TargetPath = "$InstallPath\vps\start-services.bat"
$shortcut.WorkingDirectory = $InstallPath
$shortcut.IconLocation = "shell32.dll,137"
$shortcut.Save()

# Stop shortcut
$shortcut = $WshShell.CreateShortcut("$Desktop\Stop Trademify.lnk")
$shortcut.TargetPath = "$InstallPath\vps\stop-services.bat"
$shortcut.WorkingDirectory = $InstallPath
$shortcut.IconLocation = "shell32.dll,131"
$shortcut.Save()

# Status shortcut
$shortcut = $WshShell.CreateShortcut("$Desktop\Trademify Status.lnk")
$shortcut.TargetPath = "$InstallPath\vps\check-status.bat"
$shortcut.WorkingDirectory = $InstallPath
$shortcut.IconLocation = "shell32.dll,23"
$shortcut.Save()

# Auto-start on boot
$StartupFolder = [Environment]::GetFolderPath("Startup")
Copy-Item "$Desktop\Start Trademify.lnk" "$StartupFolder\Start Trademify.lnk" -Force

Write-OK "Shortcuts created on Desktop"
Write-OK "Auto-start on boot enabled"

# ============================================
# Cleanup
# ============================================
Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue

# ============================================
# Get VPS IP
# ============================================
$IP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { 
    $_.InterfaceAlias -notlike "*Loopback*" -and 
    $_.IPAddress -notlike "169.*" -and
    $_.IPAddress -notlike "127.*"
} | Select-Object -First 1).IPAddress

if (-not $IP) { $IP = "localhost" }

# ============================================
# Done!
# ============================================
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                                                           ║" -ForegroundColor Green
Write-Host "║   ✅ INSTALLATION COMPLETE!                              ║" -ForegroundColor Green
Write-Host "║                                                           ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  📍 Installed to: $InstallPath" -ForegroundColor White
Write-Host "  🌐 Your IP: $IP" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  📋 NEXT STEPS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Install MetaTrader 5 from your broker" -ForegroundColor White
Write-Host "     - Download from Exness/XM/IC Markets etc." -ForegroundColor Gray
Write-Host "     - Login to your account" -ForegroundColor Gray
Write-Host "     - Enable 'Allow Algo Trading'" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Edit MT5 credentials:" -ForegroundColor White
Write-Host "     notepad $InstallPath\backend\.env" -ForegroundColor Cyan
Write-Host ""
Write-Host "     Set these values:" -ForegroundColor Gray
Write-Host "     MT5_LOGIN=your_account_number" -ForegroundColor DarkGray
Write-Host "     MT5_PASSWORD=your_password" -ForegroundColor DarkGray
Write-Host "     MT5_SERVER=Your-Broker-Server" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  3. Start Trademify (double-click on Desktop):" -ForegroundColor White
Write-Host "     'Start Trademify'" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  🔗 ACCESS URLS:" -ForegroundColor Yellow
Write-Host "     API Docs:  http://${IP}:8000/docs" -ForegroundColor Cyan
Write-Host "     Health:    http://${IP}:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""

# Ask to start
$response = Read-Host "Start Trademify now? (Y/n)"
if ($response -eq "" -or $response -match "^[Yy]") {
    Write-Host ""
    Write-Host "Starting Trademify..." -ForegroundColor Yellow
    Start-Process "$InstallPath\vps\start-services.bat"
    Write-Host ""
    Write-Host "✅ Trademify is starting! Check the new window." -ForegroundColor Green
    Write-Host "   API will be available at http://${IP}:8000" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Press any key to close..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
