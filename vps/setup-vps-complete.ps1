# ============================================
# Trademify - Complete VPS Setup Script
# รันครั้งเดียว ติดตั้งทุกอย่างอัตโนมัติ
# ============================================

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Trademify VPS Setup - One Click Install" -ForegroundColor Cyan  
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Enable TLS 1.2
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Create temp directory
$TempDir = "C:\trademify-temp"
if (!(Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

# ============================================
# Step 1: Install Python 3.11
# ============================================
Write-Host "[1/6] Installing Python 3.11..." -ForegroundColor Yellow

$PythonInstalled = Get-Command python -ErrorAction SilentlyContinue
if (!$PythonInstalled) {
    Write-Host "  Downloading Python..." -ForegroundColor Gray
    $PythonUrl = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
    $PythonInstaller = "$TempDir\python-installer.exe"
    Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonInstaller
    
    Write-Host "  Installing Python..." -ForegroundColor Gray
    Start-Process -FilePath $PythonInstaller -ArgumentList '/quiet', 'InstallAllUsers=1', 'PrependPath=1', 'Include_test=0' -Wait
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}
Write-Host "  OK Python installed" -ForegroundColor Green

# ============================================
# Step 2: Install Node.js
# ============================================
Write-Host "[2/6] Installing Node.js 20..." -ForegroundColor Yellow

$NodeInstalled = Get-Command node -ErrorAction SilentlyContinue
if (!$NodeInstalled) {
    Write-Host "  Downloading Node.js..." -ForegroundColor Gray
    $NodeUrl = "https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi"
    $NodeInstaller = "$TempDir\node-installer.msi"
    Invoke-WebRequest -Uri $NodeUrl -OutFile $NodeInstaller
    
    Write-Host "  Installing Node.js..." -ForegroundColor Gray
    Start-Process msiexec.exe -ArgumentList '/i', $NodeInstaller, '/quiet', '/norestart' -Wait
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}
Write-Host "  OK Node.js installed" -ForegroundColor Green

# ============================================
# Step 3: Install Git
# ============================================
Write-Host "[3/6] Installing Git..." -ForegroundColor Yellow

$GitInstalled = Get-Command git -ErrorAction SilentlyContinue
if (!$GitInstalled) {
    Write-Host "  Downloading Git..." -ForegroundColor Gray
    $GitUrl = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe"
    $GitInstaller = "$TempDir\git-installer.exe"
    Invoke-WebRequest -Uri $GitUrl -OutFile $GitInstaller
    
    Write-Host "  Installing Git..." -ForegroundColor Gray
    Start-Process -FilePath $GitInstaller -ArgumentList '/VERYSILENT', '/NORESTART' -Wait
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}
Write-Host "  OK Git installed" -ForegroundColor Green

# Refresh PATH one more time
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# ============================================
# Step 4: Clone Trademify
# ============================================
Write-Host "[4/6] Cloning Trademify..." -ForegroundColor Yellow

Set-Location C:\
if (Test-Path "C:\trademify") {
    Write-Host "  Removing existing installation..." -ForegroundColor Gray
    Remove-Item -Recurse -Force "C:\trademify" -ErrorAction SilentlyContinue
}

git clone https://github.com/ultimate-6159/trademify.git
Set-Location C:\trademify
Write-Host "  OK Repository cloned" -ForegroundColor Green

# ============================================
# Step 5: Setup Backend
# ============================================
Write-Host "[5/6] Setting up Backend (this takes 2-5 minutes)..." -ForegroundColor Yellow

Set-Location C:\trademify
python -m venv venv
& .\venv\Scripts\Activate.ps1

Set-Location backend
Write-Host "  Installing Python packages..." -ForegroundColor Gray
pip install --upgrade pip --quiet 2>$null
pip install -r requirements.txt --quiet 2>$null
pip install MetaTrader5 --quiet 2>$null

# Create .env from example
if (!(Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

# Create logs directory
if (!(Test-Path "C:\trademify\logs")) {
    New-Item -ItemType Directory -Path "C:\trademify\logs" -Force | Out-Null
}

Write-Host "  OK Backend ready" -ForegroundColor Green

# ============================================
# Step 6: Setup Frontend
# ============================================
Write-Host "[6/6] Setting up Frontend..." -ForegroundColor Yellow

Set-Location C:\trademify\frontend
Write-Host "  Installing npm packages..." -ForegroundColor Gray
npm install --silent 2>$null
Write-Host "  OK Frontend ready" -ForegroundColor Green

# ============================================
# Configure Firewall
# ============================================
Write-Host ""
Write-Host "Configuring Firewall..." -ForegroundColor Yellow
netsh advfirewall firewall delete rule name="Trademify API" 2>$null
netsh advfirewall firewall delete rule name="Trademify Frontend" 2>$null
netsh advfirewall firewall add rule name="Trademify API" dir=in action=allow protocol=tcp localport=8000 | Out-Null
netsh advfirewall firewall add rule name="Trademify Frontend" dir=in action=allow protocol=tcp localport=5173 | Out-Null
Write-Host "  OK Firewall configured" -ForegroundColor Green

# ============================================
# Create Desktop Shortcuts
# ============================================
Write-Host ""
Write-Host "Creating Desktop Shortcuts..." -ForegroundColor Yellow
$Desktop = [Environment]::GetFolderPath("Desktop")

# Start All shortcut
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$Desktop\Start Trademify.lnk")
$Shortcut.TargetPath = "C:\trademify\vps\start-services.bat"
$Shortcut.WorkingDirectory = "C:\trademify"
$Shortcut.IconLocation = "shell32.dll,137"
$Shortcut.Save()

# Stop All shortcut
$Shortcut = $WshShell.CreateShortcut("$Desktop\Stop Trademify.lnk")
$Shortcut.TargetPath = "C:\trademify\vps\stop-services.bat"
$Shortcut.WorkingDirectory = "C:\trademify"
$Shortcut.IconLocation = "shell32.dll,131"
$Shortcut.Save()

# Status Check shortcut
$Shortcut = $WshShell.CreateShortcut("$Desktop\Trademify Status.lnk")
$Shortcut.TargetPath = "C:\trademify\vps\check-status.bat"
$Shortcut.WorkingDirectory = "C:\trademify"
$Shortcut.IconLocation = "shell32.dll,23"
$Shortcut.Save()

Write-Host "  OK Desktop shortcuts created" -ForegroundColor Green

# ============================================
# Setup Auto-Start on Boot
# ============================================
Write-Host ""
Write-Host "Setting up Auto-Start on Boot..." -ForegroundColor Yellow
$StartupFolder = [Environment]::GetFolderPath("Startup")
Copy-Item "$Desktop\Start Trademify.lnk" "$StartupFolder\Start Trademify.lnk" -Force
Write-Host "  OK Auto-start configured" -ForegroundColor Green

# ============================================
# Cleanup
# ============================================
Write-Host ""
Write-Host "Cleaning up..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
Write-Host "  OK Cleanup complete" -ForegroundColor Green

# ============================================
# Get VPS IP
# ============================================
$IP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress

# ============================================
# Done!
# ============================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "   Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your VPS IP: $IP" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Install MT5 from your broker and login" -ForegroundColor White
Write-Host ""
Write-Host "2. Edit credentials:" -ForegroundColor White
Write-Host "   notepad C:\trademify\backend\.env" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start services (double-click on Desktop):" -ForegroundColor White
Write-Host "   'Start Trademify'" -ForegroundColor Gray
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Yellow
Write-Host "   Frontend:  http://${IP}:5173" -ForegroundColor Cyan
Write-Host "   API Docs:  http://${IP}:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to start services now..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Start services
Start-Process "C:\trademify\vps\start-services.bat"

