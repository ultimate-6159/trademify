# ============================================
# Trademify AI Trading Bot - Windows Service
# ระบบเทรดอัตโนมัติด้วย AI เพียงหนึ่งเดียว
# ============================================

param(
    [string]$Broker = "MT5",
    [string]$Symbols = "EURUSD,GBPUSD,XAUUSD",
    [string]$Timeframe = "H1",
    [string]$Quality = "HIGH",
    [int]$Interval = 60,
    [switch]$Real
)

$ErrorActionPreference = "Continue"

# Configuration
$ProjectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectPath "venv\Scripts\python.exe"
$BotScript = Join-Path $ProjectPath "backend\ai_trading_bot.py"
$LogPath = Join-Path $ProjectPath "logs"

# Create logs directory
if (!(Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
}

# Log file
$LogFile = Join-Path $LogPath "ai_bot_service.log"

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

Write-Log "============================================"
Write-Log "Trademify AI Trading Bot Service"
Write-Log "Broker: $Broker"
Write-Log "Symbols: $Symbols"
Write-Log "Timeframe: $Timeframe"
Write-Log "Quality: $Quality"
Write-Log "Interval: $Interval seconds"
Write-Log "Real Trading: $Real"
Write-Log "============================================"

# Check MT5 is running (for MT5 broker)
function Test-MT5Running {
    $mt5Process = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
    return $null -ne $mt5Process
}

# Start MT5 if not running
function Start-MT5 {
    $mt5Paths = @(
        "C:\Program Files\MetaTrader 5\terminal64.exe",
        "C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        "$env:APPDATA\..\Local\Programs\MetaTrader 5\terminal64.exe"
    )
    
    foreach ($path in $mt5Paths) {
        if (Test-Path $path) {
            Write-Log "Starting MT5 from: $path"
            Start-Process -FilePath $path
            Start-Sleep -Seconds 30
            return $true
        }
    }
    
    Write-Log "WARNING: MT5 not found in default locations"
    return $false
}

# Build command arguments
$Arguments = @(
    $BotScript,
    "--broker", $Broker,
    "--symbols", $Symbols,
    "--timeframe", $Timeframe,
    "--quality", $Quality,
    "--interval", $Interval
)

if ($Real) {
    $Arguments += "--real"
}

# Main loop with auto-restart
$RestartCount = 0
$MaxRestarts = 100

while ($RestartCount -lt $MaxRestarts) {
    # Check MT5 (if using MT5 broker)
    if ($Broker -eq "MT5" -and !(Test-MT5Running)) {
        Write-Log "MT5 not running, attempting to start..."
        Start-MT5
        
        if (!(Test-MT5Running)) {
            Write-Log "ERROR: Could not start MT5. Waiting 60 seconds..."
            Start-Sleep -Seconds 60
            continue
        }
    }
    
    Write-Log "Starting AI Trading Bot (attempt $($RestartCount + 1))..."
    
    try {
        # Run the bot
        $Process = Start-Process -FilePath $VenvPath `
            -ArgumentList $Arguments `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput (Join-Path $LogPath "ai_bot_stdout.log") `
            -RedirectStandardError (Join-Path $LogPath "ai_bot_stderr.log")
        
        # Wait for process to exit
        $Process.WaitForExit()
        $ExitCode = $Process.ExitCode
        
        Write-Log "Bot exited with code: $ExitCode"
        
    } catch {
        Write-Log "ERROR: $($_.Exception.Message)"
    }
    
    $RestartCount++
    
    # Wait before restart
    Write-Log "Restarting in 30 seconds..."
    Start-Sleep -Seconds 30
}

Write-Log "Max restarts reached. Service stopped."
