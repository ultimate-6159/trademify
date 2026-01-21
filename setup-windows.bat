@echo off
REM ============================================
REM Trademify - Windows Setup Script for MT5
REM ============================================

echo.
echo =============================================
echo    Trademify - MT5 Trading Bot Setup
echo =============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python found

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
cd backend
pip install -r requirements.txt
pip install MetaTrader5

REM Create .env if not exists
if not exist ".env" (
    echo.
    echo Creating configuration file...
    (
        echo # Trademify MT5 Configuration
        echo DEBUG=false
        echo API_HOST=0.0.0.0
        echo API_PORT=8000
        echo.
        echo # Trading
        echo TRADING_ENABLED=true
        echo PAPER_TRADING=false
        echo BROKER_TYPE=MT5
        echo.
        echo # MT5 Credentials - EDIT THESE VALUES
        echo MT5_LOGIN=0
        echo MT5_PASSWORD=
        echo MT5_SERVER=
        echo MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
        echo.
        echo # Risk Management
        echo MAX_RISK_PER_TRADE=2.0
        echo MAX_DAILY_LOSS=5.0
        echo MAX_POSITIONS=5
        echo MIN_CONFIDENCE=70.0
    ) > .env
    
    echo.
    echo [IMPORTANT] Please edit backend\.env with your MT5 credentials!
    echo.
)

REM Test MT5 connection
echo.
echo Testing MT5 connection...
python -c "
import MetaTrader5 as mt5
if mt5.initialize():
    print('[OK] MT5 initialized successfully')
    account = mt5.account_info()
    if account:
        print(f'[OK] Account: {account.login}')
        print(f'[OK] Balance: ${account.balance:,.2f}')
        print(f'[OK] Server: {account.server}')
    mt5.shutdown()
else:
    error = mt5.last_error()
    print(f'[WARNING] MT5 not running or not logged in')
    print(f'         Error: {error}')
    print('')
    print('Make sure:')
    print('1. MetaTrader 5 is installed')
    print('2. MT5 terminal is running')
    print('3. You are logged into your account')
"

echo.
echo =============================================
echo Setup Complete!
echo =============================================
echo.
echo Next steps:
echo 1. Edit backend\.env with your MT5 credentials
echo 2. Make sure MT5 terminal is running and logged in
echo 3. Run: start-bot.bat
echo.
pause
