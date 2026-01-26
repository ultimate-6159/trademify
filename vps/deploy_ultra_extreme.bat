@echo off
REM ============================================
REM ?????? DEPLOY ULTRA EXTREME CONFIG ??????
REM ============================================
REM Run this on VPS to apply ULTRA EXTREME settings
REM Backtest: $1,000 ? $27,266,556 (+2,726,555%)
REM ============================================

echo.
echo ============================================
echo ?????? DEPLOYING ULTRA EXTREME CONFIG ??????
echo ============================================
echo.

cd /d C:\trademify

REM 1. Stop services
echo ?? Stopping services...
call vps\stop-services.bat

REM 2. Pull latest code
echo ?? Pulling latest code from GitHub...
git pull origin main

REM 3. Backup current .env
echo ?? Backing up current .env...
if exist backend\.env (
    copy backend\.env backend\.env.backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.bak
)

REM 4. Copy ULTRA EXTREME config
echo ?? Applying ULTRA EXTREME config...
copy backend\.env.ultra_extreme backend\.env

REM 5. Show new config
echo.
echo ============================================
echo ?? NEW ULTRA EXTREME SETTINGS:
echo ============================================
type backend\.env | findstr /i "MIN_PASS_RATE MIN_HIGH_QUALITY MIN_KEY_AGREEMENT MAX_RISK MAX_DAILY_LOSS"
echo ============================================
echo.

REM 6. Start services
echo ?? Starting services with ULTRA EXTREME config...
call vps\start-services.bat

echo.
echo ============================================
echo ? ULTRA EXTREME DEPLOYMENT COMPLETE!
echo ============================================
echo.
echo ?? Expected Results:
echo    - Win Rate: 92.8%%
echo    - Profit Factor: 2.43
echo    - Trades/Day: ~72
echo    - Max Drawdown: 24-27%%
echo.
echo ?? WARNING: HIGH RISK! Monitor closely!
echo ============================================

pause
