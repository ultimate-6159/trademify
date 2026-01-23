# Trademify - AI Trading Bot Expert System

## 🚨 Production Environment

**Platform**: Windows VPS (Windows Server / Windows 10/11)
**Broker**: Exness MT5 (or any MT5 broker)
**Symbols**: `EURUSDm`, `GBPUSDm`, `XAUUSDm` (Exness micro - มี "m" ต่อท้าย)

## Architecture

```
FAISS Pattern Matching → Multi-Factor AI Analysis → Quality Filter → MT5 Trading
                                                         ↓
                                                   Firebase Sync
```

**Main Entry Point**: `backend/ai_trading_bot.py`

## Key Concepts

- **Window Size**: 60 candles per pattern
- **Quality Levels**: PREMIUM (85%+), HIGH (75%+), MEDIUM (65%+), LOW (50%+)
- **Risk Management**: Max 2% per trade, 5% daily loss, 10% max drawdown

## Directory Structure

```
C:\trademify\
├── backend/
│   ├── ai_trading_bot.py      # Main trading bot
│   ├── api/main.py            # FastAPI server
│   ├── analysis/              # AI analysis modules
│   ├── trading/               # Trading engine + connectors
│   ├── config/                # Configuration
│   └── .env                   # Environment variables
├── vps/
│   ├── setup-vps-complete.ps1 # One-click installer
│   ├── start-services.bat     # Start all services
│   ├── stop-services.bat      # Stop all services
│   └── check-status.bat       # Status check
└── start-bot.bat              # Quick bot start
```

## Commands

```bash
# Start all (API + Bot)
vps\start-services.bat

# Stop all
vps\stop-services.bat

# Run bot directly
start-bot.bat MT5 EURUSDm,GBPUSDm,XAUUSDm H1 MEDIUM 60
```

## Configuration

Edit `backend\.env`:

```env
MT5_LOGIN=your_account
MT5_PASSWORD=your_password
MT5_SERVER=Your-Broker-Server
TRADING_MODE=SNIPER
MIN_CONFIDENCE=65.0
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/v1/bot/status` - Bot status
- `POST /api/v1/bot/start` - Start bot
- `POST /api/v1/bot/stop` - Stop bot

## Development

```bash
# Activate venv
.\venv\Scripts\Activate.ps1

# Run API
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Test MT5
python check_mt5.py
```
