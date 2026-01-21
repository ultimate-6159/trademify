# Trademify - AI Trading Bot Expert System

## 🚨 IMPORTANT: Production Environment

**This system runs on Windows VPS ONLY - NO MOCK MODE**

- **VPS**: Windows Server at `66.42.50.149`
- **Broker**: Exness MT5 (Exness-MT5Real39)
- **Account**: 267643655 (Standard)
- **Symbols**: `EURUSDm`, `GBPUSDm`, `XAUUSDm` (Exness micro lots - มี "m" ต่อท้าย)
- **MT5 Terminal**: Must be running on VPS with AutoTrading enabled

**Never use Mock mode** - Always connect to real MT5 terminal on VPS.

## Architecture Overview

Trademify is an **AI-powered pattern recognition trading system** with a single, unified bot:

```
Data Lake (.parquet/.npy) → FAISS Pattern Matching → Enhanced AI Analyzer → Quality Filter → Trading Engine → MT5 Broker
                                                                                    ↓
                                                                              Firebase → Vue.js Dashboard
```

**Main Entry Point**: `backend/ai_trading_bot.py` - The ONE and ONLY trading bot
**Backend** (Python/FastAPI): `backend/` - Pattern matching engine + AI Analysis + Auto Trading  
**Frontend** (Vue 3/Pinia): `frontend/` - Real-time visualization with ECharts
**Broker**: Exness MT5 (Windows VPS only)

## Critical Domain Concepts

- **Window Size**: 60 candles per pattern (configurable in `config/settings.py`)
- **Normalization**: Always Z-score normalize before pattern matching
- **Quality Levels**: PREMIUM (85%+), HIGH (75%+), MEDIUM (65%+), LOW (50%+)
- **Correlation Check**: Skip patterns with correlation < 0.85
- **Risk Management**: Max 2% risk per trade, 5% daily loss limit

## Backend Module Responsibilities

| Module | Purpose | Key Class |
|--------|---------|-----------|
| `ai_trading_bot.py` | **Main AI Trading Bot** | `AITradingBot` |
| `data_processing/data_lake.py` | Store OHLC in `.parquet` | `DataLake` |
| `data_processing/normalizer.py` | Z-score/log-return normalization | `Normalizer` |
| `similarity_engine/faiss_engine.py` | Similarity search (millions in ms) | `FAISSEngine`, `PatternMatcher` |
| `analysis/enhanced_analyzer.py` | Multi-factor AI analysis | `EnhancedAnalyzer` |
| `analysis/voting_system.py` | Generate BUY/SELL/WAIT signals | `VotingSystem` |
| `trading/engine.py` | Auto-trading core engine | `TradingEngine`, `RiskManager` |
| `trading/binance_connector.py` | Binance (Crypto) | `BinanceBroker` |
| `trading/mt5_connector.py` | MetaTrader 5 (Forex) | `MT5Broker` |

## AI Trading Bot Usage

```bash
# Production (Windows VPS) - Real Trading
python ai_trading_bot.py --broker MT5 --symbols EURUSDm,GBPUSDm,XAUUSDm --quality MEDIUM

# High Quality Only
python ai_trading_bot.py --broker MT5 --symbols EURUSDm,GBPUSDm,XAUUSDm --quality HIGH

# Premium Only (safest)
python ai_trading_bot.py --broker MT5 --symbols EURUSDm,XAUUSDm --quality PREMIUM
```

**Note**: Always use symbol names with "m" suffix for Exness broker (EURUSDm, not EURUSD)

## Key Patterns & Conventions

### Backend (Python)
- **Single Bot**: `AITradingBot` in `ai_trading_bot.py` - NO other trading bots
- Config classes in `config/settings.py` - `DataConfig`, `PatternConfig`, `VotingConfig`
- All numpy arrays must be `float32` for FAISS compatibility
- Use dataclasses with `to_dict()` for API responses

### Frontend (Vue.js)
- Pinia stores with Composition API
- `signal.js` - Pattern analysis state
- `trading.js` - Auto-trading control state
- ECharts for visualizations
- Tailwind CSS with dark theme

## Development Commands

```bash
# Start AI Bot (Paper Trading)
cd backend
python ai_trading_bot.py --broker MT5 --quality HIGH

# API Server
uvicorn api.main:app --reload --port 8000

# Docker
docker-compose up -d
docker-compose --profile bot up -d  # With AI Bot
```

## API Endpoints

### Bot Control
- `POST /api/v1/bot/start` - Start AI Bot
- `POST /api/v1/bot/stop` - Stop AI Bot
- `GET /api/v1/bot/status` - Bot Status

### Pattern Analysis
- `POST /api/v1/build-index` - Build FAISS index
- `POST /api/v1/analyze` - Analyze pattern

### Auto Trading
- `GET /api/v1/trading/status` - Get trading system status
- `GET /api/v1/trading/settings` - Get current settings
- `PUT /api/v1/trading/settings` - Update settings
- `POST /api/v1/trading/start` - Start auto-trading
- `POST /api/v1/trading/stop` - Stop auto-trading
- `GET /api/v1/trading/positions` - List open positions- `GET /api/v1/trading/status` - Get trading system status
- `GET /api/v1/trading/settings` - Get current settings
- `PUT /api/v1/trading/settings` - Update settings
- `POST /api/v1/trading/start` - Start auto-trading
- `POST /api/v1/trading/stop` - Stop auto-trading
- `GET /api/v1/trading/positions` - List open positions- `POST /api/v1/trading/positions` - Open manual position
- `DELETE /api/v1/trading/positions/{id}` - Close position
- `POST /api/v1/trading/signal` - Process signal for auto-trade

## Adding New Features

- **New symbol**: Add to `DataConfig.DEFAULT_SYMBOLS` in `settings.py`
- **New timeframe**: Add to `DataConfig.TIMEFRAMES`  
- **New signal type**: Extend `Signal` enum in `voting_system.py`
- **New broker**: Extend `BaseBroker` in `trading/engine.py`
- **New chart**: Create component in `frontend/src/components/`, use `v-chart`
