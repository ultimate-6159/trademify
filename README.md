# 🤖 Trademify - AI Trading Bot Expert System

**ระบบเทรดอัตโนมัติด้วย AI เพียงหนึ่งเดียว - แม่นยำ เสถียร ฉลาดล้ำลึก**

![Trademify AI Bot](https://img.shields.io/badge/AI-Trading%20Bot-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![FAISS](https://img.shields.io/badge/FAISS-Pattern%20Recognition-orange)
![Vue.js](https://img.shields.io/badge/Vue.js-3.x-brightgreen)

## 🎯 What is Trademify?

Trademify เป็นระบบเทรดอัตโนมัติแบบ AI ที่:

1. **🔍 Pattern Recognition** - ค้นหา Pattern กราฟในอดีตที่เหมือนกับปัจจุบัน (FAISS)
2. **🧠 Multi-Factor AI Analysis** - วิเคราะห์ RSI, MACD, Volume, MTF
3. **⭐ Quality Filtering** - กรองสัญญาณตามคุณภาพ (PREMIUM/HIGH/MEDIUM/LOW)
4. **📊 Auto Trading** - เทรดอัตโนมัติตามสัญญาณ AI
5. **🛡️ Risk Management** - บริหารความเสี่ยงอัจฉริยะ

## 🚀 Quick Start

### 1. Paper Trading (แนะนำสำหรับเริ่มต้น)

```bash
# Clone repository
git clone https://github.com/ultimate-6159/trademify.git
cd trademify

# Setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# หรือ: venv\Scripts\activate  # Windows

cd backend
pip install -r requirements.txt

# Start AI Bot (Paper Trading - Safe)
python ai_trading_bot.py --broker MT5 --symbols EURUSD,GBPUSD,XAUUSD --quality HIGH
```

### 2. Docker (Production)

```bash
# Start all services
docker-compose up -d

# Start with AI Bot
docker-compose --profile bot up -d
```

### 3. Windows VPS

```batch
# Run
start-bot.bat MT5 EURUSD,GBPUSD,XAUUSD H1 HIGH 60
```

## 📊 AI Trading Bot

### Signal Quality Levels

| Quality  | Confidence | Win Rate | คำแนะนำ |
|----------|------------|----------|---------|
| PREMIUM  | ≥85%       | 85%+     | ปลอดภัยสุด, น้อยเทรด |
| HIGH     | ≥75%       | 75-85%   | **แนะนำ** |
| MEDIUM   | ≥65%       | 65-75%   | เทรดมากขึ้น |
| LOW      | ≥50%       | 50-65%   | เสี่ยงสูง |

### Usage Examples

```bash
# Forex (MT5) - Paper Trading
python ai_trading_bot.py --broker MT5 --symbols EURUSD,GBPUSD,XAUUSD --quality HIGH

# Crypto (Binance) - Paper Trading  
python ai_trading_bot.py --broker BINANCE --symbols BTCUSDT,ETHUSDT --quality HIGH

# Live Trading (⚠️ ระวัง - ใช้เงินจริง!)
python ai_trading_bot.py --broker MT5 --symbols EURUSD --quality PREMIUM --real
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--broker` | MT5 | MT5 (Forex) หรือ BINANCE (Crypto) |
| `--symbols` | EURUSD,GBPUSD,XAUUSD | สัญลักษณ์ที่ต้องการเทรด |
| `--timeframe` | H1 | Timeframe (M5, M15, H1, H4, D1) |
| `--htf` | H4 | Higher Timeframe สำหรับ MTF |
| `--quality` | HIGH | PREMIUM, HIGH, MEDIUM, LOW |
| `--interval` | 60 | ช่วงเวลาวิเคราะห์ (วินาที) |
| `--risk` | 2.0 | % ความเสี่ยงต่อเทรด |
| `--real` | false | ⚠️ เทรดจริง (ใช้เงินจริง) |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  TRADEMIFY AI TRADING BOT                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Data Lake   │    │   FAISS     │    │  Enhanced   │     │
│  │ (.parquet)  │───▶│  Pattern    │───▶│  Analyzer   │     │
│  │             │    │  Matching   │    │  (AI)       │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                              │              │
│                                              ▼              │
│                                      ┌─────────────┐       │
│                                      │   Quality   │       │
│                                      │   Filter    │       │
│                                      └─────────────┘       │
│                                              │              │
│                            ┌─────────────────┼──────────┐  │
│                            ▼                 ▼          ▼  │
│                     ┌──────────┐      ┌──────────┐  ┌─────┐│
│                     │   MT5    │      │ Binance  │  │ API ││
│                     │ (Forex)  │      │ (Crypto) │  │     ││
│                     └──────────┘      └──────────┘  └─────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure (Simplified)

```
trademify/
├── backend/
│   ├── ai_trading_bot.py       # 🤖 AI Trading Bot (หลัก)
│   ├── api/
│   │   └── main.py             # FastAPI Server
│   ├── analysis/
│   │   ├── enhanced_analyzer.py # Multi-factor AI
│   │   └── voting_system.py    # Signal Voting
│   ├── similarity_engine/
│   │   └── faiss_engine.py     # Pattern Matching
│   ├── trading/
│   │   ├── engine.py           # Trading Engine
│   │   ├── binance_connector.py
│   │   └── mt5_connector.py
│   └── requirements.txt
├── frontend/                   # Vue.js Dashboard
├── start-bot.bat               # Windows Quick Start
├── trading-service.sh          # Linux Service
└── docker-compose.yml          # Docker Deployment
```

## ⚙️ Configuration

### Environment Variables

```bash
# MT5 (Forex)
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server

# Binance (Crypto)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

### Risk Management

```python
RiskManager(
    max_risk_per_trade=2.0,    # 2% ต่อเทรด
    max_daily_loss=5.0,        # 5% ต่อวัน
    max_positions=5,           # 5 positions สูงสุด
    max_drawdown=10.0          # 10% drawdown สูงสุด
)
```

## 📊 API Endpoints

### Bot Control
```http
POST /api/v1/bot/start   # Start AI Bot
POST /api/v1/bot/stop    # Stop AI Bot
GET  /api/v1/bot/status  # Bot Status
```

### Analysis
```http
POST /api/v1/build-index  # Build Pattern Index
POST /api/v1/analyze      # Analyze Pattern
GET  /api/v1/events       # SSE Real-time Updates
```

## 🛡️ Safety Features

1. **Paper Trading Default** - เริ่มต้นด้วย Paper Trading เสมอ
2. **Quality Filter** - กรองเฉพาะสัญญาณคุณภาพสูง
3. **Risk Limits** - จำกัดความเสี่ยงอัตโนมัติ
4. **Live Warning** - แจ้งเตือน 5 วินาทีก่อนเทรดจริง
5. **Auto Stop Loss** - ตั้ง SL/TP อัตโนมัติตามประวัติ

## ⚠️ Disclaimer

> **คำเตือน**: ซอฟต์แวร์นี้มีไว้เพื่อการศึกษาเท่านั้น การเทรดมีความเสี่ยงสูงต่อการสูญเสียเงินทุน ผลการเทรดในอดีตไม่ได้รับประกันผลลัพธ์ในอนาคต ใช้งานด้วยความรับผิดชอบของตัวเอง

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

**Made with ❤️ for Smart Traders**
