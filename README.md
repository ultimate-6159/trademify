# 🤖 Trademify - AI Trading Bot

**ระบบเทรดอัตโนมัติด้วย AI - Pattern Recognition + Multi-Factor Analysis**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![MT5](https://img.shields.io/badge/MT5-Forex-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

- **🔍 FAISS Pattern Recognition** - ค้นหา Pattern ที่คล้ายในประวัติ (milliseconds)
- **🧠 100+ AI Indicators** - RSI, MACD, Smart Money, Order Flow, Sentiment
- **⭐ Quality Filtering** - PREMIUM/HIGH/MEDIUM/LOW signal filtering
- **📊 Auto Trading** - เทรดอัตโนมัติผ่าน MT5
- **🛡️ Risk Management** - Max 2% per trade, 5% daily loss limit
- **☁️ Firebase Sync** - Real-time dashboard sync

## 🚀 Quick Install (Windows VPS)

**One-Click Installation:**

```powershell
# Run as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
irm https://raw.githubusercontent.com/ultimate-6159/trademify/main/vps/setup-vps-complete.ps1 | iex
```

หรือ download และรัน:

```powershell
# 1. Clone repository
git clone https://github.com/ultimate-6159/trademify.git C:\trademify
cd C:\trademify

# 2. Run setup script
powershell -ExecutionPolicy Bypass -File vps\setup-vps-complete.ps1
```

## 📋 Requirements

- Windows 10/11 หรือ Windows Server 2016+
- Python 3.11+
- MetaTrader 5 (จาก broker ของคุณ)
- 4GB RAM ขึ้นไป

## ⚙️ Configuration

แก้ไขไฟล์ `backend\.env`:

```env
# MT5 Credentials (จาก broker)
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=Your-Broker-Server

# Trading Settings
TRADING_MODE=SNIPER          # SNIPER/CONSERVATIVE/BALANCED/AGGRESSIVE
MIN_CONFIDENCE=65.0          # Minimum confidence %
MAX_RISK_PER_TRADE=2.0       # Max risk per trade %
MAX_DAILY_LOSS=5.0           # Max daily loss %
```

## 🎮 Usage

### Desktop Shortcuts (หลังติดตั้ง)

| Shortcut           | คำอธิบาย        |
| ------------------ | --------------- |
| `Start Trademify`  | เริ่ม API + Bot |
| `Stop Trademify`   | หยุดทุก service |
| `Trademify Status` | ดูสถานะ         |

### Command Line

```batch
:: Start API + Bot
vps\start-services.bat

:: Stop all
vps\stop-services.bat

:: Check status
vps\check-status.bat

:: Run bot directly
start-bot.bat MT5 EURUSDm,GBPUSDm,XAUUSDm H1 HIGH 60
```

### Bot Parameters

```
start-bot.bat [BROKER] [SYMBOLS] [TIMEFRAME] [QUALITY] [INTERVAL]

BROKER    : MT5 (default)
SYMBOLS   : EURUSDm,GBPUSDm,XAUUSDm (comma-separated, Exness format)
TIMEFRAME : H1 (M5/M15/H1/H4/D1)
QUALITY   : MEDIUM (PREMIUM/HIGH/MEDIUM/LOW)
INTERVAL  : 60 (seconds between analysis)
```

## 📊 Signal Quality Levels

| Level       | Confidence | คำแนะนำ              |
| ----------- | ---------- | -------------------- |
| **PREMIUM** | ≥85%       | ปลอดภัยสุด, เทรดน้อย |
| **HIGH**    | ≥75%       | แนะนำทั่วไป          |
| **MEDIUM**  | ≥65%       | เทรดบ่อยขึ้น         |
| **LOW**     | ≥50%       | เสี่ยงสูง            |

## 🏗️ Project Structure

```
C:\trademify\
├── backend/
│   ├── ai_trading_bot.py      # 🤖 Main Trading Bot
│   ├── api/main.py            # FastAPI Server
│   ├── analysis/              # AI Analysis Modules
│   ├── trading/               # Trading Engine + Intelligence
│   └── .env                   # Configuration
├── frontend/                  # Vue.js Dashboard (optional)
├── vps/
│   ├── setup-vps-complete.ps1 # One-click installer
│   ├── start-services.bat     # Start all
│   ├── stop-services.bat      # Stop all
│   └── check-status.bat       # Status check
├── start-bot.bat              # Quick bot start
└── README.md
```

## 🔗 API Endpoints

| Endpoint                    | Method | Description    |
| --------------------------- | ------ | -------------- |
| `/health`                   | GET    | Health check   |
| `/api/v1/bot/status`        | GET    | Bot status     |
| `/api/v1/bot/start`         | POST   | Start bot      |
| `/api/v1/bot/stop`          | POST   | Stop bot       |
| `/api/v1/trading/positions` | GET    | Open positions |

**API Docs:** http://localhost:8000/docs

## 🛡️ Safety Features

1. **Risk Limits** - Max 2% per trade, 5% daily, 10% drawdown
2. **Quality Filter** - Only trade high-confidence signals
3. **Session Filter** - Best during London-NY overlap
4. **News Filter** - Pause during major news
5. **Trailing Stop** - Lock profits automatically
6. **Break-Even** - Move SL to entry when profitable

## 🔧 Troubleshooting

### MT5 ไม่เชื่อมต่อ

```powershell
# ตรวจสอบ MT5
cd C:\trademify\backend
..\venv\Scripts\Activate.ps1
python check_mt5.py
```

### API ไม่ตอบสนอง

```batch
:: Restart services
vps\stop-services.bat
vps\start-services.bat
```

### ดู Logs

```powershell
Get-Content C:\trademify\backend\logs\trading_bot.log -Tail 50
```

## ⚠️ Disclaimer

> **คำเตือน**: การเทรดมีความเสี่ยงสูง ผลการเทรดในอดีตไม่รับประกันผลในอนาคต
> ใช้งานด้วยความรับผิดชอบของตัวเอง ทดสอบด้วย Demo Account ก่อนเสมอ

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

**Made with ❤️ for Smart Traders**
