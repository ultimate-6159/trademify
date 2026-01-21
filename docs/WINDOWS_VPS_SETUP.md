# ============================================
# Trademify - Windows VPS Setup
# การติดตั้งบน Windows VPS (Binance + MT5)
# ============================================

## 🖥️ ข้อกำหนดขั้นต่ำ

| Component | Requirement |
|-----------|-------------|
| OS | Windows Server 2016+ หรือ Windows 10/11 |
| RAM | 4GB+ (6GB+ สำหรับ multi-symbol) |
| Storage | 20GB+ (ต้องพื้นที่สำหรับ data) |
| CPU | 2 cores+ |
| Internet | Stable connection |
| .NET | .NET Framework 4.5+ (สำหรับ MT5) |

## 📋 VPS Providers แนะนำ

| Provider | ราคา/เดือน | Location | ข้อเหน็จ |
|----------|-----------|----------|---------|
| **Vultr** | $6-24 | Tokyo, Singapore | SSD, ดีสำหรับ crypto |
| **DigitalOcean** | $12-48 | Singapore | เสถียร, ดีสำหรับ API |
| **Contabo** | €5-10 | Germany | ราคาถูก, CPU ดี |
| **ForexVPS** | $20-35 | NY, London | เหมาะ forex, low latency |
| **BeeksFX** | £20-40 | Equinix NY4 | Equinix, ultra-low latency |

💡 **Tip**: 
- สำหรับ **Binance Futures** → เลือก Vultr Singapore (latency ต่ำ)
- สำหรับ **MetaTrader 5** → เลือก ForexVPS (latency ให้ broker)
- สำหรับ **API Server** → DigitalOcean (ความเสถียร)

---

## 🚀 ขั้นตอนการติดตั้ง

### Step 1: เชื่อมต่อ VPS

1. ใช้ **Remote Desktop Connection** (RDP)
2. เปิด `mstsc.exe` บน Windows
3. ใส่ IP address ของ VPS
4. Login ด้วย Username/Password

### Step 2: ติดตั้ง Required Software

#### 2.1 Git
1. ดาวน์โหลด Git จาก https://git-scm.com/download/win
2. ติดตั้งแบบ default

#### 2.2 Python 3.11+
1. ดาวน์โหลด Python 3.11+ จาก https://www.python.org/downloads/
2. ติดตั้ง โดยเลือก **"Add Python to PATH"** ✓
3. เปิด PowerShell ตรวจสอบ:
```powershell
python --version
pip --version
```

#### 2.3 Node.js (ถ้าต้อง Frontend)
1. ดาวน์โหลด LTS จาก https://nodejs.org/
2. ติดตั้ง default

### Step 3: ติดตั้ง Trademify

เปิด **PowerShell (Administrator)**:

```powershell
# Clone repository
git clone https://github.com/ultimate-6159/trademify.git
cd trademify

# สร้าง virtual environment
python -m venv venv
.\venv\Scripts\Activate

# ติดตั้ง dependencies
cd backend
pip install -r requirements.txt

# ติดตั้ง MetaTrader5 Python package (ถ้าใช้ MT5)
pip install MetaTrader5
```

### Step 4: Option A - ตั้งค่า Binance Futures

#### 4A.1 สร้าง API Key

1. ไปที่ https://testnet.binance.vision/ (ทดสอบ) หรือ https://www.binance.com/ (จริง)
2. Security → API Management
3. สร้าง API Key ด้วย permissions:
   - ✓ Spot Trading
   - ✓ Futures Trading
   - ✗ Withdrawal

#### 4A.2 ตั้งค่า Configuration

สร้างไฟล์ `backend\.env`:

```ini
# Environment
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# Trading Configuration
TRADING_ENABLED=true
BROKER_TYPE=BINANCE
PAPER_TRADING=false

# Binance Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true  # true = testnet, false = real

# Risk Management
MAX_RISK_PER_TRADE=2.0
MAX_DAILY_LOSS=5.0
MAX_POSITIONS=5
MIN_CONFIDENCE=70.0
```

#### 4A.3 เริ่มเทรด Binance

```powershell
cd backend

# Paper Trading Mode (Mock Broker)
python trading_bot.py --symbol BTCUSDT --timeframe H1

# Binance Testnet
python trading_bot.py --symbol BTCUSDT --timeframe H1 --real
```

### Step 4: Option B - ตั้งค่า MetaTrader 5

#### 4B.1 ติดตั้ง MT5

1. ดาวน์โหลด MT5 จาก broker หรือ https://www.metatrader5.com/
2. ติดตั้ง MT5
3. Login บัญชี Demo หรือ Real
4. **สำคัญ**: เปิด MT5 ค้างไว้ตลอดเวลา ⚠️

#### 4B.2 ตั้งค่า Configuration

สร้างไฟล์ `backend\.env`:

```ini
# Environment
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# Trading Configuration
TRADING_ENABLED=true
BROKER_TYPE=MT5
PAPER_TRADING=false

# MT5 Configuration
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# Risk Management
MAX_RISK_PER_TRADE=2.0
MAX_DAILY_LOSS=5.0
MAX_POSITIONS=5
MIN_CONFIDENCE=70.0
```

#### 4B.3 ทดสอบการเชื่อมต่อ

```powershell
cd backend
python -c "
import MetaTrader5 as mt5
if mt5.initialize():
    print('✓ MT5 initialized')
    print(f'✓ Account: {mt5.account_info().login}')
    mt5.shutdown()
else:
    print('❌ MT5 initialization failed')
"
```

#### 4B.4 เริ่มเทรด MT5

```powershell
cd backend
python trading_bot_mt5.py --symbol EURUSD --timeframe H1
```

### Step 5: เริ่มต้น API Server

```powershell
# Terminal 1: API Server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Trading Bot
python trading_bot.py --symbol BTCUSDT --timeframe H1 --real
```

URL สำหรับ API Documentation: `http://YOUR_VPS_IP:8000/docs`

---

## 🔄 ตั้งค่า Auto-Start เมื่อ VPS Restart

### วิธีที่ 1: Task Scheduler

1. เปิด **Task Scheduler** (taskschd.msc)
2. Create Basic Task:
   - Name: `Trademify Bot`
   - Trigger: `At startup`
   - Action: `Start a program`
   - Program: `C:\trademify\start-bot.bat`

### วิธีที่ 2: สร้าง Startup Script

สร้างไฟล์ `C:\trademify\start-bot.bat`:

```batch
@echo off
echo Starting Trademify Trading Bot...

REM Wait for MT5 to start
timeout /t 30

REM Activate virtual environment
cd C:\trademify
call venv\Scripts\activate

REM Start the bot
cd backend
python trading_bot.py --symbol EURUSD --timeframe H1 --interval 60 --real

pause
```

ใส่ shortcut ใน:
```
C:\Users\<username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

---

## 📊 Multi-Symbol Trading

สร้างไฟล์ `run_all_bots.py`:

```python
import asyncio
import subprocess
import sys

SYMBOLS = [
    ("EURUSD", "H1"),
    ("GBPUSD", "H1"),
    ("XAUUSD", "M15"),
]

async def run_bot(symbol, timeframe):
    cmd = [
        sys.executable, "trading_bot.py",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--interval", "60",
        "--real"
    ]
    process = await asyncio.create_subprocess_exec(*cmd)
    await process.wait()

async def main():
    tasks = [run_bot(s, tf) for s, tf in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛡️ Security Best Practices

### 1. Firewall
```powershell
# Allow only necessary ports
netsh advfirewall firewall add rule name="Trademify API" dir=in action=allow protocol=tcp localport=8000
```

### 2. Change RDP Port (แนะนำ)
```powershell
# Change from default 3389 to custom port
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' -Name PortNumber -Value 33890
```

### 3. Use Strong Passwords

### 4. Enable Auto-Updates

---

## 🔧 Troubleshooting

### ปัญหา: MT5 ไม่ connect

```powershell
# ตรวจสอบ MT5 process
Get-Process | Where-Object {$_.Name -like "*terminal*"}

# Restart MT5
Stop-Process -Name "terminal64" -Force
Start-Process "C:\Program Files\MetaTrader 5\terminal64.exe"
```

### ปัญหา: Bot หยุดทำงาน

ตรวจสอบ log:
```powershell
Get-Content backend\logs\trading.log -Tail 100
```

### ปัญหา: Connection timeout

ตรวจสอบ internet:
```powershell
Test-NetConnection -ComputerName your-broker-server.com -Port 443
```

---

## 📈 Monitoring

### ดู Status
```powershell
curl http://localhost:8000/api/v1/trading/status
```

### ดู Positions
```powershell
curl http://localhost:8000/api/v1/trading/positions
```

### ดู Log แบบ Real-time
```powershell
Get-Content backend\logs\trading.log -Wait
```

---

## 💡 Tips

1. **ใช้ Demo Account ทดสอบก่อนเสมอ** - Binance Testnet หรือ MT5 Demo
2. **เปิด MT5 ค้างไว้ตลอด** - Bot ต้องการ MT5 terminal (สำหรับ MT5 mode)
3. **ตรวจสอบ VPS ทุกวัน** - Windows อาจ auto-update restart
4. **Backup configuration** - เก็บไฟล์ .env ไว้ที่อื่นด้วย
5. **Monitor balance** - ตั้ง alert ถ้า balance ต่ำกว่าที่กำหนด
6. **ใช้ Risk Management** - อย่าเกิน max_risk_per_trade 2%
7. **Monitor API logs** - `Get-Content backend\logs\trading.log -Tail 100`

---

## 📖 Related Documentation

- [QUICKSTART.md](../QUICKSTART.md) - เริ่มต้นใช้งานเร็ว
- [README.md](../README.md) - Architecture Overview
- [vps/README.md](../vps/README.md) - VPS Automation Scripts
