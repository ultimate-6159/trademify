# Trademify VPS Scripts

สคริปต์สำหรับติดตั้งและรัน Trademify บน Windows VPS (Binance + MT5)

## 📦 ไฟล์ในโฟลเดอร์นี้

| File | Description |
|------|-------------|
| `setup-vps-complete.ps1` | ติดตั้งทุกอย่างอัตโนมัติ (รันครั้งเดียว) |
| `start-services.bat` | เริ่มต้น Backend API + Trading Bot |
| `stop-services.bat` | หยุดทุก Services |
| `start-with-monitor.bat` | เริ่มต้นพร้อม Auto-Restart เมื่อ crash |
| `service-monitor.ps1` | Monitor script สำหรับ auto-restart |
| `setup-autostart.ps1` | ตั้งค่า auto-start เมื่อ VPS reboot |
| `check-status.bat` | ตรวจสอบสถานะ services |
| `update.bat` | อัปเดตจาก GitHub + restart services |

## 🚀 Quick Start (บน VPS)

### วิธีที่ 1: One-Click Install (แนะนำ)

เปิด PowerShell (Admin) แล้วรัน:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ultimate-6159/trademify/main/vps/setup-vps-complete.ps1" -OutFile "C:\setup.ps1"
C:\setup.ps1
```

### วิธีที่ 2: Manual Install

```powershell
# Clone repo
git clone https://github.com/ultimate-6159/trademify.git C:\trademify

# Run setup
cd C:\trademify\vps
powershell -ExecutionPolicy Bypass -File setup-vps-complete.ps1
```

---

## ⚙️ Configuration

หลังจากติดตั้ง ต้องสร้างไฟล์ `C:\trademify\backend\.env`:

```ini
# Trading Configuration
TRADING_ENABLED=true
BROKER_TYPE=BINANCE  # หรือ MT5

# Binance
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
BINANCE_TESTNET=true

# หรือ MT5
# MT5_LOGIN=12345678
# MT5_PASSWORD=xxx
# MT5_SERVER=YourBroker-Server

# Risk
MAX_RISK_PER_TRADE=2.0
MAX_DAILY_LOSS=5.0
```

## 🔄 การใช้งานประจำวัน

### เริ่มต้น Services
```batch
C:\trademify\vps\start-services.bat
```

### หยุด Services
```batch
C:\trademify\vps\stop-services.bat
```

### เริ่มต้นพร้อม Auto-Restart (แนะนำ)
```batch
C:\trademify\vps\start-with-monitor.bat
```

### ตรวจสอบสถานะ
```batch
C:\trademify\vps\check-status.bat
```

## ⚙️ Auto-Start เมื่อ VPS Reboot

รัน script นี้ครั้งเดียว:

```powershell
cd C:\trademify\vps
powershell -ExecutionPolicy Bypass -File setup-autostart.ps1
```

## 📊 Logs

Logs จะถูกเก็บที่ `C:\trademify\logs\`:
- `backend.log` - Backend API logs
- `frontend.log` - Frontend logs  
- `trading.log` - Trading Bot logs
- `monitor.log` - Service Monitor logs

## 🌐 URLs

หลังจากเริ่มต้น services:

| Service | URL |
|---------|-----|
| Frontend Dashboard | `http://YOUR_VPS_IP:5173` |
| API Documentation | `http://YOUR_VPS_IP:8000/docs` |
| Trading Status | `http://YOUR_VPS_IP:8000/api/v1/trading/status` |
| Positions | `http://YOUR_VPS_IP:8000/api/v1/trading/positions` |

---

## 📖 Related Documentation

- [QUICKSTART.md](../QUICKSTART.md) - เริ่มต้นใช้งาน
- [docs/WINDOWS_VPS_SETUP.md](../docs/WINDOWS_VPS_SETUP.md) - รายละเอียดการติดตั้ง VPS
