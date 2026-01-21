# 🚀 Trademify AI Bot - Quick Start Guide

## ⚡ เริ่มต้นใน 3 นาที

### Step 1: Setup Environment

```bash
cd trademify
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

cd backend
pip install -r requirements.txt
```

### Step 2: Start AI Bot (Paper Trading)

```bash
# Forex (MT5)
python ai_trading_bot.py --broker MT5 --symbols EURUSD,GBPUSD,XAUUSD --quality HIGH

# Crypto (Binance)
python ai_trading_bot.py --broker BINANCE --symbols BTCUSDT,ETHUSDT --quality HIGH
```

### Step 3: เปิด Dashboard

```bash
# Terminal อื่น
uvicorn api.main:app --host 0.0.0.0 --port 8000

# เปิดเบราว์เซอร์
# http://localhost:8000/docs
```

---

## 📊 Quality Levels (เลือกตามความเสี่ยง)

| Level | Command | คำแนะนำ |
|-------|---------|---------|
| **PREMIUM** | `--quality PREMIUM` | ปลอดภัยสุด (85%+ confidence) |
| **HIGH** | `--quality HIGH` | แนะนำ (75%+) |
| **MEDIUM** | `--quality MEDIUM` | เทรดบ่อยขึ้น (65%+) |
| **LOW** | `--quality LOW` | เสี่ยงสูง (50%+) |

---

## 🎯 ตัวอย่างการใช้งาน

### 1. Paper Trading (ทดสอบ - ไม่ใช้เงินจริง)

```bash
# Forex ปลอดภัยสุด
python ai_trading_bot.py --broker MT5 --symbols EURUSD --quality PREMIUM

# Crypto หลายคู่
python ai_trading_bot.py --broker BINANCE --symbols BTCUSDT,ETHUSDT,BNBUSDT --quality HIGH
```

### 2. Live Trading (⚠️ ระวัง - ใช้เงินจริง!)

```bash
# ต้องตั้งค่า Environment Variables ก่อน
export MT5_LOGIN=12345678
export MT5_PASSWORD=your_password
export MT5_SERVER=YourBroker-Server

# เทรดจริง (มีการแจ้งเตือน 5 วินาที)
python ai_trading_bot.py --broker MT5 --symbols EURUSD --quality PREMIUM --real
```

### 3. Docker

```bash
# Start ทั้งระบบ
docker-compose up -d

# พร้อม AI Bot
docker-compose --profile bot up -d
```

---

## ⚙️ Options ทั้งหมด

```
--broker      MT5 หรือ BINANCE (default: MT5)
--symbols     สัญลักษณ์คั่นด้วย comma (default: EURUSD,GBPUSD,XAUUSD)
--timeframe   M5, M15, M30, H1, H4, D1 (default: H1)
--htf         Higher Timeframe for MTF (default: H4)
--quality     PREMIUM, HIGH, MEDIUM, LOW (default: HIGH)
--interval    วินาทีระหว่างรอบวิเคราะห์ (default: 60)
--risk        % ความเสี่ยงต่อเทรด (default: 2.0)
--real        ⚠️ เทรดจริง (default: false = paper trading)
--testnet     ใช้ testnet/demo (default: false)
```

---

## 🛡️ ความปลอดภัย

1. **เริ่มต้น Paper Trading เสมอ** - ไม่ต้องใส่ `--real`
2. **ใช้ PREMIUM หรือ HIGH** - สำหรับ Live Trading
3. **ตั้ง Risk 1-2%** - ป้องกันการขาดทุนหนัก
4. **Monitor Dashboard** - ดูสถานะ Real-time

---

## ❓ FAQ

**Q: ทำไมไม่เห็น Signal?**
A: รอให้ Bot สร้าง Pattern Index ก่อน (ครั้งแรกใช้เวลา 1-2 นาที)

**Q: Paper Trading คืออะไร?**
A: โหมดจำลอง ไม่ใช้เงินจริง เหมาะสำหรับทดสอบ

**Q: PREMIUM vs HIGH ต่างกันยังไง?**
A: PREMIUM เข้าเฉพาะสัญญาณแน่นอนมาก (85%+), HIGH ยืดหยุ่นกว่า (75%+)

---

**Ready to trade! 🚀**
