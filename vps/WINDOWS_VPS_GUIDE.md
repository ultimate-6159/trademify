# 🖥️ Trademify - Windows VPS Setup Guide
## สำหรับ Windows Server 2016+ / Windows 10/11

---

## 📋 สารบัญ
1. [ขั้นตอนติดตั้งครั้งแรก](#1-ขั้นตอนติดตั้งครั้งแรก)
2. [ตั้งค่า MT5 Terminal](#2-ตั้งค่า-mt5-terminal)
3. [ตั้งค่า Auto-Start & Monitor](#3-ตั้งค่า-auto-start--monitor)
4. [คำสั่งใช้งานประจำวัน](#4-คำสั่งใช้งานประจำวัน)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. ขั้นตอนติดตั้งครั้งแรก

### 1.1 เชื่อมต่อ VPS ผ่าน Remote Desktop (RDP)
```
mstsc /v:YOUR_VPS_IP
```

### 1.2 ติดตั้งอัตโนมัติ (One Click)

เปิด **PowerShell (Administrator)** แล้วรัน:

```powershell
# Enable script execution
Set-ExecutionPolicy Bypass -Scope Process -Force

# Download and run setup script
cd C:\
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ultimate-6159/trademify/main/vps/setup-vps-complete.ps1" -OutFile "setup.ps1"
.\setup.ps1
```

หรือถ้า clone มาแล้ว:
```powershell
cd C:\trademify\vps
.\setup-vps-complete.ps1
```

### 1.3 ตั้งค่า Environment (.env)

แก้ไขไฟล์ `C:\trademify\backend\.env`:

```ini
# MT5 Configuration (สำคัญ!)
MT5_LOGIN=YOUR_MT5_LOGIN
MT5_PASSWORD=YOUR_MT5_PASSWORD
MT5_SERVER=YOUR_BROKER_SERVER
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# ปิด Mock Mode (เพราะรันบน Windows จริง)
MT5_MOCK_MODE=false
```

---

## 2. ตั้งค่า MT5 Terminal

### 2.1 ดาวน์โหลดและติดตั้ง MT5
1. ดาวน์โหลดจาก broker (เช่น Exness, XM, ICMarkets)
2. ติดตั้งที่ `C:\Program Files\MetaTrader 5\`
3. Login เข้าบัญชีเทรด

### 2.2 ตั้งค่า MT5 สำหรับ Auto Trading

1. **เปิด MT5 Terminal**
2. **Tools → Options → Expert Advisors**
   - ✅ Allow automated trading
   - ✅ Allow DLL imports
3. **Tools → Options → Server**
   - ✅ Enable news (ถ้าต้องการ)
4. **Login** เข้าบัญชีเทรดของคุณ

### 2.3 ตั้งค่า MT5 Auto-Start

สร้าง Shortcut ใน Startup:
```powershell
# รันใน PowerShell (Admin)
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\MT5.lnk")
$Shortcut.TargetPath = "C:\Program Files\MetaTrader 5\terminal64.exe"
$Shortcut.Arguments = "/portable"
$Shortcut.Save()

Write-Host "MT5 will auto-start on Windows boot" -ForegroundColor Green
```

---

## 3. ตั้งค่า Auto-Start & Monitor

### 3.1 ติดตั้ง Auto-Start Task

```powershell
cd C:\trademify\vps
.\setup-autostart.ps1
```

สิ่งที่จะเกิดขึ้น:
- ✅ สร้าง Scheduled Task "Trademify Auto Start"
- ✅ เริ่มทำงานอัตโนมัติหลัง Windows boot 60 วินาที
- ✅ Monitor และ auto-restart ถ้า crash

### 3.2 เริ่มใช้งานทันที

```batch
C:\trademify\vps\start-with-monitor.bat
```

### 3.3 ตรวจสอบสถานะ

```batch
C:\trademify\vps\check-status.bat
```

---

## 4. คำสั่งใช้งานประจำวัน

### 🟢 เริ่มบริการ
```batch
C:\trademify\vps\start-services.bat
```

### 🔴 หยุดบริการ
```batch
C:\trademify\vps\stop-services.bat
```

### 🔄 อัพเดทโค้ด
```batch
C:\trademify\vps\update.bat
```

### 📊 ตรวจสอบสถานะ
```batch
C:\trademify\vps\check-status.bat
```

### 📋 ดู Logs
```powershell
# Monitor log
Get-Content C:\trademify\logs\monitor.log -Tail 50

# Trading log
Get-Content C:\trademify\backend\logs\trading_bot.log -Tail 50
```

---

## 5. Troubleshooting

### ❌ MT5 ไม่เชื่อมต่อ

1. **ตรวจสอบ MT5 Terminal รันอยู่**
   ```powershell
   Get-Process terminal64 -ErrorAction SilentlyContinue
   ```

2. **ตรวจสอบ Login**
   - ดูที่ MT5 Terminal → ข้อมูลบัญชีถูกต้อง
   - Server name ต้องตรงกับใน `.env`

3. **ตรวจสอบ Path**
   ```powershell
   Test-Path "C:\Program Files\MetaTrader 5\terminal64.exe"
   ```

### ❌ Backend ไม่ทำงาน

```powershell
# ดู error
cd C:\trademify
.\venv\Scripts\Activate.ps1
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### ❌ Service ไม่ auto-restart

```powershell
# ตรวจสอบ Task
Get-ScheduledTask -TaskName "Trademify Auto Start"

# รัน Task ด้วยมือ
Start-ScheduledTask -TaskName "Trademify Auto Start"

# ดู Task history
Get-ScheduledTask -TaskName "Trademify Auto Start" | Get-ScheduledTaskInfo
```

### ❌ Port ถูกใช้งานอยู่แล้ว

```powershell
# หา process ที่ใช้ port 8000
netstat -ano | findstr :8000

# Kill process (แทน PID)
taskkill /PID <PID> /F
```

---

## 📞 URLs หลังติดตั้งสำเร็จ

| Service | URL |
|---------|-----|
| Frontend | http://YOUR_VPS_IP:5173 |
| Backend API | http://YOUR_VPS_IP:8000 |
| API Docs | http://YOUR_VPS_IP:8000/docs |
| Health Check | http://YOUR_VPS_IP:8000/health |

---

## ⚠️ ข้อควรระวัง

1. **อย่าปิด MT5 Terminal** - Bot ต้องการ MT5 เปิดตลอด
2. **ใช้ Demo Account ก่อน** - ทดสอบให้มั่นใจก่อนใช้เงินจริง
3. **Monitor ทุกวัน** - ตรวจสอบ logs และ positions
4. **Backup .env** - เก็บไฟล์ config ไว้ที่ปลอดภัย
5. **Windows Update** - ตั้งให้ update ช่วงที่ตลาดปิด (Weekend)

---

## 🔄 Auto-Update Script

สำหรับ update อัตโนมัติทุกวัน (optional):

```powershell
# สร้าง Daily Update Task
$Action = New-ScheduledTaskAction -Execute "C:\trademify\vps\update.bat"
$Trigger = New-ScheduledTaskTrigger -Daily -At "04:00"  # 4 AM
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable

Register-ScheduledTask -TaskName "Trademify Daily Update" `
    -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Auto-update Trademify code daily"
```

---

💡 **Support**: หากมีปัญหา ให้ส่ง logs มาที่ GitHub Issues
