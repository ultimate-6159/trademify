#!/bin/bash
# ===========================================
# Trademify - Setup Trading (Binance/MT5)
# ตั้งค่าสำหรับการเทรดจริง
# ===========================================

echo "🚀 Trademify - Trading Setup"
echo "================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo -e "${RED}❌ Error: backend/.env not found${NC}"
    echo "Run this script from the project root directory"
    exit 1
fi

echo -e "${YELLOW}⚠️  Warning: Real trading involves real money!${NC}"
echo "   เทรดจริงมีความเสี่ยงที่จะสูญเสียเงิน"
echo ""

# Ask for broker type
echo "Select broker:"
echo "1) Binance Testnet (Crypto - เงินปลอม)"
echo "2) Binance Real (Crypto - เงินจริง!)"
echo "3) MetaTrader 5 (Forex/CFD/Gold)"
echo ""
read -p "Enter choice [1-3]: " broker_choice

case $broker_choice in
    1)
        echo ""
        echo -e "${GREEN}📝 Binance Testnet Setup${NC}"
        echo "1. Go to: https://testnet.binance.vision/"
        echo "2. Login with GitHub"
        echo "3. Click 'Generate HMAC_SHA256 Key'"
        echo "4. Copy your API Key and Secret"
        echo ""
        
        read -p "Enter Binance API Key: " API_KEY
        read -sp "Enter Binance API Secret: " API_SECRET
        echo ""
        
        # Update .env
        sed -i "s/^BROKER_TYPE=.*/BROKER_TYPE=BINANCE/" backend/.env
        sed -i "s/^BINANCE_API_KEY=.*/BINANCE_API_KEY=$API_KEY/" backend/.env
        sed -i "s/^BINANCE_API_SECRET=.*/BINANCE_API_SECRET=$API_SECRET/" backend/.env
        sed -i "s/^BINANCE_TESTNET=.*/BINANCE_TESTNET=true/" backend/.env
        sed -i "s/^TRADING_ENABLED=.*/TRADING_ENABLED=true/" backend/.env
        sed -i "s/^PAPER_TRADING=.*/PAPER_TRADING=false/" backend/.env
        
        echo -e "${GREEN}✓ Configuration updated for Binance Testnet${NC}"
        ;;
        
    2)
        echo ""
        echo -e "${RED}⚠️  REAL MONEY MODE ⚠️${NC}"
        read -p "Are you sure you want to trade with real money? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Cancelled."
            exit 0
        fi
        
        echo "1. Go to: https://www.binance.com/en/my/settings/api-management"
        echo "2. Create new API Key"
        echo "3. Enable 'Spot & Margin Trading'"
        echo "4. Copy your API Key and Secret"
        echo ""
        
        read -p "Enter Binance API Key: " API_KEY
        read -sp "Enter Binance API Secret: " API_SECRET
        echo ""
        
        # Update .env
        sed -i "s/^BROKER_TYPE=.*/BROKER_TYPE=BINANCE/" backend/.env
        sed -i "s/^BINANCE_API_KEY=.*/BINANCE_API_KEY=$API_KEY/" backend/.env
        sed -i "s/^BINANCE_API_SECRET=.*/BINANCE_API_SECRET=$API_SECRET/" backend/.env
        sed -i "s/^BINANCE_TESTNET=.*/BINANCE_TESTNET=false/" backend/.env
        sed -i "s/^TRADING_ENABLED=.*/TRADING_ENABLED=true/" backend/.env
        sed -i "s/^PAPER_TRADING=.*/PAPER_TRADING=false/" backend/.env
        
        echo -e "${GREEN}✓ Configuration updated for Binance Real${NC}"
        ;;
        
    3)
        echo ""
        echo -e "${BLUE}📊 MetaTrader 5 Setup${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  Note: MT5 requires Windows!${NC}"
        echo "   สำหรับ Linux/Mac ใช้ได้เฉพาะ Mock mode"
        echo ""
        echo "For Windows VPS setup, see: docs/WINDOWS_VPS_SETUP.md"
        echo ""
        
        read -p "Enter MT5 Login (account number): " MT5_LOGIN
        read -sp "Enter MT5 Password: " MT5_PASSWORD
        echo ""
        read -p "Enter MT5 Server (e.g., ICMarkets-Demo): " MT5_SERVER
        
        # Update .env
        sed -i "s/^BROKER_TYPE=.*/BROKER_TYPE=MT5/" backend/.env
        sed -i "s/^MT5_LOGIN=.*/MT5_LOGIN=$MT5_LOGIN/" backend/.env
        sed -i "s/^MT5_PASSWORD=.*/MT5_PASSWORD=$MT5_PASSWORD/" backend/.env
        sed -i "s/^MT5_SERVER=.*/MT5_SERVER=$MT5_SERVER/" backend/.env
        sed -i "s/^TRADING_ENABLED=.*/TRADING_ENABLED=true/" backend/.env
        sed -i "s/^PAPER_TRADING=.*/PAPER_TRADING=false/" backend/.env
        
        echo ""
        echo -e "${GREEN}✓ Configuration updated for MT5${NC}"
        echo ""
        echo -e "${YELLOW}Important:${NC}"
        echo "1. For real MT5 trading, use Windows VPS"
        echo "2. Install MetaTrader5 package: pip install MetaTrader5"
        echo "3. Make sure MT5 terminal is running and logged in"
        echo "4. Run: python backend/trading_bot_mt5.py --symbol EURUSD --real"
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo ""

case $broker_choice in
    1|2)
        echo "  cd backend"
        echo "  python trading_bot.py --symbol BTCUSDT --timeframe H1 --real"
        ;;
    3)
        echo "  # On Windows VPS:"
        echo "  python backend/trading_bot_mt5.py --symbol EURUSD --timeframe H1 --real"
        echo ""
        echo "  # Or use the batch file:"
        echo "  start-bot.bat EURUSD H1"
        ;;
esac

echo ""
