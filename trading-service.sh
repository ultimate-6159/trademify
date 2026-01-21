#!/bin/bash
# Trademify AI Trading Bot Service
# ระบบเทรดอัตโนมัติด้วย AI เพียงหนึ่งเดียว - รัน 24/7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
LOG_DIR="$SCRIPT_DIR/logs"
API_LOG="$LOG_DIR/api.log"
BOT_LOG="$LOG_DIR/bot.log"
API_PID="$LOG_DIR/api.pid"
BOT_PID="$LOG_DIR/bot.pid"

# สร้าง logs directory
mkdir -p "$LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

start_api() {
    if [ -f "$API_PID" ] && kill -0 $(cat "$API_PID") 2>/dev/null; then
        echo -e "${YELLOW}⚠️  API is already running (PID: $(cat $API_PID))${NC}"
        return 1
    fi

    echo -e "${GREEN}🚀 Starting API Server...${NC}"
    
    cd "$BACKEND_DIR"
    nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 >> "$API_LOG" 2>&1 &
    echo $! > "$API_PID"
    
    sleep 2
    
    if kill -0 $(cat "$API_PID") 2>/dev/null; then
        echo -e "${GREEN}✅ API started (PID: $(cat $API_PID))${NC}"
        echo "   📊 Dashboard: http://$(hostname -I | awk '{print $1}'):8000"
        echo "   📝 Logs: $API_LOG"
    else
        echo -e "${RED}❌ Failed to start API${NC}"
        rm -f "$API_PID"
        return 1
    fi
}

start_bot() {
    if [ -f "$BOT_PID" ] && kill -0 $(cat "$BOT_PID") 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Bot is already running (PID: $(cat $BOT_PID))${NC}"
        return 1
    fi

    echo -e "${GREEN}🤖 Starting AI Trading Bot...${NC}"
    
    cd "$BACKEND_DIR"
    
    # Bot parameters (customize these)
    BROKER="${BOT_BROKER:-MT5}"
    SYMBOLS="${BOT_SYMBOLS:-EURUSD,GBPUSD,XAUUSD}"
    TIMEFRAME="${BOT_TIMEFRAME:-H1}"
    QUALITY="${BOT_QUALITY:-HIGH}"
    INTERVAL="${BOT_INTERVAL:-60}"
    
    nohup python ai_trading_bot.py \
        --broker "$BROKER" \
        --symbols "$SYMBOLS" \
        --timeframe "$TIMEFRAME" \
        --quality "$QUALITY" \
        --interval "$INTERVAL" \
        >> "$BOT_LOG" 2>&1 &
    
    echo $! > "$BOT_PID"
    
    sleep 3
    
    if kill -0 $(cat "$BOT_PID") 2>/dev/null; then
        echo -e "${GREEN}✅ AI Bot started (PID: $(cat $BOT_PID))${NC}"
        echo "   🏦 Broker: $BROKER"
        echo "   📈 Symbols: $SYMBOLS"
        echo "   ⏱️  Interval: ${INTERVAL}s"
        echo "   ⭐ Quality: $QUALITY"
        echo "   📝 Logs: $BOT_LOG"
    else
        echo -e "${RED}❌ Failed to start bot${NC}"
        rm -f "$BOT_PID"
        return 1
    fi
}

stop_api() {
    if [ ! -f "$API_PID" ]; then
        echo -e "${YELLOW}⚠️  API is not running${NC}"
        return 1
    fi

    PID=$(cat "$API_PID")
    
    if kill -0 $PID 2>/dev/null; then
        echo -e "${YELLOW}🛑 Stopping API (PID: $PID)...${NC}"
        kill $PID
        sleep 2
        kill -9 $PID 2>/dev/null
    fi
    
    rm -f "$API_PID"
    echo -e "${GREEN}✅ API stopped${NC}"
}

stop_bot() {
    if [ ! -f "$BOT_PID" ]; then
        echo -e "${YELLOW}⚠️  Bot is not running${NC}"
        return 1
    fi

    PID=$(cat "$BOT_PID")
    
    if kill -0 $PID 2>/dev/null; then
        echo -e "${YELLOW}🛑 Stopping Bot (PID: $PID)...${NC}"
        kill $PID
        sleep 2
        kill -9 $PID 2>/dev/null
    fi
    
    rm -f "$BOT_PID"
    echo -e "${GREEN}✅ Bot stopped${NC}"
}

status() {
    echo ""
    echo "========================================"
    echo "   Trademify Service Status"
    echo "========================================"
    
    # API Status
    if [ -f "$API_PID" ] && kill -0 $(cat "$API_PID") 2>/dev/null; then
        echo -e "   API:  ${GREEN}● Running${NC} (PID: $(cat $API_PID))"
    else
        echo -e "   API:  ${RED}○ Stopped${NC}"
    fi
    
    # Bot Status
    if [ -f "$BOT_PID" ] && kill -0 $(cat "$BOT_PID") 2>/dev/null; then
        echo -e "   Bot:  ${GREEN}● Running${NC} (PID: $(cat $BOT_PID))"
    else
        echo -e "   Bot:  ${RED}○ Stopped${NC}"
    fi
    
    echo "========================================"
    echo ""
}

logs_api() {
    if [ -f "$API_LOG" ]; then
        tail -f "$API_LOG"
    else
        echo "No API logs found"
    fi
}

logs_bot() {
    if [ -f "$BOT_LOG" ]; then
        tail -f "$BOT_LOG"
    else
        echo "No bot logs found"
    fi
}

case "$1" in
    start)
        start_api
        echo ""
        start_bot
        ;;
    start-api)
        start_api
        ;;
    start-bot)
        start_bot
        ;;
    stop)
        stop_bot
        stop_api
        ;;
    stop-api)
        stop_api
        ;;
    stop-bot)
        stop_bot
        ;;
    restart)
        stop_bot
        stop_api
        sleep 2
        start_api
        start_bot
        ;;
    status)
        status
        ;;
    logs)
        logs_bot
        ;;
    logs-api)
        logs_api
        ;;
    logs-bot)
        logs_bot
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|logs-api|logs-bot|start-api|start-bot|stop-api|stop-bot}"
        echo ""
        echo "Commands:"
        echo "  start      - Start both API and Bot"
        echo "  stop       - Stop both API and Bot"
        echo "  restart    - Restart all services"
        echo "  status     - Show service status"
        echo "  logs       - Show bot logs (live)"
        echo "  logs-api   - Show API logs (live)"
        echo ""
        echo "Environment variables:"
        echo "  BOT_SYMBOLS   - Trading symbols (default: BTCUSDT,ETHUSDT)"
        echo "  BOT_TIMEFRAME - Timeframe (default: H1)"
        echo "  BOT_QUALITY   - Min signal quality (default: HIGH)"
        echo "  BOT_INTERVAL  - Check interval in seconds (default: 60)"
        exit 1
        ;;
esac
        kill $PID
        sleep 2
        
        if kill -0 $PID 2>/dev/null; then
            echo "⚠️  Force killing..."
            kill -9 $PID
        fi
    fi
    
    rm -f "$PID_FILE"
    echo "✅ Trading service stopped"
}

restart() {
    stop
    sleep 1
    start
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "✅ Trading service is running (PID: $(cat $PID_FILE))"
        
        # Check API health
        HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null)
        if [ ! -z "$HEALTH" ]; then
            echo "📊 API Health: OK"
            
            # Check trading status
            STATUS=$(curl -s http://localhost:8000/api/v1/trading/status 2>/dev/null)
            ENABLED=$(echo $STATUS | python3 -c "import sys,json; d=json.load(sys.stdin); print('Enabled' if d.get('enabled') else 'Disabled')" 2>/dev/null)
            POSITIONS=$(echo $STATUS | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('open_positions',0))" 2>/dev/null)
            
            echo "🤖 Auto Trading: $ENABLED"
            echo "📈 Open Positions: $POSITIONS"
        else
            echo "⚠️  API not responding"
        fi
    else
        echo "❌ Trading service is not running"
        rm -f "$PID_FILE" 2>/dev/null
    fi
}

logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file found"
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Trademify Trading Service"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start trading service"
        echo "  stop    - Stop trading service"
        echo "  restart - Restart trading service"
        echo "  status  - Check service status"
        echo "  logs    - View live logs"
        exit 1
        ;;
esac
