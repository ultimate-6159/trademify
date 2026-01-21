<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-3xl font-bold text-white">Auto Trading</h1>
      <div class="flex items-center gap-4">
        <!-- Sync Indicator -->
        <div class="flex items-center gap-2 text-sm">
          <span
            v-if="tradingStore.isSyncing"
            class="text-yellow-400 animate-pulse"
          >
            🔄 Syncing...
          </span>
          <span v-else-if="tradingStore.lastSyncTime" class="text-gray-400">
            ✅ Synced {{ formatSyncTime(tradingStore.lastSyncTime) }}
          </span>
          <button
            @click="tradingStore.forceSync"
            class="text-blue-400 hover:text-blue-300 text-xs underline"
            :disabled="tradingStore.isSyncing"
          >
            Refresh
          </button>
        </div>
        <span :class="connectionStatusClass">
          {{ connectionStatus }}
        </span>
      </div>
    </div>

    <!-- Market Status Banner -->
    <MarketStatusBanner :symbol="manualTrade.symbol" />

    <!-- API Connection Warning -->
    <div
      v-if="!tradingStore.apiConnected"
      class="bg-red-900/30 border border-red-500 rounded-lg p-4"
    >
      <div class="flex items-center gap-3">
        <span class="text-3xl">⚠️</span>
        <div>
          <h3 class="text-red-400 font-semibold">Backend API Not Connected</h3>
          <p class="text-gray-400 text-sm">
            Cannot connect to trading server. Make sure the backend is running:
          </p>
          <code
            class="text-xs text-yellow-400 bg-gray-800 px-2 py-1 rounded mt-2 block"
          >
            cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000
          </code>
        </div>
      </div>
    </div>

    <!-- AI Bot Control (Main) -->
    <BotControl />

    <!-- Trading Control (Legacy) -->
    <TradingControl />

    <div class="grid grid-cols-2 gap-6">
      <!-- Positions -->
      <PositionMonitor />

      <!-- Trade History -->
      <TradeHistory />
    </div>

    <!-- Manual Trade Section -->
    <div class="bg-gray-800 rounded-lg p-6">
      <h2 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <span class="text-2xl">✍️</span>
        Manual Trade
      </h2>

      <div class="grid grid-cols-6 gap-4">
        <div>
          <label class="text-gray-400 text-sm mb-1 block">Symbol</label>
          <select
            v-model="manualTrade.symbol"
            class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
          >
            <optgroup label="Major Pairs">
              <option value="EURUSD">EUR/USD</option>
              <option value="GBPUSD">GBP/USD</option>
              <option value="USDJPY">USD/JPY</option>
              <option value="USDCHF">USD/CHF</option>
              <option value="AUDUSD">AUD/USD</option>
              <option value="USDCAD">USD/CAD</option>
            </optgroup>
            <optgroup label="Metals">
              <option value="XAUUSD">XAU/USD (Gold)</option>
              <option value="XAGUSD">XAG/USD (Silver)</option>
            </optgroup>
            <optgroup label="Indices">
              <option value="US30">US30 (Dow Jones)</option>
              <option value="NAS100">NAS100 (Nasdaq)</option>
              <option value="US500">US500 (S&P 500)</option>
            </optgroup>
          </select>
        </div>
        <div>
          <label class="text-gray-400 text-sm mb-1 block">Side</label>
          <select
            v-model="manualTrade.side"
            class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
          >
            <option value="BUY">BUY</option>
            <option value="SELL">SELL</option>
          </select>
        </div>
        <div>
          <label class="text-gray-400 text-sm mb-1 block">Quantity</label>
          <input
            v-model.number="manualTrade.quantity"
            type="number"
            step="0.001"
            class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
          />
        </div>
        <div>
          <label class="text-gray-400 text-sm mb-1 block">Stop Loss</label>
          <input
            v-model.number="manualTrade.stop_loss"
            type="number"
            step="0.01"
            class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
          />
        </div>
        <div>
          <label class="text-gray-400 text-sm mb-1 block">Take Profit</label>
          <input
            v-model.number="manualTrade.take_profit"
            type="number"
            step="0.01"
            class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
          />
        </div>
        <div class="flex items-end">
          <button
            @click="submitManualTrade"
            :disabled="!canSubmitTrade"
            class="w-full py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg"
          >
            Open Trade
          </button>
        </div>
      </div>

      <!-- Price Info -->
      <div v-if="currentPrice" class="mt-4 text-sm text-gray-400">
        Current Price:
        <span class="text-white font-semibold">
          {{ formatPrice(currentPrice, manualTrade.symbol) }}
        </span>
        <span
          v-if="priceSource"
          class="text-xs ml-2 px-2 py-0.5 rounded"
          :class="priceSourceClass"
        >
          {{ priceSourceLabel }}
        </span>
        <span v-if="!isLivePrice" class="text-xs ml-2 text-yellow-400">
          ⚠️ Delayed/Cached
        </span>
      </div>
      <div v-else-if="priceError" class="mt-4 text-sm text-yellow-400">
        ⚠️ {{ priceError }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { useTradingStore } from "@/stores/trading";
import BotControl from "@/components/BotControl.vue";
import TradingControl from "@/components/TradingControl.vue";
import PositionMonitor from "@/components/PositionMonitor.vue";
import TradeHistory from "@/components/TradeHistory.vue";
import MarketStatusBanner from "@/components/MarketStatusBanner.vue";

const tradingStore = useTradingStore();

const manualTrade = ref({
  symbol: "EURUSD",
  side: "BUY",
  quantity: 0.1, // 0.1 lot for forex
  stop_loss: null,
  take_profit: null,
});

const currentPrice = ref(null);
const priceSource = ref(null);
const isLivePrice = ref(false);
const priceError = ref(null);
const connectionStatus = ref("Connecting...");

// Format sync time for display
function formatSyncTime(date) {
  if (!date) return "";
  const now = new Date();
  const diff = Math.floor((now - date) / 1000);
  if (diff < 5) return "just now";
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return date.toLocaleTimeString();
}

const connectionStatusClass = computed(() => ({
  "px-3 py-1 rounded-full text-sm": true,
  "bg-green-600/20 text-green-400": connectionStatus.value === "Connected",
  "bg-yellow-600/20 text-yellow-400":
    connectionStatus.value === "Connecting...",
  "bg-red-600/20 text-red-400":
    connectionStatus.value === "Disconnected" ||
    connectionStatus.value === "API Offline",
}));

const priceSourceClass = computed(() => {
  if (priceSource.value === "mt5_live") return "bg-green-900 text-green-400";
  if (priceSource.value === "mt5_mock") return "bg-yellow-900 text-yellow-400";
  return "bg-gray-700 text-gray-400";
});

const priceSourceLabel = computed(() => {
  if (priceSource.value === "mt5_live") return "🟢 MT5 Live";
  if (priceSource.value === "mt5_mock") return "🟡 MT5 Mock";
  if (priceSource.value === "fallback") return "⚠️ Fallback";
  return priceSource.value;
});

function formatPrice(price, symbol) {
  if (!price) return "N/A";
  // JPY pairs and indices need fewer decimals
  if (symbol.includes("JPY") || ["US30", "NAS100", "US500"].includes(symbol)) {
    return price.toFixed(2);
  }
  // Gold and silver
  if (symbol.includes("XAU") || symbol.includes("XAG")) {
    return price.toFixed(2);
  }
  // Forex majors
  return price.toFixed(5);
}

const canSubmitTrade = computed(() => {
  return (
    manualTrade.value.symbol &&
    manualTrade.value.quantity > 0 &&
    tradingStore.enabled &&
    tradingStore.apiConnected
  );
});

async function fetchPrice() {
  if (manualTrade.value.symbol) {
    const data = await tradingStore.getPrice(manualTrade.value.symbol);
    if (data?.price) {
      currentPrice.value = data.price;
      priceSource.value = data.source || "unknown";
      isLivePrice.value = data.is_live || false;
      priceError.value = null;
    } else if (data?.error) {
      currentPrice.value = null;
      priceSource.value = null;
      isLivePrice.value = false;
      priceError.value = data.message || "Cannot fetch price";
    }
  }
}

async function submitManualTrade() {
  if (!tradingStore.apiConnected) {
    alert("Cannot trade: Backend API is not connected");
    return;
  }

  const result = await tradingStore.openPosition(manualTrade.value);
  if (result) {
    alert("Trade opened successfully!");
  } else {
    alert("Failed to open trade - Check if trading is enabled");
  }
}

let priceInterval = null;

// Watch apiConnected to update connection status
watch(
  () => tradingStore.apiConnected,
  (connected) => {
    connectionStatus.value = connected ? "Connected" : "API Offline";
  },
  { immediate: true },
);

onMounted(async () => {
  // เริ่ม Auto-Sync - จะดึงข้อมูลจาก Backend ทุก 10 วินาที
  // เพื่อให้ทุกอุปกรณ์เห็นข้อมูลตรงกัน
  tradingStore.startAutoSync();

  // Fetch price
  await fetchPrice();

  // Price updates (5 วินาที - เร็วกว่า settings sync)
  priceInterval = setInterval(fetchPrice, 5000);
});

onUnmounted(() => {
  // หยุด Auto-Sync เมื่อออกจากหน้า
  tradingStore.stopAutoSync();
  if (priceInterval) clearInterval(priceInterval);
});
</script>
