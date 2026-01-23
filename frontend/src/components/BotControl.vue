<template>
  <div class="bg-gray-800 rounded-lg p-3 sm:p-6">
    <!-- Header -->
    <div class="flex items-center justify-between mb-4 sm:mb-6">
      <h2
        class="text-lg sm:text-xl font-bold text-white flex items-center gap-2"
      >
        <span class="text-xl sm:text-2xl">🤖</span>
        <span class="hidden xs:inline">AI Trading</span> Bot
      </h2>

      <!-- Status Badge -->
      <div :class="statusBadgeClass" class="text-xs sm:text-sm">
        {{ botStatus.running ? "🟢 RUNNING" : "⏹️ STOPPED" }}
      </div>
    </div>

    <!-- Connection Warning -->
    <div
      v-if="!isConnected"
      class="bg-red-900/30 border border-red-500 rounded-lg p-3 sm:p-4 mb-4"
    >
      <div class="flex items-start gap-2 sm:gap-3">
        <span class="text-xl sm:text-2xl">⚠️</span>
        <div class="flex-1 min-w-0">
          <div class="text-red-400 font-semibold text-sm">
            Cannot connect to Backend
          </div>
          <div class="text-gray-400 text-xs">
            Make sure backend is running on port 8000
          </div>
        </div>
      </div>
    </div>

    <!-- Bot Status Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4 mb-4 sm:mb-6">
      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Broker</div>
        <div class="text-base sm:text-xl font-bold text-blue-400">
          {{ botStatus.broker_type || "MT5" }}
        </div>
      </div>

      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Quality</div>
        <div :class="qualityClass" class="text-base sm:text-xl">
          {{ botSettings.min_quality || "HIGH" }}
        </div>
      </div>

      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Symbols</div>
        <div class="text-base sm:text-lg font-bold text-white">
          {{ symbolCount }}
        </div>
      </div>

      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Interval</div>
        <div class="text-base sm:text-xl font-bold text-white">
          {{ botSettings.interval || 60 }}s
        </div>
      </div>
    </div>

    <!-- Symbols Display -->
    <div class="bg-gray-700 rounded-lg p-3 sm:p-4 mb-4 sm:mb-6">
      <div class="text-gray-400 text-xs sm:text-sm mb-2">Trading Symbols</div>
      <div class="flex flex-wrap gap-1.5 sm:gap-2">
        <span
          v-for="symbol in symbolList"
          :key="symbol"
          class="px-2 sm:px-3 py-0.5 sm:py-1 bg-blue-600 text-white rounded-full text-xs sm:text-sm"
        >
          {{ symbol }}
        </span>
      </div>
    </div>

    <!-- Quality Selector -->
    <div class="bg-gray-700 rounded-lg p-3 sm:p-4 mb-4 sm:mb-6">
      <div class="text-gray-400 text-xs sm:text-sm mb-2">
        Signal Quality Filter
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <button
          v-for="quality in qualityLevels"
          :key="quality.value"
          @click="setQuality(quality.value)"
          :disabled="!botStatus.running"
          :class="[
            'px-2 sm:px-4 py-2 sm:py-3 rounded-lg text-xs sm:text-sm font-semibold transition-all',
            botSettings.min_quality === quality.value
              ? quality.activeClass
              : 'bg-gray-600 text-gray-300 hover:bg-gray-500',
            !botStatus.running && 'opacity-50 cursor-not-allowed',
          ]"
        >
          <div>{{ quality.label }}</div>
          <div class="text-xs opacity-75">{{ quality.confidence }}%+</div>
        </button>
      </div>
      <div class="text-xs text-gray-500 mt-2">
        💡 Higher quality = fewer but more accurate trades
      </div>
    </div>

    <!-- Last Signals -->
    <div
      v-if="lastSignals && Object.keys(lastSignals).length > 0"
      class="bg-gray-700 rounded-lg p-3 sm:p-4 mb-4 sm:mb-6"
    >
      <div class="text-gray-400 text-xs sm:text-sm mb-2 sm:mb-3">
        Latest Signals
      </div>
      <div class="space-y-2">
        <div
          v-for="(signal, symbol) in lastSignals"
          :key="symbol"
          class="flex flex-col sm:flex-row sm:items-center justify-between bg-gray-600 rounded-lg px-3 sm:px-4 py-2 gap-1 sm:gap-0"
        >
          <span class="text-white font-medium text-sm">{{ symbol }}</span>
          <div class="flex items-center gap-2 sm:gap-4 text-xs sm:text-sm">
            <span :class="signalClass(signal.signal)">
              {{ signalEmoji(signal.signal) }} {{ signal.signal }}
            </span>
            <span class="text-gray-400"
              >{{ signal.enhanced_confidence?.toFixed(1) || 0 }}%</span
            >
            <span :class="qualityBadgeClass(signal.quality)">{{
              signal.quality
            }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Control Buttons -->
    <div class="flex flex-col sm:flex-row gap-2 sm:gap-3">
      <button
        v-if="!botStatus.running"
        @click="startBot"
        :disabled="isLoading"
        class="flex-1 py-2.5 sm:py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 text-sm sm:text-base"
      >
        <span v-if="isLoading" class="animate-spin">⏳</span>
        <span v-else>🚀</span>
        Start Bot
      </button>

      <button
        v-else
        @click="stopBot"
        :disabled="isLoading"
        class="flex-1 py-2.5 sm:py-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 text-sm sm:text-base"
      >
        <span v-if="isLoading" class="animate-spin">⏳</span>
        <span v-else>🛑</span>
        Stop Bot
      </button>

      <button
        @click="showSettings = true"
        class="px-4 sm:px-6 py-2.5 sm:py-3 bg-gray-600 hover:bg-gray-500 text-white rounded-lg font-semibold transition-colors text-sm sm:text-base"
      >
        ⚙️ <span class="hidden sm:inline">Settings</span>
      </button>
    </div>

    <!-- Daily Stats -->
    <div v-if="botStatus.daily_stats" class="mt-6 bg-gray-700 rounded-lg p-4">
      <div class="text-gray-400 text-sm mb-2">Today's Performance</div>
      <div class="grid grid-cols-4 gap-4">
        <div class="text-center">
          <div class="text-2xl font-bold text-white">
            {{ botStatus.daily_stats.trades || 0 }}
          </div>
          <div class="text-xs text-gray-400">Trades</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-green-400">
            {{ botStatus.daily_stats.wins || 0 }}
          </div>
          <div class="text-xs text-gray-400">Wins</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-red-400">
            {{ botStatus.daily_stats.losses || 0 }}
          </div>
          <div class="text-xs text-gray-400">Losses</div>
        </div>
        <div class="text-center">
          <div
            :class="[
              'text-2xl font-bold',
              (botStatus.daily_stats.pnl || 0) >= 0
                ? 'text-green-400'
                : 'text-red-400',
            ]"
          >
            {{ (botStatus.daily_stats.pnl || 0) >= 0 ? "+" : ""
            }}{{ (botStatus.daily_stats.pnl || 0).toFixed(2) }}
          </div>
          <div class="text-xs text-gray-400">P&L</div>
        </div>
      </div>
    </div>

    <!-- Settings Modal -->
    <Teleport to="body">
      <div
        v-if="showSettings"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      >
        <div class="bg-gray-800 rounded-xl p-6 w-full max-w-lg">
          <div class="flex justify-between items-center mb-6">
            <h3 class="text-xl font-bold text-white">Bot Settings</h3>
            <button
              @click="showSettings = false"
              class="text-gray-400 hover:text-white text-2xl"
            >
              ×
            </button>
          </div>

          <div class="space-y-4">
            <!-- Symbols -->
            <div>
              <label class="text-white block mb-2"
                >Trading Symbols (comma-separated)</label
              >
              <input
                type="text"
                v-model="editSettings.symbols"
                placeholder="EURUSDm,GBPUSDm,XAUUSDm"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              />
            </div>

            <!-- Timeframe -->
            <div>
              <label class="text-white block mb-2">Timeframe</label>
              <select
                v-model="editSettings.timeframe"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              >
                <option value="M5">M5 (5 minutes)</option>
                <option value="M15">M15 (15 minutes)</option>
                <option value="M30">M30 (30 minutes)</option>
                <option value="H1">H1 (1 hour)</option>
                <option value="H4">H4 (4 hours)</option>
              </select>
            </div>

            <!-- Quality -->
            <div>
              <label class="text-white block mb-2">Minimum Quality</label>
              <select
                v-model="editSettings.min_quality"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              >
                <option value="PREMIUM">
                  PREMIUM (85%+ confidence - Safest)
                </option>
                <option value="HIGH">
                  HIGH (75%+ confidence - Recommended)
                </option>
                <option value="MEDIUM">
                  MEDIUM (65%+ confidence - More trades)
                </option>
                <option value="LOW">LOW (50%+ confidence - Aggressive)</option>
              </select>
            </div>

            <!-- Interval -->
            <div>
              <label class="text-white block mb-2"
                >Analysis Interval (seconds)</label
              >
              <input
                type="number"
                v-model.number="editSettings.interval"
                min="30"
                max="3600"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              />
            </div>

            <!-- Auto-start -->
            <div class="flex items-center gap-3">
              <input
                type="checkbox"
                v-model="editSettings.auto_start"
                id="auto_start"
                class="w-5 h-5"
              />
              <label for="auto_start" class="text-white"
                >Auto-start on server restart</label
              >
            </div>
          </div>

          <div class="flex gap-3 mt-6">
            <button
              @click="saveSettings"
              class="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold"
            >
              💾 Save Settings
            </button>
            <button
              @click="showSettings = false"
              class="flex-1 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";

// API Base URL
const getApiBase = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/api\/v1\/?$/, "");
  }
  const hostname = window.location.hostname;
  if (hostname !== "localhost" && hostname !== "127.0.0.1") {
    return `http://${hostname}:8000`;
  }
  return "http://localhost:8000";
};
const API_BASE = getApiBase();

// State
const botStatus = ref({
  running: false,
  broker_type: "MT5",
  symbols: [],
  min_quality: "HIGH",
  daily_stats: null,
  last_signals: {},
});

const botSettings = ref({
  symbols: "EURUSDm,GBPUSDm,XAUUSDm",
  timeframe: "H1",
  htf_timeframe: "H4",
  min_quality: "HIGH",
  interval: 60,
  auto_start: false,
});

const editSettings = ref({ ...botSettings.value });
const showSettings = ref(false);
const isLoading = ref(false);
const isConnected = ref(true);
const lastSignals = ref({});

// Polling interval
let pollInterval = null;

// Quality levels config
const qualityLevels = [
  {
    value: "PREMIUM",
    label: "⭐ PREMIUM",
    confidence: 85,
    activeClass: "bg-purple-600 text-white ring-2 ring-purple-400",
  },
  {
    value: "HIGH",
    label: "🔥 HIGH",
    confidence: 75,
    activeClass: "bg-green-600 text-white ring-2 ring-green-400",
  },
  {
    value: "MEDIUM",
    label: "📊 MEDIUM",
    confidence: 65,
    activeClass: "bg-yellow-600 text-white ring-2 ring-yellow-400",
  },
  {
    value: "LOW",
    label: "⚡ LOW",
    confidence: 50,
    activeClass: "bg-red-600 text-white ring-2 ring-red-400",
  },
];

// Computed
const statusBadgeClass = computed(() => [
  "px-4 py-2 rounded-full font-semibold text-sm",
  botStatus.value.running
    ? "bg-green-600 text-white"
    : "bg-gray-600 text-gray-300",
]);

const qualityClass = computed(() => {
  const q = botSettings.value.min_quality;
  return {
    "text-xl font-bold": true,
    "text-purple-400": q === "PREMIUM",
    "text-green-400": q === "HIGH",
    "text-yellow-400": q === "MEDIUM",
    "text-red-400": q === "LOW",
  };
});

const symbolList = computed(() => {
  const symbols = botSettings.value.symbols || "";
  if (Array.isArray(symbols)) return symbols;
  return symbols
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s);
});

const symbolCount = computed(() => symbolList.value.length);

// Methods
function signalClass(signal) {
  return {
    "font-semibold": true,
    "text-green-400": signal?.includes("BUY"),
    "text-red-400": signal?.includes("SELL"),
    "text-gray-400": signal === "WAIT",
  };
}

function signalEmoji(signal) {
  if (signal === "STRONG_BUY") return "🟢🟢";
  if (signal === "BUY") return "🟢";
  if (signal === "STRONG_SELL") return "🔴🔴";
  if (signal === "SELL") return "🔴";
  return "⚪";
}

function qualityBadgeClass(quality) {
  return {
    "px-2 py-1 rounded text-xs font-semibold": true,
    "bg-purple-600 text-white": quality === "PREMIUM",
    "bg-green-600 text-white": quality === "HIGH",
    "bg-yellow-600 text-black": quality === "MEDIUM",
    "bg-red-600 text-white": quality === "LOW",
    "bg-gray-600 text-gray-300": quality === "SKIP",
  };
}

async function fetchStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/bot/status`);
    if (!response.ok) throw new Error("API Error");

    const data = await response.json();
    isConnected.value = true;
    botStatus.value = data;

    // Update last signals
    if (data.last_signals) {
      lastSignals.value = data.last_signals;
    }

    // Update settings from status
    if (data.min_quality) {
      botSettings.value.min_quality = data.min_quality;
    }
    if (data.symbols) {
      botSettings.value.symbols = Array.isArray(data.symbols)
        ? data.symbols.join(",")
        : data.symbols;
    }
  } catch (error) {
    console.error("[BotControl] Failed to fetch status:", error);
    isConnected.value = false;
  }
}

async function fetchSettings() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/bot/settings`);
    if (!response.ok) return;

    const data = await response.json();
    botSettings.value = { ...botSettings.value, ...data };
    editSettings.value = { ...botSettings.value };
  } catch (error) {
    console.error("[BotControl] Failed to fetch settings:", error);
  }
}

async function startBot() {
  isLoading.value = true;
  try {
    const symbols = editSettings.value.symbols || botSettings.value.symbols;
    const response = await fetch(`${API_BASE}/api/v1/bot/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbols: Array.isArray(symbols)
          ? symbols
          : symbols.split(",").map((s) => s.trim()),
        timeframe: editSettings.value.timeframe || "H1",
        htf_timeframe: editSettings.value.htf_timeframe || "H4",
        min_quality: editSettings.value.min_quality || "HIGH",
        interval: editSettings.value.interval || 60,
        auto_start: editSettings.value.auto_start || false,
        broker_type: "MT5",
        allowed_signals: ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"],
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      alert(error.message || "Failed to start bot");
      return;
    }

    await fetchStatus();
  } catch (error) {
    console.error("[BotControl] Failed to start bot:", error);
    alert("Failed to start bot: " + error.message);
  } finally {
    isLoading.value = false;
  }
}

async function stopBot() {
  isLoading.value = true;
  try {
    const response = await fetch(`${API_BASE}/api/v1/bot/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ disable_auto_start: true }),
    });

    if (!response.ok) {
      const error = await response.json();
      alert(error.message || "Failed to stop bot");
      return;
    }

    await fetchStatus();
  } catch (error) {
    console.error("[BotControl] Failed to stop bot:", error);
  } finally {
    isLoading.value = false;
  }
}

async function setQuality(quality) {
  if (!botStatus.value.running) return;

  try {
    const response = await fetch(`${API_BASE}/api/v1/bot/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ min_quality: quality }),
    });

    if (response.ok) {
      botSettings.value.min_quality = quality;
      console.log(`[BotControl] Quality updated to ${quality}`);
    }
  } catch (error) {
    console.error("[BotControl] Failed to update quality:", error);
  }
}

async function saveSettings() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/bot/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(editSettings.value),
    });

    if (response.ok) {
      botSettings.value = { ...editSettings.value };
      showSettings.value = false;
      alert("Settings saved!");
    }
  } catch (error) {
    console.error("[BotControl] Failed to save settings:", error);
    alert("Failed to save settings");
  }
}

// Lifecycle
onMounted(() => {
  fetchStatus();
  fetchSettings();

  // Poll every 5 seconds
  pollInterval = setInterval(fetchStatus, 5000);
});

onUnmounted(() => {
  if (pollInterval) {
    clearInterval(pollInterval);
  }
});
</script>
