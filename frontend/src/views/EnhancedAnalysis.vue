<template>
  <div class="space-y-4 sm:space-y-6 pb-20 md:pb-0">
    <!-- Header -->
    <div
      class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-0"
    >
      <div>
        <h1 class="text-xl sm:text-3xl font-bold text-white">🤖 AI Analysis</h1>
        <p class="text-gray-400 text-sm sm:text-base mt-1">
          Multi-factor AI for High Win Rate
        </p>
      </div>

      <div class="flex items-center gap-2 sm:gap-4">
        <!-- Bot Status -->
        <div
          :class="botStatusClass"
          class="px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg flex items-center gap-1 sm:gap-2 text-xs sm:text-sm"
        >
          <span
            class="w-2 h-2 sm:w-3 sm:h-3 rounded-full"
            :class="botRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'"
          ></span>
          <span class="hidden xs:inline">{{
            botRunning ? "Running" : "Stopped"
          }}</span>
        </div>

        <!-- Start/Stop Button -->
        <button
          @click="toggleBot"
          :disabled="!canTrade"
          :class="[
            botRunning
              ? 'bg-red-600 hover:bg-red-700'
              : 'bg-green-600 hover:bg-green-700',
            !canTrade ? 'opacity-50 cursor-not-allowed' : '',
          ]"
          class="px-3 sm:px-4 py-1.5 sm:py-2 text-white rounded-lg transition-colors text-sm"
        >
          {{ botRunning ? "⏹️ Stop" : "▶️ Start" }}
        </button>
      </div>
    </div>

    <!-- View Toggle -->
    <div
      class="flex items-center gap-2 sm:gap-4 bg-gray-800 rounded-lg p-1.5 sm:p-2 w-fit"
    >
      <button
        @click="viewMode = 'compact'"
        :class="
          viewMode === 'compact'
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white'
        "
        class="px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg transition-all text-xs sm:text-sm"
      >
        📊 <span class="hidden sm:inline">Compact</span>
      </button>
      <button
        @click="viewMode = 'detailed'"
        :class="
          viewMode === 'detailed'
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white'
        "
        class="px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg transition-all text-xs sm:text-sm"
      >
        🔬 <span class="hidden sm:inline">Detailed</span>
      </button>
    </div>

    <!-- Market Status Banner -->
    <MarketStatusBanner :symbol="getFirstSymbol(botSettings.symbols)" />

    <!-- Bot Settings (Collapsible) -->
    <div class="bg-gray-800 rounded-lg p-3 sm:p-4">
      <div
        class="flex items-center justify-between cursor-pointer"
        @click="showSettings = !showSettings"
      >
        <div class="flex items-center gap-2">
          <h2 class="text-base sm:text-lg font-semibold text-white">
            ⚙️ Settings
          </h2>
          <span
            v-if="isUserEditing"
            class="text-blue-400 text-xs animate-pulse hidden sm:inline"
            >✏️ Editing...</span
          >
          <span
            v-else-if="isSyncing"
            class="text-yellow-400 text-xs animate-pulse hidden sm:inline"
            >🔄 Syncing...</span
          >
          <span v-else class="text-green-400 text-xs hidden sm:inline">✓</span>
        </div>
        <span class="text-gray-400">{{ showSettings ? "▼" : "▶" }}</span>
      </div>

      <div v-if="showSettings" class="mt-3 sm:mt-4">
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4">
          <div class="col-span-2 sm:col-span-1">
            <label class="text-gray-400 text-xs block mb-1">Symbols</label>
            <input
              v-model="botSettings.symbols"
              placeholder="EURUSD,GBPUSD"
              class="w-full bg-gray-700 text-white rounded-lg px-2 sm:px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label class="text-gray-400 text-sm block mb-1">Timeframe</label>
            <select
              v-model="botSettings.timeframe"
              class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
            >
              <option value="M5">M5</option>
              <option value="M15">M15</option>
              <option value="H1">H1</option>
              <option value="H4">H4</option>
            </select>
          </div>
          <div>
            <label class="text-gray-400 text-sm block mb-1">Min Quality</label>
            <select
              v-model="botSettings.min_quality"
              class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
            >
              <option value="PREMIUM">PREMIUM (Best)</option>
              <option value="HIGH">HIGH</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="LOW">LOW (Risky)</option>
            </select>
          </div>
          <div>
            <label class="text-gray-400 text-sm block mb-1"
              >Interval (sec)</label
            >
            <input
              v-model.number="botSettings.interval"
              type="number"
              min="30"
              max="3600"
              class="w-full bg-gray-700 text-white rounded-lg px-3 py-2"
            />
          </div>
        </div>

        <!-- Actions -->
        <div
          class="flex items-center justify-between mt-4 pt-4 border-t border-gray-700"
        >
          <button
            @click="resetSettings"
            class="text-gray-400 hover:text-white text-sm transition-colors"
          >
            🔄 Reset to Default
          </button>
          <p class="text-gray-500 text-xs">
            ⚡ Settings synced across all devices
          </p>
        </div>
      </div>
    </div>

    <!-- Real-time Signals Grid -->
    <div
      class="grid grid-cols-1 gap-6"
      :class="viewMode === 'compact' ? 'lg:grid-cols-2' : ''"
    >
      <div
        v-for="(signal, symbol) in signals"
        :key="symbol"
        class="bg-gray-800 rounded-xl p-6 border border-gray-700"
      >
        <!-- Signal Header -->
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="text-2xl font-bold text-white">{{ symbol }}</h3>
            <p class="text-gray-400 text-sm">
              Price: ${{ formatPrice(signal.current_price) }}
            </p>
          </div>
          <div class="flex items-center gap-3">
            <!-- Quality Badge -->
            <div
              :class="getQualityClass(signal.quality)"
              class="px-3 py-1 rounded-full text-sm font-semibold"
            >
              {{ getQualityEmoji(signal.quality) }} {{ signal.quality }}
            </div>
            <!-- Signal Badge -->
            <div
              :class="getSignalClass(signal.signal)"
              class="px-6 py-3 rounded-xl font-bold text-2xl shadow-lg"
            >
              {{ signal.signal }}
            </div>
          </div>
        </div>

        <!-- Overall Confidence -->
        <div class="bg-gray-700/50 rounded-lg p-4 mb-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-gray-400">Overall Confidence</span>
            <span
              class="text-2xl font-bold"
              :class="getConfidenceColor(signal.enhanced_confidence)"
            >
              {{ signal.enhanced_confidence?.toFixed(1) }}%
            </span>
          </div>
          <div class="h-3 bg-gray-600 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-all duration-500"
              :class="getConfidenceBarColor(signal.enhanced_confidence)"
              :style="{ width: `${signal.enhanced_confidence || 0}%` }"
            ></div>
          </div>
        </div>

        <!-- COMPACT VIEW: Basic Scores -->
        <div v-if="viewMode === 'compact'">
          <!-- Scores Bar Chart -->
          <div class="space-y-2 mb-4">
            <div
              v-for="(score, name) in signal.scores"
              :key="name"
              class="flex items-center gap-2"
            >
              <span
                class="text-gray-400 text-xs w-24 capitalize flex items-center gap-1"
              >
                {{ getScoreIcon(name) }} {{ name }}
              </span>
              <div class="flex-1 bg-gray-700 rounded-full h-3">
                <div
                  class="h-3 rounded-full transition-all duration-500"
                  :class="getScoreColor(score)"
                  :style="{ width: `${score}%` }"
                ></div>
              </div>
              <span class="text-white font-semibold text-sm w-12 text-right"
                >{{ score?.toFixed(0) }}%</span
              >
            </div>
          </div>

          <!-- Quick Indicators -->
          <div class="grid grid-cols-3 gap-2 mb-4 text-sm">
            <div class="bg-gray-700 rounded-lg p-3 text-center">
              <div class="text-gray-400 text-xs">RSI (14)</div>
              <div
                :class="getRsiColor(signal.indicators?.rsi)"
                class="font-bold text-lg"
              >
                {{ signal.indicators?.rsi?.toFixed(1) || "---" }}
              </div>
              <div
                class="text-xs"
                :class="getRsiLabelColor(signal.indicators?.rsi)"
              >
                {{ getRsiLabel(signal.indicators?.rsi) }}
              </div>
            </div>
            <div class="bg-gray-700 rounded-lg p-3 text-center">
              <div class="text-gray-400 text-xs">MACD</div>
              <div
                :class="
                  signal.indicators?.macd_trend === 'BULLISH'
                    ? 'text-green-400'
                    : signal.indicators?.macd_trend === 'BEARISH'
                      ? 'text-red-400'
                      : 'text-gray-300'
                "
                class="font-bold"
              >
                {{ signal.indicators?.macd_trend || "---" }}
              </div>
              <div class="text-xs text-gray-500">Histogram</div>
            </div>
            <div class="bg-gray-700 rounded-lg p-3 text-center">
              <div class="text-gray-400 text-xs">Market</div>
              <div class="text-blue-400 font-bold">
                {{ signal.market_regime || "---" }}
              </div>
              <div class="text-xs text-gray-500">Regime</div>
            </div>
          </div>

          <!-- Toggle Detailed View -->
          <button
            @click="viewMode = 'detailed'"
            class="w-full py-2 text-blue-400 hover:text-blue-300 text-sm border border-gray-700 rounded-lg hover:bg-gray-700/50 transition-all"
          >
            🔬 View Detailed Factor Analysis
          </button>
        </div>

        <!-- DETAILED VIEW: Full Factor Breakdown -->
        <div v-else>
          <FactorBreakdown
            :finalScore="signal.enhanced_confidence"
            :scores="signal.scores"
            :indicators="signal.indicators"
            :reasons="signal.factors"
            :factorDetails="signal.factor_details"
          />
        </div>

        <!-- Risk Management -->
        <div
          v-if="signal.risk_management"
          class="mt-4 bg-gradient-to-r from-gray-700 to-gray-800 rounded-lg p-4 border border-gray-600"
        >
          <h4 class="text-white font-semibold mb-3 flex items-center gap-2">
            <span>🛡️</span> Risk Management
          </h4>
          <div class="grid grid-cols-2 gap-4 text-sm">
            <div
              class="flex justify-between items-center bg-gray-800/50 rounded p-2"
            >
              <span class="text-gray-400">Stop Loss</span>
              <span class="text-red-400 font-bold"
                >${{ formatPrice(signal.risk_management.stop_loss) }}</span
              >
            </div>
            <div
              class="flex justify-between items-center bg-gray-800/50 rounded p-2"
            >
              <span class="text-gray-400">Take Profit</span>
              <span class="text-green-400 font-bold"
                >${{ formatPrice(signal.risk_management.take_profit) }}</span
              >
            </div>
            <div
              class="flex justify-between items-center bg-gray-800/50 rounded p-2"
            >
              <span class="text-gray-400">Risk:Reward</span>
              <span class="text-blue-400 font-bold"
                >1:{{ signal.risk_management.risk_reward?.toFixed(2) }}</span
              >
            </div>
            <div
              class="flex justify-between items-center bg-gray-800/50 rounded p-2"
            >
              <span class="text-gray-400">Position Size</span>
              <span class="text-yellow-400 font-bold"
                >{{ signal.risk_management.position_size }}x</span
              >
            </div>
          </div>
        </div>

        <!-- Timestamp -->
        <div class="mt-4 text-gray-500 text-xs text-right">
          Updated: {{ formatTime(signal.timestamp) }}
        </div>
      </div>

      <!-- No Signals -->
      <div
        v-if="Object.keys(signals).length === 0"
        class="bg-gray-800 rounded-lg p-12 text-center col-span-2"
      >
        <div class="text-6xl mb-4">📊</div>
        <h3 class="text-xl font-semibold text-white mb-2">No Signals Yet</h3>
        <p class="text-gray-400 mb-4">Start the bot to begin AI analysis</p>
        <button
          @click="toggleBot"
          class="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg"
        >
          ▶️ Start Auto Trading Bot
        </button>
      </div>
    </div>

    <!-- Trade History -->
    <div class="bg-gray-800 rounded-lg p-6">
      <h2 class="text-xl font-bold text-white mb-4">📜 Recent Trades</h2>

      <div v-if="trades.length === 0" class="text-gray-400 text-center py-8">
        No trades yet
      </div>

      <div v-else class="space-y-2">
        <div
          v-for="trade in trades.slice(0, 10)"
          :key="trade.order_id"
          class="flex items-center justify-between bg-gray-700 rounded p-3"
        >
          <div class="flex items-center gap-3">
            <span
              :class="trade.side === 'BUY' ? 'text-green-400' : 'text-red-400'"
              class="font-bold"
            >
              {{ trade.side }}
            </span>
            <span class="text-white">{{ trade.symbol }}</span>
            <span
              :class="getQualityClass(trade.quality)"
              class="text-xs px-2 py-1 rounded"
            >
              {{ trade.quality }}
            </span>
          </div>
          <div class="text-right">
            <div class="text-gray-300">
              ${{ formatPrice(trade.entry_price) }}
            </div>
            <div class="text-gray-500 text-xs">
              {{ formatTime(trade.timestamp) }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- System Diagnostic Panel -->
    <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div
        class="flex items-center justify-between cursor-pointer"
        @click="showDiagnostic = !showDiagnostic"
      >
        <h2 class="text-lg font-semibold text-white flex items-center gap-2">
          🔧 System Status
          <span v-if="diagnostic.will_trade_real" class="text-green-400 text-sm"
            >✅ Ready to Trade</span
          >
          <span
            v-else-if="diagnostic.bot?.paper_mode"
            class="text-yellow-400 text-sm"
            >📝 Paper Mode</span
          >
          <span v-else class="text-red-400 text-sm">❌ Not Ready</span>
        </h2>
        <span class="text-gray-400">{{ showDiagnostic ? "▼" : "▶" }}</span>
      </div>

      <div v-if="showDiagnostic" class="mt-4 space-y-3">
        <!-- Status Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div class="bg-gray-700 rounded-lg p-3">
            <div class="text-gray-400">Bot Running</div>
            <div
              :class="
                diagnostic.bot?.running ? 'text-green-400' : 'text-red-400'
              "
              class="font-bold"
            >
              {{ diagnostic.bot?.running ? "✅ Yes" : "❌ No" }}
            </div>
          </div>
          <div class="bg-gray-700 rounded-lg p-3">
            <div class="text-gray-400">Paper Mode</div>
            <div
              :class="
                diagnostic.bot?.paper_mode
                  ? 'text-yellow-400'
                  : 'text-green-400'
              "
              class="font-bold"
            >
              {{ diagnostic.bot?.paper_mode ? "📝 Paper" : "💰 Real" }}
            </div>
          </div>
          <div class="bg-gray-700 rounded-lg p-3">
            <div class="text-gray-400">Broker</div>
            <div
              :class="
                diagnostic.trading_engine?.broker_connected
                  ? 'text-green-400'
                  : 'text-red-400'
              "
              class="font-bold"
            >
              {{ diagnostic.bot?.broker_type || "N/A" }}
              {{ diagnostic.trading_engine?.broker_connected ? "✅" : "❌" }}
            </div>
          </div>
          <div class="bg-gray-700 rounded-lg p-3">
            <div class="text-gray-400">Positions</div>
            <div class="text-white font-bold">
              {{ diagnostic.trading_engine?.open_positions || 0 }}
            </div>
          </div>
        </div>

        <!-- MT5 Credentials -->
        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-gray-400 mb-2">MT5 Credentials</div>
          <div class="flex gap-4 text-sm">
            <span
              :class="
                diagnostic.mt5_credentials?.login_set
                  ? 'text-green-400'
                  : 'text-red-400'
              "
            >
              Login: {{ diagnostic.mt5_credentials?.login_set ? "✅" : "❌" }}
            </span>
            <span
              :class="
                diagnostic.mt5_credentials?.password_set
                  ? 'text-green-400'
                  : 'text-red-400'
              "
            >
              Password:
              {{ diagnostic.mt5_credentials?.password_set ? "✅" : "❌" }}
            </span>
            <span
              :class="
                diagnostic.mt5_credentials?.server_set
                  ? 'text-green-400'
                  : 'text-red-400'
              "
            >
              Server: {{ diagnostic.mt5_credentials?.server_set ? "✅" : "❌" }}
            </span>
          </div>
        </div>

        <!-- Pattern Indices Status -->
        <div
          v-if="diagnostic.pattern_indices"
          class="bg-gray-700 rounded-lg p-3"
        >
          <div class="text-gray-400 mb-2">Pattern Indices</div>
          <div class="flex gap-4 text-sm flex-wrap">
            <span
              v-if="diagnostic.pattern_indices?.loaded?.length > 0"
              class="text-green-400"
            >
              ✅ Loaded: {{ diagnostic.pattern_indices.loaded.join(", ") }}
            </span>
            <span
              v-if="diagnostic.pattern_indices?.missing?.length > 0"
              class="text-red-400"
            >
              ❌ Missing: {{ diagnostic.pattern_indices.missing.join(", ") }}
            </span>
            <span
              v-if="
                !diagnostic.pattern_indices?.loaded?.length &&
                !diagnostic.pattern_indices?.missing?.length
              "
              class="text-yellow-400"
            >
              ⏳ Not loaded yet
            </span>
          </div>
        </div>

        <!-- Issues -->
        <div
          v-if="diagnostic.issues?.length > 0"
          class="bg-red-900/30 border border-red-500/50 rounded-lg p-3"
        >
          <div class="text-red-400 font-semibold mb-2">⚠️ Issues</div>
          <ul class="space-y-1 text-sm text-red-300">
            <li v-for="issue in diagnostic.issues" :key="issue">{{ issue }}</li>
          </ul>
        </div>

        <!-- Status Message -->
        <div
          class="text-center text-lg font-bold"
          :class="
            diagnostic.will_trade_real
              ? 'text-green-400'
              : diagnostic.bot?.paper_mode
                ? 'text-yellow-400'
                : 'text-red-400'
          "
        >
          {{ diagnostic.status || "Loading..." }}
        </div>

        <button
          @click="fetchDiagnostic"
          class="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 rounded-lg transition-colors"
        >
          🔄 Refresh Status
        </button>
      </div>
    </div>

    <!-- SSE Connection Status -->
    <div class="fixed bottom-4 right-4 text-sm">
      <div
        :class="sseConnected ? 'text-green-400' : 'text-red-400'"
        class="flex items-center gap-2"
      >
        <span
          class="w-2 h-2 rounded-full"
          :class="sseConnected ? 'bg-green-400' : 'bg-red-400'"
        ></span>
        {{ sseConnected ? "Live Connected" : "Disconnected" }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import MarketStatusBanner from "@/components/MarketStatusBanner.vue";
import FactorBreakdown from "@/components/FactorBreakdown.vue";

// Auto-detect API URL (returns base with /api/v1)
function getApiBase() {
  const hostname = window.location.hostname;
  if (hostname !== "localhost" && hostname !== "127.0.0.1") {
    return `http://${hostname}:8000/api/v1`;
  }
  // Check if VITE_API_URL already has /api/v1
  const envUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
  if (envUrl.includes("/api/v1")) {
    return envUrl;
  }
  return `${envUrl}/api/v1`;
}
const API_BASE = getApiBase();

// =====================
// Bot Settings (Centralized from Backend)
// =====================

// Default settings (fallback)
function getDefaultSettings() {
  return {
    symbols: "EURUSDm,GBPUSDm,XAUUSDm",
    timeframe: "H1",
    htf_timeframe: "H4",
    min_quality: "HIGH",
    interval: 60,
    paper_mode: false,
  };
}

// Helper: Ensure symbols is always a string (Backend may return Array)
function normalizeSymbols(symbols) {
  if (Array.isArray(symbols)) {
    return symbols.join(",");
  }
  return symbols || "EURUSD,GBPUSD,XAUUSD";
}

// Helper: Get first symbol safely
function getFirstSymbol(symbols) {
  if (Array.isArray(symbols)) {
    return symbols[0] || "EURUSD";
  }
  if (typeof symbols === "string") {
    return symbols.split(",")[0] || "EURUSD";
  }
  return "EURUSD";
}

// Load settings from Backend (centralized)
async function loadSettingsFromBackend() {
  try {
    const response = await fetch(`${API_BASE}/bot/settings`);
    if (response.ok) {
      const data = await response.json();
      console.log("[Bot] Settings loaded from backend:", data);
      // Normalize symbols to string format
      return {
        ...data,
        symbols: normalizeSymbols(data.symbols),
      };
    }
  } catch (e) {
    console.warn("[Bot] Failed to load settings from backend:", e);
  }
  return getDefaultSettings();
}

// Save settings to Backend (centralized)
async function saveSettingsToBackend(settings) {
  try {
    const response = await fetch(`${API_BASE}/bot/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (response.ok) {
      console.log("[Bot] Settings saved to backend");
      return true;
    }
  } catch (e) {
    console.warn("[Bot] Failed to save settings to backend:", e);
  }
  return false;
}

// State
const botRunning = ref(false);
const showSettings = ref(false);
const showDiagnostic = ref(false);
const diagnostic = ref({
  status: "Loading...",
  will_trade_real: false,
  bot: {},
  trading_engine: {},
  mt5_credentials: {},
  issues: [],
});
const signals = ref({});
const trades = ref([]);
const sseConnected = ref(false);
const marketStatus = ref({ is_tradeable: true, status: "UNKNOWN" });
const viewMode = ref("compact"); // 'compact' or 'detailed'
const settingsLoading = ref(true);
const lastSyncTime = ref(null);
const isSyncing = ref(false);
const isUserEditing = ref(false); // Flag: user กำลังแก้ settings
let userEditTimeout = null; // Timer สำหรับ reset editing flag

const botSettings = ref(getDefaultSettings());

// Fetch diagnostic info
async function fetchDiagnostic() {
  try {
    const response = await fetch(`${API_BASE}/bot/diagnostic`);
    if (response.ok) {
      diagnostic.value = await response.json();
      console.log("[Diagnostic]", diagnostic.value);
    }
  } catch (e) {
    console.error("[Diagnostic] Failed:", e);
    diagnostic.value.status = "❌ Cannot connect to backend";
    diagnostic.value.issues = ["Backend API not reachable"];
  }
}

// Watch for settings changes and auto-save to backend
let saveTimeout = null;
watch(
  botSettings,
  (newSettings) => {
    // Mark as user editing - ป้องกัน sync overwrite
    isUserEditing.value = true;
    if (userEditTimeout) clearTimeout(userEditTimeout);
    // Reset editing flag หลังจากไม่มีการแก้ไข 3 วินาที
    userEditTimeout = setTimeout(() => {
      isUserEditing.value = false;
    }, 3000);

    // Debounce save - wait 500ms after last change
    if (saveTimeout) clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => {
      saveSettingsToBackend(newSettings);
    }, 500);
  },
  { deep: true },
);

// Auto-sync settings from backend every 30 seconds (ลดความถี่ลง)
let syncInterval = null;
async function syncSettingsFromBackend() {
  // ไม่ sync ถ้า user กำลังแก้ settings
  if (isSyncing.value || isUserEditing.value) {
    console.debug("[Bot] Skipping sync - user is editing");
    return;
  }

  isSyncing.value = true;
  try {
    const settings = await loadSettingsFromBackend();
    // Only update if different (to avoid triggering watch)
    if (JSON.stringify(settings) !== JSON.stringify(botSettings.value)) {
      botSettings.value = settings;
      console.log("[Bot] Settings synced from backend");
    }
    lastSyncTime.value = new Date();
  } finally {
    isSyncing.value = false;
  }
}

function startSettingsSync() {
  // Initial load (only if not currently editing)
  if (!isUserEditing.value) {
    syncSettingsFromBackend();
  }
  // Sync every 30 seconds (ลดความถี่เพื่อไม่รบกวน user)
  syncInterval = setInterval(syncSettingsFromBackend, 30000);
  console.log("[Bot] Settings auto-sync started (every 30s)");
}

function stopSettingsSync() {
  if (syncInterval) {
    clearInterval(syncInterval);
    syncInterval = null;
    console.log("[Bot] Settings auto-sync stopped");
  }
}

// Computed
const canTrade = computed(() => marketStatus.value.is_tradeable !== false);

// SSE Connection
let eventSource = null;

function connectSSE() {
  eventSource = new EventSource(`${API_BASE}/events`);

  eventSource.onopen = () => {
    sseConnected.value = true;
    console.log("SSE Connected");
  };

  eventSource.onerror = () => {
    sseConnected.value = false;
    console.log("SSE Error, reconnecting...");
    setTimeout(connectSSE, 5000);
  };

  eventSource.addEventListener("signal", (event) => {
    const data = JSON.parse(event.data);
    console.log("📊 Signal received:", data.symbol, data);
    if (data.symbol) {
      // Ensure scores have default values
      if (!data.scores) {
        data.scores = {
          pattern: 0,
          trend: 0,
          volume: 0,
          momentum: 0,
          session: 0,
          volatility: 0,
          recency: 0,
        };
      }
      signals.value[data.symbol] = data;
    }
  });

  eventSource.addEventListener("trade", (event) => {
    const data = JSON.parse(event.data);
    trades.value.unshift(data);
  });

  eventSource.addEventListener("bot_status", (event) => {
    const data = JSON.parse(event.data);
    botRunning.value = data.status === "running";
  });

  eventSource.addEventListener("ping", () => {
    // Keepalive
  });
}

// API calls
async function fetchBotStatus() {
  try {
    const response = await fetch(`${API_BASE}/bot/status`);
    const data = await response.json();
    botRunning.value = data.running || false;
    if (data.last_signals) {
      signals.value = data.last_signals;
    }
  } catch (error) {
    console.error("Failed to fetch bot status:", error);
  }
}

async function toggleBot() {
  if (botRunning.value) {
    // Stop bot
    try {
      await fetch(`${API_BASE}/bot/stop`, { method: "POST" });
      botRunning.value = false;
    } catch (error) {
      console.error("Failed to stop bot:", error);
    }
  } else {
    // Start bot
    try {
      // Parse symbols and ensure correct format (add 'm' suffix for Exness)
      let symbols = botSettings.value.symbols.split(",").map((s) => s.trim());
      // Auto-add 'm' suffix if not present (for Exness broker)
      symbols = symbols.map((s) => {
        if (!s.toLowerCase().endsWith("m")) {
          return s + "m";
        }
        return s;
      });

      const response = await fetch(`${API_BASE}/bot/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbols: symbols,
          timeframe: botSettings.value.timeframe,
          htf_timeframe: botSettings.value.htf_timeframe,
          min_quality: botSettings.value.min_quality,
          interval: botSettings.value.interval,
          broker_type: "MT5",
          allowed_signals: ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"],
          auto_start: false,
        }),
      });
      if (response.ok) {
        botRunning.value = true;
        // Refresh status
        setTimeout(fetchBotStatus, 1000);
      } else {
        const error = await response.json();
        console.error("Failed to start bot:", error);
        alert(
          "Failed to start bot: " +
            (error.message || error.detail || "Unknown error"),
        );
      }
    } catch (error) {
      console.error("Failed to start bot:", error);
      alert("Failed to start bot: " + error.message);
    }
  }
}

// Reset settings to default (and save to backend)
async function resetSettings() {
  const defaults = getDefaultSettings();
  botSettings.value = defaults;
  await saveSettingsToBackend(defaults);
}

// Computed
const botStatusClass = computed(() =>
  botRunning.value
    ? "bg-green-900/50 text-green-400"
    : "bg-gray-700 text-gray-400",
);

// Helpers
function formatPrice(price) {
  if (!price) return "0.00";
  return price > 100 ? price.toFixed(2) : price.toFixed(4);
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  return new Date(timestamp).toLocaleTimeString();
}

function getSignalClass(signal) {
  const classes = {
    STRONG_BUY: "bg-green-600 text-white",
    BUY: "bg-green-500/80 text-white",
    WAIT: "bg-gray-600 text-gray-300",
    SELL: "bg-red-500/80 text-white",
    STRONG_SELL: "bg-red-600 text-white",
  };
  return classes[signal] || "bg-gray-600 text-gray-300";
}

function getQualityClass(quality) {
  const classes = {
    PREMIUM: "bg-yellow-500 text-black",
    HIGH: "bg-green-600 text-white",
    MEDIUM: "bg-blue-600 text-white",
    LOW: "bg-orange-600 text-white",
    SKIP: "bg-red-600 text-white",
  };
  return classes[quality] || "bg-gray-600 text-white";
}

function getQualityEmoji(quality) {
  const emojis = {
    PREMIUM: "⭐⭐⭐",
    HIGH: "⭐⭐",
    MEDIUM: "⭐",
    LOW: "⚠️",
    SKIP: "❌",
  };
  return emojis[quality] || "";
}

function getScoreColor(score) {
  if (score >= 80) return "bg-green-500";
  if (score >= 60) return "bg-blue-500";
  if (score >= 40) return "bg-yellow-500";
  return "bg-red-500";
}

function getScoreIcon(name) {
  const icons = {
    pattern: "🎯",
    trend: "📈",
    volume: "📊",
    recency: "📅",
    volatility: "🌊",
    session: "🕐",
    momentum: "⚡",
  };
  return icons[name] || "•";
}

function getConfidenceColor(confidence) {
  if (!confidence) return "text-gray-400";
  if (confidence >= 85) return "text-purple-400";
  if (confidence >= 75) return "text-green-400";
  if (confidence >= 60) return "text-yellow-400";
  return "text-red-400";
}

function getConfidenceBarColor(confidence) {
  if (!confidence) return "bg-gray-500";
  if (confidence >= 85) return "bg-gradient-to-r from-purple-500 to-pink-500";
  if (confidence >= 75) return "bg-gradient-to-r from-green-500 to-emerald-400";
  if (confidence >= 60) return "bg-gradient-to-r from-yellow-500 to-orange-400";
  return "bg-gradient-to-r from-red-500 to-red-400";
}

function getRsiColor(rsi) {
  if (!rsi) return "text-gray-300";
  if (rsi > 70) return "text-red-400";
  if (rsi < 30) return "text-green-400";
  return "text-blue-400";
}

function getRsiLabelColor(rsi) {
  if (!rsi) return "text-gray-500";
  if (rsi > 70) return "text-red-400";
  if (rsi < 30) return "text-green-400";
  return "text-gray-400";
}

function getRsiLabel(rsi) {
  if (!rsi) return "";
  if (rsi > 80) return "Overbought!";
  if (rsi > 70) return "Overbought";
  if (rsi < 20) return "Oversold!";
  if (rsi < 30) return "Oversold";
  return rsi > 50 ? "Bullish" : "Bearish";
}

// Lifecycle
onMounted(async () => {
  // Load settings from backend first (centralized)
  settingsLoading.value = true;
  try {
    const settings = await loadSettingsFromBackend();
    botSettings.value = settings;
  } finally {
    settingsLoading.value = false;
  }

  // Start auto-sync for settings (every 10 seconds)
  startSettingsSync();

  fetchBotStatus();
  fetchDiagnostic(); // Load diagnostic on start
  connectSSE();

  // Fetch market status
  try {
    const symbol = getFirstSymbol(botSettings.value.symbols);
    const response = await fetch(
      `${API_BASE}/mt5/market-status?symbol=${symbol}`,
    );
    if (response.ok) {
      marketStatus.value = await response.json();
    }
  } catch (e) {
    console.warn("Could not fetch market status:", e);
  }

  // Auto-refresh diagnostic every 30 seconds
  setInterval(fetchDiagnostic, 30000);
});

onUnmounted(() => {
  // Stop auto-sync
  stopSettingsSync();

  if (eventSource) {
    eventSource.close();
  }
});
</script>
