<template>
  <div class="bg-gray-800 rounded-lg p-3 sm:p-6">
    <div class="flex items-center justify-between mb-4 sm:mb-6">
      <h2
        class="text-lg sm:text-xl font-bold text-white flex items-center gap-2"
      >
        <span class="text-xl sm:text-2xl">🤖</span>
        <span class="hidden xs:inline">Auto Trading</span>
        <span class="xs:hidden">Trading</span>
      </h2>

      <!-- Main Toggle -->
      <div class="flex items-center gap-2 sm:gap-3">
        <span class="text-xs sm:text-sm text-gray-400">
          {{ tradingStore.enabled ? "Active" : "Off" }}
        </span>
        <button
          @click="toggleTrading"
          :class="[
            'relative inline-flex h-6 sm:h-8 w-11 sm:w-14 items-center rounded-full transition-colors',
            tradingStore.enabled ? 'bg-green-500' : 'bg-gray-600',
          ]"
        >
          <span
            :class="[
              'inline-block h-4 sm:h-6 w-4 sm:w-6 transform rounded-full bg-white transition-transform',
              tradingStore.enabled
                ? 'translate-x-6 sm:translate-x-7'
                : 'translate-x-1',
            ]"
          />
        </button>
      </div>
    </div>

    <!-- Status Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4 mb-4 sm:mb-6">
      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Status</div>
        <div :class="statusClass" class="text-sm sm:text-base">
          {{ tradingStore.status }}
        </div>
      </div>

      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Positions</div>
        <div class="text-lg sm:text-2xl font-bold text-white">
          {{ tradingStore.openPositions }}
        </div>
      </div>

      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Today P&L</div>
        <div :class="pnlClass" class="text-lg sm:text-2xl">
          {{ tradingStore.todayPnl >= 0 ? "+" : ""
          }}{{ tradingStore.todayPnl.toFixed(2) }}
        </div>
      </div>

      <div class="bg-gray-700 rounded-lg p-2 sm:p-4">
        <div class="text-gray-400 text-xs mb-1">Win Rate</div>
        <div class="text-lg sm:text-2xl font-bold text-blue-400">
          {{ tradingStore.winRate.toFixed(1) }}%
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="flex flex-wrap gap-2 sm:gap-3 mb-4 sm:mb-6">
      <button
        @click="pauseTrading"
        :disabled="!tradingStore.enabled"
        class="flex-1 sm:flex-none px-3 sm:px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm"
      >
        ⏸️ <span class="hidden sm:inline">Pause</span>
      </button>

      <button
        @click="closeAllPositions"
        :disabled="tradingStore.openPositions === 0"
        class="flex-1 sm:flex-none px-3 sm:px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm"
      >
        🚫 <span class="hidden sm:inline">Close All</span>
      </button>

      <button
        @click="showSettings = true"
        class="flex-1 sm:flex-none px-3 sm:px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg transition-colors text-sm"
      >
        ⚙️ <span class="hidden sm:inline">Settings</span>
      </button>
    </div>

    <!-- Risk Meter -->
    <div class="bg-gray-700 rounded-lg p-3 sm:p-4 mb-4 sm:mb-6">
      <div class="flex justify-between mb-2 text-xs sm:text-sm">
        <span class="text-gray-400">Daily Risk</span>
        <span class="text-white"
          >{{ tradingStore.riskUsed.toFixed(1) }}% /
          {{ tradingStore.maxDailyLoss }}%</span
        >
      </div>
      <div class="w-full bg-gray-600 rounded-full h-2 sm:h-3">
        <div
          :class="[
            'h-2 sm:h-3 rounded-full transition-all duration-300',
            riskBarColor,
          ]"
          :style="{ width: `${Math.min(riskPercent, 100)}%` }"
        />
      </div>
    </div>

    <!-- Paper Trading Notice -->
    <div
      v-if="tradingStore.paperTrading"
      class="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 sm:p-4 flex items-center gap-2 sm:gap-3"
    >
      <span class="text-2xl">📝</span>
      <div>
        <div class="text-yellow-400 font-semibold">Paper Trading Mode</div>
        <div class="text-yellow-600 text-sm">
          ไม่มีการเทรดจริง - ใช้เงินจำลอง
        </div>
      </div>
    </div>

    <!-- Settings Modal -->
    <Teleport to="body">
      <div
        v-if="showSettings"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      >
        <div
          class="bg-gray-800 rounded-xl p-6 w-full max-w-lg max-h-[80vh] overflow-y-auto"
        >
          <div class="flex justify-between items-center mb-6">
            <h3 class="text-xl font-bold text-white">Trading Settings</h3>
            <button
              @click="showSettings = false"
              class="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>

          <div class="space-y-4">
            <!-- Paper Trading -->
            <div class="flex items-center justify-between">
              <label class="text-white">Paper Trading (Simulation)</label>
              <input
                type="checkbox"
                v-model="settings.paper_trading"
                class="w-5 h-5"
              />
            </div>

            <!-- Risk per Trade -->
            <div>
              <label class="text-white block mb-2">Risk per Trade (%)</label>
              <input
                type="number"
                v-model.number="settings.max_risk_per_trade"
                min="0.1"
                max="10"
                step="0.1"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              />
            </div>

            <!-- Max Daily Loss -->
            <div>
              <label class="text-white block mb-2">Max Daily Loss (%)</label>
              <input
                type="number"
                v-model.number="settings.max_daily_loss"
                min="1"
                max="20"
                step="0.5"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              />
            </div>

            <!-- Max Positions -->
            <div>
              <label class="text-white block mb-2">Max Positions</label>
              <input
                type="number"
                v-model.number="settings.max_positions"
                min="1"
                max="20"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              />
            </div>

            <!-- Min Confidence -->
            <div>
              <label class="text-white block mb-2"
                >Min Signal Confidence (%)</label
              >
              <input
                type="number"
                v-model.number="settings.min_confidence"
                min="50"
                max="100"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              />
            </div>

            <!-- Allowed Signals -->
            <div>
              <label class="text-white block mb-2">Allowed Signals</label>
              <div class="flex flex-wrap gap-2">
                <label
                  v-for="signal in allSignals"
                  :key="signal"
                  class="flex items-center gap-2 text-gray-300"
                >
                  <input
                    type="checkbox"
                    :value="signal"
                    v-model="settings.allowed_signals"
                    class="w-4 h-4"
                  />
                  {{ signal }}
                </label>
              </div>
            </div>
          </div>

          <div class="flex gap-3 mt-6">
            <button
              @click="saveSettings"
              class="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
            >
              Save Settings
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
import { ref, computed } from "vue";
import { useTradingStore } from "@/stores/trading";

const tradingStore = useTradingStore();
const showSettings = ref(false);

const allSignals = ["STRONG_BUY", "BUY", "WAIT", "SELL", "STRONG_SELL"];

const settings = ref({
  paper_trading: true,
  max_risk_per_trade: 2.0,
  max_daily_loss: 5.0,
  max_positions: 5,
  min_confidence: 70,
  allowed_signals: ["STRONG_BUY", "STRONG_SELL"],
});

const statusClass = computed(() => {
  const status = tradingStore.status;
  return {
    "text-2xl font-bold": true,
    "text-green-400": status === "running",
    "text-yellow-400": status === "paused",
    "text-red-400": status === "stopped",
    "text-gray-400": status === "disconnected",
  };
});

const pnlClass = computed(() => ({
  "text-2xl font-bold": true,
  "text-green-400": tradingStore.todayPnl >= 0,
  "text-red-400": tradingStore.todayPnl < 0,
}));

const riskPercent = computed(
  () => (tradingStore.riskUsed / tradingStore.maxDailyLoss) * 100,
);

const riskBarColor = computed(() => {
  if (riskPercent.value >= 80) return "bg-red-500";
  if (riskPercent.value >= 50) return "bg-yellow-500";
  return "bg-green-500";
});

async function toggleTrading() {
  if (tradingStore.enabled) {
    await tradingStore.stopTrading();
  } else {
    await tradingStore.startTrading();
  }
}

async function pauseTrading() {
  await tradingStore.pauseTrading();
}

async function closeAllPositions() {
  if (confirm("Are you sure you want to close all positions?")) {
    await tradingStore.closeAllPositions();
  }
}

async function saveSettings() {
  await tradingStore.updateSettings(settings.value);
  showSettings.value = false;
}

// Load settings on mount
tradingStore.fetchSettings().then((data) => {
  if (data) {
    settings.value = { ...settings.value, ...data };
  }
});
</script>
