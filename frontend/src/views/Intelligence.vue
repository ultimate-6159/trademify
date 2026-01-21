<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white flex items-center gap-3">
          <span class="text-4xl">🧠</span>
          AI Intelligence Center
        </h1>
        <p class="text-gray-400 mt-1">16-Layer Deep Analysis System</p>
      </div>

      <div class="flex items-center gap-4">
        <!-- Symbol Selector -->
        <select
          v-model="selectedSymbol"
          class="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
        >
          <option v-for="symbol in symbols" :key="symbol" :value="symbol">
            {{ symbol }}
          </option>
        </select>

        <!-- Refresh Button -->
        <button
          @click="refreshAll"
          :disabled="isLoading"
          class="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-semibold transition-colors flex items-center gap-2"
        >
          <span :class="{ 'animate-spin': isLoading }">🔄</span>
          Refresh
        </button>
      </div>
    </div>

    <!-- Tab Navigation -->
    <div class="flex gap-2 border-b border-gray-700 pb-2">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        @click="activeTab = tab.id"
        :class="[
          'px-4 py-2 rounded-t-lg font-medium transition-colors',
          activeTab === tab.id
            ? 'bg-gray-700 text-white border-b-2 border-blue-500'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50',
        ]"
      >
        <span class="mr-2">{{ tab.icon }}</span>
        {{ tab.name }}
      </button>
    </div>

    <!-- Tab Content -->
    <div class="tab-content">
      <!-- Overview Tab -->
      <div v-if="activeTab === 'overview'" class="space-y-6">
        <!-- Quick Stats -->
        <div class="grid grid-cols-5 gap-4">
          <div
            class="bg-gray-800 rounded-lg p-4 text-center border border-purple-500/30"
          >
            <div class="text-gray-400 text-xs mb-1">TITAN SCORE</div>
            <div class="text-3xl font-bold text-purple-400">
              {{ titanScore }}%
            </div>
          </div>
          <div
            class="bg-gray-800 rounded-lg p-4 text-center border border-indigo-500/30"
          >
            <div class="text-gray-400 text-xs mb-1">OMEGA GRADE</div>
            <div class="text-3xl font-bold text-indigo-400">
              {{ omegaGrade }}
            </div>
          </div>
          <div
            class="bg-gray-800 rounded-lg p-4 text-center border border-orange-500/30"
          >
            <div class="text-gray-400 text-xs mb-1">ALPHA GRADE</div>
            <div class="text-3xl font-bold text-orange-400">
              {{ alphaGrade }}
            </div>
          </div>
          <div
            class="bg-gray-800 rounded-lg p-4 text-center border border-green-500/30"
          >
            <div class="text-gray-400 text-xs mb-1">RISK LEVEL</div>
            <div class="text-3xl font-bold" :class="riskLevelClass">
              {{ riskLevel }}
            </div>
          </div>
          <div
            class="bg-gray-800 rounded-lg p-4 text-center border border-blue-500/30"
          >
            <div class="text-gray-400 text-xs mb-1">CONSENSUS</div>
            <div class="text-3xl font-bold text-blue-400">{{ consensus }}</div>
          </div>
        </div>

        <!-- Intelligence Panel -->
        <IntelligencePanel
          :auto-refresh="true"
          @module-selected="handleModuleSelected"
        />
      </div>

      <!-- Titan Core Tab -->
      <div v-if="activeTab === 'titan'">
        <TitanDashboard :titan-data="titanData" />
      </div>

      <!-- Omega Brain Tab -->
      <div v-if="activeTab === 'omega'">
        <OmegaPanel :omega-data="omegaData" />
      </div>

      <!-- Risk Management Tab -->
      <div v-if="activeTab === 'risk'">
        <RiskDashboard
          :risk-data="riskData"
          @pause-trading="handlePauseTrading"
          @close-all="handleCloseAll"
          @reset-daily="handleResetDaily"
        />
      </div>

      <!-- Analysis Tab -->
      <div v-if="activeTab === 'analysis'" class="space-y-6">
        <!-- Signal Analysis -->
        <div class="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>🎯</span> Current Signal Analysis
          </h3>

          <div class="grid grid-cols-3 gap-4">
            <!-- Pattern Signal -->
            <div class="bg-gray-700 rounded-lg p-4">
              <div class="text-gray-400 text-sm mb-2">Pattern Recognition</div>
              <div
                class="text-2xl font-bold"
                :class="getSignalClass(analysis.pattern_signal)"
              >
                {{ analysis.pattern_signal || "WAIT" }}
              </div>
              <div class="text-sm text-gray-400 mt-1">
                Confidence: {{ analysis.pattern_confidence?.toFixed(1) || 0 }}%
              </div>
            </div>

            <!-- Technical Signal -->
            <div class="bg-gray-700 rounded-lg p-4">
              <div class="text-gray-400 text-sm mb-2">Technical Analysis</div>
              <div
                class="text-2xl font-bold"
                :class="getSignalClass(analysis.technical_signal)"
              >
                {{ analysis.technical_signal || "WAIT" }}
              </div>
              <div class="text-sm text-gray-400 mt-1">
                Confidence:
                {{ analysis.technical_confidence?.toFixed(1) || 0 }}%
              </div>
            </div>

            <!-- AI Signal -->
            <div class="bg-gray-700 rounded-lg p-4">
              <div class="text-gray-400 text-sm mb-2">AI Ensemble</div>
              <div
                class="text-2xl font-bold"
                :class="getSignalClass(analysis.ai_signal)"
              >
                {{ analysis.ai_signal || "WAIT" }}
              </div>
              <div class="text-sm text-gray-400 mt-1">
                Confidence: {{ analysis.ai_confidence?.toFixed(1) || 0 }}%
              </div>
            </div>
          </div>
        </div>

        <!-- Module Breakdown -->
        <div class="bg-gray-800 rounded-lg p-6 border border-yellow-500/30">
          <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>📊</span> Module Analysis Breakdown
          </h3>

          <div class="grid grid-cols-4 gap-4">
            <div
              v-for="mod in moduleAnalysis"
              :key="mod.name"
              class="bg-gray-700 rounded-lg p-3"
            >
              <div class="flex items-center gap-2 mb-2">
                <span>{{ mod.icon }}</span>
                <span class="text-white font-medium text-sm">{{
                  mod.name
                }}</span>
              </div>
              <div
                class="text-xl font-bold"
                :class="getSignalClass(mod.signal)"
              >
                {{ mod.signal }}
              </div>
              <div class="text-xs text-gray-400">
                {{ mod.score }}% confidence
              </div>
            </div>
          </div>
        </div>

        <!-- Final Recommendation -->
        <div class="p-6 rounded-lg" :class="recommendationClass">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-4">
              <span class="text-5xl">{{ recommendationEmoji }}</span>
              <div>
                <div class="text-2xl font-bold">{{ recommendation }}</div>
                <div class="text-lg opacity-75">{{ recommendationReason }}</div>
              </div>
            </div>
            <div class="text-right">
              <div class="text-sm opacity-75">Position Multiplier</div>
              <div class="text-4xl font-bold">{{ positionMultiplier }}x</div>
            </div>
          </div>
        </div>
      </div>

      <!-- History Tab -->
      <div v-if="activeTab === 'history'" class="space-y-6">
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>📜</span> Signal History
          </h3>

          <div class="overflow-x-auto">
            <table class="w-full">
              <thead>
                <tr class="text-gray-400 text-sm border-b border-gray-700">
                  <th class="text-left py-3 px-4">Time</th>
                  <th class="text-left py-3 px-4">Symbol</th>
                  <th class="text-left py-3 px-4">Signal</th>
                  <th class="text-left py-3 px-4">Titan Score</th>
                  <th class="text-left py-3 px-4">Omega Grade</th>
                  <th class="text-left py-3 px-4">Result</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(item, idx) in signalHistory"
                  :key="idx"
                  class="border-b border-gray-700/50 hover:bg-gray-700/30"
                >
                  <td class="py-3 px-4 text-gray-300">
                    {{ formatTime(item.timestamp) }}
                  </td>
                  <td class="py-3 px-4 text-white font-medium">
                    {{ item.symbol }}
                  </td>
                  <td class="py-3 px-4">
                    <span :class="getSignalBadgeClass(item.signal)">{{
                      item.signal
                    }}</span>
                  </td>
                  <td class="py-3 px-4 text-purple-400">
                    {{ item.titan_score?.toFixed(1) }}%
                  </td>
                  <td class="py-3 px-4 text-indigo-400">
                    {{ item.omega_grade }}
                  </td>
                  <td class="py-3 px-4">
                    <span :class="getResultClass(item.result)">{{
                      item.result
                    }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import IntelligencePanel from "../components/IntelligencePanel.vue";
import TitanDashboard from "../components/TitanDashboard.vue";
import OmegaPanel from "../components/OmegaPanel.vue";
import RiskDashboard from "../components/RiskDashboard.vue";
import api from "../services/api";

// State
const isLoading = ref(false);
const selectedSymbol = ref("EURUSDm");
const activeTab = ref("overview");
let refreshInterval = null;
const symbols = ref([
  "EURUSDm",
  "GBPUSDm",
  "XAUUSDm",
  "EURUSD",
  "GBPUSD",
  "XAUUSD",
]);

// Tabs configuration
const tabs = [
  { id: "overview", name: "Overview", icon: "📊" },
  { id: "titan", name: "Titan Core", icon: "🏛️" },
  { id: "omega", name: "Omega Brain", icon: "🧠" },
  { id: "risk", name: "Risk", icon: "🛡️" },
  { id: "analysis", name: "Analysis", icon: "🎯" },
  { id: "history", name: "History", icon: "📜" },
];

// Data for each tab
const titanData = ref({
  grade: "🏛️ TITAN SUPREME",
  titan_score: 92.5,
  consensus: "STRONG",
  market_condition: "FAVORABLE",
  prediction: { direction: "BUY", confidence: 85 },
  raw_confidence: 80,
  calibrated_confidence: 78,
  historical_accuracy: 72,
  position_multiplier: 0.9,
  improvements: [],
  module_weights: {
    neural: 0.15,
    deep: 0.15,
    quantum: 0.15,
    alpha: 0.2,
    omega: 0.2,
    pattern: 0.15,
  },
});

const omegaData = ref({
  grade: "Ω+",
  omega_score: 88.5,
  institutional_flow: "ACCUMULATION",
  manipulation: "NONE",
  sentiment: "BULLISH",
  volume_anomaly: 1.8,
  big_money: "ACTIVE",
  smart_money: "BUYING",
  current_regime: "UPTREND",
  predicted_regime: "UPTREND",
  regime_confidence: 80,
  suggested_allocation: 85,
  volatility_weight: 0.9,
  risk_contribution: "NORMAL",
  entry_style: "SCALE_IN",
  scale_points: 3,
  exit_plan: "PARTIAL",
  trailing_stop: true,
  position_multiplier: 0.95,
});

const riskData = ref({
  risk_level: "SAFE",
  balance: 10000,
  equity: 10250,
  daily_pnl: 2.5,
  open_positions: 1,
  max_positions: 3,
  risk_per_trade: 2,
  max_daily_loss: 5,
  leverage: 2000,
  risk_score: 20,
  can_trade: true,
  can_open_position: true,
  daily_limit_hit: false,
  losing_streak_limit: false,
  losing_streak: 0,
  max_losing_streak: 5,
});

const analysis = ref({
  pattern_signal: "BUY",
  pattern_confidence: 78,
  technical_signal: "BUY",
  technical_confidence: 82,
  ai_signal: "BUY",
  ai_confidence: 85,
});

const moduleAnalysis = ref([
  { name: "Neural", icon: "🧠", signal: "BUY", score: 82 },
  { name: "Deep", icon: "🔮", signal: "BUY", score: 78 },
  { name: "Quantum", icon: "⚛️", signal: "BUY", score: 75 },
  { name: "Alpha", icon: "🔶", signal: "BUY", score: 85 },
  { name: "Omega", icon: "⚡", signal: "BUY", score: 88 },
  { name: "Titan", icon: "🏛️", signal: "BUY", score: 92 },
  { name: "Smart", icon: "💡", signal: "BUY", score: 80 },
  { name: "Pro", icon: "⭐", signal: "BUY", score: 85 },
]);

const signalHistory = ref([
  {
    timestamp: new Date(),
    symbol: "EURUSDm",
    signal: "BUY",
    titan_score: 92.5,
    omega_grade: "Ω+",
    result: "WIN",
  },
  {
    timestamp: new Date(Date.now() - 3600000),
    symbol: "GBPUSDm",
    signal: "SELL",
    titan_score: 85.0,
    omega_grade: "Ω",
    result: "WIN",
  },
  {
    timestamp: new Date(Date.now() - 7200000),
    symbol: "XAUUSDm",
    signal: "BUY",
    titan_score: 78.5,
    omega_grade: "α+",
    result: "LOSS",
  },
]);

// Computed
const titanScore = computed(() => titanData.value.titan_score?.toFixed(0) || 0);
const omegaGrade = computed(() => omegaData.value.grade || "N/A");
const alphaGrade = computed(() => "A+");
const riskLevel = computed(() => riskData.value.risk_level || "SAFE");
const consensus = computed(() => titanData.value.consensus || "N/A");

const riskLevelClass = computed(() => {
  const level = riskLevel.value;
  if (level === "SAFE") return "text-green-400";
  if (level === "CAUTION") return "text-yellow-400";
  if (level === "WARNING") return "text-orange-400";
  return "text-red-400";
});

const recommendation = computed(() => {
  const score = titanData.value.titan_score || 0;
  if (score >= 85) return "STRONG BUY";
  if (score >= 70) return "BUY";
  if (score >= 55) return "WEAK BUY";
  return "WAIT";
});

const recommendationEmoji = computed(() => {
  const score = titanData.value.titan_score || 0;
  if (score >= 85) return "🚀";
  if (score >= 70) return "✅";
  if (score >= 55) return "⚠️";
  return "🛑";
});

const recommendationReason = computed(() => {
  return `Titan: ${titanScore.value}% | Omega: ${omegaGrade.value} | Risk: ${riskLevel.value}`;
});

const recommendationClass = computed(() => {
  const score = titanData.value.titan_score || 0;
  if (score >= 85)
    return "bg-green-500/20 border border-green-400 text-green-300";
  if (score >= 70) return "bg-blue-500/20 border border-blue-400 text-blue-300";
  if (score >= 55)
    return "bg-yellow-500/20 border border-yellow-400 text-yellow-300";
  return "bg-red-500/20 border border-red-400 text-red-300";
});

const positionMultiplier = computed(() => {
  return Math.min(
    titanData.value.position_multiplier || 1,
    omegaData.value.position_multiplier || 1,
  ).toFixed(2);
});

// Methods
const refreshAll = async () => {
  isLoading.value = true;
  try {
    // Fetch real data from API
    const [intelligence, botStatus, titan, omega, risk, history] =
      await Promise.all([
        api.getIntelligenceStatus(),
        api.getBotStatus(),
        api.getTitanData(selectedSymbol.value),
        api.getOmegaData(selectedSymbol.value),
        api.getRiskData(),
        api.getSignalHistory(20),
      ]);

    console.log("API Data:", { intelligence, titan, omega, risk, history });

    // Update Titan data
    if (titan && !titan._isMock && titan.status === "active") {
      titanData.value = {
        ...titanData.value,
        grade: titan.grade || "N/A",
        titan_score: titan.titan_score || 0,
        consensus: titan.consensus || "N/A",
        market_condition: titan.market_condition || "UNKNOWN",
        prediction: titan.prediction || { direction: "WAIT", confidence: 0 },
        position_multiplier: titan.position_multiplier || 1.0,
        agreeing_modules: titan.agreeing_modules || 0,
        total_modules: titan.total_modules || 0,
        edge_factors: titan.edge_factors || [],
        risk_factors: titan.risk_factors || [],
        final_verdict: titan.final_verdict || "",
      };
    }

    // Update Omega data
    if (omega && !omega._isMock && omega.status === "active") {
      omegaData.value = {
        ...omegaData.value,
        grade: omega.grade || "N/A",
        omega_score: omega.omega_score || 0,
        institutional_flow: omega.institutional_flow || "N/A",
        smart_money: omega.smart_money || "N/A",
        manipulation: omega.manipulation_detected || "NONE",
        sentiment: omega.sentiment || 0,
        current_regime: omega.current_regime || "N/A",
        predicted_regime: omega.predicted_regime || "N/A",
        position_multiplier: omega.position_multiplier || 1.0,
        final_verdict: omega.final_verdict || "",
        edge_factors: omega.edge_factors || [],
        risk_factors: omega.risk_factors || [],
      };
    }

    // Update Risk data
    if (risk && !risk._isMock) {
      riskData.value = {
        ...riskData.value,
        risk_level: risk.risk_level || "SAFE",
        balance: risk.balance || 0,
        equity: risk.equity || 0,
        daily_pnl: risk.daily_pnl || 0,
        open_positions: risk.open_positions || 0,
        max_positions: risk.max_positions || 3,
        risk_per_trade: risk.risk_per_trade || 2,
        max_daily_loss: risk.max_daily_loss || 5,
        leverage: risk.leverage || 2000,
        risk_score: risk.risk_score || 0,
        can_trade: risk.can_trade ?? true,
        can_open_position: risk.can_open_position ?? true,
        losing_streak: risk.losing_streak || 0,
      };
    }

    // Update Signal History
    if (history && history.signals && history.signals.length > 0) {
      signalHistory.value = history.signals.map((s) => ({
        timestamp: new Date(s.timestamp),
        symbol: s.symbol,
        signal: s.signal,
        titan_score: s.titan_score || 0,
        omega_grade: s.omega_grade || "N/A",
        result: s.result || "PENDING",
      }));
    }

    // Update module analysis from intelligence status
    if (intelligence && !intelligence._isMock) {
      moduleAnalysis.value = [
        {
          name: "Neural",
          icon: "🧠",
          signal: intelligence.neural?.active ? "ACTIVE" : "OFF",
          score: 0,
        },
        {
          name: "Deep",
          icon: "🔮",
          signal: intelligence.deep?.active ? "ACTIVE" : "OFF",
          score: 0,
        },
        {
          name: "Quantum",
          icon: "⚛️",
          signal: intelligence.quantum?.active ? "ACTIVE" : "OFF",
          score: 0,
        },
        {
          name: "Alpha",
          icon: "🔶",
          signal: intelligence.alpha?.active ? "ACTIVE" : "OFF",
          score: intelligence.alpha?.score || 0,
        },
        {
          name: "Omega",
          icon: "⚡",
          signal: intelligence.omega?.active ? "ACTIVE" : "OFF",
          score: intelligence.omega?.score || 0,
        },
        {
          name: "Titan",
          icon: "🏛️",
          signal: intelligence.titan?.active ? "ACTIVE" : "OFF",
          score: intelligence.titan?.score || 0,
        },
        {
          name: "Smart",
          icon: "💡",
          signal: intelligence.smart?.active ? "ACTIVE" : "OFF",
          score: 0,
        },
        {
          name: "Pro",
          icon: "⭐",
          signal: intelligence.pro?.active ? "ACTIVE" : "OFF",
          score: 0,
        },
      ];
    }
  } catch (e) {
    console.error("Failed to refresh:", e);
  } finally {
    isLoading.value = false;
  }
};

const handleModuleSelected = (moduleName) => {
  console.log("Module selected:", moduleName);
};

const handlePauseTrading = () => {
  console.log("Pause trading requested");
  api.pauseTrading();
};

const handleCloseAll = () => {
  if (confirm("Are you sure you want to close all positions?")) {
    console.log("Close all positions requested");
  }
};

const handleResetDaily = () => {
  if (confirm("Reset daily statistics?")) {
    riskData.value.daily_pnl = 0;
    riskData.value.losing_streak = 0;
  }
};

const formatTime = (timestamp) => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
};

const getSignalClass = (signal) => {
  if (!signal) return "text-gray-400";
  if (signal === "BUY" || signal === "STRONG_BUY") return "text-green-400";
  if (signal === "SELL" || signal === "STRONG_SELL") return "text-red-400";
  return "text-yellow-400";
};

const getSignalBadgeClass = (signal) => {
  if (!signal) return "px-2 py-1 rounded text-xs bg-gray-600 text-gray-300";
  if (signal === "BUY" || signal === "STRONG_BUY")
    return "px-2 py-1 rounded text-xs bg-green-500/20 text-green-400 border border-green-400/30";
  if (signal === "SELL" || signal === "STRONG_SELL")
    return "px-2 py-1 rounded text-xs bg-red-500/20 text-red-400 border border-red-400/30";
  return "px-2 py-1 rounded text-xs bg-yellow-500/20 text-yellow-400 border border-yellow-400/30";
};

const getResultClass = (result) => {
  if (result === "WIN") return "text-green-400 font-bold";
  if (result === "LOSS") return "text-red-400 font-bold";
  return "text-gray-400";
};

// Watch for symbol changes
watch(selectedSymbol, () => {
  refreshAll();
});

onMounted(() => {
  refreshAll();
  // Auto-refresh every 30 seconds
  refreshInterval = setInterval(() => {
    refreshAll();
  }, 30000);
});

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
});
</script>
