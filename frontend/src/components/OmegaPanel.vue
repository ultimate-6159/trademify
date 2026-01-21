<template>
  <div
    class="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-6 border border-indigo-500/30"
  >
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-bold text-white flex items-center gap-2">
        <span class="text-3xl">🧠⚡</span>
        OMEGA BRAIN
        <span class="text-sm font-normal text-indigo-400"
          >Institutional Intelligence</span
        >
      </h2>

      <!-- Omega Grade Badge -->
      <div :class="omegaGradeClass">
        {{ omegaData.grade || "Ω" }}
      </div>
    </div>

    <!-- Main Metrics -->
    <div class="grid grid-cols-4 gap-4 mb-6">
      <!-- Omega Score -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-indigo-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">OMEGA SCORE</div>
        <div class="text-3xl font-bold" :class="omegaScoreClass">
          {{ omegaData.omega_score?.toFixed(1) || 0 }}%
        </div>
      </div>

      <!-- Institutional Flow -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-blue-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">INST. FLOW</div>
        <div class="text-xl font-bold" :class="flowClass">
          {{ flowEmoji }} {{ omegaData.institutional_flow || "NEUTRAL" }}
        </div>
      </div>

      <!-- Manipulation Alert -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border"
        :class="manipulationBorderClass"
      >
        <div class="text-gray-400 text-xs mb-1">MANIPULATION</div>
        <div class="text-xl font-bold" :class="manipulationClass">
          {{ manipulationEmoji }} {{ omegaData.manipulation || "NONE" }}
        </div>
      </div>

      <!-- Sentiment -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-purple-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">SENTIMENT</div>
        <div class="text-xl font-bold" :class="sentimentClass">
          {{ sentimentEmoji }} {{ omegaData.sentiment || "NEUTRAL" }}
        </div>
      </div>
    </div>

    <!-- Institutional Flow Analysis -->
    <div class="bg-gray-700/50 rounded-lg p-4 mb-4 border border-blue-500/20">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>🏦</span> Institutional Flow Detection
      </h3>

      <div class="grid grid-cols-3 gap-4">
        <!-- Volume Anomaly -->
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Volume Anomaly</div>
          <div
            class="text-2xl font-bold"
            :class="getAnomalyClass(omegaData.volume_anomaly)"
          >
            {{ omegaData.volume_anomaly?.toFixed(2) || 1.0 }}x
          </div>
          <div class="text-xs text-gray-500">vs Average</div>
        </div>

        <!-- Big Money Activity -->
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Big Money</div>
          <div
            class="text-2xl font-bold"
            :class="getBigMoneyClass(omegaData.big_money)"
          >
            {{ omegaData.big_money || "INACTIVE" }}
          </div>
        </div>

        <!-- Smart Money Direction -->
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Smart Money</div>
          <div
            class="text-2xl font-bold"
            :class="getSmartMoneyClass(omegaData.smart_money)"
          >
            {{ smartMoneyEmoji }} {{ omegaData.smart_money || "NEUTRAL" }}
          </div>
        </div>
      </div>
    </div>

    <!-- Manipulation Scanner -->
    <div
      class="bg-gray-700/50 rounded-lg p-4 mb-4 border"
      :class="manipulationScannerBorder"
    >
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>🔍</span> Manipulation Scanner
        <span v-if="hasManipulation" class="text-red-400 animate-pulse"
          >⚠️ ALERT</span
        >
      </h3>

      <div v-if="manipulationAlerts.length > 0" class="space-y-2">
        <div
          v-for="(alert, idx) in manipulationAlerts"
          :key="idx"
          class="flex items-center gap-3 p-2 rounded-lg"
          :class="getAlertClass(alert.severity)"
        >
          <span>{{ getAlertEmoji(alert.type) }}</span>
          <div class="flex-1">
            <div class="font-semibold">{{ alert.type }}</div>
            <div class="text-xs text-gray-400">{{ alert.description }}</div>
          </div>
          <div class="text-right">
            <div :class="getSeverityClass(alert.severity)">
              {{ alert.severity }}
            </div>
            <div class="text-xs text-gray-400">{{ alert.confidence }}%</div>
          </div>
        </div>
      </div>
      <div v-else class="text-green-400 text-center py-4">
        ✅ No manipulation detected - Market is clean
      </div>
    </div>

    <!-- Regime Prediction -->
    <div class="grid grid-cols-2 gap-4 mb-4">
      <!-- Current Regime -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-cyan-500/20">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>📊</span> Regime Transition
        </h3>
        <div class="flex items-center justify-between">
          <div class="text-center flex-1">
            <div class="text-gray-400 text-xs mb-1">Current</div>
            <div
              :class="getRegimeClass(omegaData.current_regime)"
              class="text-lg font-bold"
            >
              {{ omegaData.current_regime || "N/A" }}
            </div>
          </div>
          <div class="text-2xl text-gray-500 px-4">→</div>
          <div class="text-center flex-1">
            <div class="text-gray-400 text-xs mb-1">Predicted</div>
            <div
              :class="getRegimeClass(omegaData.predicted_regime)"
              class="text-lg font-bold"
            >
              {{ omegaData.predicted_regime || "N/A" }}
            </div>
          </div>
        </div>
        <div class="mt-2 text-center text-sm text-gray-400">
          Confidence: {{ omegaData.regime_confidence?.toFixed(0) || 0 }}%
        </div>
      </div>

      <!-- Risk Parity -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-green-500/20">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>⚖️</span> Risk Parity Allocation
        </h3>
        <div class="space-y-2">
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Suggested Allocation</span>
            <span class="text-white font-bold"
              >{{ omegaData.suggested_allocation?.toFixed(0) || 0 }}%</span
            >
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Volatility Weight</span>
            <span class="text-white">{{
              omegaData.volatility_weight?.toFixed(2) || 1.0
            }}</span>
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Risk Contribution</span>
            <span
              :class="getRiskContributionClass(omegaData.risk_contribution)"
            >
              {{ omegaData.risk_contribution || "NORMAL" }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Position Orchestration -->
    <div class="bg-gray-700/50 rounded-lg p-4 border border-yellow-500/20">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>🎯</span> Position Orchestration
      </h3>

      <div class="grid grid-cols-4 gap-4">
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Entry Style</div>
          <div class="text-white font-bold">
            {{ omegaData.entry_style || "SINGLE" }}
          </div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Scale Points</div>
          <div class="text-white font-bold">
            {{ omegaData.scale_points || 1 }}
          </div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Exit Plan</div>
          <div class="text-white font-bold">
            {{ omegaData.exit_plan || "FULL" }}
          </div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Trailing Stop</div>
          <div
            :class="
              omegaData.trailing_stop ? 'text-green-400' : 'text-gray-400'
            "
          >
            {{ omegaData.trailing_stop ? "✓ ACTIVE" : "○ OFF" }}
          </div>
        </div>
      </div>
    </div>

    <!-- Final Omega Decision -->
    <div class="mt-6 p-4 rounded-lg" :class="finalDecisionClass">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="text-4xl">{{ finalDecisionEmoji }}</span>
          <div>
            <div class="text-lg font-bold">{{ finalDecisionText }}</div>
            <div class="text-sm opacity-75">{{ finalDecisionReason }}</div>
          </div>
        </div>
        <div class="text-right">
          <div class="text-sm text-gray-400">Position Multiplier</div>
          <div class="text-2xl font-bold">
            {{ omegaData.position_multiplier?.toFixed(2) || 1.0 }}x
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  omegaData: {
    type: Object,
    default: () => ({
      grade: "Ω+",
      omega_score: 85.5,
      institutional_flow: "ACCUMULATION",
      manipulation: "NONE",
      sentiment: "BULLISH",
      volume_anomaly: 1.5,
      big_money: "ACTIVE",
      smart_money: "BUYING",
      current_regime: "UPTREND",
      predicted_regime: "UPTREND",
      regime_confidence: 75,
      suggested_allocation: 80,
      volatility_weight: 0.85,
      risk_contribution: "NORMAL",
      entry_style: "SCALE_IN",
      scale_points: 3,
      exit_plan: "PARTIAL",
      trailing_stop: true,
      position_multiplier: 0.9,
    }),
  },
});

// Computed Classes
const omegaGradeClass = computed(() => {
  const grade = props.omegaData.grade || "";
  if (grade === "Ω+")
    return "px-4 py-2 bg-yellow-500/20 border border-yellow-400 rounded-lg text-yellow-300 font-bold text-xl";
  if (grade === "Ω")
    return "px-4 py-2 bg-purple-500/20 border border-purple-400 rounded-lg text-purple-300 font-bold text-xl";
  if (grade === "α+")
    return "px-4 py-2 bg-blue-500/20 border border-blue-400 rounded-lg text-blue-300 font-bold text-xl";
  if (grade === "α")
    return "px-4 py-2 bg-cyan-500/20 border border-cyan-400 rounded-lg text-cyan-300 font-bold text-xl";
  if (grade === "β")
    return "px-4 py-2 bg-green-500/20 border border-green-400 rounded-lg text-green-300 font-bold text-xl";
  if (grade === "γ")
    return "px-4 py-2 bg-gray-500/20 border border-gray-400 rounded-lg text-gray-300 font-bold text-xl";
  return "px-4 py-2 bg-red-500/20 border border-red-400 rounded-lg text-red-300 font-bold text-xl";
});

const omegaScoreClass = computed(() => {
  const score = props.omegaData.omega_score || 0;
  if (score >= 85) return "text-yellow-300";
  if (score >= 70) return "text-green-400";
  if (score >= 55) return "text-blue-400";
  return "text-red-400";
});

const flowEmoji = computed(() => {
  const flow = props.omegaData.institutional_flow || "";
  if (flow === "ACCUMULATION") return "📈";
  if (flow === "DISTRIBUTION") return "📉";
  return "➖";
});

const flowClass = computed(() => {
  const flow = props.omegaData.institutional_flow || "";
  if (flow === "ACCUMULATION") return "text-green-400";
  if (flow === "DISTRIBUTION") return "text-red-400";
  return "text-yellow-400";
});

const manipulationEmoji = computed(() => {
  const manip = props.omegaData.manipulation || "";
  if (manip === "NONE") return "✅";
  if (manip === "STOP_HUNT") return "🎯";
  if (manip === "FAKEOUT") return "🎭";
  if (manip === "LIQUIDITY_GRAB") return "💰";
  return "⚠️";
});

const manipulationClass = computed(() => {
  const manip = props.omegaData.manipulation || "";
  if (manip === "NONE") return "text-green-400";
  return "text-red-400 animate-pulse";
});

const manipulationBorderClass = computed(() => {
  const manip = props.omegaData.manipulation || "";
  if (manip === "NONE") return "border-green-500/20";
  return "border-red-500/50 animate-pulse";
});

const sentimentEmoji = computed(() => {
  const sentiment = props.omegaData.sentiment || "";
  if (sentiment === "BULLISH") return "🟢";
  if (sentiment === "BEARISH") return "🔴";
  return "🟡";
});

const sentimentClass = computed(() => {
  const sentiment = props.omegaData.sentiment || "";
  if (sentiment === "BULLISH") return "text-green-400";
  if (sentiment === "BEARISH") return "text-red-400";
  return "text-yellow-400";
});

const hasManipulation = computed(() => {
  return (
    props.omegaData.manipulation && props.omegaData.manipulation !== "NONE"
  );
});

const manipulationAlerts = computed(() => {
  return props.omegaData.manipulation_alerts || [];
});

const manipulationScannerBorder = computed(() => {
  if (hasManipulation.value) return "border-red-500/50";
  return "border-gray-600/50";
});

const smartMoneyEmoji = computed(() => {
  const sm = props.omegaData.smart_money || "";
  if (sm === "BUYING") return "🟢";
  if (sm === "SELLING") return "🔴";
  return "⚪";
});

const finalDecisionEmoji = computed(() => {
  const score = props.omegaData.omega_score || 0;
  if (hasManipulation.value) return "🛑";
  if (score >= 85) return "🚀";
  if (score >= 70) return "✅";
  if (score >= 55) return "⚠️";
  return "🛑";
});

const finalDecisionText = computed(() => {
  if (hasManipulation.value) return "MANIPULATION DETECTED";
  const score = props.omegaData.omega_score || 0;
  if (score >= 85) return "INSTITUTIONAL GRADE SIGNAL";
  if (score >= 70) return "STRONG SIGNAL";
  if (score >= 55) return "MODERATE SIGNAL";
  return "WEAK SIGNAL";
});

const finalDecisionReason = computed(() => {
  if (hasManipulation.value) return `Alert: ${props.omegaData.manipulation}`;
  return `Score: ${(props.omegaData.omega_score || 0).toFixed(1)}% | Flow: ${props.omegaData.institutional_flow || "N/A"}`;
});

const finalDecisionClass = computed(() => {
  if (hasManipulation.value)
    return "bg-red-500/20 border border-red-400 text-red-300";
  const score = props.omegaData.omega_score || 0;
  if (score >= 85)
    return "bg-green-500/20 border border-green-400 text-green-300";
  if (score >= 70) return "bg-blue-500/20 border border-blue-400 text-blue-300";
  if (score >= 55)
    return "bg-yellow-500/20 border border-yellow-400 text-yellow-300";
  return "bg-red-500/20 border border-red-400 text-red-300";
});

// Helper functions
const getAnomalyClass = (anomaly) => {
  if (!anomaly) return "text-gray-400";
  if (anomaly >= 2.0) return "text-green-400";
  if (anomaly >= 1.5) return "text-blue-400";
  if (anomaly >= 1.0) return "text-yellow-400";
  return "text-red-400";
};

const getBigMoneyClass = (bigMoney) => {
  if (!bigMoney) return "text-gray-400";
  if (bigMoney === "ACTIVE") return "text-green-400";
  if (bigMoney === "ACCUMULATING") return "text-blue-400";
  if (bigMoney === "DISTRIBUTING") return "text-red-400";
  return "text-gray-400";
};

const getSmartMoneyClass = (sm) => {
  if (!sm) return "text-gray-400";
  if (sm === "BUYING") return "text-green-400";
  if (sm === "SELLING") return "text-red-400";
  return "text-yellow-400";
};

const getAlertClass = (severity) => {
  if (severity === "HIGH") return "bg-red-900/30 border border-red-500/30";
  if (severity === "MEDIUM")
    return "bg-yellow-900/30 border border-yellow-500/30";
  return "bg-blue-900/30 border border-blue-500/30";
};

const getAlertEmoji = (type) => {
  if (type === "STOP_HUNT") return "🎯";
  if (type === "FAKEOUT") return "🎭";
  if (type === "LIQUIDITY_GRAB") return "💰";
  if (type === "SPOOFING") return "👻";
  return "⚠️";
};

const getSeverityClass = (severity) => {
  if (severity === "HIGH") return "text-red-400 font-bold";
  if (severity === "MEDIUM") return "text-yellow-400 font-bold";
  return "text-blue-400";
};

const getRegimeClass = (regime) => {
  if (!regime) return "text-gray-400";
  if (regime === "UPTREND") return "text-green-400";
  if (regime === "DOWNTREND") return "text-red-400";
  if (regime === "RANGING") return "text-yellow-400";
  if (regime === "VOLATILE") return "text-orange-400";
  return "text-gray-400";
};

const getRiskContributionClass = (contribution) => {
  if (!contribution) return "text-gray-400";
  if (contribution === "LOW") return "text-green-400";
  if (contribution === "NORMAL") return "text-blue-400";
  if (contribution === "HIGH") return "text-yellow-400";
  if (contribution === "EXCESSIVE") return "text-red-400";
  return "text-gray-400";
};
</script>
