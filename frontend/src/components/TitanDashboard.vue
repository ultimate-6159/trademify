<template>
  <div
    class="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-6 border border-purple-500/30"
  >
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-bold text-white flex items-center gap-2">
        <span class="text-3xl">🏛️</span>
        TITAN CORE SYNTHESIS
        <span class="text-sm font-normal text-purple-400"
          >Meta-Intelligence</span
        >
      </h2>

      <!-- Titan Grade Badge -->
      <div :class="titanGradeClass">
        {{ titanData.grade || "🏛️ ANALYZING..." }}
      </div>
    </div>

    <!-- Main Score Display -->
    <div class="grid grid-cols-3 gap-4 mb-6">
      <!-- Titan Score -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-purple-500/20"
      >
        <div class="text-gray-400 text-sm mb-1">TITAN SCORE</div>
        <div class="text-4xl font-bold" :class="titanScoreClass">
          {{ titanData.titan_score?.toFixed(1) || 0 }}%
        </div>
        <div class="mt-2 w-full bg-gray-600 rounded-full h-2">
          <div
            class="h-2 rounded-full transition-all duration-500"
            :class="getScoreBarClass(titanData.titan_score)"
            :style="{ width: (titanData.titan_score || 0) + '%' }"
          ></div>
        </div>
      </div>

      <!-- Consensus -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-blue-500/20"
      >
        <div class="text-gray-400 text-sm mb-1">CONSENSUS</div>
        <div class="text-2xl font-bold" :class="consensusClass">
          {{ consensusEmoji }} {{ titanData.consensus || "ANALYZING" }}
        </div>
        <div class="text-xs text-gray-400 mt-2">Module Agreement Level</div>
      </div>

      <!-- Market Condition -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-green-500/20"
      >
        <div class="text-gray-400 text-sm mb-1">MARKET CONDITION</div>
        <div class="text-2xl font-bold" :class="marketConditionClass">
          {{ marketEmoji }} {{ titanData.market_condition || "UNKNOWN" }}
        </div>
        <div class="text-xs text-gray-400 mt-2">Overall Market Assessment</div>
      </div>
    </div>

    <!-- Prediction Ensemble -->
    <div class="bg-gray-700/50 rounded-lg p-4 mb-6 border border-cyan-500/20">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-white font-semibold flex items-center gap-2">
          <span>🎯</span> Prediction Ensemble
        </h3>
        <div :class="predictionClass">
          {{ titanData.prediction?.direction || "WAIT" }}
          ({{ (titanData.prediction?.confidence || 0).toFixed(1) }}%)
        </div>
      </div>

      <!-- Prediction Methods -->
      <div class="grid grid-cols-4 gap-3">
        <div
          v-for="method in predictionMethods"
          :key="method.name"
          class="bg-gray-600/50 rounded-lg p-3 text-center"
        >
          <div class="text-gray-400 text-xs mb-1">{{ method.name }}</div>
          <div
            :class="getPredictionMethodClass(method.value)"
            class="font-bold"
          >
            {{ method.value }}
          </div>
          <div class="text-xs text-gray-500">{{ method.weight }}</div>
        </div>
      </div>
    </div>

    <!-- Calibration & Self-Improvement -->
    <div class="grid grid-cols-2 gap-4 mb-6">
      <!-- Confidence Calibration -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-yellow-500/20">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>⚖️</span> Confidence Calibration
        </h3>
        <div class="space-y-2">
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Raw Confidence</span>
            <span class="text-white"
              >{{ titanData.raw_confidence?.toFixed(1) || 0 }}%</span
            >
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Calibrated</span>
            <span :class="calibratedClass"
              >{{ titanData.calibrated_confidence?.toFixed(1) || 0 }}%</span
            >
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Historical Accuracy</span>
            <span class="text-blue-400"
              >{{ titanData.historical_accuracy?.toFixed(1) || 50 }}%</span
            >
          </div>
        </div>
      </div>

      <!-- Self-Improvement -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-green-500/20">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>📈</span> Self-Improvement Engine
        </h3>
        <div
          v-if="titanData.improvements && titanData.improvements.length > 0"
          class="space-y-2"
        >
          <div
            v-for="(imp, idx) in titanData.improvements.slice(0, 3)"
            :key="idx"
            class="flex items-center gap-2 text-sm"
          >
            <span :class="getImprovementIcon(imp.type)">{{
              getImprovementEmoji(imp.type)
            }}</span>
            <span class="text-gray-300 flex-1 truncate">{{ imp.message }}</span>
          </div>
        </div>
        <div v-else class="text-gray-400 text-sm text-center py-2">
          ✓ No issues detected
        </div>
      </div>
    </div>

    <!-- Module Weights Visualization -->
    <div class="bg-gray-700/50 rounded-lg p-4 border border-blue-500/20">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>⚙️</span> Dynamic Module Weights
      </h3>
      <div class="space-y-2">
        <div
          v-for="(weight, module) in moduleWeights"
          :key="module"
          class="flex items-center gap-3"
        >
          <span class="text-gray-300 text-sm w-24 truncate">{{
            getModuleLabel(module)
          }}</span>
          <div
            class="flex-1 bg-gray-600 rounded-full h-3 relative overflow-hidden"
          >
            <div
              class="h-full rounded-full transition-all duration-500"
              :class="getWeightBarClass(weight)"
              :style="{ width: weight * 100 + '%' }"
            ></div>
          </div>
          <span class="text-gray-400 text-xs w-12 text-right"
            >{{ (weight * 100).toFixed(0) }}%</span
          >
        </div>
      </div>
    </div>

    <!-- Final Decision -->
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
            {{ titanData.position_multiplier?.toFixed(2) || 1.0 }}x
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  titanData: {
    type: Object,
    default: () => ({
      grade: "🏛️ TITAN SUPREME",
      titan_score: 92.5,
      consensus: "STRONG",
      market_condition: "FAVORABLE",
      prediction: { direction: "BUY", confidence: 85 },
      raw_confidence: 80,
      calibrated_confidence: 75,
      historical_accuracy: 72,
      position_multiplier: 0.9,
      improvements: [],
      module_weights: {},
    }),
  },
});

// Computed Classes
const titanGradeClass = computed(() => {
  const grade = props.titanData.grade || "";
  if (grade.includes("SUPREME"))
    return "px-4 py-2 bg-yellow-500/20 border border-yellow-400 rounded-lg text-yellow-300 font-bold";
  if (grade.includes("ELITE"))
    return "px-4 py-2 bg-purple-500/20 border border-purple-400 rounded-lg text-purple-300 font-bold";
  if (grade.includes("PRIME"))
    return "px-4 py-2 bg-blue-500/20 border border-blue-400 rounded-lg text-blue-300 font-bold";
  if (grade.includes("CORE"))
    return "px-4 py-2 bg-cyan-500/20 border border-cyan-400 rounded-lg text-cyan-300 font-bold";
  if (grade.includes("BASE"))
    return "px-4 py-2 bg-gray-500/20 border border-gray-400 rounded-lg text-gray-300 font-bold";
  if (grade.includes("MORTAL"))
    return "px-4 py-2 bg-orange-500/20 border border-orange-400 rounded-lg text-orange-300 font-bold";
  return "px-4 py-2 bg-red-500/20 border border-red-400 rounded-lg text-red-300 font-bold";
});

const titanScoreClass = computed(() => {
  const score = props.titanData.titan_score || 0;
  if (score >= 90) return "text-yellow-300";
  if (score >= 80) return "text-green-400";
  if (score >= 70) return "text-blue-400";
  if (score >= 60) return "text-cyan-400";
  if (score >= 50) return "text-yellow-400";
  return "text-red-400";
});

const consensusEmoji = computed(() => {
  const consensus = props.titanData.consensus || "";
  if (consensus === "UNANIMOUS") return "🎯";
  if (consensus === "STRONG") return "💪";
  if (consensus === "MODERATE") return "🤝";
  if (consensus === "WEAK") return "🤔";
  return "⚠️";
});

const consensusClass = computed(() => {
  const consensus = props.titanData.consensus || "";
  if (consensus === "UNANIMOUS") return "text-green-400";
  if (consensus === "STRONG") return "text-blue-400";
  if (consensus === "MODERATE") return "text-yellow-400";
  if (consensus === "WEAK") return "text-orange-400";
  return "text-red-400";
});

const marketEmoji = computed(() => {
  const condition = props.titanData.market_condition || "";
  if (condition.includes("HIGHLY_FAVORABLE")) return "🚀";
  if (condition.includes("FAVORABLE")) return "✅";
  if (condition.includes("NEUTRAL")) return "➖";
  if (condition.includes("UNFAVORABLE")) return "⚠️";
  return "🛑";
});

const marketConditionClass = computed(() => {
  const condition = props.titanData.market_condition || "";
  if (condition.includes("HIGHLY_FAVORABLE")) return "text-green-400";
  if (condition.includes("FAVORABLE")) return "text-blue-400";
  if (condition.includes("NEUTRAL")) return "text-yellow-400";
  if (condition.includes("UNFAVORABLE")) return "text-orange-400";
  return "text-red-400";
});

const predictionClass = computed(() => {
  const direction = props.titanData.prediction?.direction || "";
  if (direction === "BUY" || direction === "BULLISH")
    return "text-green-400 font-bold text-lg";
  if (direction === "SELL" || direction === "BEARISH")
    return "text-red-400 font-bold text-lg";
  return "text-yellow-400 font-bold text-lg";
});

const predictionMethods = computed(() => {
  const pred = props.titanData.prediction || {};
  return [
    { name: "Momentum", value: pred.momentum || "N/A", weight: "25%" },
    { name: "Mean Revert", value: pred.mean_reversion || "N/A", weight: "25%" },
    { name: "Volatility", value: pred.volatility || "N/A", weight: "25%" },
    { name: "Pattern", value: pred.pattern || "N/A", weight: "25%" },
  ];
});

const calibratedClass = computed(() => {
  const raw = props.titanData.raw_confidence || 0;
  const calibrated = props.titanData.calibrated_confidence || 0;
  if (calibrated > raw) return "text-green-400 font-bold";
  if (calibrated < raw - 10) return "text-red-400 font-bold";
  return "text-yellow-400 font-bold";
});

const moduleWeights = computed(() => {
  return (
    props.titanData.module_weights || {
      neural: 0.15,
      deep: 0.15,
      quantum: 0.15,
      alpha: 0.2,
      omega: 0.2,
      pattern: 0.15,
    }
  );
});

const finalDecisionEmoji = computed(() => {
  const score = props.titanData.titan_score || 0;
  if (score >= 85) return "🚀";
  if (score >= 70) return "✅";
  if (score >= 55) return "⚠️";
  return "🛑";
});

const finalDecisionText = computed(() => {
  const score = props.titanData.titan_score || 0;
  if (score >= 85) return "STRONG TRADE SIGNAL";
  if (score >= 70) return "TRADE WITH CAUTION";
  if (score >= 55) return "REDUCED POSITION";
  if (score >= 40) return "MINIMAL POSITION";
  return "TRADE BLOCKED";
});

const finalDecisionReason = computed(() => {
  const score = props.titanData.titan_score || 0;
  const consensus = props.titanData.consensus || "";
  return `Score: ${score.toFixed(1)}% | Consensus: ${consensus}`;
});

const finalDecisionClass = computed(() => {
  const score = props.titanData.titan_score || 0;
  if (score >= 85)
    return "bg-green-500/20 border border-green-400 text-green-300";
  if (score >= 70) return "bg-blue-500/20 border border-blue-400 text-blue-300";
  if (score >= 55)
    return "bg-yellow-500/20 border border-yellow-400 text-yellow-300";
  if (score >= 40)
    return "bg-orange-500/20 border border-orange-400 text-orange-300";
  return "bg-red-500/20 border border-red-400 text-red-300";
});

// Helper functions
const getScoreBarClass = (score) => {
  if (score >= 85) return "bg-gradient-to-r from-green-500 to-green-400";
  if (score >= 70) return "bg-gradient-to-r from-blue-500 to-blue-400";
  if (score >= 55) return "bg-gradient-to-r from-yellow-500 to-yellow-400";
  return "bg-gradient-to-r from-red-500 to-red-400";
};

const getPredictionMethodClass = (value) => {
  if (value === "BUY" || value === "BULLISH" || value === "UP")
    return "text-green-400";
  if (value === "SELL" || value === "BEARISH" || value === "DOWN")
    return "text-red-400";
  return "text-yellow-400";
};

const getModuleLabel = (module) => {
  const labels = {
    neural: "🧠 Neural",
    deep: "🔮 Deep",
    quantum: "⚛️ Quantum",
    alpha: "🔶 Alpha",
    omega: "⚡ Omega",
    pattern: "📊 Pattern",
  };
  return labels[module] || module;
};

const getWeightBarClass = (weight) => {
  if (weight >= 0.2) return "bg-gradient-to-r from-green-500 to-green-400";
  if (weight >= 0.15) return "bg-gradient-to-r from-blue-500 to-blue-400";
  if (weight >= 0.1) return "bg-gradient-to-r from-yellow-500 to-yellow-400";
  return "bg-gradient-to-r from-gray-500 to-gray-400";
};

const getImprovementEmoji = (type) => {
  if (type === "WARNING") return "⚠️";
  if (type === "SUGGESTION") return "💡";
  if (type === "ERROR") return "❌";
  return "ℹ️";
};

const getImprovementIcon = (type) => {
  if (type === "WARNING") return "text-yellow-400";
  if (type === "SUGGESTION") return "text-blue-400";
  if (type === "ERROR") return "text-red-400";
  return "text-gray-400";
};
</script>
