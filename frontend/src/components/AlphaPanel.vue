<template>
  <div
    class="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-6 border border-orange-500/30"
  >
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-bold text-white flex items-center gap-2">
        <span class="text-3xl">🔶</span>
        ALPHA ENGINE
        <span class="text-sm font-normal text-orange-400"
          >Professional Grade Analysis</span
        >
      </h2>

      <!-- Alpha Grade Badge -->
      <div :class="alphaGradeBadgeClass">
        {{ alphaData.grade || "A" }}
      </div>
    </div>

    <!-- Main Metrics -->
    <div class="grid grid-cols-4 gap-4 mb-6">
      <!-- Alpha Score -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-orange-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">ALPHA SCORE</div>
        <div class="text-3xl font-bold" :class="alphaScoreClass">
          {{ alphaData.alpha_score?.toFixed(1) || 0 }}%
        </div>
      </div>

      <!-- Order Flow -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-blue-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">ORDER FLOW</div>
        <div class="text-xl font-bold" :class="orderFlowClass">
          {{ orderFlowEmoji }} {{ alphaData.order_flow_bias || "NEUTRAL" }}
        </div>
      </div>

      <!-- Divergence -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-purple-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">DIVERGENCE</div>
        <div class="text-xl font-bold" :class="divergenceClass">
          {{ alphaData.divergence || "NONE" }}
        </div>
      </div>

      <!-- Momentum Wave -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-cyan-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">MOMENTUM</div>
        <div class="text-xl font-bold" :class="momentumClass">
          {{ alphaData.momentum_wave || "N/A" }}
        </div>
      </div>
    </div>

    <!-- Order Flow Analysis -->
    <div class="bg-gray-700/50 rounded-lg p-4 mb-4 border border-blue-500/20">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>📊</span> Order Flow Analysis
      </h3>

      <div class="grid grid-cols-4 gap-4">
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Buy Volume</div>
          <div class="text-xl font-bold text-green-400">
            {{ alphaData.buy_volume?.toFixed(0) || 0 }}%
          </div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Sell Volume</div>
          <div class="text-xl font-bold text-red-400">
            {{ alphaData.sell_volume?.toFixed(0) || 0 }}%
          </div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">Delta</div>
          <div
            class="text-xl font-bold"
            :class="getDeltaClass(alphaData.delta)"
          >
            {{ alphaData.delta?.toFixed(0) || 0 }}
          </div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-xs mb-1">CVD</div>
          <div
            class="text-xl font-bold"
            :class="getCvdClass(alphaData.cvd_trend)"
          >
            {{ alphaData.cvd_trend || "FLAT" }}
          </div>
        </div>
      </div>
    </div>

    <!-- Liquidity Zones -->
    <div class="bg-gray-700/50 rounded-lg p-4 mb-4 border border-yellow-500/20">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>💧</span> Liquidity Zones
      </h3>

      <div class="space-y-2">
        <div
          v-for="(zone, idx) in liquidityZones"
          :key="idx"
          class="flex items-center justify-between bg-gray-600/50 rounded-lg px-4 py-2"
        >
          <div class="flex items-center gap-3">
            <span :class="getZoneTypeClass(zone.type)">{{
              getZoneEmoji(zone.type)
            }}</span>
            <span class="text-white">{{ zone.type }}</span>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-gray-400">{{ zone.price?.toFixed(5) }}</span>
            <span :class="getZoneStrengthClass(zone.strength)">{{
              zone.strength
            }}</span>
          </div>
        </div>
        <div
          v-if="liquidityZones.length === 0"
          class="text-center text-gray-400 py-2"
        >
          No significant liquidity zones detected
        </div>
      </div>
    </div>

    <!-- Market Profile -->
    <div class="grid grid-cols-2 gap-4 mb-4">
      <!-- Value Area -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-green-500/20">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>📈</span> Market Profile
        </h3>
        <div class="space-y-2">
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">VAH</span>
            <span class="text-green-400">{{
              alphaData.vah?.toFixed(5) || "N/A"
            }}</span>
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">POC</span>
            <span class="text-yellow-400 font-bold">{{
              alphaData.poc?.toFixed(5) || "N/A"
            }}</span>
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">VAL</span>
            <span class="text-red-400">{{
              alphaData.val?.toFixed(5) || "N/A"
            }}</span>
          </div>
        </div>
      </div>

      <!-- Risk Metrics -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-red-500/20">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>⚠️</span> Risk Metrics
        </h3>
        <div class="grid grid-cols-2 gap-2 text-sm">
          <div class="text-center">
            <div class="text-gray-400 text-xs">Sharpe</div>
            <div class="text-white font-bold">
              {{ alphaData.sharpe?.toFixed(2) || 0 }}
            </div>
          </div>
          <div class="text-center">
            <div class="text-gray-400 text-xs">Sortino</div>
            <div class="text-white font-bold">
              {{ alphaData.sortino?.toFixed(2) || 0 }}
            </div>
          </div>
          <div class="text-center">
            <div class="text-gray-400 text-xs">Calmar</div>
            <div class="text-white font-bold">
              {{ alphaData.calmar?.toFixed(2) || 0 }}
            </div>
          </div>
          <div class="text-center">
            <div class="text-gray-400 text-xs">Max DD</div>
            <div class="text-red-400 font-bold">
              {{ ((alphaData.max_dd || 0) * 100).toFixed(1) }}%
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Final Alpha Decision -->
    <div class="p-4 rounded-lg" :class="finalDecisionClass">
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
            {{ alphaData.position_multiplier?.toFixed(2) || 1.0 }}x
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  alphaData: {
    type: Object,
    default: () => ({
      grade: "A+",
      alpha_score: 85.0,
      order_flow_bias: "BULLISH",
      divergence: "NONE",
      momentum_wave: "IMPULSE",
      buy_volume: 60,
      sell_volume: 40,
      delta: 200,
      cvd_trend: "RISING",
      vah: 1.09,
      poc: 1.085,
      val: 1.08,
      sharpe: 1.5,
      sortino: 2.0,
      calmar: 1.2,
      max_dd: 0.05,
      position_multiplier: 0.9,
      liquidity_zones: [],
    }),
  },
});

// Computed
const alphaGradeBadgeClass = computed(() => {
  const grade = props.alphaData.grade || "";
  if (grade.includes("A+") || grade.includes("ELITE"))
    return "px-4 py-2 bg-yellow-500/20 border border-yellow-400 rounded-lg text-yellow-300 font-bold text-xl";
  if (grade.includes("A") || grade.includes("PRIME"))
    return "px-4 py-2 bg-green-500/20 border border-green-400 rounded-lg text-green-300 font-bold text-xl";
  if (grade.includes("B"))
    return "px-4 py-2 bg-blue-500/20 border border-blue-400 rounded-lg text-blue-300 font-bold text-xl";
  if (grade.includes("C"))
    return "px-4 py-2 bg-yellow-500/20 border border-yellow-400 rounded-lg text-yellow-300 font-bold text-xl";
  return "px-4 py-2 bg-red-500/20 border border-red-400 rounded-lg text-red-300 font-bold text-xl";
});

const alphaScoreClass = computed(() => {
  const score = props.alphaData.alpha_score || 0;
  if (score >= 85) return "text-yellow-300";
  if (score >= 70) return "text-green-400";
  if (score >= 55) return "text-blue-400";
  return "text-red-400";
});

const orderFlowEmoji = computed(() => {
  const flow = props.alphaData.order_flow_bias || "";
  if (flow === "BULLISH") return "📈";
  if (flow === "BEARISH") return "📉";
  return "➖";
});

const orderFlowClass = computed(() => {
  const flow = props.alphaData.order_flow_bias || "";
  if (flow === "BULLISH") return "text-green-400";
  if (flow === "BEARISH") return "text-red-400";
  return "text-yellow-400";
});

const divergenceClass = computed(() => {
  const div = props.alphaData.divergence || "";
  if (div === "NONE") return "text-gray-400";
  if (div.includes("BULLISH")) return "text-green-400";
  if (div.includes("BEARISH")) return "text-red-400";
  return "text-yellow-400";
});

const momentumClass = computed(() => {
  const mom = props.alphaData.momentum_wave || "";
  if (mom === "IMPULSE") return "text-green-400";
  if (mom === "CORRECTION") return "text-red-400";
  return "text-yellow-400";
});

const liquidityZones = computed(() => {
  return props.alphaData.liquidity_zones || [];
});

const finalDecisionEmoji = computed(() => {
  const score = props.alphaData.alpha_score || 0;
  if (score >= 85) return "🚀";
  if (score >= 70) return "✅";
  if (score >= 55) return "⚠️";
  return "🛑";
});

const finalDecisionText = computed(() => {
  const score = props.alphaData.alpha_score || 0;
  if (score >= 85) return "ELITE TRADE SETUP";
  if (score >= 70) return "PRIME TRADE SETUP";
  if (score >= 55) return "STANDARD SETUP";
  return "WEAK SETUP";
});

const finalDecisionReason = computed(() => {
  return `Score: ${(props.alphaData.alpha_score || 0).toFixed(1)}% | Flow: ${props.alphaData.order_flow_bias || "N/A"}`;
});

const finalDecisionClass = computed(() => {
  const score = props.alphaData.alpha_score || 0;
  if (score >= 85)
    return "bg-green-500/20 border border-green-400 text-green-300";
  if (score >= 70) return "bg-blue-500/20 border border-blue-400 text-blue-300";
  if (score >= 55)
    return "bg-yellow-500/20 border border-yellow-400 text-yellow-300";
  return "bg-red-500/20 border border-red-400 text-red-300";
});

// Helper functions
const getDeltaClass = (delta) => {
  if (!delta) return "text-gray-400";
  if (delta > 0) return "text-green-400";
  if (delta < 0) return "text-red-400";
  return "text-yellow-400";
};

const getCvdClass = (trend) => {
  if (!trend) return "text-gray-400";
  if (trend === "RISING") return "text-green-400";
  if (trend === "FALLING") return "text-red-400";
  return "text-yellow-400";
};

const getZoneEmoji = (type) => {
  if (type === "SUPPORT") return "🟢";
  if (type === "RESISTANCE") return "🔴";
  if (type === "LIQUIDITY_POOL") return "💧";
  return "⚪";
};

const getZoneTypeClass = (type) => {
  if (type === "SUPPORT") return "text-green-400";
  if (type === "RESISTANCE") return "text-red-400";
  return "text-blue-400";
};

const getZoneStrengthClass = (strength) => {
  if (strength === "STRONG") return "text-green-400 font-bold";
  if (strength === "MODERATE") return "text-yellow-400";
  return "text-gray-400";
};
</script>
