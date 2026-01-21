<template>
  <div
    class="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-6 border border-red-500/30"
  >
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-bold text-white flex items-center gap-2">
        <span class="text-3xl">🛡️</span>
        RISK MANAGEMENT CENTER
        <span class="text-sm font-normal text-red-400">Account Protection</span>
      </h2>

      <!-- Risk Level Badge -->
      <div :class="riskLevelBadgeClass">
        {{ riskLevelEmoji }} {{ riskData.risk_level || "SAFE" }}
      </div>
    </div>

    <!-- Main Risk Metrics -->
    <div class="grid grid-cols-4 gap-4 mb-6">
      <!-- Account Balance -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-blue-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">BALANCE</div>
        <div class="text-2xl font-bold text-blue-400">
          ${{ formatNumber(riskData.balance || 0) }}
        </div>
      </div>

      <!-- Equity -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-green-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">EQUITY</div>
        <div class="text-2xl font-bold" :class="equityClass">
          ${{ formatNumber(riskData.equity || 0) }}
        </div>
      </div>

      <!-- Daily P/L -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border"
        :class="dailyPnlBorderClass"
      >
        <div class="text-gray-400 text-xs mb-1">DAILY P/L</div>
        <div class="text-2xl font-bold" :class="dailyPnlClass">
          {{ riskData.daily_pnl >= 0 ? "+" : ""
          }}{{ riskData.daily_pnl?.toFixed(2) || 0 }}%
        </div>
      </div>

      <!-- Open Positions -->
      <div
        class="bg-gray-700/50 rounded-lg p-4 text-center border border-purple-500/20"
      >
        <div class="text-gray-400 text-xs mb-1">POSITIONS</div>
        <div class="text-2xl font-bold text-purple-400">
          {{ riskData.open_positions || 0 }} / {{ riskData.max_positions || 3 }}
        </div>
      </div>
    </div>

    <!-- Risk Gauges -->
    <div class="grid grid-cols-3 gap-4 mb-6">
      <!-- Daily Loss Gauge -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-red-500/20">
        <div class="text-gray-400 text-sm mb-2 text-center">
          Daily Loss Limit
        </div>
        <div class="relative h-24 flex items-end justify-center">
          <!-- Gauge Background -->
          <div
            class="w-full h-4 bg-gray-600 rounded-full relative overflow-hidden"
          >
            <!-- Warning Zone -->
            <div
              class="absolute right-0 top-0 h-full w-1/4 bg-red-900/50"
            ></div>
            <div
              class="absolute right-1/4 top-0 h-full w-1/4 bg-yellow-900/50"
            ></div>
            <!-- Fill -->
            <div
              class="h-full rounded-full transition-all"
              :class="getDailyLossBarClass"
              :style="{ width: dailyLossPercent + '%' }"
            ></div>
          </div>
        </div>
        <div class="text-center mt-2">
          <span class="text-2xl font-bold" :class="dailyLossClass">
            {{ Math.abs(riskData.daily_pnl || 0).toFixed(2) }}%
          </span>
          <span class="text-gray-400 text-sm">
            / {{ riskData.max_daily_loss || 5 }}%</span
          >
        </div>
        <div
          v-if="isDailyLossWarning"
          class="text-center text-yellow-400 text-xs mt-1 animate-pulse"
        >
          ⚠️ Approaching daily loss limit
        </div>
      </div>

      <!-- Risk Per Trade -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-orange-500/20">
        <div class="text-gray-400 text-sm mb-2 text-center">Risk Per Trade</div>
        <div class="text-center py-4">
          <div class="text-4xl font-bold text-orange-400">
            {{ riskData.risk_per_trade || 2 }}%
          </div>
          <div class="text-gray-400 text-sm mt-2">
            Max: ${{
              formatNumber(
                ((riskData.balance || 0) * (riskData.risk_per_trade || 2)) /
                  100,
              )
            }}
          </div>
        </div>
      </div>

      <!-- Leverage -->
      <div class="bg-gray-700/50 rounded-lg p-4 border border-yellow-500/20">
        <div class="text-gray-400 text-sm mb-2 text-center">Leverage</div>
        <div class="text-center py-4">
          <div class="text-4xl font-bold" :class="leverageClass">
            1:{{ riskData.leverage || 2000 }}
          </div>
          <div class="text-gray-400 text-sm mt-2">
            {{ leverageWarning }}
          </div>
        </div>
      </div>
    </div>

    <!-- Risk Assessment -->
    <div
      class="bg-gray-700/50 rounded-lg p-4 mb-4 border"
      :class="riskAssessmentBorder"
    >
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>📊</span> Risk Assessment
      </h3>

      <div class="grid grid-cols-2 gap-4">
        <!-- Risk Factors -->
        <div class="space-y-2">
          <div
            v-for="factor in riskFactors"
            :key="factor.name"
            class="flex items-center justify-between"
          >
            <div class="flex items-center gap-2">
              <span :class="getFactorStatusClass(factor.status)">
                {{
                  factor.status === "OK"
                    ? "✓"
                    : factor.status === "WARNING"
                      ? "⚠️"
                      : "✗"
                }}
              </span>
              <span class="text-gray-300 text-sm">{{ factor.name }}</span>
            </div>
            <span
              :class="getFactorValueClass(factor.status)"
              class="text-sm font-medium"
            >
              {{ factor.value }}
            </span>
          </div>
        </div>

        <!-- Risk Score -->
        <div class="text-center flex flex-col items-center justify-center">
          <div class="text-gray-400 text-sm mb-1">Overall Risk Score</div>
          <div class="text-5xl font-bold" :class="riskScoreClass">
            {{ riskData.risk_score || 0 }}
          </div>
          <div class="text-gray-400 text-sm mt-1">/ 100</div>
        </div>
      </div>
    </div>

    <!-- Trading Permissions -->
    <div class="bg-gray-700/50 rounded-lg p-4 mb-4 border border-cyan-500/20">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>🔐</span> Trading Permissions
      </h3>

      <div class="grid grid-cols-4 gap-4">
        <div
          class="text-center p-3 rounded-lg"
          :class="getPermissionClass(riskData.can_trade)"
        >
          <div class="text-2xl mb-1">
            {{ riskData.can_trade ? "✅" : "🚫" }}
          </div>
          <div class="text-sm font-medium">Can Trade</div>
        </div>
        <div
          class="text-center p-3 rounded-lg"
          :class="getPermissionClass(riskData.can_open_position)"
        >
          <div class="text-2xl mb-1">
            {{ riskData.can_open_position ? "✅" : "🚫" }}
          </div>
          <div class="text-sm font-medium">Open Position</div>
        </div>
        <div
          class="text-center p-3 rounded-lg"
          :class="getPermissionClass(!riskData.daily_limit_hit)"
        >
          <div class="text-2xl mb-1">
            {{ !riskData.daily_limit_hit ? "✅" : "🚫" }}
          </div>
          <div class="text-sm font-medium">Daily Limit</div>
        </div>
        <div
          class="text-center p-3 rounded-lg"
          :class="getPermissionClass(!riskData.losing_streak_limit)"
        >
          <div class="text-2xl mb-1">
            {{ !riskData.losing_streak_limit ? "✅" : "🚫" }}
          </div>
          <div class="text-sm font-medium">Streak Limit</div>
        </div>
      </div>
    </div>

    <!-- Losing Streak Monitor -->
    <div
      class="bg-gray-700/50 rounded-lg p-4 mb-4 border"
      :class="losingStreakBorder"
    >
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>📉</span> Losing Streak Monitor
      </h3>

      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <div class="text-center">
            <div class="text-gray-400 text-xs mb-1">Current Streak</div>
            <div class="text-3xl font-bold" :class="losingStreakClass">
              {{ riskData.losing_streak || 0 }}
            </div>
          </div>
          <div class="text-gray-500 text-2xl">/</div>
          <div class="text-center">
            <div class="text-gray-400 text-xs mb-1">Max Allowed</div>
            <div class="text-3xl font-bold text-gray-400">
              {{ riskData.max_losing_streak || 5 }}
            </div>
          </div>
        </div>

        <!-- Streak Visualization -->
        <div class="flex gap-1">
          <div
            v-for="i in riskData.max_losing_streak || 5"
            :key="i"
            class="w-8 h-8 rounded"
            :class="
              i <= (riskData.losing_streak || 0) ? 'bg-red-500' : 'bg-gray-600'
            "
          ></div>
        </div>
      </div>

      <div
        v-if="riskData.losing_streak >= 3"
        class="mt-3 p-2 rounded bg-yellow-900/30 border border-yellow-500/30"
      >
        <div class="flex items-center gap-2 text-yellow-400 text-sm">
          <span>⚠️</span>
          <span
            >Multiple consecutive losses detected. Consider pausing
            trading.</span
          >
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="grid grid-cols-3 gap-4">
      <button
        @click="$emit('pause-trading')"
        class="p-4 rounded-lg bg-yellow-600 hover:bg-yellow-700 text-white font-semibold transition-colors flex items-center justify-center gap-2"
      >
        <span>⏸️</span> Pause Trading
      </button>
      <button
        @click="$emit('close-all')"
        class="p-4 rounded-lg bg-red-600 hover:bg-red-700 text-white font-semibold transition-colors flex items-center justify-center gap-2"
      >
        <span>🛑</span> Close All Positions
      </button>
      <button
        @click="$emit('reset-daily')"
        class="p-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold transition-colors flex items-center justify-center gap-2"
      >
        <span>🔄</span> Reset Daily Stats
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  riskData: {
    type: Object,
    default: () => ({
      risk_level: "SAFE",
      balance: 10000,
      equity: 10150,
      daily_pnl: 1.5,
      open_positions: 1,
      max_positions: 3,
      risk_per_trade: 2,
      max_daily_loss: 5,
      leverage: 2000,
      risk_score: 25,
      can_trade: true,
      can_open_position: true,
      daily_limit_hit: false,
      losing_streak_limit: false,
      losing_streak: 0,
      max_losing_streak: 5,
    }),
  },
});

const emit = defineEmits(["pause-trading", "close-all", "reset-daily"]);

// Format number helper
const formatNumber = (num) => {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(num);
};

// Computed
const riskLevelEmoji = computed(() => {
  const level = props.riskData.risk_level || "";
  if (level === "SAFE") return "🟢";
  if (level === "CAUTION") return "🟡";
  if (level === "WARNING") return "🟠";
  if (level === "DANGER") return "🔴";
  if (level === "CRITICAL") return "⚫";
  return "⚪";
});

const riskLevelBadgeClass = computed(() => {
  const level = props.riskData.risk_level || "";
  if (level === "SAFE")
    return "px-4 py-2 bg-green-500/20 border border-green-400 rounded-lg text-green-300 font-bold";
  if (level === "CAUTION")
    return "px-4 py-2 bg-yellow-500/20 border border-yellow-400 rounded-lg text-yellow-300 font-bold";
  if (level === "WARNING")
    return "px-4 py-2 bg-orange-500/20 border border-orange-400 rounded-lg text-orange-300 font-bold animate-pulse";
  if (level === "DANGER")
    return "px-4 py-2 bg-red-500/20 border border-red-400 rounded-lg text-red-300 font-bold animate-pulse";
  return "px-4 py-2 bg-red-900/50 border border-red-600 rounded-lg text-red-200 font-bold animate-pulse";
});

const equityClass = computed(() => {
  const diff = (props.riskData.equity || 0) - (props.riskData.balance || 0);
  if (diff > 0) return "text-green-400";
  if (diff < 0) return "text-red-400";
  return "text-blue-400";
});

const dailyPnlClass = computed(() => {
  const pnl = props.riskData.daily_pnl || 0;
  if (pnl > 0) return "text-green-400";
  if (pnl < -2) return "text-red-400 animate-pulse";
  if (pnl < 0) return "text-red-400";
  return "text-gray-400";
});

const dailyPnlBorderClass = computed(() => {
  const pnl = props.riskData.daily_pnl || 0;
  if (pnl > 0) return "border-green-500/20";
  if (pnl < -2) return "border-red-500/50";
  if (pnl < 0) return "border-red-500/20";
  return "border-gray-500/20";
});

const dailyLossPercent = computed(() => {
  const loss = Math.abs(Math.min(0, props.riskData.daily_pnl || 0));
  const max = props.riskData.max_daily_loss || 5;
  return Math.min(100, (loss / max) * 100);
});

const getDailyLossBarClass = computed(() => {
  const pct = dailyLossPercent.value;
  if (pct >= 80) return "bg-red-500";
  if (pct >= 50) return "bg-yellow-500";
  return "bg-green-500";
});

const dailyLossClass = computed(() => {
  const pct = dailyLossPercent.value;
  if (pct >= 80) return "text-red-400";
  if (pct >= 50) return "text-yellow-400";
  return "text-green-400";
});

const isDailyLossWarning = computed(() => {
  return dailyLossPercent.value >= 60;
});

const leverageClass = computed(() => {
  const lev = props.riskData.leverage || 100;
  if (lev >= 500) return "text-red-400";
  if (lev >= 200) return "text-yellow-400";
  return "text-green-400";
});

const leverageWarning = computed(() => {
  const lev = props.riskData.leverage || 100;
  if (lev >= 500) return "⚠️ High Risk Leverage";
  if (lev >= 200) return "⚠️ Moderate Risk";
  return "✓ Conservative";
});

const riskFactors = computed(() => {
  const data = props.riskData;
  return [
    {
      name: "Position Limit",
      value: `${data.open_positions || 0}/${data.max_positions || 3}`,
      status:
        (data.open_positions || 0) >= (data.max_positions || 3)
          ? "DANGER"
          : "OK",
    },
    {
      name: "Daily P/L",
      value: `${(data.daily_pnl || 0).toFixed(2)}%`,
      status:
        (data.daily_pnl || 0) <= -(data.max_daily_loss || 5)
          ? "DANGER"
          : (data.daily_pnl || 0) <= -((data.max_daily_loss || 5) * 0.6)
            ? "WARNING"
            : "OK",
    },
    {
      name: "Losing Streak",
      value: `${data.losing_streak || 0} trades`,
      status:
        (data.losing_streak || 0) >= (data.max_losing_streak || 5)
          ? "DANGER"
          : (data.losing_streak || 0) >= 3
            ? "WARNING"
            : "OK",
    },
    {
      name: "Risk Per Trade",
      value: `${data.risk_per_trade || 2}%`,
      status: (data.risk_per_trade || 2) > 3 ? "WARNING" : "OK",
    },
  ];
});

const riskScoreClass = computed(() => {
  const score = props.riskData.risk_score || 0;
  if (score <= 25) return "text-green-400";
  if (score <= 50) return "text-yellow-400";
  if (score <= 75) return "text-orange-400";
  return "text-red-400";
});

const riskAssessmentBorder = computed(() => {
  const score = props.riskData.risk_score || 0;
  if (score <= 25) return "border-green-500/20";
  if (score <= 50) return "border-yellow-500/20";
  if (score <= 75) return "border-orange-500/20";
  return "border-red-500/50";
});

const losingStreakClass = computed(() => {
  const streak = props.riskData.losing_streak || 0;
  if (streak === 0) return "text-green-400";
  if (streak <= 2) return "text-yellow-400";
  if (streak <= 4) return "text-orange-400";
  return "text-red-400 animate-pulse";
});

const losingStreakBorder = computed(() => {
  const streak = props.riskData.losing_streak || 0;
  if (streak === 0) return "border-green-500/20";
  if (streak <= 2) return "border-yellow-500/20";
  if (streak >= 3) return "border-red-500/30";
  return "border-gray-600";
});

// Helper functions
const getFactorStatusClass = (status) => {
  if (status === "OK") return "text-green-400";
  if (status === "WARNING") return "text-yellow-400";
  return "text-red-400";
};

const getFactorValueClass = (status) => {
  if (status === "OK") return "text-green-400";
  if (status === "WARNING") return "text-yellow-400";
  return "text-red-400";
};

const getPermissionClass = (allowed) => {
  return allowed
    ? "bg-green-500/20 border border-green-400/30"
    : "bg-red-500/20 border border-red-400/30";
};
</script>
